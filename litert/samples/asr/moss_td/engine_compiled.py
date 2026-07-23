# Copyright 2026. Apache-2.0.
"""CompiledModel-based decode engine with buffer-bound (in==out) KV cache.

The decoder's KV tensors live inside LiteRT TensorBuffers that are bound as
BOTH input and output of every prefill/decode invocation, so the cache never
crosses the host boundary. The embedder (embed/logits over the tied matrix)
also runs on CompiledModel to avoid a second resident runtime.
"""

from __future__ import annotations

import re

import numpy as np

NEG_INF = float("-inf")
_KV_PAT = re.compile(r"kv_(k|v)_(\d+)$|kv_cache_(k|v)_(\d+)$")


def _kvkey(name):
    m = _KV_PAT.search(name)
    if not m:
        return None
    return (m.group(1) or m.group(3), int(m.group(2) or m.group(4)))


def _named(names, needle):
    return [j for j, n in enumerate(names) if needle in n][0]


class CompiledDecoder:
    """prefill_*/decode signatures with one shared KV TensorBuffer set."""

    def __init__(self, path: str, hidden: int = 1024):
        from ai_edge_litert import compiled_model as cm_lib

        self.hidden = hidden
        self.cm = cm_lib.CompiledModel.from_file(path)
        sigs = self.cm.get_signature_list()

        self.dec_idx = self.cm.get_signature_index("decode")
        din = sigs["decode"]["inputs"]
        dout = sigs["decode"]["outputs"]
        self.dec_in = self.cm.create_input_buffers(self.dec_idx)
        self.dec_out = self.cm.create_output_buffers(self.dec_idx)
        self.kvbuf = {}
        for j, n in enumerate(din):
            k = _kvkey(n)
            if k is not None:
                self.kvbuf[k] = self.dec_in[j]
        for j, n in enumerate(dout):
            k = _kvkey(n)
            if k is not None:
                self.dec_out[j] = self.kvbuf[k]
        self.d_e = _named(din, "input_embeds")
        self.d_p = _named(din, "input_pos")
        self.d_m = _named(din, "mask")
        self.d_h = _named(dout, "hidden")

        # kv length from decode mask input shape: (1,1,1,KV)
        det = self.cm.get_input_tensor_details("decode")
        mask_name = sigs["decode"]["inputs"][self.d_m]
        self.kv_len = int(det[mask_name]["shape"][-1])
        # KV dtype + zero-init (buffers are not guaranteed zeroed)
        kv0_name = [n for n in din if _kvkey(n)][0]
        kv0 = det[kv0_name]
        self.kv_dtype = np.dtype(str(kv0["dtype"]).replace("<class 'numpy.", "")
                                 .replace("'>", "")) if not isinstance(
                                     kv0["dtype"], type) else np.dtype(kv0["dtype"])
        z = np.zeros(tuple(kv0["shape"]), dtype=self.kv_dtype)
        for buf in self.kvbuf.values():
            buf.write(z)

        self.prefills = {}
        for name in sigs:
            if not name.startswith("prefill_"):
                continue
            n = int(name.split("_")[1])
            idx = self.cm.get_signature_index(name)
            ins = self.cm.create_input_buffers(idx)
            outs = self.cm.create_output_buffers(idx)
            innames = sigs[name]["inputs"]
            outnames = sigs[name]["outputs"]
            for j, nm in enumerate(innames):
                k = _kvkey(nm)
                if k is not None:
                    ins[j] = self.kvbuf[k]
            for j, nm in enumerate(outnames):
                k = _kvkey(nm)
                if k is not None:
                    outs[j] = self.kvbuf[k]
            self.prefills[n] = (idx, ins, outs,
                                _named(innames, "input_embeds"),
                                _named(innames, "input_pos"),
                                _named(innames, "mask"),
                                _named(outnames, "hidden"))

    def prefill(self, fused: np.ndarray):
        """Prefill the whole prompt; returns hidden of the last position."""
        S = fused.shape[0]
        plens = sorted(self.prefills.keys(), reverse=True)
        pos = 0
        last_hidden = None
        while pos < S:
            n = next((l for l in plens if l <= S - pos), plens[-1])
            idx, ins, outs, e_i, p_i, m_i, h_o = self.prefills[n]
            real = min(n, S - pos)
            emb = np.zeros((1, n, self.hidden), dtype=np.float32)
            emb[0, :real] = fused[pos:pos + real]
            mask = np.full((1, 1, n, self.kv_len), NEG_INF, dtype=np.float32)
            for r in range(n):
                mask[0, 0, r, :pos + r + 1] = 0.0
            ins[e_i].write(np.ascontiguousarray(emb))
            ins[p_i].write(np.arange(pos, pos + n, dtype=np.int32))
            ins[m_i].write(np.ascontiguousarray(mask))
            self.cm.run_by_index(idx, ins, outs)
            hid = np.asarray(outs[h_o].read(n * self.hidden, np.float32))
            last_hidden = hid.reshape(n, self.hidden)[real - 1]
            pos += real
        return last_hidden, S

    def step(self, embed: np.ndarray, pos: int) -> np.ndarray:
        """One decode step at absolute position pos; returns hidden (H,)."""
        mask = np.full((1, 1, 1, self.kv_len), NEG_INF, dtype=np.float32)
        mask[0, 0, 0, :pos + 1] = 0.0
        self.dec_in[self.d_e].write(
            np.ascontiguousarray(embed.reshape(1, 1, self.hidden)))
        self.dec_in[self.d_p].write(np.array([pos], dtype=np.int32))
        self.dec_in[self.d_m].write(mask)
        self.cm.run_by_index(self.dec_idx, self.dec_in, self.dec_out)
        return np.asarray(self.dec_out[self.d_h].read(self.hidden, np.float32))


class CompiledEmbedder:
    """embed_N / logits signatures on CompiledModel."""

    def __init__(self, path: str, hidden: int = 1024):
        from ai_edge_litert import compiled_model as cm_lib

        self.hidden = hidden
        self.cm = cm_lib.CompiledModel.from_file(path)
        sigs = self.cm.get_signature_list()
        self.embed_sigs = {}
        for name in sigs:
            if name.startswith("embed_"):
                n = int(name.split("_")[1])
                idx = self.cm.get_signature_index(name)
                ins = self.cm.create_input_buffers(idx)
                outs = self.cm.create_output_buffers(idx)
                self.embed_sigs[n] = (idx, ins, outs)
        li = self.cm.get_signature_index("logits")
        self.l_idx = li
        self.l_in = self.cm.create_input_buffers(li)
        self.l_out = self.cm.create_output_buffers(li)
        det = self.cm.get_output_tensor_details("logits")
        self.vocab = int(list(det.values())[0]["shape"][-1])

    def embed(self, ids) -> np.ndarray:
        lens = sorted(self.embed_sigs.keys(), reverse=True)
        out = np.empty((len(ids), self.hidden), dtype=np.float32)
        i = 0
        while i < len(ids):
            n = next((l for l in lens if l <= len(ids) - i), lens[-1])
            idx, ins, outs = self.embed_sigs[n]
            chunk = list(ids[i:i + n])
            pad = n - len(chunk)
            ins[0].write(np.array([chunk + [0] * pad], dtype=np.int32))
            self.cm.run_by_index(idx, ins, outs)
            res = np.asarray(outs[0].read(n * self.hidden, np.float32))
            out[i:i + len(chunk)] = res.reshape(n, self.hidden)[:len(chunk)]
            i += len(chunk)
        return out

    def logits(self, hidden: np.ndarray) -> np.ndarray:
        self.l_in[0].write(np.ascontiguousarray(
            hidden.reshape(1, 1, self.hidden).astype(np.float32)))
        self.cm.run_by_index(self.l_idx, self.l_in, self.l_out)
        return np.asarray(self.l_out[0].read(self.vocab, np.float32))
