"""LiteRT engine for the MOSS-Transcribe-Diarize Space (CPU/XNNPACK).

Self-contained vendored port of litert/samples/asr/moss_td (vieenrose/LiteRT
branch moss-td-port): 3 flatbuffers (encoder q8, tied embedder q8, decoder
int4-b32) run via the LiteRT CompiledModel API with the KV cache held in
TensorBuffers aliased as BOTH input and output of every prefill/decode call --
the cache never crosses the host boundary and exists exactly once.

Parity status of this stack (see the model card):
  * f32 pipeline byte-identical to the project's pinned f32 reference on all
    3 golden clips (jfk / en 5-min / zh 5-min).
  * this int4-b32 decoder: 98.99% text-fidelity (timestamps excluded) /
    95.45% full-text vs the pinned f32 reference on the zh 90s window;
    100% text on jfk.

Exposes the SAME call contract windowing.py consumed from the C engine:
  transcribe(pcm_f32, max_new, on_event) with
    on_event(kind, text, cur, total): kind 1 = phase ("encode"/"prefill"/
    "decode"), kind 0 = token event where text is the FULL partial transcript.
"""
from __future__ import annotations

import re
import time

import numpy as np

SR = 16000
HIDDEN = 1024
AUDIO_TOKEN_ID = 151671
EOS_TOKEN_ID = 151645
AUDIO_TOKENS_PER_SECOND = 12.5
AUDIO_MERGE_SIZE = 4
TIME_MARKER_EVERY_SECONDS = 5  # processor_config.json -- NOT 2
MEL_BINS, MEL_FRAMES, CHUNK_SAMPLES = 80, 3000, 480000
NEG_INF = float("-inf")

DEFAULT_PROMPT = (
    "请将音频转写为文本，每一段需以起始时间戳和说话人编号"
    "（[S01]、[S02]、[S03]…）开头，正文为对应的语音内容，"
    "并在段末标注结束时间戳，以清晰标明该段语音范围。"
)

_KV_PAT = re.compile(r"kv_(k|v)_(\d+)$|kv_cache_(k|v)_(\d+)$")


def _kvkey(name):
    m = _KV_PAT.search(name)
    return None if not m else (m.group(1) or m.group(3),
                               int(m.group(2) or m.group(4)))


def _named(names, needle):
    return [j for j, n in enumerate(names) if needle in n][0]


# --------------------------------------------------------------- prompt ------
def chunk_token_lengths(num_samples: int) -> list[int]:
    stride = 160 * 2 * AUDIO_MERGE_SIZE  # 1280
    return [(min(CHUNK_SAMPLES, num_samples - s) - 1) // stride + 1
            for s in range(0, num_samples, CHUNK_SAMPLES)]


def audio_span_ids(n: int, digit_ids: dict[str, int]) -> list[int]:
    if n <= 0:
        return []
    every = TIME_MARKER_EVERY_SECONDS
    tokens_per_marker = int(AUDIO_TOKENS_PER_SECOND * every)
    out, consumed = [], 0
    for sec in range(every, int(n / AUDIO_TOKENS_PER_SECOND) + 1, every):
        pos = (sec // every) * tokens_per_marker
        seg = pos - consumed
        if seg > 0:
            out.extend([AUDIO_TOKEN_ID] * seg)
            consumed += seg
        out.extend(digit_ids[d] for d in str(sec))
    out.extend([AUDIO_TOKEN_ID] * (n - consumed))
    return out


def build_input_ids(tokenizer, num_audio_tokens: int) -> list[int]:
    digit_ids = {d: tokenizer.encode(d, add_special_tokens=False)[0]
                 for d in "0123456789"}
    prompt = (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\n<|audio_start|><|audio_pad|><|audio_end|>\n"
        + DEFAULT_PROMPT + "<|im_end|>\n<|im_start|>assistant\n")
    before, after = prompt.split("<|audio_pad|>")
    return (tokenizer.encode(before, add_special_tokens=False)
            + audio_span_ids(num_audio_tokens, digit_ids)
            + tokenizer.encode(after, add_special_tokens=False))


# ----------------------------------------------------- compiled components ---
class CompiledDecoder:
    """prefill_*/decode signatures over ONE shared, aliased KV buffer set."""

    def __init__(self, path: str):
        from ai_edge_litert import compiled_model as cm_lib

        self.cm = cm_lib.CompiledModel.from_file(path)
        sigs = self.cm.get_signature_list()
        self.dec_idx = self.cm.get_signature_index("decode")
        din, dout = sigs["decode"]["inputs"], sigs["decode"]["outputs"]
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
        self.d_e, self.d_p = _named(din, "input_embeds"), _named(din, "input_pos")
        self.d_m, self.d_h = _named(din, "mask"), _named(dout, "hidden")

        det = self.cm.get_input_tensor_details("decode")
        self.kv_len = int(det[din[self.d_m]]["shape"][-1])
        kv0 = det[[n for n in din if _kvkey(n)][0]]
        z = np.zeros(tuple(kv0["shape"]), dtype=np.dtype(kv0["dtype"]))
        self._kv_zero = z
        for b in self.kvbuf.values():
            b.write(z)

        self.prefills = {}
        for name in sigs:
            if not name.startswith("prefill_"):
                continue
            n = int(name.split("_")[1])
            idx = self.cm.get_signature_index(name)
            ins = self.cm.create_input_buffers(idx)
            outs = self.cm.create_output_buffers(idx)
            innm, outnm = sigs[name]["inputs"], sigs[name]["outputs"]
            for j, nm in enumerate(innm):
                k = _kvkey(nm)
                if k is not None:
                    ins[j] = self.kvbuf[k]
            for j, nm in enumerate(outnm):
                k = _kvkey(nm)
                if k is not None:
                    outs[j] = self.kvbuf[k]
            self.prefills[n] = (idx, ins, outs, _named(innm, "input_embeds"),
                                _named(innm, "input_pos"), _named(innm, "mask"),
                                _named(outnm, "hidden"))

    def reset(self):
        """Zero the KV cache between windows. Cheap relative to a decode and
        removes any dependence on stale rows (they are masked anyway, but a
        zeroed cache keeps every window bit-reproducible in isolation)."""
        for b in self.kvbuf.values():
            b.write(self._kv_zero)

    def prefill(self, fused: np.ndarray, on_progress=None):
        S = fused.shape[0]
        plens = sorted(self.prefills.keys(), reverse=True)
        pos, last_hidden = 0, None
        while pos < S:
            n = next((l for l in plens if l <= S - pos), plens[-1])
            idx, ins, outs, e_i, p_i, m_i, h_o = self.prefills[n]
            real = min(n, S - pos)
            emb = np.zeros((1, n, HIDDEN), dtype=np.float32)
            emb[0, :real] = fused[pos:pos + real]
            mask = np.full((1, 1, n, self.kv_len), NEG_INF, dtype=np.float32)
            for r in range(n):
                mask[0, 0, r, :pos + r + 1] = 0.0
            ins[e_i].write(np.ascontiguousarray(emb))
            ins[p_i].write(np.arange(pos, pos + n, dtype=np.int32))
            ins[m_i].write(np.ascontiguousarray(mask))
            self.cm.run_by_index(idx, ins, outs)
            hid = np.asarray(outs[h_o].read(n * HIDDEN, np.float32))
            last_hidden = hid.reshape(n, HIDDEN)[real - 1]
            pos += real
            if on_progress is not None:
                on_progress(pos, S)
        return last_hidden, S

    def step(self, embed: np.ndarray, pos: int) -> np.ndarray:
        mask = np.full((1, 1, 1, self.kv_len), NEG_INF, dtype=np.float32)
        mask[0, 0, 0, :pos + 1] = 0.0
        self.dec_in[self.d_e].write(
            np.ascontiguousarray(embed.reshape(1, 1, HIDDEN)))
        self.dec_in[self.d_p].write(np.array([pos], dtype=np.int32))
        self.dec_in[self.d_m].write(mask)
        self.cm.run_by_index(self.dec_idx, self.dec_in, self.dec_out)
        return np.asarray(self.dec_out[self.d_h].read(HIDDEN, np.float32))


class CompiledEmbedder:
    def __init__(self, path: str):
        from ai_edge_litert import compiled_model as cm_lib

        self.cm = cm_lib.CompiledModel.from_file(path)
        sigs = self.cm.get_signature_list()
        self.embed_sigs = {}
        for name in sigs:
            if name.startswith("embed_"):
                idx = self.cm.get_signature_index(name)
                self.embed_sigs[int(name.split("_")[1])] = (
                    idx, self.cm.create_input_buffers(idx),
                    self.cm.create_output_buffers(idx))
        li = self.cm.get_signature_index("logits")
        self.l_idx, self.l_in = li, self.cm.create_input_buffers(li)
        self.l_out = self.cm.create_output_buffers(li)
        det = self.cm.get_output_tensor_details("logits")
        self.vocab = int(list(det.values())[0]["shape"][-1])

    def embed(self, ids) -> np.ndarray:
        lens = sorted(self.embed_sigs.keys(), reverse=True)
        out = np.empty((len(ids), HIDDEN), dtype=np.float32)
        i = 0
        while i < len(ids):
            n = next((l for l in lens if l <= len(ids) - i), lens[-1])
            idx, ins, outs = self.embed_sigs[n]
            chunk = list(ids[i:i + n])
            ins[0].write(np.array([chunk + [0] * (n - len(chunk))],
                                  dtype=np.int32))
            self.cm.run_by_index(idx, ins, outs)
            res = np.asarray(outs[0].read(n * HIDDEN, np.float32))
            out[i:i + len(chunk)] = res.reshape(n, HIDDEN)[:len(chunk)]
            i += len(chunk)
        return out

    def logits(self, hidden: np.ndarray) -> np.ndarray:
        self.l_in[0].write(np.ascontiguousarray(
            hidden.reshape(1, 1, HIDDEN).astype(np.float32)))
        self.cm.run_by_index(self.l_idx, self.l_in, self.l_out)
        return np.asarray(self.l_out[0].read(self.vocab, np.float32))


# ------------------------------------------------------------------ engine ---
class MossLiteRT:
    """Resident engine; one transcribe() call at a time (app serializes)."""

    def __init__(self, encoder_path: str, embedder_path: str,
                 decoder_path: str, tokenizer_dir: str, threads: int = 16):
        from ai_edge_litert.interpreter import Interpreter
        from transformers import AutoTokenizer, WhisperFeatureExtractor

        self.tok = AutoTokenizer.from_pretrained(tokenizer_dir)
        self.fe = WhisperFeatureExtractor.from_pretrained(tokenizer_dir)
        # Encoder stays on the classic Interpreter (single static signature).
        self._enc = Interpreter(model_path=encoder_path, num_threads=threads)
        self._enc_runner = self._enc.get_signature_runner(
            list(self._enc.get_signature_list())[0])
        self._enc_input = list(self._enc_runner.get_input_details())[0]
        self.emb = CompiledEmbedder(embedder_path)
        self.dec = CompiledDecoder(decoder_path)
        self.kv_len = self.dec.kv_len

    # -- windowing.py-facing contract --------------------------------------
    def transcribe(self, pcm: np.ndarray, max_new: int, on_event=None) -> str:
        """on_event(kind, text, cur, total): kind1 phases encode/prefill/
        decode; kind0 tokens with the FULL partial transcript."""
        pcm = np.ascontiguousarray(pcm, dtype=np.float32)
        tok_lens = chunk_token_lengths(len(pcm))

        if on_event:
            on_event(1, "encode", 0, len(tok_lens))
        chunks = [np.pad(pcm[i * CHUNK_SAMPLES:(i + 1) * CHUNK_SAMPLES],
                         (0, max(0, CHUNK_SAMPLES - len(
                             pcm[i * CHUNK_SAMPLES:(i + 1) * CHUNK_SAMPLES]))))
                  for i in range(len(tok_lens))]
        feats = self.fe(chunks, sampling_rate=SR, padding="max_length",
                        return_tensors="np")["input_features"]
        outs = []
        for i, tl in enumerate(tok_lens):
            o = self._enc_runner(**{self._enc_input:
                                    feats[i:i + 1].astype(np.float32)})
            outs.append(list(o.values())[0][0, :tl])
            if on_event:
                on_event(1, "encode", i + 1, len(tok_lens))
        audio_embeds = np.concatenate(outs, axis=0)

        ids = build_input_ids(self.tok, audio_embeds.shape[0])
        fused = self.emb.embed(ids)
        apos = [i for i, t in enumerate(ids) if t == AUDIO_TOKEN_ID]
        fused[apos] = audio_embeds
        S = len(fused)

        if on_event:
            on_event(1, "prefill", 0, S)
        self.dec.reset()
        last_hidden, _ = self.dec.prefill(
            fused, on_progress=(lambda c, t: on_event(1, "prefill", c, t))
            if on_event else None)

        # Static-KV budget: never let generation run past the cache.
        budget = min(max_new, self.kv_len - S - 1)
        if on_event:
            on_event(1, "decode", 0, budget)
        logits = self.emb.logits(last_hidden)
        new_ids: list[int] = []
        p = S
        while len(new_ids) < budget:
            t = int(np.argmax(logits))
            new_ids.append(t)
            if t == EOS_TOKEN_ID:
                break
            if on_event:
                partial = self.tok.decode(new_ids, skip_special_tokens=True)
                on_event(0, partial, len(new_ids), budget)
            e = self.emb.embed([t])
            h = self.dec.step(e[0], p)
            logits = self.emb.logits(h)
            p += 1
        return self.tok.decode(new_ids, skip_special_tokens=True).strip()
