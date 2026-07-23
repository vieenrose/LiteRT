# Copyright 2026. Apache-2.0.
"""Host-side LiteRT pipeline runner for MOSS-Transcribe-Diarize.

Mirrors the reference C++ pipeline (rapidspeech moss_td) exactly:
  wav -> 30s-chunked Whisper mels -> encoder.tflite -> per-chunk slice ->
  prompt build (time-marker interleaved audio span) -> embed + masked_scatter
  -> chunked prefill -> greedy decode until EOS/max_new -> detokenize.
"""

from __future__ import annotations

import argparse
import re
import time

import numpy as np

from moss_td import common

NEG_INF = float("-inf")


class Sig:
    """Small wrapper around a LiteRT signature runner."""

    def __init__(self, interp, name):
        self.runner = interp.get_signature_runner(name)
        self.inputs = self.runner.get_input_details()
        self.outputs = self.runner.get_output_details()

    def __call__(self, **kwargs):
        return self.runner(**kwargs)


def load_interpreter(path, threads):
    from ai_edge_litert.interpreter import Interpreter
    return Interpreter(model_path=path, num_threads=threads)


class KvIo:
    """Maps flattened KV-cache input/output tensor names of a signature."""

    def __init__(self, sig: Sig):
        pat = re.compile(r"kv_(k|v)_(\d+)$|kv_cache_(k|v)_(\d+)$")
        self.in_names = {}
        for name in sig.inputs:
            m = pat.search(name)
            if m:
                kind = m.group(1) or m.group(3)
                idx = int(m.group(2) or m.group(4))
                self.in_names[(kind, idx)] = name
        self.out_names = {}
        for name in sig.outputs:
            m = pat.search(name)
            if m:
                kind = m.group(1) or m.group(3)
                idx = int(m.group(2) or m.group(4))
                self.out_names[(kind, idx)] = name
        assert self.in_names and self.in_names.keys() == self.out_names.keys(), (
            list(sig.inputs), list(sig.outputs))

    def zeros(self, sig: Sig):
        return {k: np.zeros(sig.inputs[v]["shape"], dtype=np.float32)
                for k, v in self.in_names.items()}


class MossTdLiteRT:
    def __init__(self, encoder_path, embedder_path, decoder_path,
                 checkpoint=None, threads=8):
        from transformers import AutoTokenizer, WhisperFeatureExtractor

        snap = common.resolve_snapshot(checkpoint)
        self.tok = AutoTokenizer.from_pretrained(snap)
        self.fe = WhisperFeatureExtractor.from_pretrained(snap)

        enc_i = load_interpreter(encoder_path, threads)
        enc_names = list(enc_i.get_signature_list())
        self.enc = Sig(enc_i, enc_names[0])
        self.enc_input = list(self.enc.inputs)[0]

        emb_i = load_interpreter(embedder_path, threads)
        emb_sigs = emb_i.get_signature_list()
        self.embed_sigs = {}
        for name in emb_sigs:
            if name.startswith("embed_"):
                self.embed_sigs[int(name.split("_")[1])] = Sig(emb_i, name)
        self.logits_sig = Sig(emb_i, "logits")

        dec_i = load_interpreter(decoder_path, threads)
        dec_sigs = dec_i.get_signature_list()
        self.prefills = {}
        for name in dec_sigs:
            if name.startswith("prefill_"):
                self.prefills[int(name.split("_")[1])] = Sig(dec_i, name)
        self.decode_sig = Sig(dec_i, "decode")
        self.kv_io = KvIo(self.decode_sig)
        kv_shape = self.decode_sig.inputs[self.kv_io.in_names[("k", 0)]]["shape"]
        self.kv_len = int(kv_shape[1])  # BTNH layout: (1, kv_len, 8, 128)
        self.timings = {}

    # -- stages ------------------------------------------------------------

    def encode_audio(self, audio: np.ndarray) -> np.ndarray:
        tok_lens = common.chunk_token_lengths(len(audio))
        outs = []
        t0 = time.perf_counter()
        chunks = []
        for i in range(len(tok_lens)):
            c = audio[i * 480000:(i + 1) * 480000]
            chunks.append(np.pad(c, (0, 480000 - len(c))))
        feats = self.fe(chunks, sampling_rate=16000, padding="max_length",
                        return_tensors="np")["input_features"]
        self.timings["mel_s"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        for i, tl in enumerate(tok_lens):
            out = self.enc(**{self.enc_input:
                              feats[i:i + 1].astype(np.float32)})
            arr = list(out.values())[0]  # (1,375,1024)
            outs.append(arr[0, :tl])
        self.timings["encoder_s"] = time.perf_counter() - t0
        return np.concatenate(outs, axis=0)  # (n_audio, 1024)

    def embed_tokens(self, ids: list[int]) -> np.ndarray:
        """Batch embed with the largest available embed signature."""
        lens = sorted(self.embed_sigs.keys(), reverse=True)
        out = np.empty((len(ids), common.HIDDEN), dtype=np.float32)
        i = 0
        while i < len(ids):
            n = next((l for l in lens if l <= len(ids) - i), lens[-1])
            sig = self.embed_sigs[n]
            chunk = ids[i:i + n]
            pad = n - len(chunk)
            arr = np.array([chunk + [0] * pad], dtype=np.int32)
            name = list(sig.inputs)[0]
            res = list(sig(**{name: arr}).values())[0]  # (1,n,1024)
            out[i:i + len(chunk)] = res[0, :len(chunk)]
            i += len(chunk)
        return out

    def logits(self, hidden: np.ndarray) -> np.ndarray:
        name = list(self.logits_sig.inputs)[0]
        res = self.logits_sig(**{name: hidden.reshape(1, 1, -1)
                                 .astype(np.float32)})
        return list(res.values())[0].reshape(-1)

    # -- pipeline ----------------------------------------------------------

    def transcribe(self, audio: np.ndarray, max_new=5120, prompt=None,
                   progress=False, free_encoder=False):
        audio_embeds = self.encode_audio(audio)
        if free_encoder:  # low-memory devices: drop encoder before decode
            self.enc = None
            import gc
            gc.collect()
        n_audio = audio_embeds.shape[0]
        ids = common.build_input_ids(self.tok, n_audio,
                                     prompt or common.DEFAULT_PROMPT)
        S = len(ids)
        assert S + max(1, 1) <= self.kv_len, (S, self.kv_len)

        t0 = time.perf_counter()
        fused = self.embed_tokens(ids)
        apos = [i for i, t in enumerate(ids) if t == common.AUDIO_TOKEN_ID]
        assert len(apos) == n_audio
        fused[apos] = audio_embeds
        self.timings["fuse_s"] = time.perf_counter() - t0

        # chunked prefill
        t0 = time.perf_counter()
        kv = self.kv_io.zeros(self.decode_sig)
        plens = sorted(self.prefills.keys(), reverse=True)
        pos = 0
        last_hidden = None
        while pos < S:
            n = next((l for l in plens if l <= S - pos), plens[-1])
            sig = self.prefills[n]
            kvio = KvIo(sig)
            real = min(n, S - pos)
            emb = np.zeros((1, n, common.HIDDEN), dtype=np.float32)
            emb[0, :real] = fused[pos:pos + real]
            ipos = np.arange(pos, pos + n, dtype=np.int32)
            mask = np.full((1, 1, n, self.kv_len), NEG_INF, dtype=np.float32)
            for r in range(n):
                mask[0, 0, r, :pos + r + 1] = 0.0
            feed = {kvio.in_names[k]: kv[k] for k in kvio.in_names}
            name_of = {tuple(k): v for k, v in kvio.in_names.items()}
            inp = {"input_embeds": emb, "input_pos": ipos, "mask": mask}
            # map generic arg names to actual signature input names
            call = dict(feed)
            for want, arr in inp.items():
                match = [nm for nm in sig.inputs if want in nm]
                assert match, (want, list(sig.inputs))
                call[match[0]] = arr
            out = sig(**call)
            for k, nm in kvio.out_names.items():
                kv[k] = out[nm]
            hid = [v for nm, v in out.items() if "hidden" in nm][0]
            last_hidden = hid[0, real - 1]
            pos += real
        self.timings["prefill_s"] = time.perf_counter() - t0
        self.timings["prompt_tokens"] = S

        # greedy decode
        t0 = time.perf_counter()
        logits = self.logits(last_hidden)
        new_ids = []
        dec = self.decode_sig
        dkv = KvIo(dec)
        arg_embeds = [nm for nm in dec.inputs if "input_embeds" in nm][0]
        arg_pos = [nm for nm in dec.inputs if "input_pos" in nm][0]
        arg_mask = [nm for nm in dec.inputs if "mask" in nm][0]
        p = S
        while True:
            t = int(np.argmax(logits))
            new_ids.append(t)
            if t == common.EOS_TOKEN_ID or len(new_ids) >= max_new:
                break
            if p + 1 > self.kv_len:
                break
            emb = self.embed_tokens([t]).reshape(1, 1, -1)
            mask = np.full((1, 1, 1, self.kv_len), NEG_INF, dtype=np.float32)
            mask[0, 0, 0, :p + 1] = 0.0
            call = {dkv.in_names[k]: kv[k] for k in dkv.in_names}
            call[arg_embeds] = emb
            call[arg_pos] = np.array([p], dtype=np.int32)
            call[arg_mask] = mask
            out = dec(**call)
            for k, nm in dkv.out_names.items():
                kv[k] = out[nm]
            hid = [v for nm, v in out.items() if "hidden" in nm][0]
            logits = self.logits(hid[0, 0])
            p += 1
            if progress and len(new_ids) % 50 == 0:
                print(f"  ...{len(new_ids)} tokens", flush=True)
        self.timings["decode_s"] = time.perf_counter() - t0
        self.timings["new_tokens"] = len(new_ids)

        text = self.tok.decode(new_ids, skip_special_tokens=True).strip()
        return text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", required=True)
    ap.add_argument("--encoder", required=True)
    ap.add_argument("--embedder", required=True)
    ap.add_argument("--decoder", required=True)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--max-new", type=int, default=5120)
    ap.add_argument("--out", default=None)
    ap.add_argument("--progress", action="store_true")
    ap.add_argument("--free-encoder", action="store_true",
                    help="release the encoder interpreter after audio encode "
                         "(low-memory devices)")
    args = ap.parse_args()

    import soundfile as sf
    audio, sr = sf.read(args.wav, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    assert sr == 16000, f"expected 16 kHz, got {sr}"

    t0 = time.perf_counter()
    rt = MossTdLiteRT(args.encoder, args.embedder, args.decoder,
                      args.checkpoint, args.threads)
    load_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    text = rt.transcribe(audio, max_new=args.max_new, progress=args.progress,
                         free_encoder=args.free_encoder)
    total = time.perf_counter() - t0
    print(text)
    import sys
    stats = dict(rt.timings)
    stats.update(load_s=round(load_s, 2), total_s=round(total, 2),
                 audio_s=round(len(audio) / 16000, 2))
    print("STATS " + " ".join(f"{k}={v if isinstance(v,int) else round(v,3)}"
                              for k, v in stats.items()), file=sys.stderr)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text + "\n")


if __name__ == "__main__":
    main()
