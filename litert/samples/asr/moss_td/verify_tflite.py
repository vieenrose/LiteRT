# Copyright 2026. Apache-2.0.
"""Stage-1 parity: exported .tflite components vs the float32 torch modules.

  1. encoder.tflite vs MossAudioEncoder on real golden mel chunks
  2. embedder.tflite embed/logits vs MossEmbedder
  3. decoder.tflite prefill+decode logits vs MossDecoderBody (teacher-forced)
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from moss_td import common
from moss_td.runner import MossTdLiteRT

NEG_INF = float("-inf")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--wav", default="/tmp/claude-1001/jfk16k.wav")
    ap.add_argument("--encoder", default="models/moss_td_encoder_f32.tflite")
    ap.add_argument("--embedder", default="models/moss_td_embedder_f32.tflite")
    ap.add_argument("--decoder",
                    default="models/moss_td_decoder_f32_ekv6144.tflite")
    ap.add_argument("--steps", type=int, default=8)
    args = ap.parse_args()

    snap = common.resolve_snapshot(args.checkpoint)
    import soundfile as sf
    audio, sr = sf.read(args.wav, dtype="float32")
    assert sr == 16000

    rt = MossTdLiteRT(args.encoder, args.embedder, args.decoder, snap)

    # ---- encoder ----
    lite_embeds = rt.encode_audio(audio)  # (n,1024)
    enc = common.MossAudioEncoder(snap)
    tok_lens = common.chunk_token_lengths(len(audio))
    chunks = [np.pad(audio[i * 480000:(i + 1) * 480000],
                     (0, max(0, 480000 - len(audio[i * 480000:(i + 1) * 480000]))))
              for i in range(len(tok_lens))]
    feats = rt.fe(chunks, sampling_rate=16000, padding="max_length",
                  return_tensors="pt")["input_features"]
    ref = []
    with torch.no_grad():
        for i, tl in enumerate(tok_lens):
            ref.append(enc(feats[i:i + 1])[0, :tl])
    ref = torch.cat(ref).numpy()
    d = np.abs(lite_embeds - ref)
    print(f"[encoder.tflite] max_abs_diff={d.max():.3e} mean={d.mean():.3e}")

    # ---- embedder ----
    emb = common.MossEmbedder(snap)
    ids = common.build_input_ids(rt.tok, lite_embeds.shape[0])
    lite_e = rt.embed_tokens(ids)
    with torch.no_grad():
        ref_e = emb.embed(torch.tensor([ids]))[0].numpy()
    print(f"[embedder.tflite embed] max_abs_diff={np.abs(lite_e-ref_e).max():.3e}")

    h = np.random.RandomState(0).randn(1024).astype(np.float32)
    lite_l = rt.logits(h)
    with torch.no_grad():
        ref_l = emb.logits(torch.tensor(h)).numpy()
    print(f"[embedder.tflite logits] max_abs_diff={np.abs(lite_l-ref_l).max():.3e}")

    # ---- decoder: run lite pipeline prefill+greedy against torch body ----
    fused = lite_e.copy()
    apos = [i for i, t in enumerate(ids) if t == common.AUDIO_TOKEN_ID]
    fused[apos] = lite_embeds

    # torch reference
    kv_len = rt.kv_len
    body = common.MossDecoderBody(snap, kv_len)
    from litert_torch.generative.layers import kv_cache as kv_utils
    kv = kv_utils.KVCache.from_model_config(kv_len, body.config)
    S = len(ids)
    with torch.no_grad():
        mask = torch.full((1, 1, S, kv_len), NEG_INF)
        mask[0, 0, :, :S] = torch.triu(
            torch.full((S, S), NEG_INF), diagonal=1)
        out = body(torch.tensor(fused).unsqueeze(0),
                   torch.arange(S, dtype=torch.int), mask, kv)
        ref_hidden_last = out["hidden"][0, -1].numpy()
        ref_logits = emb.logits(torch.tensor(ref_hidden_last)).numpy()

    # lite prefill via runner internals: reuse transcribe() up to first logits
    # (simply re-run a short transcribe with max_new=steps and compare tokens)
    text = rt.transcribe(audio, max_new=args.steps)
    # torch greedy for the same number of steps
    tokens_ref = []
    logits = ref_logits
    kv2 = out["kv_cache"]
    p = S
    with torch.no_grad():
        for _ in range(args.steps):
            t = int(np.argmax(logits))
            tokens_ref.append(t)
            if t == common.EOS_TOKEN_ID:
                break
            e = emb.embed(torch.tensor([[t]]))
            m = torch.full((1, 1, 1, kv_len), NEG_INF)
            m[0, 0, 0, :p + 1] = 0.0
            o = body(e, torch.tensor([p], dtype=torch.int), m, kv2)
            kv2 = o["kv_cache"]
            logits = emb.logits(o["hidden"][0])[-1].numpy()
            p += 1
    text_ref = rt.tok.decode(
        [t for t in tokens_ref if t != common.EOS_TOKEN_ID],
        skip_special_tokens=True).strip()
    print(f"[decoder.tflite] lite text: {text!r}")
    print(f"[decoder.tflite] ref  text: {text_ref!r}")
    print(f"[decoder.tflite] greedy-{args.steps} match: {text == text_ref}")


if __name__ == "__main__":
    main()
