# Copyright 2026. Apache-2.0.
"""Stage-0 parity: re-authored torch modules vs the official MOSS-TD model.

Checks (all in float32 on CPU):
  1. encoder: MossAudioEncoder vs official get_audio_features (max abs diff)
  2. decoder: teacher-forced logits from MossDecoderBody + MossEmbedder vs the
     official full model on a real prompt built from golden audio.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import torch

from moss_td import common


def load_official(snapshot: str, ref_pkg: str | None):
    if ref_pkg:
        sys.path.insert(0, ref_pkg)
    from moss_transcribe_diarize.modeling_moss_transcribe_diarize import (
        MossTranscribeDiarizeForConditionalGeneration,
    )

    model = MossTranscribeDiarizeForConditionalGeneration.from_pretrained(
        snapshot, dtype=torch.float32, attn_implementation="eager")
    model.eval()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--ref-pkg", default="/tmp/claude-1001/ref/MOSS-Transcribe-Diarize")
    ap.add_argument("--wav", default="/tmp/claude-1001/jfk16k.wav")
    ap.add_argument("--steps", type=int, default=8)
    args = ap.parse_args()

    snap = common.resolve_snapshot(args.checkpoint)
    torch.manual_seed(0)

    import soundfile as sf
    from transformers import AutoTokenizer, WhisperFeatureExtractor

    audio, sr = sf.read(args.wav, dtype="float32")
    assert sr == 16000
    fe = WhisperFeatureExtractor.from_pretrained(snap)
    tok = AutoTokenizer.from_pretrained(snap)

    tok_lens = common.chunk_token_lengths(len(audio))
    chunks = []
    for i in range(len(tok_lens)):
        c = audio[i * 480000:(i + 1) * 480000]
        c = np.pad(c, (0, 480000 - len(c)))
        chunks.append(c)
    feats = fe(chunks, sampling_rate=16000, padding="max_length",
               return_tensors="pt")["input_features"]  # (n,80,3000)

    official = load_official(snap, args.ref_pkg)

    # ---- encoder parity ----
    ref_audio = official.get_audio_features(
        input_features=feats,
        audio_feature_lengths=torch.tensor(tok_lens),
        audio_chunk_mapping=torch.zeros(len(tok_lens), dtype=torch.long),
    )[0][0]  # (n_tokens, 1024)

    enc = common.MossAudioEncoder(snap)
    mine = []
    with torch.no_grad():
        for i, tl in enumerate(tok_lens):
            out = enc(feats[i:i + 1])[0, :tl]
            mine.append(out)
    mine = torch.cat(mine, dim=0)
    d = (mine - ref_audio).abs()
    print(f"[encoder] tokens={mine.shape[0]} max_abs_diff={d.max():.3e} "
          f"mean_abs_diff={d.mean():.3e}")

    # ---- decoder parity (teacher forced) ----
    n_audio = sum(tok_lens)
    ids = common.build_input_ids(tok, n_audio)
    input_ids = torch.tensor([ids])
    with torch.no_grad():
        ref_out = official(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            input_features=feats,
            audio_feature_lengths=torch.tensor(tok_lens),
            audio_chunk_mapping=torch.zeros(len(tok_lens), dtype=torch.long),
        )
    ref_logits = ref_out.logits[0]  # (S, vocab)

    emb = common.MossEmbedder(snap)
    kv_len = len(ids) + args.steps + 8
    body = common.MossDecoderBody(snap, kv_len)
    from litert_torch.generative.layers import kv_cache as kv_utils
    kv = kv_utils.KVCache.from_model_config(kv_len, body.config)

    with torch.no_grad():
        fused = emb.embed(input_ids).clone()  # (1,S,1024)
        pos = [i for i, t in enumerate(ids) if t == common.AUDIO_TOKEN_ID]
        assert len(pos) == n_audio
        fused[0, pos] = mine
        S = len(ids)
        mask = torch.full((1, 1, S, kv_len), float("-inf"))
        causal = torch.triu(torch.full((S, S), float("-inf")), diagonal=1)
        mask[0, 0, :, :S] = causal
        out = body(fused, torch.arange(S, dtype=torch.int), mask, kv)
        my_logits = emb.logits(out["hidden"][0])  # (S, vocab)

    dl = (my_logits - ref_logits).abs()
    last = dl[-1]
    print(f"[decoder] prompt_len={S} logits max_abs_diff={dl.max():.3e} "
          f"last_pos max={last.max():.3e}")
    print(f"[decoder] argmax match (all pos): "
          f"{(my_logits.argmax(-1) == ref_logits.argmax(-1)).float().mean():.4f}")

    # greedy continuation comparison for a few steps
    kv2 = out["kv_cache"]
    my_tokens, cur = [], int(my_logits[-1].argmax())
    p = S
    with torch.no_grad():
        for _ in range(args.steps):
            my_tokens.append(cur)
            e = emb.embed(torch.tensor([[cur]]))
            mask = torch.full((1, 1, 1, kv_len), float("-inf"))
            mask[0, 0, 0, :p + 1] = 0.0
            out = body(e, torch.tensor([p], dtype=torch.int), mask, kv2)
            kv2 = out["kv_cache"]
            cur = int(emb.logits(out["hidden"][0])[-1].argmax())
            p += 1

    with torch.no_grad():
        gen = official.generate(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            input_features=feats,
            audio_feature_lengths=torch.tensor(tok_lens),
            audio_chunk_mapping=torch.zeros(len(tok_lens), dtype=torch.long),
            max_new_tokens=args.steps, do_sample=False)
    ref_tokens = gen[0][S:].tolist()
    print(f"[greedy]  mine={my_tokens}")
    print(f"[greedy]  ref ={ref_tokens}")
    print(f"[greedy]  match={my_tokens == ref_tokens}")


if __name__ == "__main__":
    main()
