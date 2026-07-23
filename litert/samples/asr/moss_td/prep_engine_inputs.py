# Copyright 2026. Apache-2.0.
"""Dump mel.bin / lens.bin / ids.bin for the C++ engine; decode tokens.bin."""

from __future__ import annotations

import argparse

import numpy as np

from moss_td import common


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav")
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--decode-tokens", default=None,
                    help="tokens.bin to detokenize instead of prepping")
    args = ap.parse_args()

    from transformers import AutoTokenizer, WhisperFeatureExtractor
    snap = common.resolve_snapshot(args.checkpoint)
    tok = AutoTokenizer.from_pretrained(snap)

    if args.decode_tokens:
        ids = np.fromfile(args.decode_tokens, dtype=np.int32).tolist()
        print(tok.decode(ids, skip_special_tokens=True).strip())
        return

    import soundfile as sf
    fe = WhisperFeatureExtractor.from_pretrained(snap)
    audio, sr = sf.read(args.wav, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    assert sr == 16000
    tok_lens = common.chunk_token_lengths(len(audio))
    chunks = [np.pad(audio[i * 480000:(i + 1) * 480000],
                     (0, 480000 - len(audio[i * 480000:(i + 1) * 480000])))
              for i in range(len(tok_lens))]
    feats = fe(chunks, sampling_rate=16000, padding="max_length",
               return_tensors="np")["input_features"].astype(np.float32)
    feats.tofile(args.prefix + "_mel.bin")
    np.array(tok_lens, dtype=np.int32).tofile(args.prefix + "_lens.bin")
    ids = np.array(common.build_input_ids(tok, sum(tok_lens)), dtype=np.int32)
    ids.tofile(args.prefix + "_ids.bin")
    print(f"wrote {args.prefix}_mel.bin ({feats.shape}), lens {tok_lens}, "
          f"ids {len(ids)}")


if __name__ == "__main__":
    main()
