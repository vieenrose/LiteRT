# Copyright 2026. Apache-2.0.
"""Compose per-signature on-device latencies into a pipeline estimate.

Inputs: per-signature avg latencies (us) measured with the LiteRT
android_aarch64_benchmark_model tool, plus the per-clip pipeline counts
(mel chunks, prompt tokens, generated tokens) printed by the host runner.

Estimate:
  wall = n_chunks * t_encoder
       + n_prefill_1024 * t_prefill_1024 + n_prefill_128 * t_prefill_128
       + n_gen * (t_decode + t_logits + t_embed_1)
       + t_logits          (first logits after prefill)

Host-side mel/tokenizer/argmax cost is negligible next to these terms.
"""

from __future__ import annotations

import argparse


def plan_prefill(prompt: int, lens=(1024, 128)) -> dict[int, int]:
    """Greedy chunking identical to runner.py."""
    counts = {l: 0 for l in lens}
    pos = 0
    while pos < prompt:
        n = next((l for l in lens if l <= prompt - pos), lens[-1])
        counts[n] += 1
        pos += min(n, prompt - pos)
    return counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", required=True)
    ap.add_argument("--audio-s", type=float, required=True)
    ap.add_argument("--n-chunks", type=int, required=True)
    ap.add_argument("--prompt-tokens", type=int, required=True)
    ap.add_argument("--gen-tokens", type=int, required=True)
    # measured on-device signature latencies, milliseconds
    ap.add_argument("--enc-ms", type=float, required=True)
    ap.add_argument("--prefill1024-ms", type=float, required=True)
    ap.add_argument("--prefill128-ms", type=float, required=True)
    ap.add_argument("--decode-ms", type=float, required=True)
    ap.add_argument("--logits-ms", type=float, required=True)
    ap.add_argument("--embed1-ms", type=float, default=0.0)
    args = ap.parse_args()

    counts = plan_prefill(args.prompt_tokens)
    enc = args.n_chunks * args.enc_ms
    pre = (counts[1024] * args.prefill1024_ms + counts[128] * args.prefill128_ms)
    per_tok = args.decode_ms + args.logits_ms + args.embed1_ms
    dec = args.gen_tokens * per_tok + args.logits_ms
    total_s = (enc + pre + dec) / 1e3
    print(f"clip={args.clip} audio={args.audio_s:.1f}s")
    print(f"  encoder : {args.n_chunks} x {args.enc_ms:.0f} ms = {enc/1e3:.1f} s")
    print(f"  prefill : {counts[1024]}x1024 + {counts[128]}x128 = {pre/1e3:.1f} s")
    print(f"  decode  : {args.gen_tokens} tok x {per_tok:.0f} ms = {dec/1e3:.1f} s"
          f"  ({1e3/per_tok:.2f} tok/s)")
    print(f"  TOTAL est {total_s:.1f} s  RTF {total_s/args.audio_s:.2f}")


if __name__ == "__main__":
    main()
