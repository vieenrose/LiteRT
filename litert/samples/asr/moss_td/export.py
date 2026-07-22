# Copyright 2026. Apache-2.0.
"""Export MOSS-TD components to LiteRT (.tflite).

Usage:
  python -m moss_td.export --component encoder  --quantize none --out models/
  python -m moss_td.export --component embedder --quantize none --out models/
  python -m moss_td.export --component decoder  --quantize none \
      --kv-cache-max-len 6144 --prefill-lens 128,1024 --out models/

Quantize options: none (f32), fp16, dynamic_int8.
"""

from __future__ import annotations

import argparse
import os

import torch

import litert_torch
from litert_torch.generative.quantize import quant_recipes

from moss_td import common


def quant_config(name: str, model_config=None):
    if name == "none":
        return None
    if name == "fp16":
        return quant_recipes.full_fp16_recipe()
    if name == "dynamic_int8":
        return quant_recipes.full_dynamic_recipe(mcfg=model_config)
    raise ValueError(name)


def suffix(name: str) -> str:
    return {"none": "f32", "fp16": "fp16", "dynamic_int8": "q8"}[name]


def export_encoder(snap, out_dir, quantize):
    enc = common.MossAudioEncoder(snap)
    sample = (torch.zeros(1, common.MEL_BINS, common.MEL_FRAMES),)
    edge = litert_torch.convert(enc.eval(), sample,
                                quant_config=quant_config(quantize))
    path = os.path.join(out_dir, f"moss_td_encoder_{suffix(quantize)}.tflite")
    edge.export(path)
    return path


def export_embedder(snap, out_dir, quantize, embed_lens=(1, 128)):
    emb = common.MossEmbedder(snap)
    conv = litert_torch.signature(
        f"embed_{embed_lens[0]}", common.EmbedWrapper(emb),
        (torch.zeros(1, embed_lens[0], dtype=torch.int),))
    for l in embed_lens[1:]:
        conv = conv.signature(f"embed_{l}", common.EmbedWrapper(emb),
                              (torch.zeros(1, l, dtype=torch.int),))
    conv = conv.signature("logits", common.LogitsWrapper(emb),
                          (torch.zeros(1, 1, common.HIDDEN),))
    edge = conv.convert(quant_config=quant_config(quantize))
    path = os.path.join(out_dir, f"moss_td_embedder_{suffix(quantize)}.tflite")
    edge.export(path)
    return path


def export_decoder(snap, out_dir, quantize, kv_len, prefill_lens):
    from litert_torch.generative.layers import kv_cache as kv_utils

    body = common.MossDecoderBody(snap, kv_len)
    kv = kv_utils.KVCache.from_model_config(kv_len, body.config)

    conv = None
    for S in prefill_lens:
        kwargs = {
            "input_embeds": torch.zeros(1, S, common.HIDDEN),
            "input_pos": torch.arange(0, S, dtype=torch.int),
            "mask": torch.zeros(1, 1, S, kv_len),
            "kv_cache": kv,
        }
        name = f"prefill_{S}"
        conv = (litert_torch.signature(name, body, sample_kwargs=kwargs)
                if conv is None else
                conv.signature(name, body, sample_kwargs=kwargs))
    dec_kwargs = {
        "input_embeds": torch.zeros(1, 1, common.HIDDEN),
        "input_pos": torch.zeros(1, dtype=torch.int),
        "mask": torch.zeros(1, 1, 1, kv_len),
        "kv_cache": kv,
    }
    conv = conv.signature("decode", body, sample_kwargs=dec_kwargs)
    edge = conv.convert(quant_config=quant_config(quantize, body.config))
    path = os.path.join(
        out_dir, f"moss_td_decoder_{suffix(quantize)}_ekv{kv_len}.tflite")
    edge.export(path)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--component", required=True,
                    choices=["encoder", "embedder", "decoder"])
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--quantize", default="none",
                    choices=["none", "fp16", "dynamic_int8"])
    ap.add_argument("--kv-cache-max-len", type=int, default=6144)
    ap.add_argument("--prefill-lens", default="128,1024")
    ap.add_argument("--out", default="models")
    args = ap.parse_args()

    snap = common.resolve_snapshot(args.checkpoint)
    os.makedirs(args.out, exist_ok=True)
    if args.component == "encoder":
        p = export_encoder(snap, args.out, args.quantize)
    elif args.component == "embedder":
        p = export_embedder(snap, args.out, args.quantize)
    else:
        lens = [int(x) for x in args.prefill_lens.split(",")]
        p = export_decoder(snap, args.out, args.quantize,
                           args.kv_cache_max_len, lens)
    print("exported:", p, f"{os.path.getsize(p)/1e6:.1f} MB")


if __name__ == "__main__":
    main()
