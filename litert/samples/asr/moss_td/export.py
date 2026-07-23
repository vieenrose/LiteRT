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
    if name == "dynamic_int4_block32":
        from litert_torch.generative.quantize import quant_attrs
        return quant_recipes.full_dynamic_recipe(
            mcfg=model_config, weight_dtype=quant_attrs.Dtype.INT4,
            granularity=quant_attrs.Granularity.BLOCKWISE_32)
    if name == "dynamic_int4_block128":
        from litert_torch.generative.quantize import quant_attrs
        return quant_recipes.full_dynamic_recipe(
            mcfg=model_config, weight_dtype=quant_attrs.Dtype.INT4,
            granularity=quant_attrs.Granularity.BLOCKWISE_128)
    raise ValueError(name)


def suffix(name: str) -> str:
    return {"none": "f32", "fp16": "fp16", "dynamic_int8": "q8",
            "dynamic_int4_block32": "q4b32",
            "dynamic_int4_block128": "q4b128"}[name]


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


def apply_f16_kv_patch():
    """Store the KV cache in float16: cast K/V slices to f16 before the
    dynamic_update_slice and cast the (updated) cache back to f32 for the
    attention math. Halves KV I/O tensors and their runtime buffers."""
    import torch
    import litert_torch.generative.layers.sdpa_with_kv_update as skv
    from litert_torch.generative.layers import kv_cache as kv_utils
    from litert_torch.generative.layers import scaled_dot_product_attention as sdpa

    def _f16_default(query, key, value, kv, input_pos, mask, config,
                     enable_hlfb, alibi_bias=None):
        b, seq_len, _, _ = query.shape
        if kv is not None:
            kv = kv_utils.update(kv, input_pos,
                                 key.to(torch.float16),
                                 value.to(torch.float16))
            key = kv.k_cache.to(torch.float32)
            value = kv.v_cache.to(torch.float32)
        sdpa_func = (sdpa.scaled_dot_product_attention_with_hlfb
                     if enable_hlfb else sdpa.scaled_dot_product_attention)
        out = sdpa_func(query, key, value, config.head_dim, mask=mask,
                        softcap=config.logit_softcap, alibi_bias=alibi_bias)
        return out.reshape(b, seq_len, -1), kv

    skv._sdpa_with_kv_update_default = _f16_default


def export_decoder(snap, out_dir, quantize, kv_len, prefill_lens,
                   kv_dtype="f32"):
    import torch as _t
    from litert_torch.generative.layers import kv_cache as kv_utils

    if kv_dtype == "f16":
        apply_f16_kv_patch()
    body = common.MossDecoderBody(snap, kv_len)
    kv = kv_utils.KVCache.from_model_config(
        kv_len, body.config,
        dtype=_t.float16 if kv_dtype == "f16" else _t.float32)

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
    kvtag = "_kvf16" if kv_dtype == "f16" else ""
    path = os.path.join(
        out_dir, f"moss_td_decoder_{suffix(quantize)}{kvtag}_ekv{kv_len}.tflite")
    edge.export(path)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--component", required=True,
                    choices=["encoder", "embedder", "decoder"])
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--quantize", default="none",
                    choices=["none", "fp16", "dynamic_int8",
                             "dynamic_int4_block32", "dynamic_int4_block128"])
    ap.add_argument("--kv-cache-max-len", type=int, default=6144)
    ap.add_argument("--prefill-lens", default="128,1024")
    ap.add_argument("--kv-dtype", default="f32", choices=["f32", "f16"])
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
                           args.kv_cache_max_len, lens, args.kv_dtype)
    print("exported:", p, f"{os.path.getsize(p)/1e6:.1f} MB")


if __name__ == "__main__":
    main()
