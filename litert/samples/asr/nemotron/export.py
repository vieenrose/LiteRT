#!/usr/bin/env python3
# Copyright 2026. Apache-2.0.
"""Export nvidia/nemotron-3.5-asr-streaming-0.6b (transformers) to LiteRT as a
q4-mix build: INT4 encoder + fp32 decoder/joint.

The transformers FastConformer-RNNT traces cleanly through litert_torch.convert
(no torch rebuild needed). Three graphs mirror the reference split:

  encoder  : input_features (1,T,128) f32       -> hidden (1,T',1024) f32
  decoder  : token (1,1) i32 + h,c (2,1,640)    -> dec_out (1,1,640) + h,c
  joint    : enc (1,1,1024) + dec (1,1,640)     -> logits (1,1,13088)

Only FULLY_CONNECTED weights in the encoder are quantized to INT4 (blockwise-128,
min_max) — convs / norms / the prompt_projector stay fp32 (INT4 on the RNN-T
decoder/joint or the prompt projector collapses the model). With an optional QAT
checkpoint (label-based RNN-T self-distillation), pre-baking the fake-quant grid
before export makes the exported INT4 reproduce the QAT weights (near-lossless).

Usage:
  python -m nemotron.export --component encoder --checkpoint qat_q4mix_enc.pt --prebake --out models/
  python -m nemotron.export --component decoder --out models/
  python -m nemotron.export --component joint   --out models/

Requires: transformers>=5.13, litert_torch (ai-edge-torch) 0.9.1,
ai_edge_quantizer 0.7.0, ai-edge-litert==2.1.5 (the quantizer pins 2.1.5).
"""
from __future__ import annotations
import argparse, os
import torch, torch.nn as nn, transformers, litert_torch

MODEL_ID = "nvidia/nemotron-3.5-asr-streaming-0.6b"
BLOCK = 128


def load_model(checkpoint=None):
    Model = getattr(transformers, "Nemotron3_5AsrForRNNT")
    m = Model.from_pretrained(MODEL_ID, dtype=torch.float32).eval()
    if checkpoint:
        sd = torch.load(checkpoint, map_location="cpu")
        remap = {k.replace(".lin.weight", ".weight").replace(".lin.bias", ".bias"): v
                 for k, v in sd.items()}
        _, unexpected = m.load_state_dict(remap, strict=False)
        assert not unexpected, f"unexpected checkpoint keys: {unexpected[:3]}"
    return m


def fake_quant_int4(W, group=BLOCK):
    """Symmetric blockwise-INT4 (amax/7, -8..7) along the input dim — the QAT grid."""
    out, inn = W.shape
    if group and inn % group == 0:
        Wg = W.float().view(out, inn // group, group)
        s = Wg.abs().amax(2, keepdim=True).clamp_min(1e-8) / 7.0
        return (torch.clamp(torch.round(Wg / s), -8, 7) * s).view(out, inn).to(W.dtype)
    s = W.float().abs().amax(1, keepdim=True).clamp_min(1e-8) / 7.0
    return (torch.clamp(torch.round(W.float() / s), -8, 7) * s).to(W.dtype)


class EncOffline(nn.Module):
    def __init__(self, enc): super().__init__(); self.enc = enc
    def forward(self, input_features):
        return self.enc(input_features, num_lookahead_tokens=3).last_hidden_state


class DecStep(nn.Module):
    def __init__(self, dec):
        super().__init__()
        self.embedding, self.lstm, self.proj = dec.embedding, dec.lstm, dec.decoder_projector
    def forward(self, token, h, c):
        out, (h2, c2) = self.lstm(self.embedding(token), (h, c))
        return self.proj(out), h2, c2


class JointStep(nn.Module):
    """Folds encoder_projector (1024->640) so joint takes the raw 1024 encoder dim."""
    def __init__(self, model):
        super().__init__()
        self.encoder_projector, self.joint = model.encoder_projector, model.joint
    def forward(self, enc, dec):
        return self.joint(decoder_hidden_states=dec, encoder_hidden_states=self.encoder_projector(enc))


class PromptFuse(nn.Module):
    """Language conditioning: hidden[1,Tp,1024] + one_hot[1,128] -> fused[1,Tp,1024].
    fp32 (INT4 collapses it). Runs once per utterance, between encoder and greedy."""
    def __init__(self, model): super().__init__(); self.pp = model.prompt_projector
    def forward(self, hidden, one_hot):
        oh = one_hot[:, None, :].expand(-1, hidden.shape[1], -1)
        return self.pp(torch.cat([hidden, oh], -1))


def int4_fc_recipe(block=BLOCK):
    from ai_edge_quantizer import recipe, qtyping
    from ai_edge_quantizer.recipe import AlgorithmName
    r = recipe.dynamic_wi4_afp32()
    for e in r:
        e["algorithm_key"] = AlgorithmName.MIN_MAX_UNIFORM_QUANT
        e["operation"] = qtyping.TFLOperationName.FULLY_CONNECTED
        e["op_config"]["weight_tensor_config"]["granularity"] = "BLOCKWISE"
        e["op_config"]["weight_tensor_config"]["block_size"] = block
    return r


def fp16_recipe():
    from ai_edge_quantizer import recipe, qtyping
    from ai_edge_quantizer.recipe import AlgorithmName
    r = recipe.dynamic_wi8_afp32()
    for e in r:
        e["algorithm_key"] = AlgorithmName.FLOAT_CASTING
        e["operation"] = qtyping.TFLOperationName.ALL_SUPPORTED
        e["op_config"]["weight_tensor_config"]["num_bits"] = 16
        e["op_config"]["weight_tensor_config"]["dtype"] = qtyping.TensorDataType.FLOAT
        e["op_config"]["compute_precision"] = qtyping.ComputePrecision.FLOAT
    return r


def _to_fp16(fp32_path):
    """fp16-cast a converted fp32 graph (numerically lossless; halves size)."""
    from ai_edge_quantizer import quantizer
    q = quantizer.Quantizer(float_model=fp32_path)
    q.load_quantization_recipe(fp16_recipe())
    out = fp32_path.replace("_fp32.tflite", "_fp16.tflite")
    with open(out, "wb") as f:
        f.write(q.quantize().quantized_model)
    os.remove(fp32_path)
    return out


def export_encoder(m, out_dir, T, prebake):
    if prebake:
        n = 0
        for _, mod in m.encoder.named_modules():
            if isinstance(mod, nn.Linear):
                with torch.no_grad():
                    mod.weight.copy_(fake_quant_int4(mod.weight))
                n += 1
        print(f"pre-baked QAT fake-quant into {n} encoder Linears")
    wrap = EncOffline(m.encoder).eval()
    fp32 = os.path.join(out_dir, "nemotron_encoder_fp32.tflite")
    litert_torch.convert(wrap, (torch.zeros(1, T, 128),), quant_config=None).export(fp32)
    from ai_edge_quantizer import quantizer
    q = quantizer.Quantizer(float_model=fp32)
    q.load_quantization_recipe(int4_fc_recipe())
    out = os.path.join(out_dir, "nemotron_encoder_q4.tflite")
    with open(out, "wb") as f:
        f.write(q.quantize().quantized_model)
    os.remove(fp32)
    print(f"encoder -> {out} ({os.path.getsize(out)/1e6:.0f} MB)")


def export_decoder(m, out_dir, fp16=True):
    dec = DecStep(m.decoder).eval()
    s = (torch.zeros(1, 1, dtype=torch.int32), torch.zeros(2, 1, 640), torch.zeros(2, 1, 640))
    out = os.path.join(out_dir, "nemotron_decoder_fp32.tflite")
    litert_torch.convert(dec, s, quant_config=None).export(out)
    if fp16:
        out = _to_fp16(out)
    print(f"decoder -> {out} ({os.path.getsize(out)/1e6:.0f} MB)")


def export_joint(m, out_dir, fp16=True):
    jnt = JointStep(m).eval()
    out = os.path.join(out_dir, "nemotron_joint_fp32.tflite")
    litert_torch.convert(jnt, (torch.zeros(1, 1, 1024), torch.zeros(1, 1, 640)), quant_config=None).export(out)
    if fp16:
        out = _to_fp16(out)
    print(f"joint -> {out} ({os.path.getsize(out)/1e6:.0f} MB)")


def export_prompt_fuse(m, out_dir, T):
    with torch.no_grad():
        Tp = EncOffline(m.encoder)(torch.zeros(1, T, 128)).shape[1]  # encoder output frames for this T
    pf = PromptFuse(m).eval()
    out = os.path.join(out_dir, "nemotron_prompt_fuse_fp32.tflite")
    litert_torch.convert(pf, (torch.zeros(1, Tp, 1024), torch.zeros(1, 128)), quant_config=None).export(out)
    print(f"prompt_fuse -> {out} ({os.path.getsize(out)/1e6:.1f} MB, Tp={Tp})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--component", required=True,
                    choices=["encoder", "decoder", "joint", "prompt_fuse", "all"])
    ap.add_argument("--checkpoint", default=None, help="QAT encoder.layers state_dict (optional)")
    ap.add_argument("--prebake", action="store_true", help="bake QAT fake-quant grid before export")
    ap.add_argument("--T", type=int, default=1101, help="fixed encoder mel-frame length (offline)")
    ap.add_argument("--keep-fp32", action="store_true", help="keep decoder/joint at fp32 (default fp16)")
    ap.add_argument("--out", default="models")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    fp16 = not a.keep_fp32
    m = load_model(a.checkpoint)
    if a.component in ("encoder", "all"):
        export_encoder(m, a.out, a.T, a.prebake)
    if a.component in ("decoder", "all"):
        export_decoder(m, a.out, fp16)
    if a.component in ("joint", "all"):
        export_joint(m, a.out, fp16)
    if a.component in ("prompt_fuse", "all"):
        export_prompt_fuse(m, a.out, a.T)


if __name__ == "__main__":
    main()
