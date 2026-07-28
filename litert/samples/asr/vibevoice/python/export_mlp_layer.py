"""Export one REAL Qwen2.5 MLP block through the ternary custom op, and check it.

Why the MLP first: of the 46.8 M weights in a decoder layer, 41.3 M (88%) are the
three MLP projections. Decode is bandwidth-bound, so verifying the MLP verifies most
of the cost — and it needs no RoPE or KV cache, so a mistake here is a mistake in the
ternary path itself rather than in attention plumbing.

Weights come straight out of the shipped I2_S GGUF via gguf_read.unpack_i2s (layout
established against ggml itself, cosine 1.000000). Nothing is requantized.

  python export_mlp_layer.py --layer 0
"""

import argparse

import numpy as np
import torch

import gguf_read as G
import ternary_op as T

LM = "/home/luigi/VibeASR.cpp/models/vibeasr/vibeasr-lm-i2_s-embed-q6_k.gguf"


def load_proj(g, name):
    """-> (dense ternary [out, in] float32, scale). GGUF dims are (in, out)."""
    t = g.tensors[name]
    k, n = int(t.dims[0]), int(t.dims[1])
    raw = g.raw(name)
    return G.unpack_i2s(raw, k, n).astype(np.float32), G.i2s_scale(raw, k, n)


class TernaryMlpRef(torch.nn.Module):
    """Dense reference: exactly what the block computes, no packing involved."""

    def __init__(self, norm_w, gate, gate_s, up, up_s, down, down_s, eps=1e-6):
        super().__init__()
        self.eps = eps
        for k, v in dict(norm_w=norm_w, gate=gate, up=up, down=down).items():
            self.register_buffer(k, torch.from_numpy(v))
        self.gate_s, self.up_s, self.down_s = gate_s, up_s, down_s

    def forward(self, x):
        h = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.norm_w
        g = (h @ self.gate.T) * self.gate_s
        u = (h @ self.up.T) * self.up_s
        return (torch.nn.functional.silu(g) * u) @ self.down.T * self.down_s


class TernaryMlpCustomOp(torch.nn.Module):
    """Same block, projections routed through voxsum::ternary_matmul so the export
    carries PACKED weights instead of expanding them to int8."""

    def __init__(self, norm_w, gate, gate_s, up, up_s, down, down_s, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.register_buffer("norm_w", torch.from_numpy(norm_w))
        for tag, w, s in (("gate", gate, gate_s), ("up", up, up_s), ("down", down, down_s)):
            packed = T.pack_ternary(torch.from_numpy(w).to(torch.int8))
            self.register_buffer(f"{tag}_w", packed)
            # Per-tensor scale broadcast to per-row: the kernel takes one scale per
            # output channel, and I2_S stores a single scale for the whole tensor.
            self.register_buffer(f"{tag}_s", torch.full((w.shape[0],), float(s)))

    def _proj(self, x, tag):
        return torch.ops.voxsum.ternary_matmul(
            x, getattr(self, f"{tag}_w"), getattr(self, f"{tag}_s"))

    def forward(self, x, gate_w, gate_s, up_w, up_s, down_w, down_s):
        h = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.norm_w
        g = torch.ops.voxsum.ternary_matmul(h, gate_w, gate_s)
        u = torch.ops.voxsum.ternary_matmul(h, up_w, up_s)
        return torch.ops.voxsum.ternary_matmul(
            torch.nn.functional.silu(g) * u, down_w, down_s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--out", default="mlp_layer.tflite")
    args = ap.parse_args()

    g = Gguf = G.Gguf(LM)
    p = f"blk.{args.layer}."
    gate, gate_s = load_proj(g, p + "ffn_gate.weight")
    up, up_s = load_proj(g, p + "ffn_up.weight")
    down, down_s = load_proj(g, p + "ffn_down.weight")
    norm_w = np.frombuffer(g.raw(p + "ffn_norm.weight").tobytes(), dtype=np.float32).copy()
    print(f"layer {args.layer}: gate{gate.shape} up{up.shape} down{down.shape} "
          f"scales={gate_s:.5g}/{up_s:.5g}/{down_s:.5g}")

    dim = gate.shape[1]
    x = torch.randn(1, dim) * 0.5

    ref = TernaryMlpRef(norm_w, gate, gate_s, up, up_s, down, down_s).eval()
    with torch.no_grad():
        y_ref = ref(x)
    print(f"reference out: rms={y_ref.pow(2).mean().sqrt():.6g}")

    mod = TernaryMlpCustomOp(norm_w, gate, gate_s, up, up_s, down, down_s).eval()
    # Weights as ARGUMENTS: the LiteRT dispatcher will not hand constant tensors to
    # a custom kernel, so they have to enter through the signature.
    sample = (x, mod.gate_w, mod.gate_s, mod.up_w, mod.up_s, mod.down_w, mod.down_s)
    with torch.no_grad():
        y_op = mod(*sample)
    print(f"custom-op eager: rms={y_op.pow(2).mean().sqrt():.6g}  "
          f"cos={torch.nn.functional.cosine_similarity(y_ref, y_op).item():.6f}")

    import litert_torch
    litert_torch.convert(mod, sample).export(args.out)
    print(f"wrote {args.out}")

    np.save("mlp_input.npy", x.numpy())
    np.save("mlp_expected.npy", y_ref.numpy())
    for tag in ("gate", "up", "down"):
        getattr(mod, f"{tag}_w").numpy().tofile(f"mlp_{tag}_w.bin")
        getattr(mod, f"{tag}_s").numpy().tofile(f"mlp_{tag}_s.bin")
    print("wrote inputs/weights for the C++ runner")


if __name__ == "__main__":
    main()
