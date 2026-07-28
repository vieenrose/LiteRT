"""Export one REAL Qwen2.5 attention block through the ternary custom op.

Completes the projection coverage: with the MLP already verified (gate/up/down),
this adds q/k/v/o — all seven ternary projections in a decoder layer.

Scope: ONE decode step with the KV history supplied as an input rather than held in
a persistent aliased cache. That deliberately separates two concerns — the maths
(GQA, RoPE, softmax, four ternary projections) is verified here, leaving cache
aliasing as engine plumbing rather than an unknown. MossLiteEngine already has the
aliased-buffer pattern for that.

Qwen2.5-1.5B: hidden 1536, 12 query heads, 2 KV heads (GQA), head_dim 128,
rope_theta 1e6. q/k/v carry biases; o does not.

  python export_attn_layer.py --layer 0 --past 8
"""

import argparse

import numpy as np
import torch

import gguf_read as G
import ternary_op as T

LM = "/home/luigi/VibeASR.cpp/models/vibeasr/vibeasr-lm-i2_s-embed-q6_k.gguf"
N_HEAD, N_KV, HEAD_DIM, ROPE_THETA = 12, 2, 128, 1_000_000.0


def load_proj(g, name):
    t = g.tensors[name]
    k, n = int(t.dims[0]), int(t.dims[1])
    raw = g.raw(name)
    return G.unpack_i2s(raw, k, n).astype(np.float32), G.i2s_scale(raw, k, n)


def f32(g, name):
    return np.frombuffer(g.raw(name).tobytes(), dtype=np.float32).copy()


def rope(x, pos, theta=ROPE_THETA):
    """Rotary embedding, NeoX-style split-halves, matching Qwen2."""
    d = x.shape[-1]
    half = d // 2
    inv = 1.0 / (theta ** (torch.arange(0, half, dtype=torch.float32) / half))
    ang = pos.float().unsqueeze(-1) * inv          # [T, half]
    cos, sin = ang.cos(), ang.sin()
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


class AttnRef(torch.nn.Module):
    """Dense reference over the same weights — no packing anywhere."""

    def __init__(self, w, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.w = w

    def forward(self, x, k_past, v_past, pos):
        w = self.w
        h = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * w["norm"]
        q = (h @ w["q"].T) * w["qs"] + w["qb"]
        k = (h @ w["k"].T) * w["ks"] + w["kb"]
        v = (h @ w["v"].T) * w["vs"] + w["vb"]
        T_ = h.shape[0]
        q = q.view(T_, N_HEAD, HEAD_DIM)
        k = k.view(T_, N_KV, HEAD_DIM)
        v = v.view(T_, N_KV, HEAD_DIM)
        q = rope(q.transpose(0, 1), pos).transpose(0, 1)
        k = rope(k.transpose(0, 1), pos).transpose(0, 1)
        # Append to history, then GQA-expand the KV heads to match the query heads.
        k_all = torch.cat([k_past, k], dim=0)      # [P+T, N_KV, D]
        v_all = torch.cat([v_past, v], dim=0)
        rep = N_HEAD // N_KV
        k_all = k_all.repeat_interleave(rep, dim=1)
        v_all = v_all.repeat_interleave(rep, dim=1)
        att = torch.einsum("thd,shd->hts", q, k_all) / (HEAD_DIM ** 0.5)
        att = att.softmax(-1)
        o = torch.einsum("hts,shd->thd", att, v_all).reshape(T_, N_HEAD * HEAD_DIM)
        return (o @ w["o"].T) * w["os"]


class AttnCustomOp(AttnRef):
    """Same maths, projections routed through voxsum::ternary_matmul so the export
    carries PACKED weights."""

    def forward(self, x, k_past, v_past, pos, qw, qs, kw, ks, vw, vs, ow, os_):
        w = self.w
        h = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * w["norm"]
        q = torch.ops.voxsum.ternary_matmul(h, qw, qs) + w["qb"]
        k = torch.ops.voxsum.ternary_matmul(h, kw, ks) + w["kb"]
        v = torch.ops.voxsum.ternary_matmul(h, vw, vs) + w["vb"]
        T_ = h.shape[0]
        q = rope(q.view(T_, N_HEAD, HEAD_DIM).transpose(0, 1), pos).transpose(0, 1)
        k = rope(k.view(T_, N_KV, HEAD_DIM).transpose(0, 1), pos).transpose(0, 1)
        v = v.view(T_, N_KV, HEAD_DIM)
        k_all = torch.cat([k_past, k], dim=0).repeat_interleave(N_HEAD // N_KV, dim=1)
        v_all = torch.cat([v_past, v], dim=0).repeat_interleave(N_HEAD // N_KV, dim=1)
        att = torch.einsum("thd,shd->hts", q, k_all) / (HEAD_DIM ** 0.5)
        o = torch.einsum("hts,shd->thd", att.softmax(-1), v_all).reshape(T_, N_HEAD * HEAD_DIM)
        return torch.ops.voxsum.ternary_matmul(o, ow, os_)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--past", type=int, default=8)
    ap.add_argument("--out", default="attn_layer.tflite")
    args = ap.parse_args()

    g = G.Gguf(LM)
    p = f"blk.{args.layer}."
    qw_d, qs = load_proj(g, p + "attn_q.weight")
    kw_d, ks = load_proj(g, p + "attn_k.weight")
    vw_d, vs = load_proj(g, p + "attn_v.weight")
    ow_d, os_ = load_proj(g, p + "attn_output.weight")
    w = {
        "norm": torch.from_numpy(f32(g, p + "attn_norm.weight")),
        "q": torch.from_numpy(qw_d), "qs": qs, "qb": torch.from_numpy(f32(g, p + "attn_q.bias")),
        "k": torch.from_numpy(kw_d), "ks": ks, "kb": torch.from_numpy(f32(g, p + "attn_k.bias")),
        "v": torch.from_numpy(vw_d), "vs": vs, "vb": torch.from_numpy(f32(g, p + "attn_v.bias")),
        "o": torch.from_numpy(ow_d), "os": os_,
    }
    print(f"layer {args.layer}: q{tuple(qw_d.shape)} k{tuple(kw_d.shape)} "
          f"v{tuple(vw_d.shape)} o{tuple(ow_d.shape)}")

    torch.manual_seed(0)
    dim = qw_d.shape[1]
    x = torch.randn(1, dim) * 0.5
    k_past = torch.randn(args.past, N_KV, HEAD_DIM) * 0.3
    v_past = torch.randn(args.past, N_KV, HEAD_DIM) * 0.3
    pos = torch.tensor([args.past])

    with torch.no_grad():
        y_ref = AttnRef(w)(x, k_past, v_past, pos)
    print(f"reference out: rms={y_ref.pow(2).mean().sqrt():.6g}")

    packed = {}
    for tag, dense, sc in (("q", qw_d, qs), ("k", kw_d, ks), ("v", vw_d, vs), ("o", ow_d, os_)):
        packed[tag + "w"] = T.pack_ternary(torch.from_numpy(dense).to(torch.int8))
        packed[tag + "s"] = torch.full((dense.shape[0],), float(sc))

    mod = AttnCustomOp(w).eval()
    sample = (x, k_past, v_past, pos,
              packed["qw"], packed["qs"], packed["kw"], packed["ks"],
              packed["vw"], packed["vs"], packed["ow"], packed["os"])
    with torch.no_grad():
        y_op = mod(*sample)
    cos = torch.nn.functional.cosine_similarity(y_ref, y_op).item()
    print(f"custom-op eager: rms={y_op.pow(2).mean().sqrt():.6g}  cos={cos:.6f}")

    import litert_torch
    litert_torch.convert(mod, sample).export(args.out)
    print(f"wrote {args.out}")

    names = ["attn_x", "attn_kpast", "attn_vpast", "attn_pos",
             "attn_qw", "attn_qs", "attn_kw", "attn_ks",
             "attn_vw", "attn_vs", "attn_ow", "attn_os"]
    for n, t in zip(names, sample):
        arr = t.numpy()
        arr.tofile(f"{n}.bin")
    y_ref.numpy().tofile("attn_expected.bin")
    print("wrote inputs/weights for the C++ runner")


if __name__ == "__main__":
    main()
