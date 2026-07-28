"""Export an N-layer Qwen2.5 decoder step with a KV cache, on the ternary custom op.

One decode step: a single token in, updated caches out. The caches enter and leave
through the signature so the engine can ALIAS each pair to one TensorBuffer — the
same trick MossLiteEngine uses, which is what keeps the cache off the host boundary.

Weight tensors are inputs, not constants, because the LiteRT dispatcher will not
hand constant tensors to a custom kernel. For 28 layers that is 196 weight inputs
plus 56 scales plus 56 cache pairs; they are bound once and reused every step.

  python export_decoder.py --layers 2 --ctx 64      # validate the pattern
  python export_decoder.py --layers 28 --ctx 512    # the real thing
"""

import argparse

import numpy as np
import torch

from litert_torch.generative.custom_ops.dynamic_update_slice import dynamic_update_slice as dus

import gguf_read as G
import ternary_op as T

LM = "/home/luigi/VibeASR.cpp/models/vibeasr/vibeasr-lm-i2_s-embed-q6_k.gguf"
N_HEAD, N_KV, HEAD_DIM, ROPE_THETA, EPS = 12, 2, 128, 1_000_000.0, 1e-6


def load_proj(g, name):
    """-> (packed int8 [n, k/4], scale). Stays int8 throughout: a float32 detour
    would cost 55 MB per FFN tensor and there are 196 of them."""
    t = g.tensors[name]
    k, n = int(t.dims[0]), int(t.dims[1])
    raw = g.raw(name)
    dense = G.unpack_i2s(raw, k, n)                      # int8 {-1,0,1}
    return T.pack_ternary(torch.from_numpy(dense)), G.i2s_scale(raw, k, n), n


def f32(g, name):
    return torch.from_numpy(np.frombuffer(g.raw(name).tobytes(), dtype=np.float32).copy())


def rope_tables(pos, half=HEAD_DIM // 2):
    """cos/sin for this position, shaped to broadcast over [T, heads, dim].

    Hoisted out of the layer loop: it depends only on `pos`, so computing it inside
    rope() meant 56 identical evaluations per decode step (two per layer).
    """
    inv = 1.0 / (ROPE_THETA ** (torch.arange(0, half, dtype=torch.float32) / half))
    ang = pos.float().reshape(-1, 1) * inv                 # [T, half]
    return ang.cos().reshape(-1, 1, half), ang.sin().reshape(-1, 1, half)


def rope(x, cos, sin):
    """x is [T, heads, dim] — no transposes. The earlier version moved heads to the
    front and back again purely so a [T, half] table would broadcast, which for
    T=1 decode is pure copying."""
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


def rms(x, w):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS) * w


class Decoder(torch.nn.Module):
    """Norm weights and biases stay baked in (they are float and tiny). Only the
    ternary projections come in as arguments, since only they need the custom op."""

    def __init__(self, layers, ctx, meta, n_tok=1):
        super().__init__()
        # n_tok > 1 exports a PREFILL graph: the same weights and the same maths,
        # but T tokens ingested in one pass. Decode remains n_tok=1. Prompt
        # ingestion is otherwise T sequential decode steps, each re-reading all
        # 328 MB of weights, when one batched pass reads them once.
        self.L, self.ctx, self.T = layers, ctx, n_tok
        for i, m in enumerate(meta):
            for k, v in m.items():
                self.register_buffer(f"l{i}_{k}", v)

    def forward(self, x, pos, *rest):
        # rest = per layer: qkv_w,qkv_s, o_w,o_s, gateup_w,gateup_s, down_w,down_s
        # then the interleaved k,v caches.
        n_w = self.L * 8
        W, caches = rest[:n_w], rest[n_w:]
        k_out, v_out = [], []
        # Causal mask over the cache: only slots already written (< pos+1) count.
        idx = torch.arange(self.ctx).reshape(1, 1, -1)
        mask = (idx <= pos.reshape(-1, 1, 1)).float()
        neg = (1.0 - mask) * -1e30
        cos, sin = rope_tables(pos)

        for i in range(self.L):
            qkv_w, qkv_s, ow, os_, gu_w, gu_s, dw, ds = W[i * 8:(i + 1) * 8]
            k_cache, v_cache = caches[2 * i], caches[2 * i + 1]

            h = rms(x, getattr(self, f"l{i}_attn_norm"))
            qkv = torch.ops.voxsum.ternary_matmul(h, qkv_w, qkv_s)
            dq, dk = N_HEAD * HEAD_DIM, N_KV * HEAD_DIM
            q = qkv[:, :dq] + getattr(self, f"l{i}_qb")
            k = qkv[:, dq:dq + dk] + getattr(self, f"l{i}_kb")
            v = qkv[:, dq + dk:] + getattr(self, f"l{i}_vb")

            q = rope(q.view(self.T, N_HEAD, HEAD_DIM), cos, sin)
            k = rope(k.view(self.T, N_KV, HEAD_DIM), cos, sin)
            v = v.view(self.T, N_KV, HEAD_DIM)

            # Write this step into the cache at `pos`. dynamic_update_slice touches
            # ONE row; the obvious one-hot scatter
            #     k_cache * (1 - hot) + k * hot
            # is O(ctx) instead, reading and rewriting the whole cache per layer per
            # token. Measured at ctx=64 that cost ~2 ms/layer on top of the block
            # time, and it grows linearly with context.
            # Indices must be 0-D scalars; a shape-[1] tensor is rejected with
            # "operand #2 must be variadic of 0D tensor".
            p0 = pos[0]   # batch is contiguous, so it starts at pos[0]
            z0 = torch.zeros_like(p0)
            k_new = dus(k_cache, k, (p0, z0, z0))
            v_new = dus(v_cache, v, (p0, z0, z0))
            k_out.append(k_new)
            v_out.append(v_new)

            # GQA without materializing expanded K/V: group the queries instead.
            qg = q.view(self.T, N_KV, N_HEAD // N_KV, HEAD_DIM)
            att = torch.einsum("tgnd,sgd->gnts", qg, k_new) / (HEAD_DIM ** 0.5)
            # att is [group, head_in_group, T, ctx]; neg is [T, 1, ctx]. Broadcast
            # it as [1, 1, T, ctx] so each query row masks the slots after its own
            # position — which is also what makes prefill causal WITHIN the batch,
            # since pos = 0..T-1.
            att = (att + neg[:, 0, :].reshape(1, 1, self.T, self.ctx)).softmax(-1)
            o = torch.einsum("gnts,sgd->tgnd", att, v_new).reshape(self.T, N_HEAD * HEAD_DIM)
            x = x + torch.ops.voxsum.ternary_matmul(o, ow, os_)

            h2 = rms(x, getattr(self, f"l{i}_ffn_norm"))
            gu = torch.ops.voxsum.ternary_matmul(h2, gu_w, gu_s)
            ff = gu.shape[-1] // 2
            x = x + torch.ops.voxsum.ternary_matmul(
                torch.nn.functional.silu(gu[:, :ff]) * gu[:, ff:], dw, ds)

        return (x, *k_out, *v_out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--ctx", type=int, default=64)
    ap.add_argument("--out", default=None)
    # A REAL embedding, not noise. Real rows have rms ~0.02-0.05; torch.randn*0.5
    # is ~20x larger, and with residual connections that compounds through the
    # stack (rms 21.7 by layer 8) and wrecks the int8 activation quantization.
    ap.add_argument("--token", type=int, default=9707)
    ap.add_argument("--tokens", type=int, default=1,
                    help=">1 exports a batched PREFILL graph instead of a decode step")
    args = ap.parse_args()
    out = args.out or (f"decoder_{args.layers}L_{args.ctx}.tflite" if args.tokens == 1
                       else f"prefill_{args.layers}L_{args.ctx}_t{args.tokens}.tflite")

    g = G.Gguf(LM)
    meta, weights = [], []
    for i in range(args.layers):
        p = f"blk.{i}."
        m = {"attn_norm": f32(g, p + "attn_norm.weight"),
             "ffn_norm": f32(g, p + "ffn_norm.weight"),
             "qb": f32(g, p + "attn_q.bias"),
             "kb": f32(g, p + "attn_k.bias"),
             "vb": f32(g, p + "attn_v.bias")}
        meta.append(m)
        # FUSE projections that share an input. q/k/v all consume the same post-norm
        # h, and gate/up both consume h2, so concatenating their rows turns 3 custom
        # ops into 1 and 2 into 1: seven per layer become four.
        #
        # This is about PARTITION COUNT, not arithmetic. Each custom op is opaque to
        # XNNPACK and splits the graph, and 7 of them per layer produced 282
        # partitions across 28 layers — every boundary hands control back to the
        # interpreter. Row concatenation is free with this packing (rows are
        # independent) and per-row scales absorb the differing per-tensor scales.
        def fused(names):
            ps, ss = [], []
            for nm in names:
                packed, sc, rows = load_proj(g, p + nm + ".weight")
                ps.append(packed)
                ss.append(torch.full((rows,), float(sc)))
            return torch.cat(ps, dim=0), torch.cat(ss, dim=0)

        for group in (["attn_q", "attn_k", "attn_v"], ["attn_output"],
                      ["ffn_gate", "ffn_up"], ["ffn_down"]):
            pw, ps_ = fused(group)
            weights.append(pw)
            weights.append(ps_)
        print(f"  layer {i} loaded", end="\r")
    print(f"loaded {args.layers} layers ({len(weights)} weight tensors)      ")

    dim = 1536
    T = args.tokens
    rows = [G.token_embedding(g, args.token + i, dim) for i in range(T)]
    x = torch.from_numpy(np.stack(rows)).reshape(T, dim)
    print(f"input: {T} token(s) from {args.token}, rms={x.pow(2).mean().sqrt():.5f}")
    pos = torch.arange(T)
    caches = []
    for _ in range(args.layers):
        caches.append(torch.zeros(args.ctx, N_KV, HEAD_DIM))
    for _ in range(args.layers):
        caches.append(torch.zeros(args.ctx, N_KV, HEAD_DIM))
    # interleave k,v per layer to match the forward signature
    inter = []
    for i in range(args.layers):
        inter += [caches[i], caches[args.layers + i]]

    mod = Decoder(args.layers, args.ctx, meta, n_tok=T).eval()
    sample = (x, pos, *weights, *inter)
    with torch.no_grad():
        y = mod(*sample)
    print(f"eager ok: out rms={y[0].pow(2).mean().sqrt():.6g}, "
          f"{len(y) - 1} cache outputs")

    import litert_torch
    litert_torch.convert(mod, sample).export(out)
    print(f"wrote {out}")

    tag = "dec" if T == 1 else f"pre{T}"
    np.asarray(y[0].numpy()).tofile(f"{tag}_{args.layers}L_expected.bin")
    names = [f"{tag}_x.bin", f"{tag}_pos.bin"]
    x.numpy().tofile(f"{tag}_x.bin")
    pos.numpy().astype(np.int64).tofile(f"{tag}_pos.bin")
    # Weights and caches are IDENTICAL between decode and prefill — same tensors,
    # same order — so the engine loads them once and binds them to both graphs.
    for j, t in enumerate(weights):
        if T == 1: t.numpy().tofile(f"dec_w{j:03d}.bin")
        names.append(f"dec_w{j:03d}.bin")
    for j, t in enumerate(inter):
        if T == 1: t.numpy().tofile(f"dec_c{j:03d}.bin")
        names.append(f"dec_c{j:03d}.bin")
    with open(f"{tag}_{args.layers}L_manifest.txt", "w") as f:
        f.write("\n".join(names) + "\n")
    print(f"wrote {len(names)} input files + manifest")


if __name__ == "__main__":
    main()
