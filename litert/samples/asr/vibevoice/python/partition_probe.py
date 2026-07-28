"""Does interleaving ternary custom ops break XNNPACK delegation of the ops between them?

Custom ops are opaque to the XNNPACK delegate. In a real decoder ~224 ternary
projections sit between norms, RoPE and softmax; if each custom op splits the graph
and drops its neighbours out of delegation, the ternary win gets handed straight
back. A one-op toy graph cannot show this.

Three graphs, so the cost of interleaving is isolated rather than inferred:

  norms      L x (RMSNorm -> GELU)                  fully delegable
  ternary    L x (ternary_matmul)                   custom ops only
  mixed      L x (ternary_matmul -> RMSNorm -> GELU)

If mixed ~= norms + ternary there is no partitioning penalty. If mixed is much
larger, the delegate is losing the ops between the custom ops.

  python partition_probe.py --layers 12 --dim 1024
"""

import argparse

import torch

import ternary_op as T


class Norms(torch.nn.Module):
    def __init__(self, layers, dim):
        super().__init__()
        self.norms = torch.nn.ModuleList([torch.nn.RMSNorm(dim) for _ in range(layers)])

    def forward(self, x):
        for n in self.norms:
            x = torch.nn.functional.gelu(n(x))
        return x


# Weights arrive as ARGUMENTS, not buffers: the dispatcher will not hand constant
# tensors to a custom kernel, so a constant-weight version of these graphs cannot
# run at all. One weight tensor is shared across layers to keep the signature small
# — interleaving is what is under test, not weight variety.
class Ternary(torch.nn.Module):
    def __init__(self, layers, dim):
        super().__init__()
        self.layers = layers

    def forward(self, x, w, s):
        for _ in range(self.layers):
            x = torch.ops.voxsum.ternary_matmul(x, w, s)
        return x


class Mixed(torch.nn.Module):
    def __init__(self, layers, dim):
        super().__init__()
        self.norms = torch.nn.ModuleList([torch.nn.RMSNorm(dim) for _ in range(layers)])
        self.layers = layers

    def forward(self, x, w, s):
        for i in range(self.layers):
            x = torch.ops.voxsum.ternary_matmul(x, w, s)
            x = torch.nn.functional.gelu(self.norms[i](x))
        return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=12)
    ap.add_argument("--dim", type=int, default=1024)
    args = ap.parse_args()

    import litert_torch
    torch.manual_seed(0)
    x = torch.randn(1, args.dim)

    w = T.pack_ternary(torch.randint(-1, 2, (args.dim, args.dim), dtype=torch.int8))
    sc = torch.full((args.dim,), 0.02)
    for name, mod, sample in [
        ("norms", Norms(args.layers, args.dim), (x,)),
        ("ternary", Ternary(args.layers, args.dim), (x, w, sc)),
        ("mixed", Mixed(args.layers, args.dim), (x, w, sc)),
    ]:
        out = f"probe_{name}.tflite"
        litert_torch.convert(mod.eval(), sample).export(out)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
