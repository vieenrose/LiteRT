"""Export output_norm + the int8 LM head: hidden [1,1536] -> logits [1,151936].

Separate graph from the layers, for two reasons: the head's weights bake in as
CONSTANTS (no custom op is involved, so the dispatcher's constant restriction does
not apply), and keeping it separate lets the layer graph be re-exported for
different context buckets without redoing a 233 MB quantization.
"""
import argparse
import numpy as np
import torch
import gguf_read as G

LM = "/home/luigi/VibeASR.cpp/models/vibeasr/vibeasr-lm-i2_s-embed-q6_k.gguf"
DIM, VOCAB, EPS = 1536, 151936, 1e-6


class Head(torch.nn.Module):
    def __init__(self, norm_w, w):
        super().__init__()
        self.register_buffer("norm_w", norm_w)
        self.lin = torch.nn.Linear(DIM, w.shape[0], bias=False)
        with torch.no_grad():
            self.lin.weight.copy_(w)

    def forward(self, x):
        h = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS) * self.norm_w
        return self.lin(h)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=VOCAB, help="vocab slice, for quick tests")
    ap.add_argument("--out", default="head_q8.tflite")
    args = ap.parse_args()

    g = G.Gguf(LM)
    norm_w = torch.from_numpy(
        np.frombuffer(g.raw("output_norm.weight").tobytes(), dtype=np.float32).copy())
    w = np.frombuffer(g.raw("output.weight").tobytes(), dtype=np.float16)
    w = w[: args.rows * DIM].reshape(args.rows, DIM).astype(np.float32)
    print(f"head {w.shape}: f16 {args.rows*DIM*2/1e6:.0f} MB -> int8 ~{args.rows*DIM/1e6:.0f} MB")

    m = Head(norm_w, torch.from_numpy(w)).eval()
    x = torch.randn(1, DIM) * 0.5
    with torch.no_grad():
        ref = m(x)

    from litert_torch.quantize import pt2e_quantizer, quant_config as qcfg
    from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e
    import litert_torch
    qz = pt2e_quantizer.PT2EQuantizer().set_global(
        pt2e_quantizer.get_symmetric_quantization_config(is_per_channel=True, is_dynamic=True))
    ep = torch.export.export(m, (x,)).module()
    prep = prepare_pt2e(ep, qz)
    with torch.no_grad():
        prep(x)
    qm = convert_pt2e(prep, fold_quantize=False)
    litert_torch.convert(qm, (x,), quant_config=qcfg.QuantConfig(pt2e_quantizer=qz)).export(args.out)

    import os
    print(f"wrote {args.out} ({os.path.getsize(args.out)/1e6:.1f} MB)")
    x.numpy().tofile("head_x.bin")
    ref.numpy().tofile("head_expected.bin")
    open("head_manifest.txt", "w").write("head_x.bin\n")


if __name__ == "__main__":
    main()
