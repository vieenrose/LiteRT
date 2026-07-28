"""Export the VibeVoice-ASR audio front end to LiteRT (.tflite).

Why: on a Boox Tab Mini C the ggml front end is ~70% of VibeASR's runtime, while
MOSS-TD's comparable LiteRT/XNNPACK encoder runs 2.2x faster than ggml's on the same
device. Generation is the opposite — ggml beats LiteRT by an order of magnitude there.
So the encoder moves to LiteRT and the Qwen2.5 decoder stays on ggml.

The graph is fixed-length: LiteRT wants static shapes, and the runtime already slices
audio into fixed windows (VAE_WINDOW_FRAMES), so a bucket per supported window length
costs nothing. Frames = samples / 3200.

  python export_litert.py --secs 10 --out vibe_front_10s.tflite
"""

import argparse

import torch

import build_encoder

HOP = 3200          # total encoder stride; one output frame per HOP input samples
SR = 24_000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="/home/luigi/VibeASR.cpp/models/vibeasr")
    ap.add_argument("--secs", type=float, default=10.0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--quantize", choices=["none", "dynamic_int8"], default="none")
    args = ap.parse_args()

    n_samples = int(round(args.secs * SR / HOP)) * HOP     # keep it a whole number of frames
    out = args.out or f"vibe_front_{int(args.secs)}s_{args.quantize}.tflite"

    print(f"loading front end from {args.model_dir}")
    model = build_encoder.load(args.model_dir)

    sample = (torch.zeros(1, n_samples),)
    print(f"tracing at {n_samples} samples ({n_samples / SR:.1f}s -> {n_samples // HOP} frames)")

    import litert_torch
    kwargs = {}
    if args.quantize == "dynamic_int8":
        # Weights int8 PER CHANNEL, activations dynamic. Per-channel is the point: the
        # ggml build quantizes I8_S with ONE scale per tensor, which costs cosine 0.885
        # (acoustic) against f32 — and still transcribes correctly. A per-channel int8
        # export should land far inside that budget while being ~4x smaller than f32.
        from litert_torch.quantize import pt2e_quantizer, quant_config as qcfg
        quantizer = pt2e_quantizer.PT2EQuantizer().set_global(
            pt2e_quantizer.get_symmetric_quantization_config(
                is_per_channel=True, is_dynamic=True))
        kwargs["quant_config"] = qcfg.QuantConfig(pt2e_quantizer=quantizer)

        # PT2E needs the graph prepared and converted around a calibration run before
        # the LiteRT conversion sees it.
        from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e
        exported = torch.export.export(model.eval(), sample).module()  # export_for_training was folded into export() in torch 2.9+
        prepared = prepare_pt2e(exported, quantizer)
        with torch.no_grad():
            prepared(*sample)          # dynamic quant: activation ranges come from runtime
        model = convert_pt2e(prepared, fold_quantize=False)

    edge = litert_torch.convert(model, sample, **kwargs)
    edge.export(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
