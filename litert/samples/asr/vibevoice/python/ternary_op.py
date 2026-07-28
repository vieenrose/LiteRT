"""Torch custom op + LiteRT lowering for ternary (BitNet) matmul.

This is half (b) of the plan: teach the exporter to emit an OPAQUE op carrying
PACKED ternary weights, instead of letting the default path expand them to int8.

Why it matters: batch-1 decode is memory-bandwidth-bound, so per-token time is
weight_bytes / bandwidth. On a Boox Tab Mini C the achievable bandwidth measures
~2.6 GB/s, which for a 1.31 B-parameter decoder is ~124 ms/token at 2 bits packed
versus ~496 ms/token at int8. The packing IS the performance; a normal export
keeps the ternary values and throws the packing away.

The lowering emits stablehlo.custom_call, which the LiteRT converter turns into a
tfl.custom op. At runtime that dispatches to the kernel registered through
LiteRtAddCustomOpKernelOption (exported by the stock prebuilt libLiteRt.so — no
LiteRT fork needed), which calls src/ternary_gemm.cc.

Weights ride as an int8 tensor operand (raw bytes) in the layout ternary_pack() writes:
u = w + 1, four per byte, so the runtime can consume them with no repacking.
"""

import numpy as np
import torch
from litert_torch.backend import lowerings
from litert_converter.mlir import ir
from litert_converter.mlir.dialects import stablehlo

TERNARY_CUSTOM_CALL = "voxsum.ternary_matmul"


def pack_ternary(w: torch.Tensor) -> torch.Tensor:
    """[n_rows, k] in {-1,0,+1} -> uint8 [n_rows, k/4], matching ternary_pack()."""
    if w.shape[-1] % 4:
        raise ValueError(f"k must be a multiple of 4, got {w.shape[-1]}")
    u = (w.to(torch.int16) + 1).clamp_(0, 3)                      # {-1,0,1}->{0,1,2}
    q = u.reshape(w.shape[0], -1, 4)
    packed = q[..., 0] | (q[..., 1] << 2) | (q[..., 2] << 4) | (q[..., 3] << 6)
    # Carried as INT8, not uint8: the converter rejects uint8 graph tensors
    # ("RuntimeError: torch.uint8"). These are opaque bytes either way — the
    # runtime reads them through a uint8_t*, so the reinterpretation is free and
    # the top bit of a 4-code byte simply reads back as a sign bit here.
    return packed.to(torch.uint8).view(torch.int8).contiguous()


@torch.library.custom_op("voxsum::ternary_matmul", mutates_args=())
def ternary_matmul(
    x: torch.Tensor,          # [m, k] float activations
    packed_w: torch.Tensor,   # [n_rows, k/4] int8 (raw bytes), packed u = w+1
    w_scale: torch.Tensor,    # [n_rows] float, per output channel
) -> torch.Tensor:
    """Eager reference. Unpacks and does a plain matmul — correctness only, no
    attempt at speed; the packed path exists for the exported graph."""
    n_rows, kq = packed_w.shape
    k = kq * 4
    p = packed_w.view(torch.uint8).to(torch.int16)
    codes = torch.stack([(p >> s) & 0x3 for s in (0, 2, 4, 6)], dim=-1)  # [n,k/4,4]
    w = codes.reshape(n_rows, k).to(x.dtype) - 1.0
    return (x @ w.T) * w_scale.to(x.dtype)


@ternary_matmul.register_fake
def _(x, packed_w, w_scale):
    return torch.empty(x.shape[:-1] + (packed_w.shape[0],), dtype=x.dtype, device=x.device)


@lowerings.lower(torch.ops.voxsum.ternary_matmul)
def _ternary_matmul_lower(lctx, x: ir.Value, packed_w: ir.Value, w_scale: ir.Value):
    """Emit an opaque custom_call so the converter cannot decompose it — the whole
    point is that the packed uint8 operand survives to the runtime intact."""
    out_ty = ir.RankedTensorType.get(
        (*x.type.shape[:-1], packed_w.type.shape[0]), x.type.element_type
    )
    return stablehlo.custom_call(
        result=[out_ty],
        inputs=[x, packed_w, w_scale],
        call_target_name=TERNARY_CUSTOM_CALL,
        has_side_effect=False,
    )


class TernaryLinear(torch.nn.Module):
    """Drop-in for nn.Linear whose weights are ternary. Bias stays float and is
    added outside the custom op, so the kernel only has to do the GEMM."""

    def __init__(self, w_ternary: torch.Tensor, w_scale: torch.Tensor, bias=None):
        super().__init__()
        self.register_buffer("packed_w", pack_ternary(w_ternary))
        self.register_buffer("w_scale", w_scale.float())
        self.register_buffer("bias", bias.float() if bias is not None else None)

    def forward(self, x):
        y = torch.ops.voxsum.ternary_matmul(x, self.packed_w, self.w_scale)
        return y + self.bias if self.bias is not None else y
