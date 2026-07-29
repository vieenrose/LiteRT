#!/usr/bin/env python3
"""Compact PLE table generator: int8 or int4 with per-column symmetric scales.

Source: model.safetensors bf16 tensor described by ple.json (raw values;
the global x16.0 dequant scale is carried in the header and applied by the
engine on top of the per-column scale).

Format (extends make_ple_table.py's PLETBL01):
  header (32 B): magic "PLETBL01", u32 dtype (3=int8, 4=int4),
                 u32 rows, u32 cols, f32 global scale, 8 pad
  then cols x f32 per-column quant scales (raw_value ~= q * colscale)
  then rows*cols int8 (dtype 3) or rows*cols/2 bytes int4 (dtype 4,
       little nibble first, signed [-8,7])

Usage: make_ple_table_intq.py ple.json out.bin {int8|int4}
"""
import json, struct, sys
import numpy as np

meta = json.load(open(sys.argv[1]))
out = sys.argv[2]
dt = sys.argv[3]
assert dt in ("int8", "int4")
rows, cols, off = meta["rows"], meta["cols"], meta["offset"]
src = np.memmap(meta["path"], dtype=np.uint16, mode="r",
                offset=off, shape=(rows, cols))
CH = 8192

# pass 1: per-column absmax
absmax = np.zeros(cols, dtype=np.float32)
for r0 in range(0, rows, CH):
    f32 = (np.asarray(src[r0:r0+CH]).astype(np.uint32) << 16).view(np.float32)
    np.maximum(absmax, np.abs(f32).max(axis=0), out=absmax)

qmax = 127 if dt == "int8" else 7
colscale = (absmax / qmax).astype(np.float32)
colscale[colscale == 0] = 1.0

hdr = struct.pack("<8sIIIf8x", b"PLETBL01", 3 if dt == "int8" else 4,
                  rows, cols, float(meta.get("scale", 16.0)))
maxerr = 0.0
with open(out, "wb") as f:
    f.write(hdr)
    f.write(colscale.tobytes())
    inv = 1.0 / colscale
    for r0 in range(0, rows, CH):
        f32 = (np.asarray(src[r0:r0+CH]).astype(np.uint32) << 16).view(np.float32)
        q = np.clip(np.round(f32 * inv), -qmax - 1, qmax).astype(np.int8)
        maxerr = max(maxerr, float(np.abs(q.astype(np.float32) * colscale - f32).max()))
        if dt == "int8":
            f.write(q.tobytes())
        else:
            u = q.astype(np.uint8) & 0x0F
            f.write((u[:, 0::2] | (u[:, 1::2] << 4)).tobytes())
print(f"wrote {out} dtype={dt} rows={rows} cols={cols} "
      f"max_abs_raw_err={maxerr:.3e} (x16 => {maxerr*16:.3e})")
