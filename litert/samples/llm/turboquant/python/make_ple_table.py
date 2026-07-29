#!/usr/bin/env python3
"""Extract the per-layer-embedding table from model.safetensors into a flat
fp16 (or fp32) binary that engine2 mmaps directly (--ple-table).
Header: 32 bytes = magic "PLETBL01", u32 dtype(0=fp32,1=fp16), u32 rows, u32 cols,
float scale, 8 pad. Data: rows*cols values, row-major, RAW (scale applied at runtime).
Also reports whether bf16->fp16 is lossless."""
import json, sys, numpy as np, struct

meta = json.load(open(sys.argv[1]))          # ple.json
out  = sys.argv[2]                            # output .bin
dt   = sys.argv[3] if len(sys.argv) > 3 else "fp16"
rows, cols, off = meta["rows"], meta["cols"], meta["offset"]
src = np.memmap(meta["path"], dtype=np.uint16, mode="r",
                offset=off, shape=(rows, cols))
hdr = struct.pack("<8sIIIf8x", b"PLETBL01", 1 if dt=="fp16" else 0,
                  rows, cols, float(meta.get("scale", 16.0)))
lossy = 0
with open(out, "wb") as f:
    f.write(hdr)
    CH = 8192
    for r0 in range(0, rows, CH):
        blk = src[r0:r0+CH]
        f32 = (blk.astype(np.uint32) << 16).view(np.float32)
        if dt == "fp16":
            h = f32.astype(np.float16)
            lossy += int((h.astype(np.float32) != f32).sum())
            f.write(h.tobytes())
        else:
            f.write(f32.tobytes())
print(f"wrote {out} dtype={dt} rows={rows} cols={cols} lossy_values={lossy}")
