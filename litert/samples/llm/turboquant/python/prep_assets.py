#!/usr/bin/env python
"""Phase 2b asset prep: export rotation matrices, codebooks, PLE mmap metadata,
and prompt/teacher token files for the C++ engine.

Everything the C++ engine needs that would otherwise require torch/safetensors:
  assets/rot_d256.bin, rot_d512.bin   : fp32 row-major Pi (seed-42 QR, torch)
  assets/cb_d256_b3.bin, cb_d512_b3.bin : 8 fp32 centroids + 7 fp32 decision boundaries
  assets/ple.json                     : safetensors path + absolute byte offset of the
                                        PLE table (bf16, rows x 8960), engine mmaps it
  assets/prompt_p{0,1}.json           : prompt token ids + teacher tokens (baseline gen)
  assets/prompt_p2.json               : long zh prompt (window-check / free-run)
"""
import json, os, struct, sys
import numpy as np

sys.path.insert(0, os.path.expanduser("~/turboquant/turboquant"))
sys.path.insert(0, os.path.expanduser("~/turboquant"))

OUT = os.path.expanduser("~/turboquant/engine/assets")
os.makedirs(OUT, exist_ok=True)
SNAP = os.path.expanduser(
    "~/.cache/huggingface/hub/models--google--gemma-4-E2B-it/snapshots/3e22461f65e89153144f8adb70e3b8c2cc9845a7")

# 1. rotation matrices (must be bit-identical to TurboQuantMSE's buffer)
import torch
from turboquant.rotation import generate_rotation_matrix
for d in (256, 512):
    Pi = generate_rotation_matrix(d, torch.device("cpu"), torch.float32, seed=42)
    Pi.numpy().astype(np.float32).tofile(f"{OUT}/rot_d{d}.bin")
    print(f"rot_d{d}.bin: {Pi.shape}, checksum {float(Pi.abs().sum()):.6f}")

# 2. codebooks: centroids + decision boundaries exactly as the quantizer sees them
from turboquant.codebook import get_codebook_tensors
for d in (256, 512):
    centroids, boundaries = get_codebook_tensors(d, 3, torch.device("cpu"), torch.float32)
    dec = boundaries[1:-1].contiguous()  # 7 interior boundaries
    with open(f"{OUT}/cb_d{d}_b3.bin", "wb") as f:
        f.write(centroids.numpy().astype(np.float32).tobytes())
        f.write(dec.numpy().astype(np.float32).tobytes())
    print(f"cb_d{d}_b3.bin: centroids {centroids.numpy().round(5).tolist()}")

# 3. PLE metadata: locate tensor data span inside model.safetensors
path = f"{SNAP}/model.safetensors"
with open(path, "rb") as f:
    hlen = struct.unpack("<Q", f.read(8))[0]
    header = json.loads(f.read(hlen))
name = "model.language_model.embed_tokens_per_layer.weight"
info = header[name]
assert info["dtype"] == "BF16", info
rows, cols = info["shape"]
data_start = 8 + hlen + info["data_offsets"][0]
meta = {"path": path, "offset": data_start, "rows": rows, "cols": cols,
        "dtype": "bf16", "scale": 16.0}
json.dump(meta, open(f"{OUT}/ple.json", "w"), indent=1)
print("ple.json:", meta)

# 4. prompts + teacher tokens
from transformers import AutoTokenizer
from phase2a_harness import get_prompts
tok = AutoTokenizer.from_pretrained(SNAP)
for p in range(3):
    prompt = get_prompts()[p]
    txt = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                  add_generation_prompt=True, tokenize=False)
    ids = tok(txt, add_special_tokens=False)["input_ids"]
    d = {"prompt_ids": ids, "eos": [tok.eos_token_id, 106]}
    npz = os.path.expanduser(f"~/turboquant/out_p{p}_base.npz")
    if os.path.exists(npz):
        base = np.load(npz, allow_pickle=True)
        d["teacher"] = [int(t) for t in base["gen"]]
    json.dump(d, open(f"{OUT}/prompt_p{p}.json", "w"))
    print(f"prompt_p{p}.json: {len(ids)} tokens, teacher={len(d.get('teacher', []))}")
