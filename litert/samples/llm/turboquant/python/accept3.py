#!/usr/bin/env python
"""Phase 3 acceptance driver.

Runs engine2 on p0/p1 in FUSED mode (model_tq3.tflite) and STAGING mode
(model_quantized.tflite, exact Phase 2b math) with teacher forcing +
--dump-logits, plus fused free-running passes; computes
  (a) top-1 vs the Phase 2a fp32-baseline teacher tokens  (gate >= 0.94)
  (b) max |logits| diff fused vs staging per step
  (c) memory numbers from the engine JSONs
and decodes the free-run text for a human read.
"""
import json, os, subprocess, sys
import numpy as np

D = os.path.expanduser("~/turboquant/phase3")
ENG = f"{D}/build/engine2"
FINAL = os.path.expanduser("~/turboquant/export/e2b_16k/final")
ASSETS = os.path.expanduser("~/turboquant/engine/assets")
OUTD = f"{D}/out"
SNAP = os.path.expanduser(
    "~/.cache/huggingface/hub/models--google--gemma-4-E2B-it/snapshots/3e22461f65e89153144f8adb70e3b8c2cc9845a7")
VOCAB = 262144

def run(prompt, tag, extra, fused):
    out = f"{OUTD}/{tag}.json"
    cmd = [ENG, "--final", FINAL, "--assets", ASSETS,
           "--prompt-file", f"{ASSETS}/prompt_p{prompt}.json", "--out", out,
           "--threads", "32"] + extra
    if fused:
        cmd += ["--model", f"{D}/model_tq3.tflite"]
    print(">>", " ".join(cmd), flush=True)
    with open(f"{OUTD}/{tag}.log", "w") as log:
        subprocess.run(cmd, check=True, stderr=subprocess.STDOUT, stdout=log)
    return json.load(open(out))

def logit_diff(a_path, b_path):
    a = np.fromfile(a_path, dtype=np.float32).reshape(-1, VOCAB)
    b = np.fromfile(b_path, dtype=np.float32).reshape(-1, VOCAB)
    n = min(len(a), len(b))
    d = np.abs(a[:n] - b[:n])
    return dict(steps=int(n), max_abs=float(d.max()),
                max_abs_step0=float(d.max(1)[0]),
                argmax_agree=float((a[:n].argmax(1) == b[:n].argmax(1)).mean()))

def main():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(SNAP)
    res = {}
    for p in (0, 1):
        res[f"p{p}_fused_teacher"] = run(
            p, f"p{p}_fused_teacher",
            ["--teacher", "--dump-logits", f"{OUTD}/p{p}_fused.logits"], True)
        res[f"p{p}_staging_teacher"] = run(
            p, f"p{p}_staging_teacher",
            ["--teacher", "--dump-logits", f"{OUTD}/p{p}_staging.logits"], False)
        r = run(p, f"p{p}_fused_free", ["--free"], True)
        r["text"] = tok.decode(r["gen"], skip_special_tokens=False)
        res[f"p{p}_fused_free"] = r
        res[f"p{p}_logits_ab"] = logit_diff(
            f"{OUTD}/p{p}_fused.logits", f"{OUTD}/p{p}_staging.logits")
        res[f"p{p}_logits_ab_catchup"] = logit_diff(
            f"{OUTD}/p{p}_fused.logits.catchup", f"{OUTD}/p{p}_staging.logits.catchup")
    json.dump(res, open(f"{OUTD}/acceptance3.json", "w"), ensure_ascii=False, indent=1)

    print("\n==== PHASE 3 ACCEPTANCE ====")
    for p in (0, 1):
        f, s = res[f"p{p}_fused_teacher"], res[f"p{p}_staging_teacher"]
        ab, abc = res[f"p{p}_logits_ab"], res[f"p{p}_logits_ab_catchup"]
        ok = f["top1"] >= 0.94
        print(f"p{p} FUSED : top1={f['top1']:.4f} ({'PASS' if ok else 'FAIL'} >=0.94) "
              f"diverge={f['diverge_step']} steps={f['steps_compared']} | "
              f"prefill={f['prefill_tok_s']:.0f} decode={f['decode_tok_s']:.2f} tok/s | "
              f"rss_hwm={f['rss_hwm_mb']} MB packed={f['packed_mb']} MB "
              f"memo={f['memo_mb']} MB staging={f['staging_mb']} MB")
        print(f"p{p} STAGE : top1={s['top1']:.4f} | prefill={s['prefill_tok_s']:.0f} "
              f"decode={s['decode_tok_s']:.2f} tok/s | rss_hwm={s['rss_hwm_mb']} MB "
              f"staging={s['staging_mb']} MB")
        print(f"p{p} logits A/B: catchup step0 max|d|={abc['max_abs_step0']:.4g}, "
              f"gen max|d|={ab['max_abs']:.4g}, argmax agree={ab['argmax_agree']:.4f}")
        print(f"p{p} free text:\n{res[f'p{p}_fused_free']['text']}\n")

if __name__ == "__main__":
    main()
