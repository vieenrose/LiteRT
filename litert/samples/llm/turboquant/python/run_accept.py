#!/usr/bin/env python
"""Phase 2b acceptance driver: runs the C++ engine on p0/p1 (teacher-forced +
free-running), reads its JSON, decodes free-run text, prints a summary table."""
import json, os, subprocess, sys

ENG = os.path.expanduser("~/turboquant/engine/build/engine")
FINAL = os.path.expanduser("~/turboquant/export/e2b_16k/final")
ASSETS = os.path.expanduser("~/turboquant/engine/assets")
OUTD = os.path.expanduser("~/turboquant/engine/out")
os.makedirs(OUTD, exist_ok=True)
SNAP = os.path.expanduser(
    "~/.cache/huggingface/hub/models--google--gemma-4-E2B-it/snapshots/3e22461f65e89153144f8adb70e3b8c2cc9845a7")

def run(prompt, tag, extra):
    out = f"{OUTD}/{tag}.json"
    cmd = [ENG, "--final", FINAL, "--assets", ASSETS,
           "--prompt-file", f"{ASSETS}/prompt_p{prompt}.json", "--out", out] + extra
    print(">>", " ".join(cmd), flush=True)
    log = open(f"{OUTD}/{tag}.log", "w")
    subprocess.run(cmd, check=True, stderr=subprocess.STDOUT, stdout=log)
    return json.load(open(out))

def main():
    threads = ["--threads", sys.argv[1] if len(sys.argv) > 1 else "32"]
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(SNAP)
    results = {}
    for p in (1, 0):
        results[f"p{p}_tq_teacher"] = run(p, f"p{p}_tq_teacher",
                                          ["--mode", "tq", "--teacher"] + threads)
        r = run(p, f"p{p}_tq_free", ["--mode", "tq", "--free"] + threads)
        r["text"] = tok.decode(r["gen"], skip_special_tokens=False)
        results[f"p{p}_tq_free"] = r
    json.dump(results, open(f"{OUTD}/acceptance.json", "w"), ensure_ascii=False, indent=1)
    print("\n==== ACCEPTANCE ====")
    for p in (0, 1):
        t = results[f"p{p}_tq_teacher"]
        ok = t["top1"] >= 0.94
        print(f"p{p}: top1={t['top1']:.4f} ({'PASS' if ok else 'FAIL'} >=0.94) "
              f"diverge={t['diverge_step']} steps={t['steps_compared']} "
              f"prefill={t['prefill_tok_s']:.1f} tok/s decode={t['decode_tok_s']:.2f} tok/s "
              f"rss_hwm={t['rss_hwm_mb']} MB")
        print(f"p{p} free text:\n{results[f'p{p}_tq_free']['text']}\n")

if __name__ == "__main__":
    main()
