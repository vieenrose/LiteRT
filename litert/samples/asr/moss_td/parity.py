# Copyright 2026. Apache-2.0.
"""Transcript parity vs the rs.cpp f32 reference outputs.

Compares full text (incl. timestamps/speaker tags) with
difflib.SequenceMatcher(autojunk=False) and reports the agreement ratio.
"""

from __future__ import annotations

import argparse
import difflib
import re

TS_RE = re.compile(r"\[\d+\.\d+\]")


def agreement(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a, b, autojunk=False).ratio()


def strip_ts(s: str) -> str:
    return TS_RE.sub("", s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pairs", nargs="+",
                    help="name=lite.txt:ref.txt triples")
    args = ap.parse_args()
    for pair in args.pairs:
        name, rest = pair.split("=", 1)
        lite_p, ref_p = rest.split(":", 1)
        a = open(lite_p).read().strip()
        b = open(ref_p).read().strip()
        r = agreement(a, b)
        rt = agreement(strip_ts(a), strip_ts(b))
        print(f"{name}: agreement={r*100:.3f}%  text_only={rt*100:.3f}%  "
              f"identical={a == b}  len_lite={len(a)} len_ref={len(b)}")
        if a != b:
            sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
            for tag, i1, i2, j1, j2 in sm.get_opcodes():
                if tag != "equal":
                    print(f"  {tag}: lite[{i1}:{i2}]={a[i1:i2]!r} "
                          f"ref[{j1}:{j2}]={b[j1:j2]!r}")


if __name__ == "__main__":
    main()
