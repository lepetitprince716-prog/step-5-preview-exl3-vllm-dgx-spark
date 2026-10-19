#!/usr/bin/env python3
"""Print one line per p6_kld.py result: overall KL / top-1 / NLL, then per-domain KL / top-1 / positions.

    p6_summary.py runs/p6/b444 runs/p6/b566 ...      (each dir holds kld.json)
"""
from __future__ import annotations

import json
import os
import sys


def main() -> int:
    for d in sys.argv[1:]:
        r = json.load(open(os.path.join(d, "kld.json")))
        dom = " ".join(f"{k}:{v['kld']:.4f}/{v['top1']:.3f}/{v['positions']}" for k, v in r["by_domain"].items())
        print(f"{os.path.basename(d)} kl={r['kld_mean']:.5f} p99={r['kld_p99']:.4f} top1={r['top1_agreement']:.4f} "
              f"nll={r['nll_ours']:.4f} teacher={r['nll_teacher']:.4f} | {dom}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
