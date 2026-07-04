"""Does slingshot's disadvantage vs. Nash-MD under stochastic gradients (see
run_poc.py's stochastic_gradient regime, K=32) shrink as the per-step sample
budget K grows, or is it present even at large K? Answers whether that gap
is a generic property of the mechanism or a "too noisy, wrong K" artifact
of one specific value -- the evidence poc/report.tex's interpretation of the
stochastic regime rests on.

Usage: python poc/noise_sweep.py (run from the project root)
"""
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from run_poc import _run_one, RESULTS_DIR  # noqa: E402
from preference_game import RPS, spectral_bounds  # noqa: E402
from slingshot import chebyshev  # noqa: E402

K_VALUES = (8, 32, 128, 512, 2048)
SEEDS = range(5)
ITERS = 4000


def main():
    F = RPS.F
    m_spec, M_spec = spectral_bounds(F)
    eta = 1.0 / np.sqrt(M_spec)
    schedule = chebyshev(T=ITERS, m=m_spec, M=M_spec)

    rows = []
    for K in K_VALUES:
        for algo in ("nash_md", "slingshot"):
            for seed in SEEDS:
                run_rows = _run_one(RPS, algo, seed, ITERS, ITERS, eta, schedule, noisy=True, K=K)
                gap = run_rows[-1]["duality_gap"]
                rows.append({"K": K, "algo": algo, "seed": seed, "final_duality_gap": gap})
                print(f"K={K:5d}  {algo:9s}  seed={seed}  gap={gap:.4g}")

    out_dir = os.path.join(RESULTS_DIR, "stochastic_gradient", "rps3")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "noise_sweep.csv")
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
