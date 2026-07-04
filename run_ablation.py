"""Runs the full experiment suite and writes results to results/<game>/...,
partitioned per game (and, for chizat_example_4_1, further split into
global/ vs local/ init regimes -- see References/CLAUDE.md for why that
split matters). One command produces the entire on-disk results tree this
project's findings are based on; nothing here is exploratory plumbing kept
in /tmp.

Usage:
    python run_ablation.py            # run every game
    python run_ablation.py --game bilinear chizat_example_4_1
"""
import argparse
import csv
import os

import numpy as np

from cli import run_experiment
from metrics import fit_log_linear_rate

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


def _run_one(out_dir, game, algo_L, mode, seed, iters, log_every, init_mode,
             eta=None, m=None, M=None, lipschitz_L=None, init_noise=0.01,
             n_x=30, n_y=30, grid_size=400):
    rows, state = run_experiment(
        game, algo_L, mode, n_x, n_y, seed, iters, log_every, grid_size,
        eta=eta, m=m, M=M, lipschitz_L=lipschitz_L,
        init_mode=init_mode, init_noise=init_noise,
    )

    csv_name = f"L{algo_L}_{mode}_init-{init_mode}_seed{seed}.csv"
    csv_path = os.path.join(out_dir, csv_name)
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    iters_logged = [r["t"] for r in rows]
    gaps = [r["duality_gap"] for r in rows]
    w2_sum = [r["w2_mu"] + r["w2_nu"] for r in rows if r["w2_mu"] is not None]

    bary_x = float(state.a @ state.x)
    bary_y = float(state.b @ state.y)

    return {
        "game": game, "algo_L": algo_L, "stepsize_mode": mode,
        "init_mode": init_mode, "seed": seed, "iters": iters,
        "final_duality_gap": gaps[-1],
        "gap_log_linear_rate": fit_log_linear_rate(gaps, iters_logged),
        "final_w2_sum": w2_sum[-1] if w2_sum else "",
        "w2_log_linear_rate": fit_log_linear_rate(w2_sum, iters_logged[:len(w2_sum)]) if w2_sum else "",
        "final_entropy_a": rows[-1]["entropy_a"],
        "final_entropy_b": rows[-1]["entropy_b"],
        "final_barycenter_x": bary_x,
        "final_barycenter_y": bary_y,
        "csv_path": os.path.relpath(csv_path, RESULTS_DIR),
    }


def _write_summary(out_dir, summary_rows):
    if not summary_rows:
        return
    path = os.path.join(out_dir, "summary.csv")
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"  wrote {path}  ({len(summary_rows)} rows)")
    return path


def run_chizat_global(seeds=(0, 1, 2, 3, 4), iters=6000, log_every=250):
    """Random init across the whole torus -- Chizat's actual experimental
    setup. Headline finding: both modes plateau/cycle indistinguishably; see
    References/CLAUDE.md 'Implementation findings'."""
    out_dir = os.path.join(RESULTS_DIR, "chizat_example_4_1", "global")
    os.makedirs(out_dir, exist_ok=True)
    summary = []
    for algo_L in (1, 2):
        for mode in ("vanilla", "slingshot"):
            for seed in seeds:
                summary.append(_run_one(
                    out_dir, "chizat_example_4_1", algo_L, mode, seed,
                    iters, log_every, init_mode="uniform"))
    _write_summary(out_dir, summary)
    return summary


def run_chizat_local(seeds=(0, 1, 2, 3, 4), iters=5000, log_every=250, noise=0.01):
    """Warm-started near the known MNE atoms -- the local regime Shugart's
    slingshot guarantee actually covers. Finding: both modes collapse to a
    tiny residual fast; no separation."""
    out_dir = os.path.join(RESULTS_DIR, "chizat_example_4_1", "local")
    os.makedirs(out_dir, exist_ok=True)
    summary = []
    for mode in ("vanilla", "slingshot"):
        for seed in seeds:
            summary.append(_run_one(
                out_dir, "chizat_example_4_1", 1, mode, seed,
                iters, log_every, init_mode="near_mne", init_noise=noise))
    _write_summary(out_dir, summary)
    return summary


def run_bilinear(seeds=(0, 1, 2), iters=3000, log_every=200):
    """Calibrated small step (h=0.01) to stay below the position-step's
    adaptive clip -- bilinear's spectrum is degenerate (m=M=1), so the
    *default* stepsize saturates the clip for both modes (see
    run_bilinear_pitfall). Finding: slingshot's duality gap -> ~1e-14 in
    every seed; vanilla stays stuck at ~106-117. The cleanest reproduction
    of Shugart's headline claim in this codebase."""
    out_dir = os.path.join(RESULTS_DIR, "bilinear", "calibrated")
    os.makedirs(out_dir, exist_ok=True)
    summary = []
    for seed in seeds:
        summary.append(_run_one(
            out_dir, "bilinear", 1, "vanilla", seed, iters, log_every,
            init_mode="uniform", eta=0.01))
        summary.append(_run_one(
            out_dir, "bilinear", 1, "slingshot", seed, iters, log_every,
            init_mode="uniform", m=10000, M=10000))
    _write_summary(out_dir, summary)
    return summary


def run_bilinear_pitfall(seed=0, iters=2000, log_every=200):
    """Default stepsize (h=1/sqrt(M)=1.0) on bilinear's degenerate spectrum
    (m=M=1) saturates the adaptive position-step clip immediately for BOTH
    modes -- kept here, clearly labeled, as a documented negative/pitfall
    run, not a finding about slingshot itself. See run_bilinear for the
    calibrated comparison that actually isolates the sign-schedule effect."""
    out_dir = os.path.join(RESULTS_DIR, "bilinear", "pitfall_default_stepsize")
    os.makedirs(out_dir, exist_ok=True)
    summary = [
        _run_one(out_dir, "bilinear", 1, "vanilla", seed, iters, log_every, init_mode="uniform"),
        _run_one(out_dir, "bilinear", 1, "slingshot", seed, iters, log_every, init_mode="uniform"),
    ]
    _write_summary(out_dir, summary)
    with open(os.path.join(out_dir, "README.md"), "w") as f:
        f.write(
            "# Pitfall: default stepsize on bilinear's degenerate spectrum\n\n"
            "bilinear uses B=[[1]], so m=M=1. The vanilla baseline (h=1/sqrt(M)=1.0) and the\n"
            "Chebyshev/slingshot schedule (which collapses to the same constant magnitude when\n"
            "m=M) both pick step magnitude 1.0 -- far above the position-step's adaptive clip\n"
            "cap (min(1, 0.1/a_i) <= 1). Both modes instantly saturate the clip and slam\n"
            "particles to the domain boundary within a handful of steps, then freeze there.\n"
            "Vanilla and slingshot look identical here -- but for the wrong reason\n"
            "(clip-domination, not the sign schedule). See ../calibrated/ for the run that\n"
            "stays below clip saturation and actually isolates slingshot's effect.\n"
        )
    return summary


def run_convex_concave(seeds=(0, 1, 2), iters=2000, log_every=100):
    out_dir = os.path.join(RESULTS_DIR, "convex_concave")
    os.makedirs(out_dir, exist_ok=True)
    summary = []
    for algo_L in (1, 2):
        for mode in ("vanilla", "slingshot"):
            for seed in seeds:
                summary.append(_run_one(
                    out_dir, "convex_concave", algo_L, mode, seed,
                    iters, log_every, init_mode="uniform"))
    _write_summary(out_dir, summary)
    return summary


def run_sc_sc(seeds=(0, 1, 2), iters=2000, log_every=100):
    out_dir = os.path.join(RESULTS_DIR, "sc_sc")
    os.makedirs(out_dir, exist_ok=True)
    summary = []
    for algo_L in (1, 2):
        for mode in ("vanilla", "slingshot"):
            for seed in seeds:
                summary.append(_run_one(
                    out_dir, "sc_sc", algo_L, mode, seed,
                    iters, log_every, init_mode="uniform"))
    _write_summary(out_dir, summary)
    return summary


GAME_RUNNERS = {
    "chizat_example_4_1": lambda: (run_chizat_global() , run_chizat_local()),
    "bilinear": lambda: (run_bilinear_pitfall(), run_bilinear()),
    "convex_concave": lambda: (run_convex_concave(),),
    "sc_sc": lambda: (run_sc_sc(),),
}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--game", nargs="+", choices=list(GAME_RUNNERS.keys()),
                         default=list(GAME_RUNNERS.keys()))
    args = parser.parse_args(argv)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    all_summary_rows = []
    for game in args.game:
        print(f"=== {game} ===")
        results = GAME_RUNNERS[game]()
        for summary in results:
            all_summary_rows.extend(summary)

    if all_summary_rows:
        path = os.path.join(RESULTS_DIR, "summary_all.csv")
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(all_summary_rows)
        print(f"\nwrote {path}  ({len(all_summary_rows)} rows total)")


if __name__ == "__main__":
    main()
