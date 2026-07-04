"""Reads the per-run CSVs that run_ablation.py writes under results/ and
renders PNG comparison panels (vanilla vs slingshot, all seeds overlaid)
next to them, in a results/<game>/<...>/plots/ subfolder -- the dashboard
view the source notebooks had (plot_dashboard/plot_comparison), rebuilt
against this package's actual CSV schema instead of being regenerated
inside a notebook.

Requires matplotlib (not a core dependency of the CLI/algo -- only this
script and environment.yml's optional plotting extra need it).

Usage:
    python plot_results.py            # plot every comparison group found
    python plot_results.py --game bilinear chizat_example_4_1
"""
import argparse
import csv
import glob
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

RUN_RE = re.compile(r"^L(\d+)_(vanilla|slingshot)_init-(\w+)_seed(\d+)\.csv$")

COLORS = {"vanilla": "#6a0dad", "slingshot": "#e63946"}


def _load_run(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    out = {"t": np.array([int(r["t"]) for r in rows])}
    for key in ("duality_gap", "entropy_a", "entropy_b", "barycenter_x", "barycenter_y"):
        out[key] = np.array([float(r[key]) for r in rows])
    w2_mu = [r["w2_mu"] for r in rows]
    out["w2_sum"] = (np.array([float(a) + float(b) for a, b in zip(w2_mu, [r["w2_nu"] for r in rows])])
                      if w2_mu[0] not in ("", "None") else None)
    return out


def _discover_groups(run_dir):
    """Group filenames in run_dir by (algo_L, init_mode) -- each group is one
    panel figure with vanilla vs slingshot overlaid across seeds."""
    groups = {}
    for path in glob.glob(os.path.join(run_dir, "*.csv")):
        name = os.path.basename(path)
        m = RUN_RE.match(name)
        if not m:
            continue
        algo_L, mode, init_mode, seed = m.groups()
        groups.setdefault((algo_L, init_mode), {}).setdefault(mode, []).append(path)
    return groups


def _semilogy_safe(ax, x, y, **kw):
    y = np.asarray(y, dtype=float)
    mask = y > 0
    if mask.any():
        ax.semilogy(x[mask], y[mask], **kw)


def plot_group(run_dir, plots_dir, game, algo_L, init_mode, mode_paths):
    fig, axs = plt.subplots(2, 2, figsize=(11, 9))
    fig.suptitle(f"{game} — L={algo_L} — init={init_mode}", fontsize=13, fontweight="bold")

    any_w2 = False
    for mode, paths in mode_paths.items():
        color = COLORS[mode]
        for i, path in enumerate(sorted(paths)):
            d = _load_run(path)
            label = mode if i == 0 else None
            _semilogy_safe(axs[0, 0], d["t"], d["duality_gap"], color=color, alpha=0.7, lw=1.3, label=label)
            if d["w2_sum"] is not None:
                any_w2 = True
                _semilogy_safe(axs[0, 1], d["t"], d["w2_sum"], color=color, alpha=0.7, lw=1.3, label=label)
            axs[1, 0].plot(d["t"], d["entropy_a"], color=color, alpha=0.7, lw=1.3, label=label)
            bary_dist = np.sqrt(d["barycenter_x"] ** 2 + d["barycenter_y"] ** 2)
            _semilogy_safe(axs[1, 1], d["t"], bary_dist, color=color, alpha=0.7, lw=1.3, label=label)

    axs[0, 0].set_title("Duality gap (log)"); axs[0, 0].set_xlabel("t"); axs[0, 0].legend(fontsize=8)
    axs[0, 0].grid(True, ls="--", alpha=0.4)

    axs[0, 1].set_title("W2(mu,nu) to MNE, summed (log)" if any_w2 else "W2 to MNE -- no known MNE")
    axs[0, 1].set_xlabel("t"); axs[0, 1].legend(fontsize=8); axs[0, 1].grid(True, ls="--", alpha=0.4)

    axs[1, 0].set_title("Entropy H(a)"); axs[1, 0].set_xlabel("t"); axs[1, 0].legend(fontsize=8)
    axs[1, 0].grid(True, ls="--", alpha=0.4)

    axs[1, 1].set_title("||barycenter|| (log)"); axs[1, 1].set_xlabel("t"); axs[1, 1].legend(fontsize=8)
    axs[1, 1].grid(True, ls="--", alpha=0.4)

    plt.tight_layout()
    out_path = os.path.join(plots_dir, f"L{algo_L}_init-{init_mode}.png")
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_dir(run_dir):
    groups = _discover_groups(run_dir)
    if not groups:
        return []
    plots_dir = os.path.join(run_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    game = os.path.basename(os.path.dirname(run_dir)) if os.path.basename(run_dir) in (
        "global", "local", "calibrated", "pitfall_default_stepsize") else os.path.basename(run_dir)
    written = []
    for (algo_L, init_mode), mode_paths in sorted(groups.items()):
        out_path = plot_group(run_dir, plots_dir, game, algo_L, init_mode, mode_paths)
        written.append(out_path)
        print(f"  wrote {out_path}")
    return written


GAME_RUN_DIRS = {
    "chizat_example_4_1": [
        os.path.join(RESULTS_DIR, "chizat_example_4_1", "global"),
        os.path.join(RESULTS_DIR, "chizat_example_4_1", "local"),
    ],
    "bilinear": [
        os.path.join(RESULTS_DIR, "bilinear", "calibrated"),
        os.path.join(RESULTS_DIR, "bilinear", "pitfall_default_stepsize"),
    ],
    "convex_concave": [os.path.join(RESULTS_DIR, "convex_concave")],
    "sc_sc": [os.path.join(RESULTS_DIR, "sc_sc")],
}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--game", nargs="+", choices=list(GAME_RUN_DIRS.keys()),
                         default=list(GAME_RUN_DIRS.keys()))
    args = parser.parse_args(argv)

    total = 0
    for game in args.game:
        print(f"=== {game} ===")
        for run_dir in GAME_RUN_DIRS[game]:
            if os.path.isdir(run_dir):
                total += len(plot_dir(run_dir))
    print(f"\nwrote {total} PNGs total")


if __name__ == "__main__":
    main()
