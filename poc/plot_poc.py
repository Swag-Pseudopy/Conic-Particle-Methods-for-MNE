"""Renders PNG comparison panels for run_poc.py's and noise_sweep.py's CSVs,
next to them under poc/results/.../plots/ -- same visual language as the
main project's plot_results.py (vanilla=purple, slingshot=red), plus a third
color for the Nash-MD baseline.

Usage (run from the project root):
    python poc/plot_poc.py
"""
import csv
import glob
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

POC_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(POC_DIR, "results")

COLORS = {"vanilla": "#6a0dad", "nash_md": "#1f77b4", "slingshot": "#e63946"}
ALGOS = ("vanilla", "nash_md", "slingshot")


def _load_run(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    out = {"t": np.array([int(r["t"]) for r in rows])}
    for key in ("duality_gap", "tv_to_ne_a", "tv_to_ne_b", "entropy_a", "entropy_b"):
        out[key] = np.array([float(r[key]) for r in rows])
    return out


def _semilogy_safe(ax, x, y, **kw):
    y = np.asarray(y, dtype=float)
    mask = y > 0
    if mask.any():
        ax.semilogy(x[mask], y[mask], **kw)


def plot_game_regime(run_dir, plots_dir, game, regime):
    fig, axs = plt.subplots(2, 2, figsize=(11, 9))
    fig.suptitle(f"{game} -- {regime}", fontsize=13, fontweight="bold")

    for algo in ALGOS:
        paths = sorted(glob.glob(os.path.join(run_dir, f"{algo}_seed*.csv")))
        color = COLORS[algo]
        for i, path in enumerate(paths):
            d = _load_run(path)
            label = algo if i == 0 else None
            _semilogy_safe(axs[0, 0], d["t"], d["duality_gap"], color=color, alpha=0.7, lw=1.3, label=label)
            tv = 0.5 * (d["tv_to_ne_a"] + d["tv_to_ne_b"])
            _semilogy_safe(axs[0, 1], d["t"], tv, color=color, alpha=0.7, lw=1.3, label=label)
            axs[1, 0].plot(d["t"], d["entropy_a"], color=color, alpha=0.7, lw=1.3, label=label)
            axs[1, 1].plot(d["t"], d["entropy_b"], color=color, alpha=0.7, lw=1.3, label=label)

    axs[0, 0].set_title("Duality gap (log)"); axs[0, 0].set_xlabel("t"); axs[0, 0].legend(fontsize=8)
    axs[0, 0].grid(True, ls="--", alpha=0.4)
    axs[0, 1].set_title("TV distance to uniform NE, mean(a,b) (log)"); axs[0, 1].set_xlabel("t")
    axs[0, 1].legend(fontsize=8); axs[0, 1].grid(True, ls="--", alpha=0.4)
    axs[1, 0].set_title("Entropy H(a)"); axs[1, 0].set_xlabel("t"); axs[1, 0].legend(fontsize=8)
    axs[1, 0].grid(True, ls="--", alpha=0.4)
    axs[1, 1].set_title("Entropy H(b)"); axs[1, 1].set_xlabel("t"); axs[1, 1].legend(fontsize=8)
    axs[1, 1].grid(True, ls="--", alpha=0.4)

    plt.tight_layout()
    out_path = os.path.join(plots_dir, f"{game}_{regime}.png")
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_noise_sweep(csv_path, out_path):
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    Ks = sorted(set(int(r["K"]) for r in rows))

    fig, ax = plt.subplots(figsize=(6.5, 5))
    for algo in ("nash_md", "slingshot"):
        means, mins, maxs = [], [], []
        for K in Ks:
            vals = [float(r["final_duality_gap"]) for r in rows if r["algo"] == algo and int(r["K"]) == K]
            means.append(np.mean(vals)); mins.append(np.min(vals)); maxs.append(np.max(vals))
        means, mins, maxs = np.array(means), np.array(mins), np.array(maxs)
        ax.plot(Ks, means, "o-", color=COLORS[algo], label=algo, lw=1.8)
        ax.fill_between(Ks, mins, maxs, color=COLORS[algo], alpha=0.15)

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("K (sampled comparisons per step)")
    ax.set_ylabel("final duality gap (mean over 5 seeds, band = min/max)")
    ax.set_title("rps3 -- does slingshot's noise penalty vs. Nash-MD shrink with K?")
    ax.legend(); ax.grid(True, ls="--", alpha=0.4)
    plt.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main():
    written = []
    for regime in ("exact_gradient", "stochastic_gradient"):
        for game in ("rps3", "cyclic7"):
            run_dir = os.path.join(RESULTS_DIR, regime, game)
            if not os.path.isdir(run_dir):
                continue
            plots_dir = os.path.join(run_dir, "plots")
            os.makedirs(plots_dir, exist_ok=True)
            out_path = plot_game_regime(run_dir, plots_dir, game, regime)
            written.append(out_path)
            print(f"wrote {out_path}")

    sweep_csv = os.path.join(RESULTS_DIR, "stochastic_gradient", "rps3", "noise_sweep.csv")
    if os.path.isfile(sweep_csv):
        plots_dir = os.path.join(RESULTS_DIR, "stochastic_gradient", "rps3", "plots")
        os.makedirs(plots_dir, exist_ok=True)
        out_path = plot_noise_sweep(sweep_csv, os.path.join(plots_dir, "noise_sweep.png"))
        written.append(out_path)
        print(f"wrote {out_path}")

    print(f"\nwrote {len(written)} PNGs total")


if __name__ == "__main__":
    main()
