"""Stage 1 of References/preference_optimization_ideas.md: the cheapest
possible test of whether slingshot's cycling fix transfers to the
Nash-equilibrium-seeking preference-optimization problem class (Munos et
al.'s Nash-MD; Swamy et al.'s minimax-winner framing), before any LLM-scale
investment.

Three self-play algorithms, on the same finite intransitive preference
game(s):
  vanilla    -- plain multiplicative-weights self-play (algo.conic_particle's
                already-validated _weight_step, constant stepsize). Expected
                to cycle, exactly as classical GDA cycles on matching
                pennies / rock-paper-scissors.
  nash_md    -- the same update, anchored toward the uniform reference policy
                (poc.nash_md.nash_md_step) -- a tabular reduction of Munos et
                al.'s actual fix for this cycling problem.
  slingshot  -- the identical _weight_step, but with Shugart & Altschuler's
                Chebyshev-node slingshot schedule (this game is bilinear in
                the mixed strategies, the same schedule family
                games/bilinear.py uses) in place of a constant stepsize.

Two gradient regimes: exact (the opponent's exact mixed strategy and the
exact preference matrix are used every step) and stochastic (only a finite
batch of sampled comparisons is used each step, mimicking real self-play
RLHF) -- see preference_game.sample_noisy_gradients. The stochastic regime
is the actual point of this exercise: slingshot's mechanism is a
second-order cancellation effect, and whether it survives realistic
gradient noise is the open question motivating this whole proof of concept.

Usage (run from the project root, so algo/, metrics/, slingshot/ resolve as
siblings):
    python poc/run_poc.py
"""
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from algo.conic_particle import _weight_step  # noqa: E402  (deliberate reuse, see module docstring)
from metrics import entropy, fit_log_linear_rate  # noqa: E402
from slingshot import chebyshev  # noqa: E402

from preference_game import (  # noqa: E402
    RPS, CYCLIC7, spectral_bounds, exact_duality_gap, tv_distance, sample_noisy_gradients,
)
from nash_md import nash_md_step  # noqa: E402

POC_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(POC_DIR, "results")

GAMES = (RPS, CYCLIC7)
SEEDS = range(5)
TAU = 0.1  # Nash-MD anchor strength
ALGOS = ("vanilla", "nash_md", "slingshot")


def _write_csv(rows, path):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _log_row(t, F, a, b, ne):
    return {
        "t": t,
        "duality_gap": exact_duality_gap(F, a, b),
        "tv_to_ne_a": tv_distance(a, ne),
        "tv_to_ne_b": tv_distance(b, ne),
        "entropy_a": entropy(a),
        "entropy_b": entropy(b),
    }


def _apply_step(algo, a, b, grad_a, grad_b, eta, schedule, step):
    if algo == "vanilla":
        return _weight_step(a, b, grad_a, grad_b, eta, eta)
    if algo == "nash_md":
        n, m = len(a), len(b)
        return nash_md_step(a, b, grad_a, grad_b, eta, TAU, np.ones(n) / n, np.ones(m) / m)
    alpha, beta = schedule(step)
    return _weight_step(a, b, grad_a, grad_b, alpha, beta)


def _run_one(game, algo, seed, iters, log_every, eta, schedule, noisy, K):
    F, ne = game.F, game.uniform_ne
    n = len(ne)
    rng = np.random.default_rng(seed)
    a, b = rng.dirichlet(np.ones(n)), rng.dirichlet(np.ones(n))

    rows = [_log_row(0, F, a, b, ne)]
    for step in range(iters):
        if noisy:
            grad_a, grad_b = sample_noisy_gradients(F, a, b, K, rng)
        else:
            grad_a, grad_b = F @ b, a @ F
        a, b = _apply_step(algo, a, b, grad_a, grad_b, eta, schedule, step)

        n_steps = step + 1
        if n_steps % log_every == 0 or n_steps == iters:
            rows.append(_log_row(n_steps, F, a, b, ne))
    return rows


def run_regime(game, regime_dir, iters, log_every, noisy, K=None, seeds=SEEDS):
    F = game.F
    m_spec, M_spec = spectral_bounds(F)
    eta = 1.0 / np.sqrt(M_spec)
    schedule = chebyshev(T=iters, m=m_spec, M=M_spec)

    out_dir = os.path.join(regime_dir, game.name)
    os.makedirs(out_dir, exist_ok=True)

    summary = []
    for algo in ALGOS:
        for seed in seeds:
            rows = _run_one(game, algo, seed, iters, log_every, eta, schedule, noisy, K)
            csv_path = os.path.join(out_dir, f"{algo}_seed{seed}.csv")
            _write_csv(rows, csv_path)

            iters_logged = [r["t"] for r in rows]
            gaps = [r["duality_gap"] for r in rows]
            summary.append({
                "game": game.name, "algo": algo, "seed": seed,
                "final_duality_gap": gaps[-1],
                "gap_log_linear_rate": fit_log_linear_rate(gaps, iters_logged),
                "final_tv_a": rows[-1]["tv_to_ne_a"],
                "final_tv_b": rows[-1]["tv_to_ne_b"],
                "final_entropy_a": rows[-1]["entropy_a"],
                "csv_path": os.path.relpath(csv_path, RESULTS_DIR),
            })

    summary_path = os.path.join(out_dir, "summary.csv")
    _write_csv(summary, summary_path)
    print(f"  wrote {summary_path}  ({len(summary)} rows)")
    return summary


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    all_rows = []

    for game in GAMES:
        gap0 = exact_duality_gap(game.F, game.uniform_ne, game.uniform_ne)
        assert abs(gap0) < 1e-9, f"{game.name}: uniform should be an exact NE, got gap {gap0}"
        print(f"=== {game.name} (uniform-NE duality gap = {gap0:.2e}, sanity OK) ===")

        print("-- exact gradient --")
        all_rows += run_regime(game, os.path.join(RESULTS_DIR, "exact_gradient"),
                                iters=3000, log_every=25, noisy=False)

        print("-- stochastic gradient (K=32 samples/step) --")
        all_rows += run_regime(game, os.path.join(RESULTS_DIR, "stochastic_gradient"),
                                iters=6000, log_every=50, noisy=True, K=32)

    path = os.path.join(RESULTS_DIR, "summary_all.csv")
    _write_csv(all_rows, path)
    print(f"\nwrote {path}  ({len(all_rows)} rows total)")


if __name__ == "__main__":
    main()
