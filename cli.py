"""One run = one configuration: pick a game, an algorithm (L), and a
stepsize-mode (vanilla baseline or slingshot), then log convergence
diagnostics to a CSV. Sweeps are a thin loop over this module's Python API
(see run_experiment), not shelled-out subprocess calls -- see
References/CLAUDE.md for the agreed 2x2 ablation grid this is meant to drive.
"""
import argparse
import csv
import sys

import numpy as np

from games import get_game, list_games
from algo import init_state, init_state_near_mne, wfr_step
from slingshot import vanilla, chebyshev, randomized_sign
from metrics import entropy, duality_gap_grid, wasserstein_to_mne, fit_log_linear_rate


def build_schedule(game, mode: str, iters: int, eta, m, M, lipschitz_L, rng):
    family = game.schedule_family
    spectral = game.default_spectral

    if mode == "vanilla":
        if family == "chebyshev":
            default_eta = 1.0 / np.sqrt(M if M is not None else spectral["M"])
        else:
            default_eta = 1.0 / (3.0 * (lipschitz_L if lipschitz_L is not None else spectral["L"]))
        return vanilla(eta if eta is not None else default_eta)

    if family == "chebyshev":
        return chebyshev(
            T=iters,
            m=m if m is not None else spectral["m"],
            M=M if M is not None else spectral["M"],
        )
    return randomized_sign(
        L=lipschitz_L if lipschitz_L is not None else spectral["L"],
        rng=rng,
    )


def run_experiment(game_name: str, algo_L: int, stepsize_mode: str, n_x: int, n_y: int,
                    seed: int, iters: int, log_every: int, grid_size: int,
                    eta=None, m=None, M=None, lipschitz_L=None,
                    init_mode: str = "uniform", init_noise: float = 0.01):
    game = get_game(game_name)
    rng = np.random.default_rng(seed)
    if init_mode == "near_mne":
        state = init_state_near_mne(n_x, n_y, game, rng, noise=init_noise)
    else:
        state = init_state(n_x, n_y, game, rng)
    schedule = build_schedule(game, stepsize_mode, iters, eta, m, M, lipschitz_L, rng)

    rows = []
    for t in range(iters):
        alpha, beta = schedule(t)
        state = wfr_step(state, game, alpha, beta, algo_L)

        if t % log_every == 0 or t == iters - 1:
            gap = duality_gap_grid(game, state, grid_size=grid_size)
            w_mu, w_nu = wasserstein_to_mne(state, game)
            rows.append({
                "t": t,
                "duality_gap": gap,
                "w2_mu": w_mu,
                "w2_nu": w_nu,
                "entropy_a": entropy(state.a),
                "entropy_b": entropy(state.b),
                "barycenter_x": float(state.a @ state.x),
                "barycenter_y": float(state.b @ state.y),
            })

    return rows, state


def write_csv(rows, out_path: str):
    if not rows:
        return
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows, label: str):
    iters = [r["t"] for r in rows]
    gaps = [r["duality_gap"] for r in rows]
    w2_sum = [r["w2_mu"] + r["w2_nu"] for r in rows if r["w2_mu"] is not None]

    gap_rate = fit_log_linear_rate(gaps, iters)
    w2_rate = fit_log_linear_rate(w2_sum, iters[:len(w2_sum)]) if w2_sum else float("nan")

    print(f"[{label}] final duality_gap={gaps[-1]:.6g}  "
          f"log-linear rate={gap_rate:.6g}", end="")
    if w2_sum:
        print(f"  |  final W2(mu)+W2(nu)={w2_sum[-1]:.6g}  log-linear rate={w2_rate:.6g}")
    else:
        print("  |  no known MNE for this game -- W2 metric unavailable")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--game", choices=list_games(), required=True)
    parser.add_argument("--algo-L", type=int, choices=[1, 2], required=True,
                         help="1 = CP-MDA (explicit), 2 = CP-MP (extragradient/mirror-prox)")
    parser.add_argument("--stepsize-mode", choices=["vanilla", "slingshot"], required=True)
    parser.add_argument("--n-x", type=int, default=30)
    parser.add_argument("--n-y", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--iters", type=int, default=1000)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--grid-size", type=int, default=400)
    parser.add_argument("--eta", type=float, default=None, help="override vanilla stepsize")
    parser.add_argument("--m", type=float, default=None, help="override chebyshev spectrum lower bound")
    parser.add_argument("--M", type=float, default=None, help="override chebyshev spectrum upper bound")
    parser.add_argument("--lipschitz-L", type=float, default=None,
                         help="override randomized-sign Lipschitz constant")
    parser.add_argument("--out", type=str, default=None, help="CSV output path")
    parser.add_argument("--init", choices=["uniform", "near_mne"], default="uniform",
                         help="uniform = random over the whole domain (Chizat's setup); "
                              "near_mne = warm-start particles near the known MNE support, "
                              "the local regime Shugart's slingshot guarantee covers")
    parser.add_argument("--init-noise", type=float, default=0.01,
                         help="stddev of the perturbation around each MNE atom when --init near_mne")
    args = parser.parse_args(argv)

    rows, _ = run_experiment(
        args.game, args.algo_L, args.stepsize_mode, args.n_x, args.n_y,
        args.seed, args.iters, args.log_every, args.grid_size,
        eta=args.eta, m=args.m, M=args.M, lipschitz_L=args.lipschitz_L,
        init_mode=args.init, init_noise=args.init_noise,
    )

    label = f"{args.game} L={args.algo_L} {args.stepsize_mode} seed={args.seed}"
    summarize(rows, label)

    if args.out:
        write_csv(rows, args.out)
        print(f"wrote {len(rows)} rows -> {args.out}")


if __name__ == "__main__":
    sys.exit(main())
