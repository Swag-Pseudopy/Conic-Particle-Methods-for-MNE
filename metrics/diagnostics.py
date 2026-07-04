"""Convergence diagnostics, shared by every (game, algo-L, stepsize-mode) run
so they are computed identically across the whole ablation grid.

Two metrics are primary (per the agreed design in References/CLAUDE.md):
  - duality_gap_grid: the Nikaido-Isoda error / duality gap, a true sup over
    continuous single-player deviations (via a fine grid), not just over the
    opponent's current particles -- this is the actual Nash optimality
    certificate and is always computable, even with no known closed-form MNE.
  - wasserstein_to_mne: distance from the current particle measures to the
    known closed-form MNE, where one exists. Promoted to co-primary (not just
    a secondary plot) because, unlike entropy, it cannot mistake the bilinear
    frozen-position pathology (entropy -> 0 while duality gap stays nonzero,
    documented in References/Notes___Conic_Particle_Methods_with_Slingshot_GDA.pdf)
    for convergence -- it directly measures distance to the right answer.

Entropy is logged but kept secondary for exactly that reason.
"""
import numpy as np

from games.registry import GameSpec
from algo.conic_particle import ParticleState


def entropy(weights: np.ndarray) -> float:
    """Shannon entropy H(w) = -sum w_i log w_i. Zero iff w is a Dirac mass."""
    w = np.maximum(weights, 1e-15)
    return float(-np.sum(w * np.log(w)))


def duality_gap_grid(game: GameSpec, state: ParticleState, grid_size: int = 400) -> float:
    """Nikaido-Isoda error: max_y' F(mu,y') - min_x' F(x',nu), both sup/inf
    taken over a fine grid covering the game's actual domain (not just the
    opponent's current particles). >= 0, equals 0 iff (mu,nu) is an MNE.
    Ported from the grid-based compute_ni_error in
    References/particle_random_fourier_1D.ipynb, generalized to euclidean
    domains as well as the torus.
    """
    if game.domain == "torus":
        grid = np.linspace(0.0, 1.0, grid_size, endpoint=False)
    else:
        lo, hi = game.bounds
        grid = np.linspace(lo, hi, grid_size)

    F_y = game.payoff(state.x[:, None], grid[None, :])  # (n, grid)
    max_y = np.max(state.a @ F_y)

    F_x = game.payoff(grid[:, None], state.y[None, :])  # (grid, m)
    min_x = np.min(F_x @ state.b)

    return float(max_y - min_x)


def wasserstein_distance_1d(u_values, v_values, u_weights=None, v_weights=None) -> float:
    """1D Wasserstein-2 distance between weighted empirical distributions on
    the real line (no wraparound). Ported faithfully from the pure-numpy
    quantile-matching implementation shared by References/CPMDA_Slingshot.ipynb
    and References/CPMP_Slingshot.ipynb."""
    u_values = np.asarray(u_values, dtype=float)
    v_values = np.asarray(v_values, dtype=float)

    u_weights = np.ones_like(u_values) / len(u_values) if u_weights is None \
        else np.asarray(u_weights, dtype=float) / np.sum(u_weights)
    v_weights = np.ones_like(v_values) / len(v_values) if v_weights is None \
        else np.asarray(v_weights, dtype=float) / np.sum(v_weights)

    u_sorter = np.argsort(u_values)
    u_values, u_weights = u_values[u_sorter], u_weights[u_sorter]
    v_sorter = np.argsort(v_values)
    v_values, v_weights = v_values[v_sorter], v_weights[v_sorter]

    u_cdf = np.cumsum(u_weights); u_cdf[-1] = 1.0
    v_cdf = np.cumsum(v_weights); v_cdf[-1] = 1.0

    cdf_axis = np.insert(np.unique(np.concatenate((u_cdf, v_cdf))), 0, 0.0)
    widths = np.diff(cdf_axis)
    midpoints = cdf_axis[:-1] + widths / 2

    u_q = u_values[0] if len(u_values) == 1 else u_values[np.searchsorted(u_cdf, midpoints)]
    v_q = v_values[0] if len(v_values) == 1 else v_values[np.searchsorted(v_cdf, midpoints)]

    return float(np.sqrt(np.sum((u_q - v_q) ** 2 * widths)))


def _empirical_cdf_torus(positions: np.ndarray, weights: np.ndarray, t: np.ndarray) -> np.ndarray:
    pos = np.mod(positions, 1.0)
    return np.array([np.sum(weights[pos <= tt]) for tt in t])


def wasserstein_distance_torus(positions, weights, target_positions, target_weights,
                                grid_size: int = 2000) -> float:
    """Circular Wasserstein-1 distance via the Werman-Peleg-Rosenfeld weighted-
    median trick: on the circle, W1 = min_s mean(|F(t)-G(t)-s|), with the
    minimizing s the median of (F-G). Needed because chizat_example_4_1's
    domain is the torus [0,1), where a naive line-distance would overstate
    the cost of particles that wrap around 0/1 (e.g. between a particle at
    0.95 and an MNE atom at 0.05, the true circular distance is 0.1, not 0.9).
    """
    t = np.linspace(0.0, 1.0, grid_size, endpoint=False)
    F = _empirical_cdf_torus(positions, weights, t)
    G = _empirical_cdf_torus(target_positions, target_weights, t)
    diff = F - G
    s = np.median(diff)
    return float(np.mean(np.abs(diff - s)))


def wasserstein_to_mne(state: ParticleState, game: GameSpec):
    """Distance from the current (mu_t, nu_t) to the known closed-form MNE,
    dispatched by domain. Returns (None, None) if the game has no known MNE
    (e.g. fourier_random_k once revived)."""
    if game.known_mne is None:
        return None, None

    mu_pos, mu_w = game.known_mne["mu"]
    nu_pos, nu_w = game.known_mne["nu"]

    if game.domain == "torus":
        w_mu = wasserstein_distance_torus(state.x, state.a, mu_pos, mu_w)
        w_nu = wasserstein_distance_torus(state.y, state.b, nu_pos, nu_w)
    else:
        w_mu = wasserstein_distance_1d(state.x, mu_pos, state.a, mu_w)
        w_nu = wasserstein_distance_1d(state.y, nu_pos, state.b, nu_w)

    return float(w_mu), float(w_nu)


def fit_log_linear_rate(values, iterations=None) -> float:
    """Slope of log(metric) vs iteration, fit by least squares over the
    points where metric > 0. A single comparable "improvement" number per
    run -- more negative means faster decay means stronger convergence.
    Returns nan if fewer than 2 usable points."""
    values = np.asarray(values, dtype=float)
    if iterations is None:
        iterations = np.arange(len(values))
    else:
        iterations = np.asarray(iterations, dtype=float)

    mask = values > 0
    if mask.sum() < 2:
        return float("nan")

    slope, _ = np.polyfit(iterations[mask], np.log(values[mask]), 1)
    return float(slope)
