from dataclasses import dataclass
from typing import Callable, Optional
import numpy as np


@dataclass
class GameSpec:
    """A continuous two-player zero-sum game min_mu max_nu F(mu,nu) = E[f(x,y)].

    payoff(x, y) and grad(x, y) -> (df/dx, df/dy) must broadcast elementwise,
    so they can be called both on scalars and on (n,1)/(1,m) grids.

    schedule_family / default_spectral tell the CLI which slingshot schedule
    applies and with what spectral parameters, per Shugart & Altschuler:
    "chebyshev" for linear-gradient (bilinear/quadratic) games needs {m, M}
    (the singular-value spectrum), "randomized_sign" for nonlinear
    convex-concave games needs {L} (a Lipschitz constant of the gradient).
    The two families are not interchangeable -- see References/CLAUDE.md.
    """
    name: str
    domain: str  # "torus" or "euclidean"
    payoff: Callable[[np.ndarray, np.ndarray], np.ndarray]
    grad: Callable[[np.ndarray, np.ndarray], tuple]
    schedule_family: str  # "chebyshev" or "randomized_sign"
    default_spectral: dict
    bounds: Optional[tuple] = None  # (low, high), required iff domain == "euclidean"
    known_mne: Optional[dict] = None  # {"mu": (positions, weights), "nu": (positions, weights)}


_REGISTRY: dict[str, GameSpec] = {}


def register(spec: GameSpec) -> GameSpec:
    _REGISTRY[spec.name] = spec
    return spec


def get_game(name: str) -> GameSpec:
    if name not in _REGISTRY:
        raise KeyError(f"Unknown game {name!r}. Available: {sorted(_REGISTRY)}")
    return _REGISTRY[name]


def list_games() -> list[str]:
    return sorted(_REGISTRY)


def sanity_check(name: str, fd_eps: float = 1e-6, fd_tol: float = 1e-3, grid_size: int = 2000) -> None:
    """Checks a registered game against its own closed-form ground truth:
    1. grad matches a central finite-difference of payoff at random points.
    2. if known_mne is given, the duality gap at the known MNE is ~0 (grid search).
    Raises AssertionError on failure; prints nothing on success.
    """
    game = get_game(name)
    rng = np.random.default_rng(0)

    if game.domain == "torus":
        xs = rng.uniform(0.0, 1.0, 20)
        ys = rng.uniform(0.0, 1.0, 20)
    else:
        lo, hi = game.bounds
        xs = rng.uniform(lo, hi, 20)
        ys = rng.uniform(lo, hi, 20)

    gx, gy = game.grad(xs, ys)
    fd_x = (game.payoff(xs + fd_eps, ys) - game.payoff(xs - fd_eps, ys)) / (2 * fd_eps)
    fd_y = (game.payoff(xs, ys + fd_eps) - game.payoff(xs, ys - fd_eps)) / (2 * fd_eps)
    assert np.allclose(gx, fd_x, atol=fd_tol), f"{name}: grad_x disagrees with finite difference"
    assert np.allclose(gy, fd_y, atol=fd_tol), f"{name}: grad_y disagrees with finite difference"

    if game.known_mne is not None:
        from metrics.diagnostics import duality_gap_grid
        from algo.conic_particle import ParticleState

        mu_pos, mu_w = game.known_mne["mu"]
        nu_pos, nu_w = game.known_mne["nu"]
        state = ParticleState(a=np.asarray(mu_w, dtype=float), x=np.asarray(mu_pos, dtype=float),
                               b=np.asarray(nu_w, dtype=float), y=np.asarray(nu_pos, dtype=float))
        gap = duality_gap_grid(game, state, grid_size=grid_size)
        assert gap < 1e-2, f"{name}: duality gap at known MNE is {gap}, expected ~0"
