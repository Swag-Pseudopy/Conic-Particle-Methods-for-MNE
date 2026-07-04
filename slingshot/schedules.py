"""Stepsize-getter functions only -- pure functions of (t) -> (alpha_t, beta_t).

Ported faithfully from the formulas in References/CPMDA_Slingshot.ipynb and
References/CPMP_Slingshot.ipynb (both notebooks carry an identical
get_slingshot_steps), which in turn implement Shugart & Altschuler (2025).

The two slingshot schedules are NOT interchangeable -- the paper states the
Chebyshev-node schedule is for linear-gradient (bilinear/quadratic) games and
fails on nonlinear-gradient games, while the randomized-sign schedule is for
nonlinear convex-concave games. See References/CLAUDE.md.

Every schedule here is a plain closure (t: int) -> (alpha_t, beta_t), so the
algo module never imports a schedule directly -- it is always passed in.
"""
from typing import Callable, Optional
import numpy as np

StepSchedule = Callable[[int], tuple]


def vanilla(eta: float) -> StepSchedule:
    """Constant positive stepsize, same for both players -- the no-slingshot
    baseline. The *same* algo code path runs this as runs any slingshot
    schedule, since both are just (t) -> (alpha, beta) closures."""
    def schedule(t: int) -> tuple:
        return eta, eta
    return schedule


def chebyshev(T: int, m: float, M: float) -> StepSchedule:
    """Chebyshev-node slingshot schedule for linear-gradient (bilinear/
    quadratic) games, over the singular-value spectrum [m, M]. Achieves the
    optimal O(sqrt(kappa) log(1/eps)) rate per Shugart & Altschuler. Period T
    should match the total iteration budget of the run."""
    def schedule(t: int) -> tuple:
        k = t // 2
        r_t = (M + m) / 2 + (M - m) / 2 * np.cos((2 * k + 1) * np.pi / (2 * T))
        h = 1.0 / np.sqrt(r_t)
        return (h, -h) if t % 2 == 0 else (-h, h)
    return schedule


def randomized_sign(L: float, rng: Optional[np.random.Generator] = None) -> StepSchedule:
    """Randomized-sign slingshot schedule for nonlinear convex-concave games,
    with stepsize magnitude h = 1/(3L) for a Lipschitz constant L of the
    gradient. Sign is flipped with a fair coin on even steps, held positive
    on odd steps (the "correction" half of each pair)."""
    rng = rng if rng is not None else np.random.default_rng()
    h = 1.0 / (3.0 * L)

    def schedule(t: int) -> tuple:
        if t % 2 == 0:
            return (h, -h) if rng.random() < 0.5 else (-h, h)
        return h, h
    return schedule


SCHEDULE_FAMILIES = {
    "chebyshev": chebyshev,
    "randomized_sign": randomized_sign,
}
