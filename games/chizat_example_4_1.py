import numpy as np
from .registry import GameSpec, register


def payoff(x, y):
    return np.sin(4 * np.pi * x) + np.sin(4 * np.pi * y) + 2 * np.cos(2 * np.pi * (x + y))


def grad(x, y):
    # Ported from References/particle_random_fourier_1D.ipynb, fixing two bugs found there:
    # (1) it returned the unused zero-initialized (gx, gy) instead of the computed (g_x, g_y),
    #     so the original notebook's grad_f always returned (0.0, 0.0);
    # (2) the y-component used `**` (exponentiation) instead of `*` (multiplication).
    gx = 4 * np.pi * np.cos(4 * np.pi * x) - 4 * np.pi * np.sin(2 * np.pi * (x + y))
    gy = 4 * np.pi * np.cos(4 * np.pi * y) - 4 * np.pi * np.sin(2 * np.pi * (x + y))
    return gx, gy


CHIZAT_EXAMPLE_4_1 = register(GameSpec(
    name="chizat_example_4_1",
    domain="torus",
    payoff=payoff,
    grad=grad,
    schedule_family="randomized_sign",
    # Crude Hessian-norm bound (16pi^2 + 8pi^2) on grad f -- a starting point for the
    # randomized-sign schedule's h=1/(3L), not a certified Lipschitz constant. Override
    # via --lipschitz-L if a run is unstable or you want to sweep it.
    default_spectral={"L": 24 * np.pi ** 2},
    known_mne={
        "mu": (np.array([3 / 8, 7 / 8]), np.array([0.5, 0.5])),
        "nu": (np.array([1 / 8, 5 / 8]), np.array([0.5, 0.5])),
    },
))
