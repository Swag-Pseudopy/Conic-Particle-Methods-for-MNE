import numpy as np
from .registry import GameSpec, register


def _logcosh(x):
    s = np.abs(x)
    return s + np.log1p(np.exp(-2 * s)) - np.log(2.0)


def payoff(x, y):
    return _logcosh(x) - _logcosh(y)


def grad(x, y):
    return np.tanh(x), -np.tanh(y)


CONVEX_CONCAVE = register(GameSpec(
    name="convex_concave",
    domain="euclidean",
    bounds=(-10.0, 10.0),
    payoff=payoff,
    grad=grad,
    schedule_family="randomized_sign",
    default_spectral={"L": 1.0},  # tanh' = sech^2 <= 1
    known_mne={
        "mu": (np.array([0.0]), np.array([1.0])),
        "nu": (np.array([0.0]), np.array([1.0])),
    },
))
