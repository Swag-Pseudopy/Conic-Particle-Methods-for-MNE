import numpy as np
from .registry import GameSpec, register


def payoff(x, y):
    return x * y


def grad(x, y):
    return y, x


BILINEAR = register(GameSpec(
    name="bilinear",
    domain="euclidean",
    bounds=(-10.0, 10.0),
    payoff=payoff,
    grad=grad,
    schedule_family="chebyshev",
    default_spectral={"m": 1.0, "M": 1.0},  # singular value of B=[[1.0]]
    known_mne={
        "mu": (np.array([0.0]), np.array([1.0])),
        "nu": (np.array([0.0]), np.array([1.0])),
    },
))
