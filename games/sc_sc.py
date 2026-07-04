import numpy as np
from .registry import GameSpec, register

MU = 1.0


def payoff(x, y):
    return 0.5 * MU * x ** 2 - 0.5 * MU * y ** 2 + x * y


def grad(x, y):
    return MU * x + y, -MU * y + x


SC_SC = register(GameSpec(
    name="sc_sc",
    domain="euclidean",
    bounds=(-10.0, 10.0),
    payoff=payoff,
    grad=grad,
    schedule_family="randomized_sign",
    default_spectral={"L": float(np.sqrt(2))},
    known_mne={
        "mu": (np.array([0.0]), np.array([1.0])),
        "nu": (np.array([0.0]), np.array([1.0])),
    },
))
