"""Tabular/simplex specialization of Nash-MD (Munos et al., "Nash Learning
from Human Feedback", arXiv:2312.00886): a mirror-descent step on the
preference gradient, geometrically mixed with a fixed reference (anchor)
policy. tau=0 recovers plain mirror descent -- the vanilla self-play
baseline this is compared against in run_poc.py.

This is a tabular reduction of the mechanism, not a line-for-line port of
Munos et al.'s deep-RL implementation (no parametric policy, no
regularized-reward inner objective) -- see poc/report.tex for exactly what
is and isn't being claimed.
"""
import numpy as np

_LOG_FLOOR = 1e-15


def nash_md_step(a: np.ndarray, b: np.ndarray, grad_a: np.ndarray, grad_b: np.ndarray,
                  eta: float, tau: float, a_anchor: np.ndarray, b_anchor: np.ndarray) -> tuple:
    """Same log-sum-exp-stabilized multiplicative-weights update as
    algo.conic_particle._weight_step, plus a geometric pull of strength tau
    toward (a_anchor, b_anchor) at every step."""
    log_a = (1 - tau) * np.log(np.maximum(a, _LOG_FLOOR)) \
        + tau * np.log(np.maximum(a_anchor, _LOG_FLOOR)) - eta * grad_a
    log_a -= np.max(log_a)
    a_new = np.exp(log_a)
    a_new /= a_new.sum()

    log_b = (1 - tau) * np.log(np.maximum(b, _LOG_FLOOR)) \
        + tau * np.log(np.maximum(b_anchor, _LOG_FLOOR)) + eta * grad_b
    log_b -= np.max(log_b)
    b_new = np.exp(log_b)
    b_new /= b_new.sum()
    return a_new, b_new
