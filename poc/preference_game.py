"""Finite, tabular preference games -- the Nash-equilibrium-seeking
preference-optimization problem class (Munos et al.'s Nash Learning from
Human Feedback; Swamy et al.'s minimax-winner framing), at the smallest
scale that still has a closed-form Nash equilibrium to check against. No
LLM, no learned preference model: P[i, j] is a literal, hand-specified
pairwise win-probability matrix.

Both games here are "generalized rock-paper-scissors": n actions arranged in
a cycle, each beating the next k and losing to the previous k, margin p.
This is the textbook counterexample to Bradley-Terry (no scalar reward can
represent it -- if one could, the highest-reward action would beat
everything, but every action here loses to roughly half the others) and,
because the construction is circulant and antisymmetric, every row of
F = P - 0.5 sums to exactly zero, so the uniform distribution is *always* a
Nash equilibrium at game value 0 -- the same row-sum-zero argument that
makes uniform the equilibrium of ordinary rock-paper-scissors. That gives a
closed-form ground truth to measure against, exactly like the known_mne
field every other game in this project registers.

n=7 is a structured (not literally random) generalization of n=3 RPS, used
to check the headline result isn't an artifact of RPS's small size or extra
symmetry -- see References/preference_optimization_ideas.md, "Stage 1".
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class PreferenceGame:
    name: str
    F: np.ndarray           # (n, n) antisymmetric, F[i, j] = P(i beats j) - 0.5
    uniform_ne: np.ndarray  # (n,), the closed-form Nash equilibrium weights


def make_cyclic_dominance_game(n: int, p: float, name: str) -> PreferenceGame:
    """n must be odd, so every pair is compared (no ties). Action i beats the
    k = (n-1)//2 actions "behind" it cyclically with probability p, and loses
    to the k "ahead" of it. n=3, k=1, labeled (Rock, Paper, Scissors) is
    exactly ordinary rock-paper-scissors."""
    assert n % 2 == 1, "cyclic dominance needs odd n so every pair is compared"
    k = (n - 1) // 2
    F = np.zeros((n, n))
    for i in range(n):
        for d in range(1, k + 1):
            F[i, (i - d) % n] = p - 0.5
            F[i, (i + d) % n] = -(p - 0.5)
    return PreferenceGame(name=name, F=F, uniform_ne=np.ones(n) / n)


RPS = make_cyclic_dominance_game(n=3, p=0.90, name="rps3")
CYCLIC7 = make_cyclic_dominance_game(n=7, p=0.85, name="cyclic7")


def spectral_bounds(F: np.ndarray, tol: float = 1e-9) -> tuple:
    """Smallest/largest *nonzero* singular value of F. The all-ones vector is
    always in F's kernel (every row of a cyclic-dominance matrix sums to
    zero), which is irrelevant on the simplex -- you cannot move in the
    "shift every weight by a constant" direction and stay on it. Excluding
    that direction is the same m=M degenerate-spectrum situation already
    documented for games/bilinear.py, just confirmed here by direct
    computation rather than a 1x1 matrix being trivially degenerate."""
    svals = np.linalg.svd(F, compute_uv=False)
    nonzero = svals[svals > tol * svals.max()]
    return float(nonzero.min()), float(nonzero.max())


def exact_duality_gap(F: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    """Nikaido-Isoda error by exact finite enumeration -- no grid, no
    discretization error, the one methodological advantage of a finite
    action set over the continuous games elsewhere in this project."""
    max_y = float(np.max(a @ F))
    min_x = float(np.min(F @ b))
    return max_y - min_x


def tv_distance(w: np.ndarray, target: np.ndarray) -> float:
    """Total variation distance between two mixed strategies -- the
    domain-appropriate replacement for metrics.wasserstein_to_mne here: a
    finite, unordered action set has no spatial metric for a Wasserstein
    distance to be computed against (action 'Paper' is not 'between' Rock
    and Scissors), so TV distance on the simplex is the standard notion of
    distance to equilibrium, not a continuous-domain metric repurposed by
    analogy. See poc/report.tex for why this matters."""
    return float(0.5 * np.sum(np.abs(w - target)))


def sample_noisy_gradients(F: np.ndarray, a: np.ndarray, b: np.ndarray, K: int,
                            rng: np.random.Generator) -> tuple:
    """Unbiased Monte-Carlo estimate of (grad_a, grad_b) = (F @ b, a @ F) from
    K sampled opponent actions and one Bernoulli preference judgement per
    comparison -- the "real battlefield" noise model: a real self-play
    preference learner never sees the opponent's exact mixed strategy or the
    exact win-probability, only a finite batch of sampled completions and a
    finite batch of (typically binary) preference judgements over them.
    Estimator variance shrinks as O(1/K); K is the per-step comparison
    budget."""
    n, m = F.shape
    j_samples = rng.choice(m, size=K, p=b)
    p_i_beats_j = np.clip(F[:, j_samples] + 0.5, 0.0, 1.0)        # (n, K)
    grad_a_hat = (rng.random((n, K)) < p_i_beats_j).mean(axis=1) - 0.5

    i_samples = rng.choice(n, size=K, p=a)
    p_i_beats_j2 = np.clip(F[i_samples, :] + 0.5, 0.0, 1.0)       # (K, m)
    grad_b_hat = (rng.random((K, m)) < p_i_beats_j2).mean(axis=0) - 0.5

    return grad_a_hat, grad_b_hat
