"""One WFR (Wasserstein-Fisher-Rao) step primitive, parametrized by L = number
of inner gradient-recompute passes per outer iteration:
  L=1 -> CP-MDA (explicit, single gradient eval, no convergence guarantee)
  L=2 -> CP-MP  (extragradient / mirror-prox, Chizat's practical algorithm)
  L>2 -> CP-PP  (implicit proximal point) -- theory-only, not implemented.

Ported and unified from the (near-duplicate) ConicParticles/update_mda in
References/CPMDA_Slingshot.ipynb and ConicParticlesMP/update_mp in
References/CPMP_Slingshot.ipynb. Stabilization tricks preserved from there:
log-sum-exp shift on the Fisher-Rao weight update, and the pre-update weights
used to precondition the Wasserstein position step (using post-update weights
here would create a feedback loop and was the key numerical-stability fix in
both source notebooks).

The stepsize schedule is always passed in as a callable (t) -> (alpha, beta)
-- this module never imports anything from slingshot/, so swapping the
schedule never touches this code.
"""
from dataclasses import dataclass
import numpy as np

from games.registry import GameSpec

_WEIGHT_FLOOR = 1e-3   # denominator floor when preconditioning by 1/a_i
_CLIP_BUDGET = 0.1     # adaptive per-particle step clip: min(1, _CLIP_BUDGET / a_i)
_LOG_FLOOR = 1e-15     # floor before taking log of a weight


@dataclass
class ParticleState:
    a: np.ndarray  # (n,) player-1 (min) weights, simplex
    x: np.ndarray  # (n,) player-1 positions
    b: np.ndarray  # (m,) player-2 (max) weights, simplex
    y: np.ndarray  # (m,) player-2 positions

    def copy(self) -> "ParticleState":
        return ParticleState(self.a.copy(), self.x.copy(), self.b.copy(), self.y.copy())


def init_state(n: int, m: int, game: GameSpec, rng: np.random.Generator) -> ParticleState:
    if game.domain == "torus":
        x = rng.uniform(0.0, 1.0, n)
        y = rng.uniform(0.0, 1.0, m)
    else:
        lo, hi = game.bounds
        x = rng.uniform(lo, hi, n)
        y = rng.uniform(lo, hi, m)
    return ParticleState(a=np.ones(n) / n, x=x, b=np.ones(m) / m, y=y)


def _sample_near_atoms(n: int, atom_pos, atom_w, rng: np.random.Generator,
                        noise: float, game: GameSpec) -> np.ndarray:
    """n particles split across known MNE atoms (proportional to atom_w),
    each perturbed by N(0, noise^2). Lets a run start already in the local
    neighborhood of the equilibrium support, rather than spread uniformly
    over the whole domain -- the regime Shugart's slingshot guarantee
    (linearization near the saddle) actually covers."""
    atom_pos = np.asarray(atom_pos, dtype=float)
    atom_w = np.asarray(atom_w, dtype=float)
    counts = np.floor(n * atom_w / atom_w.sum()).astype(int)
    counts[-1] = n - counts[:-1].sum()  # exact total, remainder to last atom

    positions = np.concatenate([
        rng.normal(loc=atom_pos[k], scale=noise, size=counts[k])
        for k in range(len(atom_pos))
    ])
    rng.shuffle(positions)
    return _project(positions, game)


def init_state_near_mne(n: int, m: int, game: GameSpec, rng: np.random.Generator,
                         noise: float = 0.01) -> ParticleState:
    if game.known_mne is None:
        raise ValueError(f"{game.name} has no known_mne to warm-start near.")
    mu_pos, mu_w = game.known_mne["mu"]
    nu_pos, nu_w = game.known_mne["nu"]
    x = _sample_near_atoms(n, mu_pos, mu_w, rng, noise, game)
    y = _sample_near_atoms(m, nu_pos, nu_w, rng, noise, game)
    return ParticleState(a=np.ones(n) / n, x=x, b=np.ones(m) / m, y=y)


def _project(pos: np.ndarray, game: GameSpec) -> np.ndarray:
    if game.domain == "torus":
        return np.mod(pos, 1.0)
    lo, hi = game.bounds
    return np.clip(pos, lo, hi)


def _compute_gradients(state: ParticleState, game: GameSpec):
    """O(n*m) pairwise payoff/gradient sums, vectorized via broadcasting."""
    n, m = len(state.x), len(state.y)
    X = state.x[:, None]  # (n,1)
    Y = state.y[None, :]  # (1,m)
    F = game.payoff(X, Y)            # (n,m)
    Gx, Gy = game.grad(X, Y)
    # Games whose grad component depends on only one of x/y (e.g.
    # convex_concave's tanh(x)/-tanh(y)) return (n,1)/(1,m), not (n,m) --
    # broadcast explicitly so the @ state.b / state.a @ below is well-defined.
    Gx = np.broadcast_to(Gx, (n, m))
    Gy = np.broadcast_to(Gy, (n, m))
    grad_a = F @ state.b             # (n,)  E_{y~b}[f(x_i,y)]
    grad_b = state.a @ F             # (m,)  E_{x~a}[f(x,y_j)]
    grad_x = Gx @ state.b            # (n,)
    grad_y = state.a @ Gy            # (m,)
    return grad_a, grad_x, grad_b, grad_y


def _weight_step(a, b, grad_a, grad_b, alpha, beta):
    """Fisher-Rao multiplicative mirror-descent step, log-sum-exp stabilized."""
    log_a = np.log(np.maximum(a, _LOG_FLOOR)) - alpha * grad_a
    log_a -= np.max(log_a)
    a_new = np.exp(log_a)
    a_new /= a_new.sum()

    log_b = np.log(np.maximum(b, _LOG_FLOOR)) + beta * grad_b
    log_b -= np.max(log_b)
    b_new = np.exp(log_b)
    b_new /= b_new.sum()
    return a_new, b_new


def _position_step(a_ref, x, b_ref, y, grad_x, grad_y, alpha, beta, game: GameSpec):
    """Wasserstein gradient step.

    On euclidean domains (References/CPMDA_Slingshot.ipynb,
    CPMP_Slingshot.ipynb): preconditioned by 1/a_i using PRE-UPDATE weights.
    This is a deliberate exploration heuristic, not literal WFR theory --
    near-zero-weight "ghost" particles get amplified, exploratory kicks --
    paired with an adaptive clip (`min(1, 0.1/a_i)`) so it doesn't blow up.
    There is no periodic safety net on this domain, so the clip is load
    bearing.

    On the torus (References/particle_random_fourier_1D.ipynb): no 1/a_i
    division at all. That notebook's Dx[i] is built as
    `wxp[i] * sum_j wyp[j] * gx(...)` and then divided back by `wxp[i]`,
    so the mass factor cancels exactly -- the faithful position step is a
    plain unweighted gradient step `x - eta * grad_x`, with no clip (only a
    1e-15 floor against literal zero-division, which _WEIGHT_FLOOR already
    covers). Carrying over the euclidean games' 1/a_i amplification here was
    the actual bug behind flat, non-converging duality gaps in initial
    smoke-testing: with ~30 particles, a_i ~= 1/30, so every step was
    amplified by ~30x regardless of the clip constant.
    """
    if game.domain == "torus":
        return (_project(x - alpha * grad_x, game),
                _project(y + beta * grad_y, game))

    denom_x = np.maximum(a_ref, _WEIGHT_FLOOR)
    denom_y = np.maximum(b_ref, _WEIGHT_FLOOR)
    clip_x = np.minimum(1.0, _CLIP_BUDGET / denom_x)
    clip_y = np.minimum(1.0, _CLIP_BUDGET / denom_y)
    step_x = np.clip(alpha * grad_x / denom_x, -clip_x, clip_x)
    step_y = np.clip(beta * grad_y / denom_y, -clip_y, clip_y)

    return _project(x - step_x, game), _project(y + step_y, game)


def _full_step(state: ParticleState, game: GameSpec, alpha: float, beta: float) -> ParticleState:
    grad_a, grad_x, grad_b, grad_y = _compute_gradients(state, game)
    a_ref, b_ref = state.a.copy(), state.b.copy()
    a_new, b_new = _weight_step(state.a, state.b, grad_a, grad_b, alpha, beta)
    x_new, y_new = _position_step(a_ref, state.x, b_ref, state.y, grad_x, grad_y, alpha, beta, game)
    return ParticleState(a_new, x_new, b_new, y_new)


def wfr_step(state: ParticleState, game: GameSpec, alpha: float, beta: float, L: int) -> ParticleState:
    """One outer WFR iteration.

    L=1 (CP-MDA): a single explicit full_step from the current state.
    L=2 (CP-MP):  extragradient/mirror-prox -- a "prediction" full_step with
                  the (possibly negative, slingshot) signs, then gradients
                  are recomputed at the predicted state and the real
                  "correction" full_step is taken from the ORIGINAL state
                  using those lookahead gradients, but with |alpha|, |beta|.
                  The slingshot sign only ever shapes the prediction; the
                  correction is always a genuine improvement step (this
                  detail is load-bearing -- see the "KEY FIX" comments in
                  References/CPMP_Slingshot.ipynb).
    """
    if L == 1:
        return _full_step(state, game, alpha, beta)

    if L == 2:
        predicted = _full_step(state, game, alpha, beta)
        grad_a2, grad_x2, grad_b2, grad_y2 = _compute_gradients(predicted, game)
        a_ref, b_ref = state.a.copy(), state.b.copy()
        a_new, b_new = _weight_step(state.a, state.b, grad_a2, grad_b2, abs(alpha), abs(beta))
        x_new, y_new = _position_step(
            a_ref, state.x, b_ref, state.y, grad_x2, grad_y2, abs(alpha), abs(beta), game)
        return ParticleState(a_new, x_new, b_new, y_new)

    raise NotImplementedError(
        f"L={L} is not implemented. L=1 (CP-MDA) and L=2 (CP-MP) are the only "
        "practically run algorithms here; L->infinity (CP-PP, implicit proximal "
        "point) is a theory-only reference result in Chizat (2025) -- see "
        "References/CLAUDE.md."
    )
