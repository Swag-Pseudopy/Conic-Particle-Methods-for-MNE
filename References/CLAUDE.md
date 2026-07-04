# References — Conic Particle Methods + Slingshot GDA for Mixed Nash Equilibria

This folder is the literature + prototype-code base for a research project synthesizing two
2025 min-max optimization papers, with an eventual eye toward Nash-equilibrium-based LLM
preference optimization (à la Munos et al.'s NLHF). This file orients any future session on the
math, the terminology, and the state of the existing notebooks.

## The big picture

Three papers here share one theme: **last-iterate convergence to a Nash/saddle point via
mirror-descent-family dynamics**, avoiding the averaging/mixture-storage that classical
regret-minimization (fictitious play, vanilla mirror descent) needs — which is impractical when
"strategies" are continuous distributions or LLM policies.

- **Wang & Chizat (2025)** — how to find the Nash equilibrium when strategies are continuous
  (atomic-measure / particle representation, Wasserstein–Fisher–Rao geometry).
- **Shugart & Altschuler (2025)** — a stepsize trick ("slingshot") that makes plain
  gradient-descent-ascent converge last-iterate, even on the bilinear counterexample that has
  defeated GDA since 1976.
- **Munos et al. (NLHF, 2024)** — why last-iterate Nash computation matters for aligning LLMs
  with preference data, and a mirror-descent algorithm (Nash-MD) for the *finite/tabular* policy
  case.

The concrete, near-term research plan (already prototyped in the notebooks below) is to graft
Shugart's slingshot stepsizes onto Chizat's conic particle method, and check whether slingshot
buys convergence (or speed) on a case **Chizat's own paper documents as a failure case for the
naive variant of his algorithm** — see "The smoking-gun example" below. The longer-term aim
(connecting back to Munos/NLHF) is open research, not something already solved in these papers —
treat that link as a direction, not a result.

## Source files

| File | What it is |
|---|---|
| `[Lénaïc Chizat] An Exponentially Converging Particle Method...pdf` | Wang & Chizat 2025 (arXiv:2211.01280v4). The conic particle method (CP-PP/CP-MP/CP-MDA). |
| `[Shugart] Negative Stepsizes Make Gradient-Descent-Ascent Converge.pdf` | Shugart & Altschuler 2025 (arXiv:2505.01423). The "slingshot" stepsize schedules. |
| `nlhf_munos.pdf` | Munos et al., "Nash Learning from Human Feedback" (ICML 2024). Nash-MD, preference models, last-iterate KL convergence. |
| `Notes___Conic_Particle_Methods_with_Slingshot_GDA.pdf` | **Own synthesis doc.** Combines the two papers above; documents CP-MDA/CP-MP pseudocode, stability tricks, a bilinear pathology, and a "Random Fourier Torus Game" validation. Treat as the most implementation-relevant reference. |
| `CPMDA_Slingshot.ipynb` | Prototype: Conic Particle Mirror Descent-Ascent + slingshot, on bilinear / convex-concave / SC-SC games. |
| `CPMP_Slingshot.ipynb` | Prototype: Conic Particle Mirror Prox (extragradient) + slingshot, same three games. |
| `particle_random_fourier_1D.ipynb` | Prototype: 1D torus Fourier-feature game used to stress-test global (non-convex) convergence. **Currently runs Chizat's `Example 4.1` exactly, not a random K-feature game — see Known Issues.** |

## Core concepts cheat-sheet

**Problem.** Two-player zero-sum game `min_μ max_ν F(μ,ν) = E_{x~μ,y~ν}[f(x,y)]` over
probability measures (mixed strategies). Solution = Mixed Nash Equilibrium (MNE) `(μ*, ν*)`.

**Atomic/particle parametrization** (Chizat): `μ = Σ aᵢδ_{xᵢ}`, `ν = Σ bⱼδ_{yⱼ}`. Each particle
`(aᵢ, xᵢ)` is a pure strategy `xᵢ` with probability mass `aᵢ`.

**Wasserstein–Fisher–Rao (WFR) split**: two different geometries act on the two halves of each
particle.
- *Fisher–Rao* on weights `aᵢ` → multiplicative **mirror descent** (entropy mirror map),
  i.e. `aᵢ ∝ aᵢ·exp(-η ∂F/∂aᵢ)`. Reallocates mass between particles.
- *Wasserstein* on positions `xᵢ` → gradient step **preconditioned by `1/aᵢ`**. Moves particles
  through space. Heavy particles take small careful steps; near-zero-weight ("ghost") particles
  take huge exploratory steps for free. **Caveat found during the port** (see "Implementation
  findings" below): this `1/aᵢ` precondition is a deliberate exploration heuristic used only by
  the euclidean-game notebooks (paired with an adaptive clip); the torus/Fourier notebook's
  position step has no such precondition at all — the mass factor cancels out of its update
  algebraically. The two are genuinely different design choices, not one rule with an exception.
- **Gotcha (critical):** the position step must use the **pre-update** weight `aᵢ_old`, not the
  just-updated weight. Using the new weight creates a feedback loop (a particle just downweighted
  for being bad immediately gets a huge destabilizing kick). Both prototype notebooks call this
  out explicitly as "KEY FIX".

**Algorithm family — all three are the same WFR step, with different repetition count `L`:**
| Name | `L` | What it is | Convergence guarantee |
|---|---|---|---|
| CP-MDA | 1 | Explicit, single-step (Mirror **Descent**-Ascent) | **None.** Chizat's own paper: *"the explicit method CP-MDA does not always converge."* |
| CP-MP | 2 | Extragradient / Mirror **Prox**: predict, then correct with a second gradient eval | Empirically matches CP-PP; proven equal in the exact-parametrization case |
| CP-PP | ∞ | Implicit Proximal Point (solves an inner min-max exactly each step) | **Proven** local exponential rate (Thm 2.2/3.7), under non-degeneracy + a closeness-to-optimum localness condition. Not directly implementable — CP-MP is the practical stand-in. |

**Slingshot stepsizes** (Shugart): break GDA's perpetual cycling by taking **occasionally
negative** steps. Needs all three of: negative stepsizes, asymmetric (`α_t ≠ β_t`), time-varying —
all three are *provably necessary* (not just sufficient) even for the toy bilinear game `xy`.
Mechanism: positive/negative steps cancel to first order (no net displacement) but the **second**-order
cross term `~h²∇²_{xy}f` survives — this is exactly one finite-difference step of *Hamiltonian
Gradient Descent* on `Φ(x,y) = ½‖∇f(x,y)‖²`, whose minimizers are the saddle points.
- **Two non-interchangeable schedules** — this is a second gotcha, separate from the CP-MDA
  failure above:
  - *Linear* `∇f` (bilinear/quadratic): magnitudes from **Chebyshev polynomial roots** over the
    spectral interval `[m,M]`. Gives the optimal accelerated rate `O(√κ log 1/ε)`.
  - *Nonlinear* `∇f` (convex-concave): fixed magnitude `h=1/(3L)`, **randomized sign** each pair
    of steps (symmetry-breaking).
  - The paper states outright that the linear/Chebyshev schedule **fails to converge** when `∇f`
    is nonlinear — it is not a universal drop-in. Pick the schedule that matches the local
    curvature of the game being solved.

**Convergence diagnostics** (can't just watch a scalar loss — you're optimizing over measures).
Two are **co-primary**: duality gap / Nikaido-Isoda error `max_ν F(μ,ν) - min_μ F(μ,ν)` (the
game-theoretic Nash certificate, always computable even with no known closed-form MNE) and
**Wasserstein distance to the known MNE** (where one exists — promoted from a secondary plot to
co-primary on the user's request, since unlike entropy it can't be fooled by the frozen-position
pathology below: it measures distance to the *right answer*, not just to *a* Dirac). Secondary:
barycenter distance to `(x*,y*)`, strategy entropy `H(a) = -Σ aᵢlog aᵢ` (→0 = correct mass
collapse to a Dirac, *not* a pathology, when the MNE is a pure strategy — but can hit 0 while the
duality gap stays nonzero, see the pathology below), and the Hamiltonian `½‖∇f(x̄,ȳ)‖²`.

**Known pathology (Notes §9):** on the bilinear game, the position gradient `∂F/∂xᵢ = E_ν[y]` is
*independent of `xᵢ`* — every particle gets the same positional push. As `ν→δ₀` this push
vanishes for everyone simultaneously, freezing particle positions even though the *weights*
correctly collapse. Net effect: entropy → 0 correctly, but the duality gap plateaus at a nonzero
residual from frozen off-axis particles. This is a structural feature of bilinear games, not a
bug — four mitigations are listed in the Notes doc (entropy-triggered position reset, FR-only
refinement phase, SC-SC regularization, bigger iteration budget).

## The smoking-gun example (start here)

Chizat's own paper (§4.1, "Example 4.1") gives a small, closed-form 1D game built entirely from
sines and cosines:

```
f(x, y) = sin(4πx) + sin(4πy) + 2·cos(2πx + 2πy),     x, y ∈ 𝕋¹ = [0,1)
MNE:  μ* = ½δ_{3/8} + ½δ_{7/8},     ν* = ½δ_{1/8} + ½δ_{5/8}
```

The paper proves this MNE analytically (3-step argument) and then states: *"experimentally we
observe that CP-MDA does not converge, while CP-MP converges with an exponential rate that scales
as η²."* This is exactly the kind of case the user is after — a documented failure of the base
algorithm, with ground truth known in closed form, ideal as a first unit test for "does adding
slingshot rescue CP-MDA here?" Chizat's broader §4.1 setup (general random trigonometric-polynomial
payoffs on a torus, guaranteed-sparse MNE by separability) is the template the Notes doc's
"Random Fourier Torus Game" (§10) and `particle_random_fourier_1D.ipynb` generalize — but see
below, the notebook currently collapses back down to exactly this fixed example rather than a
true random K-term sum.

## Existing notebook code — structure and known issues

All three notebooks are matplotlib + numpy only (no scipy). Read in full before reusing —
they contain superseded draft cells left in place (notebook-history cruft), not just final code.

**What's duplicated almost verbatim across `CPMDA_Slingshot.ipynb` and `CPMP_Slingshot.ipynb`**
(i.e., exactly what a shared module should absorb): the slingshot scheduler
(`get_slingshot_steps`), the `wasserstein_distance` helper, all three fixed-game payoffs/gradients
(bilinear, convex-concave via stable `logcosh`, SC-SC), all diagnostics (`entropy`, `duality_gap`,
`get_weighted_barycenter`, `compute_regret`, `wasserstein_to_mne`), and all plotting
(`plot_dashboard`, `animate_convergence`, `plot_comparison`). Each notebook also carries a "raw/
unstabilized" baseline class (`ConicParticlesChizat` / `ConicParticlesMPChizat`, no log-sum-exp,
no clipping) alongside the stabilized one — useful as an ablation, worth preserving as a flag
rather than a separate copy-pasted class.

**Numerical-stability tricks actually implemented** (stabilized classes only — absent from the
raw/Chizat baseline classes and from the Fourier notebook): log-sum-exp shift before
exponentiating weights; pre-update-weight preconditioning; adaptive clip
`min(1, 0.1/max(aᵢ_old, 1e-3))`; hard domain projection `clip(x, -D, D)`; stable `logcosh`.

**GIF export** is implemented via `matplotlib.animation.FuncAnimation(...).save(..., writer='pillow')`
in all three notebooks, but disabled (commented out) in the CPMDA/CPMP drivers and live only in
the Fourier notebook. No `gifs/` directory is actually referenced in code (despite the Notes doc
mentioning one) — animation filenames currently land in the working directory.

**Bugs / inconsistencies to fix during the port, not preserve:**
- `particle_random_fourier_1D.ipynb`'s `RandomFourierTorusGame` builds random Fourier coefficients
  `c, u, v, phi` from a seed, but `f()`/`grad_f()` **don't use them** — the `for k in range(K)`
  summation is commented out, and both methods hardcode the literal Chizat `Example 4.1` function
  instead. Net effect is arguably fortunate (it means existing runs already target the right
  benchmark) but `K` and the coefficient arrays are currently dead weight, and any "random Fourier
  game" claim about current results is not accurate until that loop is restored as an option.
- Same file's `grad_f` has a likely typo: `4*np.pi**np.sin(...)` (exponentiation) where
  multiplication was almost certainly intended — double check the y-gradient before trusting any
  existing comparison plots from this notebook.
- Two different hardcoded RNG seeds coexist in the same file (`42` for the game, `1234` for
  particle init), and domain bound `D`, particle count, step count, and `eta` all differ across
  near-identical copies/cells in `CPMDA_Slingshot.ipynb` and `CPMP_Slingshot.ipynb` (e.g. `D=10`
  vs `D=2.5`, `T=1000` vs `T=500`, `eta` 0.01/0.03/0.04 at different call sites). These should
  collapse to single named constants/CLI args during the refactor, not multiply further.
- The Fourier notebook's solver (`run_chizat_torus`) does not use the slingshot scheduler at all —
  it runs CP-MDA/CP-MP (unified via an `extrasteps∈{1,2}` parameter) with a flat constant `eta`.
  Wiring the slingshot scheduler into this path is the actual experiment the user wants to run.

## Munos NLHF — the preference-optimization connection

Nash-MD treats RLHF-style alignment as a two-player constant-sum game over a **preference model**
`P(y≻y'|x)` (not a scalar reward model — this is what allows it to represent non-transitive,
non-Bradley-Terry preferences). It computes the Nash equilibrium of a KL-regularized version of
this game (regularized toward a reference policy `μ`) via mirror descent against a *geometric
mixture* of the current policy and `μ`, with proven last-iterate `O(1/T)` KL convergence to the
regularized Nash equilibrium — critically, without needing to store or average over a history of
past policies (unlike fictitious play), which matters because LLM policies are expensive to keep
around. `Nash-MD-PG`/`Nash-EMA-PG` are the deep-learning/policy-gradient versions actually used on
LLMs in the paper.

The shared thread with Chizat/Shugart is the **last-iterate-convergence-without-averaging**
property. Chizat's particle method extends the idea from a finite/discrete policy set (Nash-MD's
`Δ(Y)` simplex) to a genuine continuum of strategies whose support can move (the Wasserstein
component) rather than living on a fixed discretization grid. Where slingshot stepsizes might fit
into that picture for *preference* optimization specifically is open — nothing in these four
documents works that out yet; it's the user's research question, not a result to look up.

## Implementation plan (agreed direction)

Module split, mapping directly onto the algorithm family above:
- `slingshot/` — stepsize-getter functions only, pure functions of `(t, T, spectral/Lipschitz
  params) -> (alpha_t, beta_t)`. Must include a trivial "vanilla" getter (constant positive `eta`,
  no sign flips) so the *same* algo code path serves as the no-slingshot baseline — slingshot is
  always an injected dependency, never imported directly by the algo module.
- `algo/` — **one** WFR step primitive, parametrized by `L` (inner gradient-recompute passes per
  outer iteration): `L=1` = CP-MDA (explicit), `L=2` = CP-MP (extragradient/mirror-prox, matches
  Chizat's practical algorithm), `L→∞` = CP-PP (implicit proximal point — theory-only reference,
  not practically run). Takes the stepsize-getter as a parameter rather than importing one.
- `games/` — registry keyed by name (`bilinear`, `convex_concave`, `sc_sc`, `chizat_example_4_1`,
  `fourier_random_k`), each entry providing payoff, gradient, domain handling (Euclidean clip vs.
  torus wraparound), and known closed-form MNE where available (all but `fourier_random_k`).
  Sanity-check assertions against the known MNEs belong here.
- `cli.py` — argparse over game / `L` / stepsize-mode / particle counts / seed / iteration budget /
  output path. One run = one configuration; sweeps are a thin loop over this module's Python API
  (not shelled-out subprocess calls), dumping one row per run to a CSV/JSON metrics file.

**This stage's experimental focus** (isolating slingshot's actual contribution, not just building
plumbing): a 2x2 ablation, stepsize-mode `{vanilla, slingshot}` x algorithm `{CP-MDA (L=1), CP-MP
(L=2)}`, with seed/particle-count/iteration-budget/game held fixed between the two cells being
compared:

| | vanilla eta | slingshot |
|---|---|---|
| **CP-MDA (L=1)** | Chizat's documented failure on `chizat_example_4_1` | does slingshot rescue it? (headline question) |
| **CP-MP (L=2)** | Chizat's documented success | does slingshot add a further edge (rate, residual, robustness to fewer particles/bigger steps)? |

Primary metric: duality gap / Nikaido-Isoda error (always computable, the actual Nash certificate)
and distance-to-known-MNE where available. Entropy is logged but kept secondary — the bilinear
pathology documented above already shows it can hit 0 while the duality gap stays nonzero. Report
a fitted log-linear convergence *rate* (slope of `log(metric)` vs. iteration over the decay
window), not just pass/fail, so "improvement" is a single comparable number per cell. Run multiple
seeds per cell before drawing conclusions — particle init is stochastic.

Sequencing: start on `chizat_example_4_1` (closed-form MNE, documented failure point) before the
other games; revive `fourier_random_k`'s dead K-feature loop afterward as the
generalization/robustness check.

Environment: conda-managed (`environment.yml`), pure argparse CLI, run from a terminal — no
notebook dependency at runtime. The three `.ipynb` files remain reference prototypes only.

## Implementation findings (the package now exists — read this before trusting old notebook plots)

The package has been scaffolded: `games/` (registry + `bilinear`, `convex_concave`, `sc_sc`,
`chizat_example_4_1`, each with `sanity_check`), `slingshot/` (`vanilla`, `chebyshev`,
`randomized_sign`), `algo/` (`conic_particle.py` — one `wfr_step(state, game, alpha, beta, L)`
primitive for `L∈{1,2}`), `metrics/` (`diagnostics.py` — grid-based duality gap, domain-aware
Wasserstein-to-MNE, entropy, log-linear rate fit), `cli.py`, `environment.yml`. `metrics/` is a
fifth module beyond the original four-module plan — it absorbs diagnostics that were duplicated
near-verbatim across all three source notebooks, the same duplication rationale that motivated
splitting `games/`/`slingshot/`/`algo/` apart in the first place. The duality gap is also a
genuine upgrade over the notebooks' version: it's a true grid-based sup/inf over the domain (400
points by default), not just a max/min over the opponent's existing particles, so it's a tighter
Nash certificate.

Porting the three notebooks into one unified `algo/conic_particle.py` surfaced bugs that were
real findings, not just engineering cleanup — anything computed from the old notebooks (including
prior comparison plots) should be treated as suspect until re-run against this package:

- **`particle_random_fourier_1D.ipynb`'s `grad_f` always returned `(0.0, 0.0)`.** It computes
  `g_x, g_y` correctly (modulo the `**`/`*` typo on `g_y` below) but then returns the unused
  zero-initialized `gx, gy = 0.0, 0.0` instead. Net effect: in every run from that notebook,
  particle *positions* never moved at all — only the simplex weights `wx, wy` evolved, via mirror
  descent over a fixed random initial discretization of points. The "CP-MP converges, CP-MDA
  doesn't" result reported from that notebook was therefore a **weight-only** result on a fixed
  point cloud (effectively a finite/tabular game), not a demonstration of the conic particle
  method's actual claim — that particle *positions* adapt. `games/chizat_example_4_1.py` fixes
  this (positions now genuinely move), which is more faithful to Chizat's algorithm but is also a
  strictly harder dynamical problem; see below for why this changes the right stepsize scale.
- **Same function, separate bug:** `g_y = 4*np.pi*np.cos(4*np.pi*y) - 4*np.pi**np.sin(...)` uses
  `**` (exponentiation) where `*` (multiplication) was clearly intended. Fixed in the port.
- **The euclidean-games' `1/aᵢ` position-step precondition is not universal.** The cheat-sheet
  above already flags this, but concretely: `CPMDA_Slingshot.ipynb`/`CPMP_Slingshot.ipynb` compute
  `grad_x[i] = Σⱼ bⱼ·∇ₓf(xᵢ,yⱼ)` (no mass factor) and *then* divide by `aᵢ` as a deliberate
  ghost-particle exploration heuristic, clipped to `min(1, 0.1/aᵢ)` to keep it bounded. The
  Fourier/torus notebook instead builds `Dx[i] = aᵢ · Σⱼ bⱼ·∇ₓf(xᵢ,yⱼ)` and divides by the *same*
  `aᵢ` again — the mass cancels exactly, so its faithful position step is the **plain unweighted
  gradient step**, no precondition, no clip (only a `1e-15` zero-division floor). Unifying the two
  notebooks' position steps under one `1/aᵢ`-preconditioned-and-clipped rule (the natural first
  pass when deduplicating) silently carried the euclidean heuristic onto the torus game, where
  with ~30 particles (`aᵢ≈1/30`) it amplified every step by ~30×, regardless of the clip constant.
  Combined with the `grad_f` bug above (which had made this invisible — a zero gradient times any
  precondition is still zero), this is why initial smoke-testing showed flat, non-converging
  duality gaps on `chizat_example_4_1` even after the gradient was fixed. `algo/conic_particle.py`
  now branches on `game.domain`: euclidean keeps the `1/aᵢ`-precondition-plus-clip; torus uses the
  plain unweighted step, matching the faithful source.
- **Consequence for stepsize choice:** the Fourier notebook's literal constants (`eta=0.04`,
  `mx=my=40`) were tuned against the zero-gradient (no-op position step) regime and are too large
  by roughly an order of magnitude once positions genuinely move — applying them post-fix causes
  the duality gap to *increase*. The CLI's auto-derived default (`eta = 1/(3L)` with the crude
  Hessian-bound `L=24π²`, ≈0.0014) is the one that produces real decay; don't reach for the old
  notebook's `eta` as a sanity baseline.
- **`convex_concave`'s gradient doesn't broadcast on its own.** `grad(x,y) = (tanh(x), -tanh(y))`
  has no cross term, so calling it with `x` shape `(n,1)`, `y` shape `(1,m)` (the vectorized
  pairwise-grid pattern used for `O(n·m)` gradient sums) returns `(n,1)` and `(1,m)`, not `(n,m)` —
  a shape mismatch that crashes the subsequent `Gx @ b`. Any future game whose gradient component
  depends on only one player's position will hit this. Fixed generically in
  `algo/conic_particle.py::_compute_gradients` via `np.broadcast_to(Gx, (n, m))` rather than in
  each game module, so new games don't need to remember to broadcast their own gradients.

**Smoke-test result on `chizat_example_4_1`** (single seed, default settings, post-fixes): the
duality gap drops quickly from its random-init value, then **plateaus and oscillates** around
1.0–1.3 (rather than decaying further) over thousands of iterations, for both `L=1`/`L=2` and
both `vanilla`/`slingshot`. Entropy stays near its max (`≈ln(30)`) throughout — no mass collapse
toward the known 2-atom MNE. This looks like genuine GDA-style limit-cycling, not a bug (gaps are
bounded, not diverging). At `L=2`, vanilla and slingshot traces are nearly indistinguishable —
expected, since the Mirror-Prox correction step always forces `|alpha|,|beta|` (the slingshot sign
only shapes the discarded prediction sub-step). Whether slingshot can break this cycle is exactly
the open headline question the ablation grid above was built to answer — not yet conclusively
either way from a single seed/short horizon; needs the multi-seed, longer-horizon, swept-eta runs
the ablation plan calls for.

**Headline finding so far — CP-MDA's failure here looks global, not local.** Ran the multi-seed,
longer-horizon, swept-`L` ablation the previous paragraph called for: at the theoretically-derived
default stepsize (`L=24π²`), `L=1` (CP-MDA) plateaus/cycles at gap ≈1.0–1.5 for both
vanilla and slingshot, indistinguishably, across 5 seeds × 20000 iterations — entropy stays
pinned near `ln(n)` (no mass collapse) the whole time. Sweeping `--lipschitz-L` down (bigger
steps) makes *both* modes worse (gap rises to ≈4–5), never producing a regime where vanilla
visibly cycles while slingshot escapes.

To isolate whether this is a *local* stability failure (the kind Shugart's slingshot is proven to
fix — perpetual rotation around a saddle, like the bilinear `xy` cycle) or a *global*
basin-of-attraction failure (the dynamics never finding/concentrating onto the right support from
a random spread), added a warm-start path: `algo/conic_particle.py::init_state_near_mne` (CLI:
`--init near_mne --init-noise σ`) splits particles across the known MNE atoms proportionally to
their true weights, each perturbed by `N(0, σ²)`, instead of spreading them uniformly over the
whole domain. Result, `L=1`, σ=0.01, 5 seeds, 5000 iterations: **both vanilla and slingshot
collapse from the initial perturbation to a tiny residual gap (≈2.5e-5) within ~250 steps and hold
there** — confirmed not a grid-discretization artifact (same floor at `grid_size=400` and `2000`;
it's the inherent cost of approximating a continuous 2-atom measure with finitely many noisy
particles). No meaningful separation between vanilla and slingshot in this regime either.

Reading: CP-MDA on this game is **locally stable** near the true MNE — small perturbations shrink
fast, for both stepsize modes, contradicting a naive reading of "CP-MDA doesn't converge" as a
local-oscillation claim. The actual failure (visible in the random-init runs) is that the
Fisher-Rao weight dynamics never drives mass toward the right two points to begin with when
started from a uniform spread — an exploration/basin-of-attraction issue. Slingshot's mechanism
only reshapes the *position* update near the *current* iterate to cancel first-order GDA drift
while keeping a second-order Hessian-descent term; it has no obvious lever on whether the weight
dynamics finds the right support in the first place. This is the most likely reason slingshot
shows no measurable benefit on this particular game/parametrization, and narrows the open
question for next steps: either (a) try other games/inits where the failure mode is genuinely
local cycling (the bilinear game is the natural candidate, since that's literally Shugart's
worked example), or (b) look for a basin-of-attraction-specific fix (e.g. periodic entropy-reset,
more particles, annealed `eta`) rather than expecting slingshot itself to help here.

**Follow-up — option (a), bilinear: a clean positive result, with one calibration pitfall along
the way.** First attempt at default settings was a dead end: `bilinear`'s spectrum is degenerate
(`B=[[1]]` ⟹ `m=M=1`), so both the vanilla baseline (`h=1/√M=1`) and the Chebyshev schedule
(which collapses to the same constant magnitude when `m=M`) pick step magnitude `1.0` — far above
the position-step's adaptive clip cap (`min(1, 0.1/aᵢ) ≤ 1`). Both modes instantly saturate the
clip and slam particles to the domain boundary (`x,y → ±10`) within a handful of steps, then
freeze there per the frozen-position pathology — clip-domination, not the sign-schedule, was
driving the outcome, and vanilla/slingshot looked identical for the wrong reason.

Re-ran below clip saturation (`--eta 0.01` for vanilla; `--m 10000 --M 10000` for slingshot, same
`h=0.01` magnitude via the now-genuinely-degenerate-but-correctly-scaled Chebyshev formula), random
init, `L=1`, 3000 iterations, 3 seeds: **slingshot's duality gap collapses to machine zero
(~1e-14–1e-15) in all 3 seeds; vanilla's stays stuck around 106–117 and is still increasing.**
This is the cleanest possible replication of Shugart's headline claim inside the conic-particle
parametrization — exactly the regime (perpetual rotation around a saddle) the slingshot mechanism
is proven for, and it works.

**Calibration pitfall surfaced by this run:** the duality gap and the Wasserstein-to-MNE metric
told two different stories (gap ≈ 0, but `W2(mu)+W2(nu) ≈ 11`, not shrinking) — investigated and
it's a fact about the game, not a bug. `f(x,y)=xy` is separable, so under any product measure
`μ⊗ν`, `F(μ,ν) = E_μ[x]·E_ν[y]` depends **only on the two barycenters**, never on the shape or
support of the distributions. Checked directly: slingshot drove `barycenter_x, barycenter_y` to
`~1e-16` (true zero) while individual particles stayed spread across nearly the whole domain
(`x∈[-10, 9.6]`) and weights stayed near-uniform (entropy ≈ `ln(30)`) — vanilla's barycenters,
by contrast, sat at `(-1.69, -9.93)`, matching its ~116 gap almost exactly. So *any* zero-mean `μ`
paired with any zero-mean `ν` is a valid MNE of this game — the registered `known_mne` (a point
mass at `0`) is only the canonical representative, not the unique equilibrium. `duality_gap_grid`
correctly reports zero either way; `wasserstein_to_mne`, which measures distance to that one
representative, is not a sound convergence signal for `bilinear` specifically (it would only
flatline at zero if the dynamics happened to collapse to the point mass, which they have no reason
to). Trust the duality gap as primary for this game; treat its W2 reading as informative only
about "did we converge to *this particular* atom," not "did we reach an MNE."
