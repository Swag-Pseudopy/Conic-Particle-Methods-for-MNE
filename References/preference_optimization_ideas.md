# Slingshot self-play for Nash-based preference optimization — ideas draft

Status: brainstorm / not yet executed. Companion to `results/report.tex`
(the toy-game empirical study); see that document's "Outlook" section for
how the two connect.

## TL;DR

RLHF research has spent 2023–2026 reinventing exactly the problem this
codebase already studies: finding the **Mixed Nash Equilibrium of a
two-player constant-sum game** via self-play mirror descent, and fighting
the same **cycling/non-convergence** failure mode GDA is famous for. Every
fix tried so far in that literature is either (a) extragradient/optimism
(double the per-step cost) or (b) an added anchor/regularization term
(changes which equilibrium you converge to, adds a hyperparameter). Nobody
has tried **slingshot's periodically-negative stepsize schedule**
(Shugart & Altschuler, [arXiv:2505.01423](https://arxiv.org/abs/2505.01423))
— a fix that needs neither. That's the paper-shaped gap. This repo's
`algo/conic_particle.py` + `slingshot/` already implement the machinery
needed to test it cheaply, at toy scale, before touching a real LLM.

## 1. Why pivot away from the current toy-game study

Be honest about what `results/` currently shows: a clean win on a textbook
bilinear example, a documented non-rescue on the one case anyone would
actually care about (Chizat's Fourier game), and a careful diagnosis of why
(global basin-of-attraction failure, not local cycling). That's solid,
defensible numerical work, but as a standalone contribution it's thin —
no new theory, one positive result on a 1-dimensional toy game. It reads as
a workshop note, not a paper. To get a paper out of this line of work, the
move is to keep the *mechanism* (slingshot) and the *infrastructure*
(WFR particle dynamics, the games/slingshot/algo/metrics split) but point
them at a problem with actual stakes.

## 2. The landscape: Nash-equilibrium-seeking preference optimization

Framing RLHF as a two-player symmetric constant-sum game — rather than
fitting a Bradley-Terry reward model and maximizing it — is now a major,
active subfield. Motivation: human preferences are often **intransitive**
(rock-paper-scissors-style cycles), which no scalar reward model can
represent, but a Nash equilibrium of the preference game can. This is the
explicit motivation in Swamy et al.'s "minimaximalist" framing
([arXiv:2401.04056](https://arxiv.org/abs/2401.04056)).

Once you frame it this way, you need an algorithm that actually finds that
Nash equilibrium — and every such algorithm is some flavor of mirror
descent / no-regret self-play, which is exactly the family GDA-cycling
results (and slingshot) are about:

| Method | Mechanism | Equilibrium reached | Extra cost vs. vanilla self-play | Convergence |
|---|---|---|---|---|
| **Nash-MD** ([2312.00886](https://arxiv.org/abs/2312.00886), Munos et al., DeepMind) | Mirror descent with KL anchor to a reference policy | *regularized* NE only | none | last-iterate, but only to the regularized game |
| **SPPO** ([2405.00675](https://arxiv.org/abs/2405.00675), Wu et al.) | Multiplicative/exponential weight self-play | approx. NE of the empirical preference game | none | provable approx-NE convergence |
| **DNO** ([2404.03715](https://arxiv.org/abs/2404.03715), Rosset et al.) | Iterative batched DPO-style contrastive self-improvement | NE on average | iteration/batch overhead | average-iterate |
| **INPO** ([2407.00617](https://arxiv.org/abs/2407.00617), ICLR'25 oral) | Online mirror descent, no-regret | Nash policy, general preferences | none | no-regret / approx-NE |
| **Optimistic OMD** ([2502.16852](https://arxiv.org/abs/2502.16852)) | Optimism / predictive gradient | NE, general preferences | implicit lookahead step | improved stability |
| **Nash-Prox / Mirror-Prox NLHF** ([2505.19731](https://arxiv.org/abs/2505.19731), Tiapkin et al., 2025) | Proximal point / extragradient family | regularized or true NE | **2× gradient evals/step** | high-probability last-iterate |
| **Magnetic Mirror Descent / MPO** ([2410.16714](https://arxiv.org/abs/2410.16714)) | Annealed magnet/anchor regularizer | **true, unregularized** NE | extra anchor term + anneal schedule | last-iterate, **linear rate** |
| **RSPO** ([2503.00030](https://arxiv.org/abs/2503.00030)) | Explicit KL regularization to reference policy | regularized NE | extra KL term | empirical stability |
| **Slingshot self-play (proposed)** | Periodic negative-sign stepsize on the *same* single gradient step | true NE (proven for convex-concave/bilinear; open for stochastic/general preference games) | **none** — same step, just resigned | open question |

Note the structural pattern: the two strongest fixes (Nash-Prox, Magnetic
Mirror Descent) both pay for their cycling-fix with something —
either double the gradient evaluations, or an extra regularization term
that has to be carefully annealed to avoid converging to the wrong
(regularized) equilibrium. Slingshot's entire selling point in the pure
optimization literature is that it pays for neither: same one gradient
step per iteration, same fixed point, just a different sign schedule on
the step itself. A targeted web search (June 2026) for "slingshot" +
preference optimization / RLHF / self-play turns up nothing — this
combination has not been tried.

## 3. Why this codebase is unusually well-positioned to test it

This isn't a from-scratch project. The conic-particle WFR machinery
already built here maps onto the Nash-RLHF formalism almost directly:

- `ParticleState(a, x, b, y)`: `a`/`b` are the simplex (mixed-strategy)
  weights, `x`/`y` are the particle positions (pure strategies / support
  points). **This is structurally identical to a population/mixture
  representation of a policy.**
- The Fisher-Rao weight update inside `algo/conic_particle.py` is a
  continuous-particle generalization of **exactly** the multiplicative/
  exponential-weight self-play update SPPO performs over a finite
  candidate-response set. For a *fixed, finite* support (no particle
  movement), the WFR flow's weight half reduces precisely to entropic
  mirror descent on a simplex — i.e., the classical normal-form
  constant-sum matrix game setting (matching pennies, rock-paper-scissors)
  that GDA-cycling results were originally stated for. A finite
  intransitive preference matrix (Swamy et al.'s rock-paper-scissors-style
  counterexample to Bradley-Terry) drops into this almost for free: turn
  off the position step, treat the `n` particles as `n` fixed candidate
  responses with a known pairwise win-probability matrix, and the existing
  Fisher-Rao update *is* the self-play algorithm.
- `slingshot/`'s `chebyshev` and `randomized_sign` schedules already exist
  and are already wired into `cli.py`/`run_ablation.py` — applying one of
  them to the weight-update's implicit learning rate instead of to a
  Euclidean position step is a small, scoped code change, not a rewrite.
- `metrics/duality_gap_grid` is already a Nikaido-Isoda / Nash-optimality
  certificate; it generalizes directly to a finite-matrix-game duality gap
  (sup/inf over a finite action set instead of a grid).
- The Wasserstein/position-step half (continuous particle movement) is the
  natural analog of Nash-MD/DNO's actual mechanism — a mirror-descent or
  contrastive-loss step in *policy-parameter* space — and becomes relevant
  only at Stage 2 below (scaling past a fixed finite candidate set).

Concretely: **Stage 1 is closer to "add one new `games/` entry and re-run
`run_ablation.py`" than to "build new infrastructure."**

## 4. Core research question

> Does a slingshot-style periodically-negative stepsize schedule, applied
> to vanilla self-play mirror descent / multiplicative weights on a
> preference (Nash-equilibrium-seeking) game, recover last-iterate
> convergence to the **true** (unregularized) Nash equilibrium — at the
> same per-step cost as vanilla self-play — competitive with Magnetic
> Mirror Descent (regularization-based) and Nash-Prox (extragradient-based),
> on games where vanilla self-play is known/shown to cycle?

Falsifiable sub-hypotheses:
1. On a synthetic finite intransitive-preference matrix game (rock-paper-
   scissors-style), vanilla multiplicative-weight self-play cycles
   (oscillating duality gap, no convergence), exactly as classical GDA does
   on matching pennies.
2. Slingshot's sign schedule applied to that same update collapses the
   duality gap to ~0, matching or beating Magnetic Mirror Descent's
   reported linear rate, without an anchor term.
3. The result degrades gracefully (or doesn't — this is the real open
   question) once gradients are stochastic (sampled preference comparisons,
   minibatches) rather than exact — because slingshot's theory relies on a
   second-order cancellation argument (positive/negative steps cancel to
   first order) that noisy gradients could plausibly swamp.

Hypothesis 3 is the crux. If it survives stochasticity, this is a strong,
clean, cheap-to-adopt paper. If it doesn't, *characterizing exactly where
it breaks* (noise level vs. cancellation signal) is itself a legitimate,
publishable boundary-of-applicability result, given how much current
attention this subfield has (INPO was an ICLR'25 oral; Nash-Prox and MPO
are both 2025 papers actively competing on this exact cycling problem).

## 5. Staged experimental plan

**Stage 0 (done).** The current `results/` toy-game study — keep as the
"does the implementation behave correctly on known cases" validation layer
and as the source of the duality-gap / Wasserstein-to-MNE metric machinery.

**Stage 1 (cheap, days, mostly reuses existing code).**
- Add `games/intransitive_preference.py`: a finite constant-sum matrix game
  with a known intransitive (cyclic) preference structure — start with the
  canonical 3-action rock-paper-scissors-style payoff, then a larger
  randomly generated intransitive tournament matrix for robustness.
- Run the existing Fisher-Rao weight update with position step disabled
  (fixed finite support) under: vanilla mirror descent, `chebyshev`,
  `randomized_sign` — i.e. exactly the existing `vanilla`/`slingshot`
  comparison in `cli.py`, just on the new game.
- Add a stochastic-gradient variant (inject calibrated noise into the
  payoff gradient, or sample a finite number of pairwise comparisons per
  step instead of using the exact matrix) to directly test hypothesis 3.
- Reuse `duality_gap_grid` (adapted to enumerate the finite action set
  instead of a grid) and `run_ablation.py`'s CSV/summary/plotting pipeline
  unchanged.

**Stage 2 (if Stage 1 is positive).** Scale to a small real preference
model + small LM — comparable in spirit to SPPO's setup (a lightweight
pairwise preference model, e.g. PairRM-scale, plus a small base model,
e.g. 125M–1B parameters) rather than jumping straight to 7B-scale. Apply
the slingshot sign schedule to the self-play policy update (Nash-MD-style
mirror descent step on policy parameters, the position-step analog).
Compare against Nash-MD, SPPO, and (if compute allows) Magnetic Mirror
Descent as baselines, on win-rate against a held-out reference and on
directly measured Nash-gap against a constructed intransitive eval set.

**Stage 3 (theory, can run in parallel with Stage 1/2).** Extend Shugart &
Altschuler's convex-concave/bilinear slingshot convergence proof to (a) the
KL-regularized Nash equilibrium objective these RLHF methods actually
optimize, and (b) the stochastic-gradient setting. Several of the
surveyed papers explicitly flag stochastic-gradient extensions as open
(this is a recurring "future work" line in the last-iterate-convergence
literature), so this would not be working in a vacuum.

## 6. Honest risks

- **Stochasticity is the central risk**, not a minor caveat: slingshot's
  mechanism is a second-order finite-difference effect (positive/negative
  steps cancel to first order, net movement is second-order). Gradient
  noise from sampled preferences/minibatches is exactly the kind of thing
  that could dominate a second-order signal. This must be tested early
  (Stage 1's noise-injection ablation) before any LLM-scale investment.
- Slingshot's published theory (2505.01423) covers convex-concave and
  bilinear problems; a finite matrix game is convex-concave-shaped (bilinear
  in the mixed strategies, in fact — same structure as `games/bilinear.py`
  already in this repo), so Stage 1 is theoretically well-justified. A real
  LLM policy is neither convex nor concave in its parameters — Stage 2 is
  an empirical bet, not a theory-backed extrapolation, and should be framed
  that way.
- This is a competitive, fast-moving subfield (multiple 2025 papers on
  cycling fixes alone). Time-to-first-result matters; Stage 1 is designed
  to be answerable in days specifically so novelty risk is checked early.

## 7. Possible paper framings depending on outcome

- **Positive Stage 1 + Stage 2**: "Slingshot Self-Play" — a cheaper
  drop-in alternative to extragradient/magnetic-anchor fixes for Nash
  learning from human feedback, no extra hyperparameter, no extra gradient
  evaluation.
- **Positive Stage 1, negative/mixed Stage 2**: an honest boundary-of-
  applicability paper — "when does a deterministic-theory cycling fix
  survive contact with stochastic, non-convex self-play" — still a real
  contribution given how many of the surveyed papers gesture at this exact
  open question without answering it.
- **Negative Stage 1**: still worth writing up briefly (a clean negative
  result on the *cleanest possible test* — finite, deterministic,
  bilinear-shaped — would be informative), but would not justify Stage 2
  investment.

## 8. Immediate next steps (concrete, in this repo)

1. Add `games/intransitive_preference.py` (finite matrix game, `domain`
   likely a new `"simplex"`/finite case rather than `"euclidean"`/`"torus"`
   — check whether `_position_step`/`_project` need a third branch or
   whether disabling the position step via zero stepsize is enough).
2. Adapt `metrics/duality_gap_grid` for a finite action set (sup/inf over
   the matrix's rows/columns instead of a continuous grid) — likely a new
   `duality_gap_matrix` alongside it rather than a rewrite.
3. Wire the new game into `run_ablation.py` following the existing
   per-game-subfolder convention (`results/intransitive_preference/...`).
4. Add the noise-injection ablation (stochastic vs. exact gradient) as a
   `--noise` flag or a second game variant, specifically to test
   hypothesis 3 before anything else.
5. Re-run `plot_results.py` unchanged (it's already schema-driven) to get
   comparison panels for the new game alongside the existing four.

## References

- Shugart & Altschuler, "Negative Stepsizes Make Gradient-Descent-Ascent
  Converge," [arXiv:2505.01423](https://arxiv.org/abs/2505.01423)
- Munos et al., "Nash Learning from Human Feedback,"
  [arXiv:2312.00886](https://arxiv.org/abs/2312.00886)
- Wu et al., "Self-Play Preference Optimization for Language Model
  Alignment," [arXiv:2405.00675](https://arxiv.org/abs/2405.00675)
- Rosset et al., "Direct Nash Optimization,"
  [arXiv:2404.03715](https://arxiv.org/abs/2404.03715)
- "Iterative Nash Policy Optimization," ICLR 2025 oral,
  [arXiv:2407.00617](https://arxiv.org/abs/2407.00617)
- "Improving LLM General Preference Alignment via Optimistic Online Mirror
  Descent," [arXiv:2502.16852](https://arxiv.org/abs/2502.16852)
- Tiapkin et al., "Proximal Point Nash Learning from Human Feedback
  (Mirror Prox NLHF)," [arXiv:2505.19731](https://arxiv.org/abs/2505.19731)
- "Magnetic Preference Optimization: Achieving Last-Iterate Convergence for
  Language Model Alignment,"
  [arXiv:2410.16714](https://arxiv.org/abs/2410.16714)
- "RSPO: Regularized Self-Play Alignment of Large Language Models,"
  [arXiv:2503.00030](https://arxiv.org/abs/2503.00030)
- Swamy et al., "A Minimaximalist Approach to Reinforcement Learning from
  Human Feedback," [arXiv:2401.04056](https://arxiv.org/abs/2401.04056)
- "Beyond Bradley-Terry Models: A General Preference Model for Language
  Model Alignment," [arXiv:2410.02197](https://arxiv.org/abs/2410.02197)
