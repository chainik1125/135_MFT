# Red-team review of summary.md (SG135 slave-boson sprint)

Reviewer stance: hostile condensed-matter referee. All numerical claims below were
re-checked with an independent numpy re-implementation of the solver kernels
(`notes/_reftest_numpy.py`), run against the archived solution
`code/sg135/sol_classII_U1.npy`. Where I confirm a claim I say so; where I refute
one I give the recomputed number.

---

## A. The headline claim

**1. (BLOCKER) "Must be topologically ordered" misstates Watanabe–Po–Vishwanath–Zaletel and contradicts the sprint's own theory document.**
Location: §"Problem" — "the interacting bounds (Watanabe-Po-Vishwanath-Zaletel) allow a gapped symmetric insulator at ν=4 — which, if it exists, must be topologically ordered"; §5 end — "By Watanabe-Po-Vishwanath-Zaletel, a symmetric gapped insulator at ν = 4 in SG135 must be topologically ordered"; TL;DR — "the LSM-mandated topological phase".
`notes/theory/sg135_factorization.md` §5.1 says the opposite: SG135 is "one of the ten groups they single out where 'a sym-SRE insulator is possible at ν = 4'…, leaving open whether a stronger constraint exists", and explicitly states the construction "does not realize the open 'SRE at ν=4' option — it occupies the topologically ordered branch." No theorem cited anywhere in the repo excludes a short-range-entangled symmetric insulator at ν=4 in SG135, so "gapped + symmetric ⇒ topologically ordered" is unsupported, and the syllogism that closes the headline is broken.
**Fix:** Replace with the correct logic: the state is topologically ordered *by construction* (deconfined Z₂ parton state), it is *consistent with* the WPVZ window (no band insulator at ν=4; symmetric insulator allowed only at ν ∈ {4, 8, …}), and it sits on the LRE branch of the open WPVZ dichotomy. Delete "LSM-mandated" or cite a theorem that actually closes the SRE option.

**2. (BLOCKER) "GLOBAL mean-field ground state for all U ≳ 1" was established against a competitor set that contains no symmetry-broken state at all.**
Location: TL;DR; exec bullet "HEADLINE: … the GLOBAL ground state for all U ≳ 1"; §5 "is the global minimum … into the deep Mott regime (checked to U = 6)".
What was actually compared (per `run_sg135.py`, `stage_d.py`, `classII_sweep.py`): z-nodal, xy-nodal, uniform-mixed (a saddle), staggered-z+uniform-xy, and condensed SC seeds in the *uniform* family. Never constructed: (i) any magnetic decoupling (the ansatz has no spin/Weiss channel, so Néel/AFM — the generic Hubbard ground state at large U in 3D — was never in the race); (ii) dimerized/VBS bond order — note the KMH gate's *own* phase diagram has the dimer phase winning above U_c2 ≈ 1.98, yet no SG135 analogue was tried before claiming victory "into the deep Mott regime"; (iii) other flux patterns / staggered versions of the xy, 1, 2 channels; (iv) Class II (chargon-flux) states; (v) condensed states coexisting with the staggered channel (`cond_seed_bank` is 26-parameter uniform-only). §6 quietly concedes "no flux/Class II patterns, no broken-symmetry bond order" but never mentions magnetism, and the TL;DR/headline carry no restriction.
**Fix:** Rewrite as "lowest-energy state within the translation-invariant 4+1-channel paired ansatz family studied". Add magnetic and dimer decoupling channels before any "global" language, especially for U ≳ 2 where the t²/U regime makes ordered states the default competitors.

**3. (MAJOR) The free-fermion saddle is asserted to *be* the topologically ordered state; the deconfinement step is silent, and it is unusually fragile here.**
Location: §5 — "this mean field is precisely the Z₂ fractionalized insulator"; TL;DR — "uncondensed Z₂ fractionalized insulator".
The converged mean field is a quadratic BdG + free-boson state; "Z₂ topological order" is a statement about the projected state / the deconfined phase of the fluctuating Z₂ gauge theory. That is an assumption (plausible in 3+1d, but an assumption), and here it deserves a sentence rather than silence because the solution is *pure pairing*: in `sol_classII_U1.npy` every hopping amplitude χ_b, χ_f is 0 to solver noise (≤1e-17). The breaking SU(2)→Z₂ in the gauge structure then rests entirely on the Lagrange multiplier λ = −0.0138 — the same tiny scale that sets the spectral gap. The summary never mentions that all χ vanish.
**Fix:** State the saddle→phase inference explicitly ("the saddle's fluctuations are a Z₂ gauge theory; in its deconfined phase the state is the Z₂ fractionalized insulator"), report χ = 0 and the λ-only gap, and note the IGG/confinement question as a caveat.

**4. (MAJOR) "Symmetric up to μ_z gauge" is checked for the spinon sector only; the chargon sector's symmetry is never verified by the shipped code.**
Location: §5 symmetry bullet; `verify_sg135.py` `check_symmetries` calls only `fermion_blocks`. For the physical (projected) state to be symmetric — which the LSM/WPVZ framing requires — *both* matter sectors must transform consistently under the same Z₂ gauge assignment. The boson kernel carries the staggered channel too (Γ₄ = μ^xτ^z in `boson_blocks`), so this is not vacuous. (My audit ran the same generator/gauge check on the boson blocks: it passes, worst deviation 4.1e-9 with the same μ_z gauge — so the physics survives, but the verification as shipped does not support the claim.) Also unfulfilled: `verify_sg135.py`'s own docstring promises "(c) … boson Γ-multiplet structure (Class I check)" — no such check exists in the code; and the theory note's numerics checklist (§5.4: ⟨Q_i⟩=1 constraint audit, Class confirmation, Δ reality) is only partially discharged.
**Fix:** Add the boson-sector PSG check and the Γ-multiplet/Class identification to `verify_sg135.py`; add one sentence to the summary explaining why invariance up to an IGG gauge transformation implies symmetry of the physical projected state.

**5. (MAJOR) The symmetry verification's "≤10⁻⁷" is exactly the deliberately injected symmetry-breaking floor; the pass threshold gives only a 5× margin.**
Location: §5 — "All P4₂/mbc generators, TRS and pairing antisymmetry verified to ≤10⁻⁷"; `sg135_solver.py` `_BREAK8 = diag(linspace(0,1,8))·1e-7` is added inside `fermion_blocks`/`boson_blocks` unconditionally, and `verify_sg135.py` checks against `tol = 5e-7`. `_BREAK8` is not proportional to the identity, so it explicitly breaks every symmetry being tested at the 1e-7 level: the quoted verification number is the artifact, and a genuine violation up to ~4e-7 would have been certified "OK". My rerun with `_BREAK8` removed gives true deviations ≤ 9.5e-9 for every generator (the residual is the converged-to-zero uniform-z amplitude Δ_b,z = 9.4e-9, i.e. solver tolerance) — so the state *is* symmetric, but the shipped check could not have told you that.
**Fix:** Disable `_BREAK8` in verification builds (it is only needed for forward-mode eigh derivatives, not for plain diagonalization) and tighten the threshold to ~1e-8; quote the clean number.

**6. (BLOCKER → factual) The headline gap numbers are wrong and internally inconsistent.**
Location: §5 — "fully gapped and non-degenerate everywhere: min BdG gap 0.018 t_xy over the BZ (0.014 at A), chargon gap 0.13"; TL;DR — "gapped and non-degenerate over the whole BZ".
(a) "Min over the BZ = 0.018" with "(0.014 at A)" is self-contradictory on its face — A is in the BZ. Recomputation: the shifted nk=20 grid (the `gap_audit` default, which misses high-symmetry sets) gives 0.0184; nk=32 gives 0.0158; the true minimum is |λ| = 0.01381, attained not only at A but on the *entire lines* {k_x or k_y = ±π} ∩ {k_z = ±π} where both form factors vanish (verified at three points on the line: gap = 0.01381 each). 0.018 is a grid artifact, ~30% too optimistic.
(b) "Chargon gap 0.13" is the minimum *eigenvalue of the boson kernel matrix* (the para-diagonalizability/stability margin: my recomputation 0.1316); the physical chargon excitation gap is the minimal para-energy, 0.343 — which is what the shipped figure `sg135_bands_gapped_classII_U1.png` visibly shows (chargon bands at ±0.34). The text and its own figure disagree.
(c) "Non-degenerate" is false: with SU(2) spin symmetry, TRS and the anticommuting two-channel structure, the BdG spectrum is **8-fold degenerate at generic k** (recomputed: eigenvalues ±0.111178, each ×8). Presumably "no zero modes" was meant.
**Fix:** Quote min BdG gap = |λ| = 0.0138 attained on the stated lines; quote chargon gap 0.34 (and the 0.13 stability margin separately, labeled as such); replace "non-degenerate" with "fully gapped (no zero modes); bands are 8-fold degenerate by spin SU(2) × channel structure".

**7. (MAJOR) Production data behind the headline are not in the repo; the exec summary admits numbers are placeholders.**
Location: exec header — "(≤600 words, drafted — final numbers being filled from the production runs)"; log last line — "Production nk=12 confirmation sweep running" (never recorded as finished). The repo contains only `classII_sweep_local.npy`: **six** converged U points (1.0–6.0; the gapped branch is NaN at U = 0.6, 0.8), at an *unrecorded* nk ("local" per ground rules = lightweight, i.e. small grid). `classII_sweep.npz` (the production output), `sg135_uncond.npy`, and any condensed-sweep output are absent. The summary nowhere states the k-grid used for any SG135 energy, and no convergence-in-nk study is reported — relevant because the *nodal* competitor's energy converges slowly in nk (gapless plane), and the branch separation at U=1 is only 0.016.
**Fix:** Archive the production sweep, state nk for every quoted number, and show E(nk) for the U=1 branch competition. Remove the placeholder parenthesis or mark the whole document draft.

**8. (MAJOR) The SC boundary "U* ≈ 0.8–1" is an inference, not a computed crossing; and it is inconsistent with §3.**
Location: exec — "SC … below U* ≈ 0.8–1"; §3 — "condensed (SC) solutions appear below U* ≈ 0.6–1.0"; §5 — "below U ≈ 0.8–1 the chargon condenses and the SC takes over". In `classII_sweep_local.npy` the gapped branch simply fails to converge below U=1 (NaN); no condensed solve including the staggered channel exists (the condensed solver is 26-parameter, uniform channels only), so no energy crossing between the *actual* headline state and an SC was ever computed. The two quoted ranges also disagree.
**Fix:** Either compute the crossing (condensed solver extended to the 22+8-parameter family) or phrase as "the uncondensed gapped branch ceases to converge below U ≈ 1, where condensed solutions of the uniform family exist" — and reconcile the two ranges.

**9. (MAJOR) Stationary points, not minima: the gradient-residual solver cannot certify even a local minimum, and the headline branch's χ-directions were never probed.**
Location: §5 calls the uniform mixed state "only a mountain-pass" — establishing that this solver happily converges to saddles. The headline branch was obtained from two seeds (`sG`, `sZ` in `classII_sweep.py`) whose hopping entries are zero; the converged solution has χ ≡ 0 (an invariant subspace of the symmetric problem), so stability against χ ≠ 0 (FL*-like) admixture was never tested, and no Hessian audit is reported for any state.
**Fix:** Check the Hessian (or at least random finite perturbations including χ and other staggered channels) at the headline solution; report the lowest eigenvalue.

**10. (MINOR) Generator set and W matrices are sound — one label is wrong.**
I verified GENS = {4₂ screw, C2x|½½0, I} (+T) is a generating set of P4₂/mbc × T, and the k-actions are correct; derived elements are products. However `GENS["glide_b(mx)"]` is actually m_y (W = diag(1,−1,1); the theory note §4.1 itself defines U(m_y) = U(I)U(C2z)U(C2x)). Harmless physically (m_x is screw-conjugate to m_y) but the label and the summary's "both glides" should say which planes were checked directly.
**Fix:** Rename to `glide_b(my)`; optionally check m_x explicitly.

**11. (MINOR) The "gauge-equivalence confirmed to 10⁻⁵" check is three orders looser than every other check, unexplained.**
Location: §5 — "pure staggered-z state is exactly degenerate with the uniform-z state (gauge equivalence confirmed numerically to 10⁻⁵)". Solver residuals are 1e-10 and the validation chain quotes 1e-8..1e-16; an exact gauge equivalence should match at that level on matched grids. The likely cause (the τ-staggered gauge shifts k by half a reciprocal vector, so the two states sample different points of the shifted grid) is never stated. As the *only* cross-check that touches the new staggered channel's bookkeeping, 1e-5 is weak.
**Fix:** Repeat on a gauge-compatible grid (or do the comparison analytically per k) and report machine-precision agreement, or explain the 1e-5.

## B. Channel competition / concavity

**12. (MINOR) The mechanism is stated essentially correctly but is heuristic, and its only evidence is a missing figure.**
The matrix facts check out: [τ^x, μ^x] = 0 (commuting ⇒ linear interference) and {μ^xτ^z, τ^x} = 0 (anticommuting ⇒ quadrature; confirmed by the recomputed 8-fold degenerate spectrum E² = λ² + Σ(t_iΔ_ig_i)²). But "pairing energy is concave in |Δ(k)|" is asserted, not derived, and the supporting evidence — "The energy surface E(Δ_xy, Δ_z) shows two single-channel valleys separated by a ridge. ![channel surface](figures/channel_surface.png)" — points to a file that does not exist. Also "channel competition *forbids* the multi-channel pairing" (TL;DR) overstates: §5 itself shows the mixed state exists as a stationary point 0.03 above; "disfavors at these parameters" is the supportable claim. "The mean field refuses to mix channels" (twice) is anthropomorphic.
**Fix:** Restore the figure; downgrade "forbids"→"disfavors"; either derive the concavity statement (one line: condensation energy ~ −Σ_k√(λ²+|Δ(k)|²) is concave in the channel weights at fixed k-structure) or label it a heuristic.

**13. (MINOR) "Only channel mixtures can gap the BZ (with λ ≠ 0 gapping the A point)" understates λ's role for the actual headline state.**
The xy ⊕ staggered-z mixture leaves the entire lines {g_xy = g_z = 0} gapped *solely* by λ (issue 6) — not just the A point. For the full four-channel mixture the statement is closer to true (g₁ ≠ 0 on most of those lines), but that is not the state realized.
**Fix:** State the headline state's gap-floor honestly: "on the common zero lines of both form factors the gap equals |λ| = 0.014".

## C. The KMH gate

**14. (MAJOR) "Gate passed" headline vs the log's own "4/5 quantitative checks pass"; and the residual's two descriptions contradict each other.**
Location: exec bullet 1 — "Gate passed: the solver machinery reproduces the published KMH phase diagram"; log 00:30–01:00 — "Gate: 4/5 quantitative checks pass". The failing check (SC-boundary growth with λ_SO: 1.78 vs paper's 1.93 at λ_SO = 0.1) is disclosed — good — but §1 simultaneously says it is "improving with better branch-continuation but not converged to their line" *and* "the discrepancy is robust on our side". Both cannot hold: if continuation is still moving the boundary, the discrepancy is not yet robust. The triple-point row ("between 0.10t and 0.15t" vs paper ≈0.10t) is presented inside an "agreement" table although it is exactly in the discrepant λ_SO direction. The dead figure link (`figures/kmh_boundaries.png` missing) makes none of this checkable.
**Fix:** Say "4/5 checks pass"; resolve robust-vs-unconverged (finish the continuation or bound the boundary); move the triple point out of the agreement table; restore the figure.

**15. (MINOR) "The λ_SO sector does not transfer to SG135 (no SOC channels there)" misleads on two counts.**
(a) It reads as if SOC were symmetry-forbidden in SG135; in fact the Wieder et al. model (and the factorization note's §4.1 Hamiltonian) contains three SOC terms — the *studied solver truncates them away* (`sg135_solver.py` has only the four spin-independent channels). The summary never discloses this truncation. (b) The unresolved KMH discrepancy lives in the multi-channel primed-χ bookkeeping — structurally the closest KMH analogue to SG135's multi-channel competition — so "does not transfer" deserves an argument, not an aside. The three-way energy validation does mitigate this; say so in that sentence. Also: λ is used for the spinon Lagrange multiplier in §5 and for SOC (λ_SO) in §1 — a needless collision.
**Fix:** Disclose the SOC-free truncation explicitly (and that LSM/WPVZ applies regardless); rename the multiplier (e.g. λ_f); justify non-transfer via the energy cross-checks.

## D. The bookkeeping traps

**16. (MINOR) C = (4,4,8,4) is correct (I re-derived it by bond counting) but the supporting apparatus is sloppy: the referee script asserts nothing and the solver documentation contradicts itself.**
`test_realspace.py` prints ratios and ends with "Whatever matches defines the correct C_i … see notes" — there is no pass/fail assertion for (4,4,8,4) and no archived output. The `sg135_solver.py` header docstring says "all four channels give … 4 t_i" (contradicting C_VEC = (4,4,8,4,4) twenty lines below) and instructs "run check_ec_factor()" — a function that exists nowhere in the repo, as does the referenced `checks()`. Crucially, the real-space referee covers only the four *uniform* channels: the load-bearing staggered channel (C₄ = 4) never passed through it — its only cross-check is the loose 1e-5 of issue 11. This falsifies the TL;DR's "Every energy entering the phase competition is cross-checked against independent constructions."
**Fix:** Add assertions to `test_realspace.py`, extend `bond_list()` with the staggered z-channel, fix the docstring, delete phantom function references, and soften the TL;DR sentence.

**17. (MINOR) μ = U/2: mechanism plausible and consistent with the code, but the "every solve fails at μ=0" claim has no archived test.**
The ±(U/2 − μ) para-splitting and the 1e-3 tether to U/2 are real (`run_sg135.py`); the converged μ = U/2 exactly (sol files). But the dramatic claim "μ=0 makes every solve fail" and "the total energy is flat in μ" exist only as narrative; a 5-line test (energy vs μ at fixed OPs; min boson eigenvalue vs μ) would make this bullet referee-proof.
**Fix:** Add that test; cite its output.

## E. Writing and presentation

**18. (MAJOR) Four of the six referenced figures do not exist.**
`figures/` contains only `sg135_bands_gapped_classII_U1.png` and `sg135_bands_mixed_U1.png`. Dead references: `kmh_boundaries.png` (§1), `sg135_bands_znodal_U4.png` (§5), `channel_surface.png` (§5), `sg135_phases.png` (§5 + exec). The exec's phase-diagram bullet and the entire channel-competition argument are therefore figure-free, and one bullet says outright "[no figure — see §3]" — violating the sprint's own deliverable spec (log ground rules: "Exec summary ≤300 words, bullets ≤30 words, each backed by one self-explanatory graph"; the exec header even raises the budget to "≤600 words" unilaterally, and most bullets run 50–90 words).
**Fix:** Generate/copy the four figures (the data scripts exist: `plot_boundaries.py`, `plot_bands.py`, `channel_surface.py`, `plot_phases.py`) or cut the references; conform to the spec or renegotiate it explicitly.

**19. (MINOR) The "Class I / Class II / π-flux" terminology is internally contradictory.**
§4: "Our ansatz realizes Class I." §5 headline figure: "gapped Class II bands" (and the code/files call the new channel "CLASS II"). §6 limitations: "no flux/Class II patterns" — while §5 says "this is a genuine π-flux Z₂ ansatz". In the note's own taxonomy (§2.4), Class II means chargon χ_b(C2z) = −1; whether the τ-staggered state is Class II is exactly the unimplemented multiplet check of issue 4 (my boson check found C2z realized plainly, gauge "1", suggesting it is still Class I with a nontrivial PSG on C2x/glides — in which case every "Class II" label on the headline state is wrong). Relatedly, `sg135_solver.py`'s comment says the *screw* needs the τ^z gauge, while the summary says the screw is plain and C2x/glides need μ_z (my audit confirms the summary; the code comment is stale).
**Fix:** Run the Γ/A chargon multiplet check, then use one consistent label ("PSG-twisted Class I" if that is what it is); rename files/figure captions; fix the stale solver comment.

**20. (MINOR) Jargon and provenance polish.**
(a) LSM, PSG, EBR, IGG, corep, BdG, HK, FL*, Colpa are never expanded; the TL;DR is impenetrable to a non-specialist ("τ-staggered z-channel whose matrix structure anticommutes…"). (b) "(user directive: 'change the slave boson')" is internal sprint language in a scientific summary. (c) "to our knowledge the first self-consistent mean-field realization" — the literature agent itself said "Harden via citation sweep of 1811.11182 and 2309.15118 before claiming novelty"; that sweep is not recorded. (d) Precision claims disagree: TL;DR "agreement 1e-10..1e-16" vs §2's real-space energy match "3×10⁻⁸". (e) The exec bullet's "E → atomic-Mott 0 from below as ~t²/U" has no fit or figure behind it. (f) Negative/anthropomorphic phrasing: "channel mixing … is refused", "the mean field refuses", "[no figure]", "was not attempted".
**Fix:** One-line expansions at first use; delete sprint-internal asides; do the citation sweep or soften to "we found no prior slave-boson treatment"; reconcile precision claims; show the t²/U fit; rephrase actively ("channel mixing costs 0.03 t_xy at these parameters").

---

## Three fixes first

1. **Issue 1** — repair the headline logic (WPVZ does *not* force topological order at ν=4; the repo's own theory note says so). Every other claim can be weakened and survive; this one is a wrong theorem statement at the center of the paper.
2. **Issue 2** — re-scope "GLOBAL" to the studied ansatz family and add (or at least honestly exclude) magnetic/dimer competitors before any deep-Mott claim; the KMH gate itself demonstrates that symmetry-broken states win at large U in this method.
3. **Issue 6 (+5)** — correct the quantitative spectral claims (gap = |λ| = 0.0138 on lines, not "0.018 over the BZ"; chargon gap 0.34, not 0.13; bands 8-fold degenerate, not "non-degenerate") and re-run the symmetry verification without the `_BREAK8` artifact so the quoted numbers mean what they say. These are the numbers a referee will check first, and currently two of them are contradicted by the project's own figure and grid data.
