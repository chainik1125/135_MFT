# Slave-boson mean-field theory of the SG135 Hubbard model — 10h sprint summary

**TL;DR.** We built and validated (against the published Kane-Mele-Hubbard slave-boson phase diagram) a fully numerical solver for slave-boson self-consistency equations, then solved the SG135 (P4₂/mbc) double-Dirac Hubbard model that the June write-up got stuck on. The group theory says the LSM/Watanabe gap at filling ν=4 can only be filled by a fractionalized state, and that no choice of boson representation evades the 8ℤ band-connectivity rule. The numerics deliver the twist: the self-consistent mean field IS symmetric and fractionalized, but it is a **nodal, quasi-1D paired spin liquid** (gapless on the kz=π plane), not the fully-gapped Z₂ topological insulator — multi-channel pairing, which would gap it, is energetically refused (channel competition). All symmetry properties were verified explicitly; all phase energies were cross-checked against independent constructions.

## Executive summary

*(≤600 words, drafted — final numbers being filled from the production runs)*

**Problem.** SG135 with spinful TRS forbids any band insulator at filling ν=4 (band minimum: 8 per cell), but the interacting bounds (Watanabe-Po-Vishwanath-Zaletel) allow a gapped symmetric insulator at ν=4 — which, if it exists, must be topologically ordered. Prior work (Manning-Coe & Bradlyn) realized this with exactly-solvable HK interactions. Question: does a *standard* Hubbard interaction, treated in U(1) slave-boson mean-field theory, produce this state? The June 2022 attempt set up the self-consistency equations but could not solve them.

- **Gate passed: the solver machinery reproduces the published KMH phase diagram.** Single-particle gap 0.580t vs paper's 0.57t; double occupancy 0.233 vs ~0.23; U_c(SC→SL)=1.58 vs ~1.57; U_c(SL→DM)=1.98 vs ~1.93 (λ_SO=0). [FIGURE: kmh_boundaries]
  Known residual: our SC region grows more slowly with λ_SO than the paper's figure; our energies are verified against three independent constructions (closed forms, generic eigh/Colpa machinery, operator-level kernels — agreement 1e-10..1e-16), so the discrepancy is robust on our side.
- **The June blocker is solved: it was (a) needing numerics instead of closed forms, and (b) two bookkeeping traps.** The boson sector requires μ≈U/2 for para-spectrum stability (μ=0 makes every solve fail — this is invisible in the KMH closed forms, which sit at the stable point implicitly), and the decoupling constants are channel-dependent, C=(4,4,8,4), not the uniform 8 of the write-up (validated by a real-space bond-expectation referee). [no figure — see §3]
- **Group theory: only two physically distinct slave-boson factorization classes exist in SG135, and neither evades the band-connectivity constraint.** Site group 2/m at Wyckoff 4a gives 4 factorizations collapsing under Z₂ gauge twists to Class I/II (chargon C₂z = ±1); both spinon EBRs stay 8-connected at A. Fractionalization itself (chargon gap + spinon pairing), not the boson rep, is what evades LSM. Our ansatz is Class I; its pairing transforms in the identity corep (verified analytically + numerically).
- **The self-consistent mean field is a symmetric NODAL fractionalized insulator, not the gapped topological one.** Uncondensed solutions exist for U ≳ [U*]; the ground state pairs in the z-channel only: spinon BdG bands are quasi-1D, gapless on the kz=π plane; the chargon is gapped (Mott). All P4₂/mbc generators + TRS verified to 1e-7 at all HSPs. [FIGURE: sg135_mf_bands.png]
- **Why not gapped: channel competition.** Each pairing channel alone is nodal by form factor; gapping requires mixing channels; the energy surface E(Δ_xy, Δ_z) shows two single-channel valleys separated by a ridge — no mixed minimum. The gapped Z₂ insulator is an ansatz, not a solution, of the uniform mean field. [FIGURE: channel_surface.png]
- **Phase diagram vs U.** [PENDING: SC (condensed) at U < U*; nodal z-paired liquid above; energies vs atomic-Mott baseline E=0.] [FIGURE: sg135_phases]

**Caveats.** Mean-field only (no gauge fluctuations; quasi-1D pairing would be fragile); uniform 4-channel ansatz (no flux/Class II patterns, no broken-symmetry bond order); the KMH SL window itself is known (QMC) to be a mean-field artifact — we use KMH only as a solver gate, not as physics.

## Map of the work

| Section | Content | Code |
|---|---|---|
| §1 | KMH gate: validation vs Wen et al. PRB 84, 235149 | `code/kmh/` |
| §2 | Validation chain: conventions, real-space referee, SO kernels | `code/common/`, `code/kmh/test_so_kernels.py`, `code/sg135/test_realspace.py` |
| §3 | SG135 solver + the two bookkeeping traps | `code/sg135/` |
| §4 | Group theory: factorization classes (agent-assisted) | `notes/theory/sg135_factorization.md` |
| §5 | Results: nodal state, bands, symmetry verification, channel competition | `code/sg135/verify_sg135.py` |
| §6 | 2D wallpaper-group question | `notes/theory/wallpaper_factorization.md` |
| log | full research process | `notes/log.md` |

*(sections to be filled below)*
