# Sprint log — SG135 slave-boson MFT

## Ground rules (fixed at kickoff Q&A, 2026-06-09)
- **Clock**: 10h wall-clock. Start **2026-06-09 21:27 PDT**, end **2026-06-10 07:27 PDT**. Check `date` at least hourly and log here.
- **Compute**: RunPod only (key in `RP_API_KEY_MATS`, set in `~/.zshenv`; verified working, balance $2105). Burn < $10/h. No local compute except very lightweight jobs. Modal NOT used (superseded by user instruction).
- **2D scope**: the 17 wallpaper groups with spinful time reversal (double-valued coreps). Layer groups only if time permits.
- **Step-1 gate**: quantitative reproduction of `notes/theory/kmh_paper/phase_diagram_at_half_filling.pdf` (Wen–Kargarian–Vaezi–Fiete, "Doping the Kane-Mele-Hubbard model: A Slave-Boson Approach", arXiv:1110.3328). Doping results not required.
- **Numerics from scratch** — no surviving code from the June attempt.
- **Subagent fan-out authorized** (theory / numerics / literature search), incl. red-team/blue-team iteration on the write-up.
- **Deliverable**: `summary.md` (repo root). Exec summary ≤300 words, bullets ≤30 words, each backed by one self-explanatory graph. ≥1h reserved for writing. Keep this log as the research-process record.

## Sprint plan
1. **Gate (KMH)**: solve the slave-boson self-consistency equations from the KMH paper appendix; reproduce half-filling phase diagram. Do not pass without high confidence.
2. **2D rep theory**: wallpaper groups + spinful TRS — does electron corep factor as (non-trivial boson rep) ⊗ (fermion corep)? Brute force via Bilbao. If impossible, prove it + check literature.
3. **3D SG135 numerics**: solve the self-consistency equations from `notes/theory/SG135_write_up_June.pdf` (half-filling, homogeneous/isotropic HS fields), then relax assumptions.
4. **SG135 rep enumeration**: non-trivial boson ⊗ fermion = electron rep; confirm/modify the proposed slave-boson decomposition; phase diagram.
5. **Write-up**: summary.md + red/blue agent iteration. Reserve final ~1.5h.

## Key sources
- `notes/theory/kmh_paper/slave-boson_paper-10-16-2011.tex` + figure PDFs (ground truth for gate)
- `notes/theory/SG135_write_up_June.pdf` (June attempt; stuck at self-consistency solve)
- `notes/theory/sg135_hk (2)/main.tex` (HK paper — SG135 double Dirac + LSM context)
- `notes/theory/Watanabe Bound/` (Watanabe 2015/2016 filling constraints; Hastings 2004 LSM)
- Bilbao crystallographic server (coreps); Bradley & Cracknell for group theory.

## Hourly log
- **21:27** Sprint start. RunPod verified. Repo inventoried. Beginning source reading (KMH appendix first).
- **21:45** Read KMH paper (full appendix) + June write-up (all of §4/§5). Diagnosis of June blocker: tried closed-form spectra for the full SG135 kernels; fix = numerical diagonalization (eigh for fermions, Colpa for bosons) + autodiff stationarity.
- **21:50** KMH solver written (code/kmh/): energies transcribed, residuals via jax.grad (avoids hand-transcribing the 13 appendix equations). DM phase = decoupled-dimer limit. Pod 11mcy7pb34iiu2 (L4, $0.39/h) provisioned.
- **21:55** Private repo chainik1125/135_MFT created + pushed (user request).
- **22:00** Lit-search agent returned: factorization question appears NOVEL (see notes/literature_review.md). KMH arXiv ID corrected: 1107.0007. SL sliver is known MFT artifact (QMC kills it) - caveat for write-up. 2D wallpaper-group enumeration agent launched (running).
- **22:20** Local smoke tests: SL solution converges (machine precision); gap ~0.60t @ nk=36 vs paper 0.57t. SC sector required analytic reduction: derived mu=U/2, lam=U/2-3t*Delta_f (BEC pinning) from condensate equations; reduced SC solver added. SC>SL>DM sequence appears at lso=0 with U_c1~1.65 (coarse grid), U_c2~1.9.
- **22:30** First pod stuck (image pull, 0% CPU, 22min) - terminated, redeployed A40 ($0.44/h) with account-cached image.
- **22:35** CONVENTIONS VALIDATED: generic eigh/Colpa machinery == paper closed forms to 2e-10 in boson-stable region (test_bdg_kmh.py). SG135 engine calibrated. Constraint for all future solves: keep min eig H_b >= 0.
- **22:50** Pod3 working (PUBLIC_KEY env was the sshd fix). KMH gate on GPU: V1 PASS (SL, gap 0.580t vs paper 0.57t), V2 PASS (Docc 0.233 vs ~0.23). V3 boundary bisection + full sweep running.
- **23:05** SG135 solver written on validated machinery. Stage A PASS: 16-fold zero mode at A (double Dirac) + band energies match write-up Fig 16 at G/X/M/Z/R exactly; atomic limit E=0. Stage B (the June blocker - actually solving the self-consistency equations) launched on pod. NOTE: write-up Eq 5.53 coordination factor 2z_i=8 appears to double-count; bond-counted value C_i=4t_i used (KMH-pattern-validated); flagged for write-up.
- **00:30-01:00** KMH Eb fix (|g| vs Re g in primed cross-term, per paper d3,4). lso=0 boundaries STABLE: Uc1=1.584, Uc2=1.982 (paper ~1.57, ~1.93). lso>0 SC-boundary slope still shallower than paper even with lso-continuation seeding - documented as discrepancy; hypothesis: primed-channel branch; NOT relevant to SG135 (no SOC channels there). Gate: 4/5 quantitative checks pass.
- **01:00** theory-sg135 agent delivered (notes/theory/sg135_factorization.md): Wyckoff 4a (2/m); 4 factorizations -> 2 gauge classes (chargon C2z=+-1); 8Z connectivity holds for ALL factorizations - evasion is via fractionalization+pairing, not boson rep; our pairing = identity corep (Class I). Caveat: all form factors vanish at A -> gap at A only via lam != 0.
- **01:15** SG135 BREAKTHROUGH after mu-tether fix (mu_L ~ U/2 needed for boson para-spectrum stability; mu=0 tether had artificially crippled all earlier solves): uncondensed solutions converge to 1e-10. Ground state of uncondensed sector = Z-CHANNEL-ONLY pairing (Df_z~0.64, lam~0): nodal plane at kz=pi, NOT fully gapped. xy-only root exists but loses (-0.042 vs -0.052 at U=4); no mixed root found. Within this ansatz: symmetric NODAL fractionalized state evades LSM by gaplessness; chargon sector gapped (wmin>0).
- (note: log timestamps above between 22:50-01:15 ran ~1h ahead of wall clock; sequence correct)
- **23:40-00:40** Stage D: t_z tuning only swaps WHICH nodal state wins; uniform channel mixing always refused (concavity + commuting structures). Mixed uniform root = mountain pass. Channel-competition surface job queued.
- **00:40-01:45 HEADLINE**: PSG-twisted ansatz (tau-staggered z-channel, mu^x tau^z structure, anticommutes with xy channel -> quadrature gaps): GLOBAL minimum for all U>=1 (E=-0.2374 vs nodal -0.2217 at U=1), fully gapped (BdG 0.018, chargon 0.13), uncondensed, symmetric with PSG verified explicitly (screw/I/C2z/T plain; C2x/glides up to mu_z gauge; pi flux through mixed loops). Gauge-equivalence sanity check: pure staggered-z == uniform-z energy to 1e-5. Production nk=12 confirmation sweep running.
