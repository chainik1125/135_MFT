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
