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
