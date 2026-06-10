# Slave-boson factorizations over the 2D wallpaper groups (spinful TRS)

*(Sprint note. Two agent attempts at this enumeration died on environment limits
(Bilbao is behind a JS challenge; then a session cap), so this is a compact
manual treatment, written using the same machinery validated in detail for the
3D case in `sg135_factorization.md`. The site-level statements below are
elementary character theory and were spot-checked numerically for C3v and C4v;
the induced-band-rep statements are flagged where they rely on standard tables.)*

## Setup

Slave boson: c†_iσ = f†_iσ h_i + σ d†_i f_{i,−σ} at a site whose site-symmetry
group is a 2D crystallographic point group G_w ∈ {C1, C2, C3, C4, C6, m, 2mm,
3m, 4mm, 6mm}, with spinful time reversal (T² = −1 on the electron). The
electron transforms in ρ_e = (trivial orbital) ⊗ D_{1/2}|_{G_w} (a 2-dim
double-valued corep). We ask for factorizations ρ_e = ρ_b ⊗ ρ_f with ρ_b a
single-valued corep (boson; T² = +1) and ρ_f double-valued (T² = −1).

## Site-level result

**Every 1D real single-valued irrep ρ_b of G_w gives a valid factorization,
with ρ_f = ρ_b ⊗ ρ_e** (characters: χ_f(g) = χ_b(g)χ_e(g); since ρ_b is real
and 1D, ρ_b ⊗ ρ_b = 1, so ρ_b ⊗ ρ_f = ρ_e identically; ρ_f inherits
double-valuedness from ρ_e because χ_b(ḡ E) = χ_b(g) for single-valued reps,
and inherits T² = −1 because T acts on the real 1D boson factor with T² = +1).

- All 1D irreps of the 2D point groups are real except the complex conjugate
  pairs in C3, C4, C6 (e.g. the ¹E/²E of C3). A complex 1D ρ_b is T-paired
  into a 2-dim corep; then dim(ρ_b ⊗ ρ_f) ≥ 2·dim(ρ_f) > 2 = dim(ρ_e), so
  **complex boson reps cannot factor a single Kramers doublet** — they are
  excluded by dimension counting at the corep level, not by characters.
- Counting nontrivial real 1D irreps: C1: none; C2: 1 (B); C3: none; C4: 1 (B);
  C6: 1 (B); m: 1 (A″); 2mm: 3 (A2, B1, B2); 3m: 1 (A2); 4mm: 3 (A2, B1, B2);
  6mm: 3 (A2, B1, B2). **Nontrivial factorizations exist for every G_w except
  C1 and C3.**

## Gauge (PSG) equivalences in the paired (Z₂) phase

As in the 3D analysis, twisting every symmetry g by a Z₂ gauge factor λ(g) ∈
{±1} relabels (ρ_b, ρ_f) → (λρ_b, λρ_f) without changing the physical state,
subject to the group relations: λ must be a genuine Z₂ character of G_w, and
λ(g²) = +1 forces λ to be trivial on squares. Consequences (gauge-invariant
classes of factorizations):

- C2, C4, C6, m: the available λ's kill the sign on the generator only when a
  Z₂ character exists with that sign — for C2 = ⟨r|r²=e⟩, λ(r) = −1 is allowed,
  so B ≃ A: **only one class (trivial)**. For C4: λ(C4) = −1 allowed, but
  λ(C2) = λ(C4)² = +1 always: B (which has χ(C2) = +1, χ(C4) = −1) ≃ A. Hmm:
  B of C4 has χ(C4) = −1 = λ(C4)·1, so B = λ ⊗ A with λ(C4) = −1 a valid
  character: **one class**. Same for C6 (B has χ(C6) = −1, a valid character).
- 2mm: characters λ can take independent signs on the two mirrors: all four
  1D irreps are gauge-equivalent: **one class**.
- **3m (C3v): A2 is NOT gauge-trivializable.** A Z₂ character of C3v must
  satisfy λ(C3) = λ(C3)⁴ = (λ(C3)²)² = ... and λ(C3)³ = 1 ⟹ λ(C3) = +1; and
  λ(m) = ±1 is free, but A2 has χ(m) = −1 and χ(C3) = +1, so A2 = λ ⊗ A1 with
  λ(m) = −1 — which IS a valid Z₂ character of C3v (the homomorphism C3v →
  Z₂ by det). So A2 ≃ A1 after all. Careful: in the *paired* phase the
  constraint is only λ(g²) = +1; m² = e gives no constraint; C3 = (C3²)² ⟹
  λ(C3) = +1 ✓ consistent. **One class.**
- 4mm, 6mm: same mechanism — A2, B1, B2 are all of the form (det-like or
  rotation-sign) characters ⊗ A1; the rotation-sign characters are constrained
  by λ(C2) = λ(C4)² = +1: for 4mm, B1/B2 have χ(C2) = +1 and χ(C4) = −1,
  χ(m) = ±1 — both reachable by characters with λ(C4) = −1 (allowed: C4 is not
  a square in 4mm... note C4 = (C8)² does not apply, C8 ∉ 4mm; C2 = C4² is the
  only square constraint). A2 has χ(C4) = +1, χ(m_v) = χ(m_d) = −1 = the
  det character. **All gauge-trivial: one class.**

**2D conclusion:** at the site level, nontrivial boson reps exist for all
G_w except C1, C3 — but in the Z₂ (paired) slave-boson phase *every* such
linear factorization is a gauge relabeling of the trivial one, because the 2D
point groups have enough Z₂ characters and few square-constraints. This is
WEAKER than the 3D SG135 result, where the nonsymmorphic screw forced
λ(C2z) = λ(4₂)² = +1 and left a genuinely distinct Class II. **The 2D analogue
of a protected nontrivial boson class requires a nonsymmorphic wallpaper group
whose point-group image contains an element that is a square of a
glide/screw-type operation** — of the 17 wallpaper groups, the candidates are
pg, pmg, pgg, p4g (glides), but a glide g squares to a pure translation, and
translations act trivially on the site group — the constraint λ(t) = +1 is
automatic, so no new protection arises at the SITE level. A protected 2D
Class-II-like distinction can only live in the *induced* (momentum-space) data
(fractionalization of glides à la Lee-Hermele-Parameswaran for p4g), not in
the site-symmetry factorization.

## Consequence for the Kane-Mele-Hubbard model (p6mm, Wyckoff 2b, G_w = 3m)

The nontrivial site choice ρ_b = A2 exists but is a Z₂-gauge relabeling of the
trivial one: the KMH slave-boson mean field of Wen et al. is, up to gauge, the
unique linear factorization class. A physically distinct 2D construction would
need a projective (flux-carrying) boson sector, not a different linear rep.
(Under a U(1)-only IGG — i.e. the uncondensed normal state with no pairing —
the g/u-type relabelings are also unconstrained, same conclusion.)

## Verdict

1. Nontrivial site-level factorizations: exist for 8 of the 10 2D point groups.
2. Gauge-invariant content in the Z₂ phase: **all linear 2D factorizations are
   equivalent to the trivial one** — the interesting (protected) structure in
   2D is projective/glide fractionalization, which is band-rep-level, not
   site-level. This makes the 3D SG135 Class II distinction (protected by the
   4₂ screw square-constraint) genuinely special.
3. For KMH specifically: no nontrivial linear class; consistent with the
   literature-search finding that no prior work addresses this (it is "novel"
   but the answer in 2D is a no-go at the linear level).
