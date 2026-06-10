# The SG135 Hamiltonian and its slave-boson mean field, all terms explicit

Conventions match the code (`code/sg135/sg135_solver.py`) and every number in
`summary.md`. Lattice: tetragonal, a = b = c = 1. Four sites per cell labeled
by two sublattice indices (μ, τ) ∈ {A, B}²:

| site (μ,τ) | position |
|---|---|
| (A,A) | (0, 0, 0) |
| (A,B) | (½, ½, 0) |
| (B,A) | (0, 0, ½) |
| (B,B) | (½, ½, ½) |

τ = B adds the in-plane offset (½,½,0); μ = B adds the vertical offset
(0,0,½). Pauli matrices μ^a act on the μ index, τ^a on the τ index, σ^a on
spin. Shorthands: c_x ≡ cos(k_x/2), s_x ≡ sin(k_x/2), etc.

---

## 1. The microscopic electron model

H = H_t + H_U, with H_t the spin-orbit-free (t-only) limit of the
Wieder–Kim–Rappe–Kane SG135 model and H_U the on-site Hubbard repulsion.

### 1.1 Bloch form (what the code diagonalizes)

H_t = Σ_{k,σ} c†_{kσ} h(k) c_{kσ},   c_{kσ} = (c_{(A,A)}, c_{(A,B)}, c_{(B,A)}, c_{(B,B)})_{kσ}

h(k) = t_xy · c_x c_y · (μ⁰ ⊗ τˣ)
     + t_z  · c_z     · (μˣ ⊗ τ⁰)
     + t₁′  · (cos k_x − cos k_y) · (μᶻ ⊗ τ⁰)
     + t₂′  · s_x s_y c_z · (μʸ ⊗ τʸ)

Parameters used throughout: t_xy = 1, t_z = 0.5, t₁′ = t₂′ = 0.3
(write-up Fig. 15 values). h(k) is spin-diagonal (⊗ σ⁰); spinful TRS
T = iσʸK. At A = (π,π,π) every form factor vanishes ⇒ the 8-fold double
Dirac point.

### 1.2 Real-space form, bond by bond

(verified to machine precision against the Bloch form in
`code/sg135/test_realspace.py` / `test_realspace_ch4.py`)

**t_xy bonds** — 4 in-plane diagonal neighbors, τ=A ↔ τ=B within one μ-layer:

H_xy = (t_xy/4) Σ_{r} Σ_{μ} Σ_{δ=(±½,±½,0)} Σ_σ c†_{(μ,A),σ}(r) c_{(μ,B),σ}(r+δ) + h.c.

**t_z bonds** — 2 vertical neighbors, μ=A ↔ μ=B at fixed τ:

H_z = (t_z/2) Σ_{r} Σ_{τ} Σ_{δ=(0,0,±½)} Σ_σ c†_{(A,τ),σ}(r) c_{(B,τ),σ}(r+δ) + h.c.

**t₁′ bonds** — same-site-type neighbors at ±x̂ and ±ŷ, sign + for x-bonds,
− for y-bonds, overall sign s_μ = +1 (μ=A) / −1 (μ=B):

H₁ = (t₁′/2) Σ_{r} Σ_{(μ,τ)} s_μ Σ_σ [ c†_{(μ,τ),σ}(r) c_{(μ,τ),σ}(r+x̂)
                                       − c†_{(μ,τ),σ}(r) c_{(μ,τ),σ}(r+ŷ) ] + h.c.

**t₂′ bonds** — 8 body-diagonal neighbors connecting (A,τ) ↔ (B,τ̄):

H₂ = (t₂′/8) Σ_{r} Σ_{τ} s_τ Σ_{δ=(±½,±½,±½)} sgn(δ_x) sgn(δ_y) Σ_σ
        c†_{(A,τ),σ}(r) c_{(B,τ̄),σ}(r+δ) + h.c.,   s_τ = +1 (τ=A) / −1 (τ=B).

**Hubbard term** — every site i of every cell:

H_U = U Σ_i n_{i↑} n_{i↓},   U in units of t_xy.

Filling: ν = 4 electrons per cell (half filling).

---

## 2. Slave-boson decomposition and constraints

c†_{iσ} = f†_{iσ} h_i + σ d†_i f_{i,−σ}

(f: fermionic spinon; h: holon; d: doublon; both bosons spinless). Exact
identities used:

- on-site: U n_{i↑} n_{i↓} = U d†_i d_i
- per bond: Σ_σ c†_{iσ} c_{jσ} = χ̂^f_{ij} χ̂^b_{ji} + Δ̂^{f†}_{ij} Δ̂^b_{ij}, with

  χ̂^f_{ij} = Σ_σ f†_{iσ} f_{jσ},  χ̂^b_{ij} = h†_i h_j − d†_i d_j,
  Δ̂^f_{ij} = Σ_σ σ f_{i,−σ} f_{jσ},  Δ̂^b_{ij} = d_i h_j + h_i d_j.

Constraints, enforced on average with uniform multipliers:

- completeness: h†_i h_i + Σ_σ f†_{iσ} f_{iσ} + d†_i d_i = 1
  → term −λ Σ_i ( h†_i h_i + Σ_σ f†_{iσ} f_{iσ} + d†_i d_i − 1 )
- filling: ⟨Σ_σ c†_{iσ}c_{iσ}⟩ = 1 + x, x = 0 at half filling
  → term −μ_L Σ_i ( d†_i d_i − h†_i h_i − x )

---

## 3. Mean-field decoupling: the five channels

Each bond's quartic term is decoupled by mean fields ⟨χ̂^b⟩, ⟨Δ̂^b⟩ (entering
the spinon Hamiltonian) and ⟨χ̂^f⟩, ⟨Δ̂^f⟩ (entering the chargon Hamiltonian).
Channel structure (form factor g_i, matrix Γ_i, decoupling constant C_i):

| i | bonds | g_i(k) | Γ_i | C_i | bond mean-field pattern |
|---|---|---|---|---|---|
| xy | t_xy | c_x c_y | τˣ | 4 | uniform |
| z | t_z | c_z | μˣ | 4 | uniform |
| 1 | t₁′ | cos k_x − cos k_y | μᶻ | 8 | s_μ·(x vs y) signs of H₁ |
| 2 | t₂′ | s_x s_y c_z | μʸτʸ | 4 | signs of H₂ |
| z̃ | t_z (same bonds as z) | c_z | μˣτᶻ | 4 | **τ-staggered**: bond value = s_τ · (χ_{b,z̃}, Δ_{b,z̃}, …) |

The z̃ channel is the PSG-twisted (π-flux) direction: the *same* z-bonds, but
the bond expectation values alternate in sign between the two τ columns. Its
matrix anticommutes with the xy channel ({μˣτᶻ, τˣ} = 0), so the two pairing
gaps add in quadrature. C_i = 2 Σ_{bonds/cell} (Bloch weight), validated
numerically channel by channel.

### 3.1 Spinon (fermion) mean-field Hamiltonian, every term

With order parameters χ_{b,i}, Δ_{b,i} (i runs over the five channels) and
the multiplier λ:

H_f = Σ_{k,σ} f†_{kσ} [ −λ·𝟙₄ + Σ_i t_i χ_{b,i} g_i(k) Γ_i ] f_{kσ}
    + Σ_k Σ_i t_i Δ_{b,i} g_i(k) [ f†_{k↑} Γ_i (f†_{−k↓})ᵀ − f†_{k↓} Γ_i (f†_{−k↑})ᵀ ]/1 + h.c.

i.e. spin-singlet pairing with the same form factor and orbital matrix as the
hopping: Δ(k) = [Σ_i t_i Δ_{b,i} g_i(k) Γ_i] ⊗ (iσʸ). All Γ_i are real
symmetric and all g_i even, so Δ(k) = Δ(−k) and fermionic antisymmetry holds.
BdG kernel (16×16, basis ψ = (f_{k↑}, f_{−k↓}, f†_{k↑}, f†_{−k↓}) ⊗ 4 orbitals):

H_BdG(k) = [ [ ξ(k), Δ(k) ], [ −Δ*(k), −ξ*(k) ] ],
ξ(k) = −λ·𝟙 + Σ_i t_i χ_{b,i} g_i(k) (σ⁰⊗Γ_i).

### 3.2 Chargon (boson) mean-field Hamiltonian, every term

Basis b_k = (h_{k,1}, …, h_{k,4}, d_{k,1}, …, d_{k,4}); N^a Pauli matrices on
the holon/doublon index:

H_b = Σ_k [ (μ_L − λ) Σ_α h†_{kα}h_{kα} + (U − λ − μ_L) Σ_α d†_{kα}d_{kα} ]
    + Σ_k Σ_i t_i χ_{f,i} g_i(k) [ h†_k Γ_i h_k − d†_k Γ_i d_k ]
    + Σ_k Σ_i t_i Δ_{f,i} g_i(k) [ d_k Γ_i h_{−k} + h_k Γ_i d_{−k} ] + h.c.

(kernel form: ξ_b = (U−2λ)/2 + (U−2μ_L)/2 · N^z + Σ_i t_i χ_{f,i} g_i N^zΓ_i;
pairing block Δ_b = Σ_i t_i Δ_{f,i} g_i N^xΓ_i; para-diagonalized by Colpa.
Note the sign labeling here is the code's, which differs from the KMH paper's
by χ → −χ, Δ → −Δ in both sectors simultaneously — a pure relabeling.)

### 3.3 Constant + constraint terms

E_c = Σ_i C_i t_i ( χ_{b,i} χ_{f,i} + Δ_{b,i} Δ_{f,i} )
    + 4 (λ + x μ_L)
    + (U − λ − μ_L) Σ_α d_α² + (μ_L − λ) Σ_α h_α²
    + Σ_{i: g_i(0)≠0} t_i g_i(0) [ χ_{f,i} (h Γ_i h − d Γ_i d) + 2 Δ_{f,i} (h Γ_i d) ]

where (h_α, d_α) are k = 0 condensate amplitudes (zero in the insulating
states; nonzero in the SC below U* ≈ 0.82) and g_i(0) = (1, 1, 0, 0, 1).

---

## 4. The converged PSG-twisted state at U = 1 (explicit numbers)

All hopping order parameters vanish; only two pairing channels survive:

| quantity | value |
|---|---|
| Δ_{f,xy} | 0.2728 |
| Δ_{b,xy} | 0.1599 |
| Δ_{f,z̃} (staggered) | 0.5354 |
| Δ_{b,z̃} (staggered) | 0.3083 |
| all χ, all other Δ | 0 (≤ 10⁻⁸) |
| λ | −0.01381 |
| μ_L | 0.5000 (= U/2) |

So the realized spinon Hamiltonian is, in full:

H_f^MF = Σ_{k,σ} 0.01381 · f†_{kσ} f_{kσ}
       + Σ_k [ 1.0 · 0.1599 · c_x c_y · (τˣ) + 0.5 · 0.3083 · c_z · (μˣτᶻ) ]_{ab}
              ( f†_{k,a,↑} f†_{−k,b,↓} − f†_{k,a,↓} f†_{−k,b,↑} ) + h.c.

with BdG gap² (k) = λ² + [t_xy Δ_{b,xy} c_x c_y]² + [t_z Δ_{b,z̃} c_z]²
(exact quadrature because {τˣ, μˣτᶻ} = 0): minimum |λ| = 0.0138 on the lines
(k_x or k_y = ±π) ∩ (k_z = ±π). In real space the z̃ pairing reads

Σ_r Σ_τ s_τ (t_z/2) Δ_{b,z̃} ( f†_{(A,τ)↑}(r) f†_{(B,τ)↓}(r+ẑ/2) − ↑↔↓ ) + …,

s_τ = ±1 on the two τ columns — uniform singlet amplitude with a staggered
sign: the π-flux pattern (loops mixing z̃- and xy-bonds enclose flux π; the
sign pattern itself is gauge, the flux is not).

The chargon Hamiltonian at this point is diagonal-plus-pairing with
ξ_b = diag(μ_L−λ, U−λ−μ_L) ⊗ 𝟙₄ + staggered/uniform pairing blocks, gapped
with excitation gap 0.343 t_xy and no condensate.
