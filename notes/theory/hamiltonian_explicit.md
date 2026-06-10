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

---

## 5. What "pi-flux twist" means

The staggered sign s_tau on the z-bonds is itself gauge: spinons carry an
emergent gauge charge, so f_i -> eps_i f_i (eps_i = +-1, site-dependent)
multiplies every bond amplitude by eps_i eps_j, moving the minus signs around
(this is why the pure staggered-z and uniform-z states are exactly
degenerate - same state, two gauges). The gauge-invariant content is the
product of bond signs around closed loops. For the minimal loop using both
channels,

(0,0,0) -z-> (0,0,1/2) -xy-> (1/2,1/2,1/2) -z-> (1/2,1/2,0) -xy-> (0,0,0),

the signs multiply to (+)(+)(-)(+) = -1 = e^{i pi}: every such mixed z-xy
plaquette encloses pi flux of the emergent Z2 gauge field. No gauge
transformation removes it (the analogue of square-lattice pi-flux ansatze in
PSG classifications). It is internal flux seen only by the partons - no
physical magnetic field, no broken symmetry; all gauge-invariant observables
are fully symmetric (verified).

"Twist": because of the flux, C2x and the glides map the sign pattern to a
gauge-shuffled copy; invariance holds only as (symmetry o gauge
transformation) with the compensating sign-per-z-layer gauge G = mu_z - a
nontrivial projective symmetry group (PSG) realization. Energetic payoff: in
k-space the stagger turns the z-channel matrix mu^x into mu^x tau^z, which
anticommutes with the xy channel's tau^x, so the two pairing gaps add in
quadrature with zero interference - the mechanism that gaps both nodal
planes.

---

## 6. Time-reversal, term by term

T = i sigma^y K on electrons/spinons (T^2 = -1); T = K on the spinless
bosons (T^2 = +1). Conditions: a one-body term g(k) Gamma (x) sigma^0 is
T-invariant iff g is real and EVEN and Gamma is REAL; a singlet pairing
block D(k) requires D(-k)^dag = D(k), i.e. additionally Gamma SYMMETRIC and
the amplitude real (a relative complex phase between channels breaks T).

| term | g even? | Gamma real? | Gamma symmetric? | T-inv |
|---|---|---|---|---|
| t_xy: c_x c_y tau^x | yes | yes | yes | yes |
| t_z: c_z mu^x | yes | yes | yes | yes |
| t1': (cos kx - cos ky) mu^z | yes | yes | yes | yes |
| t2': s_x s_y c_z mu^y tau^y | (-)(-)( +) = even | (-mu^y)(-tau^y) = real | yes (antisym x antisym) | yes |
| z-staggered: c_z mu^x tau^z | yes | yes | yes | yes (PLAIN T, no gauge) |
| U n_up n_dn | - | T swaps up/dn | - | yes |

Numerical per-term check (code in conversation record; each channel in
isolation, hopping and pairing blocks, both matter sectors, 12 random k):
every entry EXACTLY zero. T^2 representations verified: (i sigma^y K)^2 = -1
(spinon Kramers), K^2 = +1 (chargon). The converged state's amplitudes are
all relatively real, so the full mean field is T-invariant term by term.

### 6.1 T-covariance of the decomposition itself

The parton T-actions are not free postulates; they must reproduce the
electron transformation through c+_sigma = f+_sigma h + sigma d+ f_{-sigma}.
With T f_up = f_dn, T f_dn = -f_up (Kramers, T^2 = -1) and T h = h,
T d = d (scalars, T^2 = +1; forced, since d+ creates the on-site singlet
|updn>, which is T-even):

  T c+_up T^-1 = f+_dn h - d+ f_up  = c+_dn      (the sigma = dn sign emerges)
  T c+_dn T^-1 = -f+_up h - d+ f_dn = -c+_up

The sigma factor in the doublon term is exactly (i sigma^y f)_sigma - the
T-covariant singlet combination. Machine-checked on the local 4-state
Hilbert space {|0>, |up>, |dn>, |updn>}: decomposition identity, electron-
and parton-level T action, covariance through the decomposition, and
T^2 = (+1, -1, -1, +1) on (|0>, |up>, |dn>, |updn>) - all exact. The derived
bond operators chi^f (spin trace), Delta^f (singlet), chi^b, Delta^b (real
boson bilinears), U d+d, and both constraint terms are then individually
T-even, which is what reduces mean-field T-invariance to reality of the
expectation values (section 6).

---

## 7. Which representations the partons carry (realized state)

Site level (Wyckoff 4a, site group 2/m, every site an inversion center):

- boson (h and d): the TRIVIAL rep A_g (single-valued, T^2 = +1);
  chi_b(C2z) = +1 - the gauge-protected Class I invariant.
- spinon f: E-bar_g, the electron's own Kramers corep (double-valued, even
  parity, T^2 = -1).

So the linear factorization is the "expected/uninteresting" one (trivial
boson x electron corep); all other linear options are Z2-gauge relabelings
of it except the unrealized Class II (chi_b(C2z) = -1). The nontriviality of
the realized state lives one level up, in the PROJECTIVE (PSG) realization
of the space group, identical in both parton sectors: screw 4_2, I, C2z, T
linear; C2x and both glides realized up to the Z2 gauge G = (-1)^{2z}
(mu_z); invariant content = pi flux through mixed z-xy plaquettes. This
projective class restricts trivially to the site group, which is why it
coexists with the trivial site-rep assignment.

Induced band representations: boson A_g -> G (4 bands, 4-fold connected at
A); spinon E-bar_g -> G (8 bands, irreducibly 8-fold at A - the connectivity
that forces spinon pairing rather than a spinon band insulator at nu_f = 4).

---

## 8. Linear vs projective splitting of the electron's symmetry
##    (why the trivial boson rep does not make the state trivial)

The project's founding question was whether the electron representation can
be split as rho_e = rho_b (x) rho_f with a NONTRIVIAL bosonic part - the
intuition being that an "interesting" parton state should divide the
electron's symmetry quantum numbers between its constituents. Two layers of
that statement must be distinguished.

**Layer 1 - linear reps, and their gauge ambiguity.** All three partons
carry the emergent gauge charge, so every symmetry g can be dressed with a
gauge rotation, relabeling (rho_b, rho_f) -> (lambda(g) rho_b,
lambda(g) rho_f) at fixed rho_e and fixed physical state. "The boson is in a
nontrivial 1D rep" is therefore largely a gauge choice. The gauge-invariant
residue in SG135 is a single bit: lambda(C2z) = lambda(screw)^2 = +1 for
every Z2 character, so chi_b(C2z) = +-1 cannot be relabeled. That bit is the
invariant version of "nontrivial boson rep" (Class II when -1). The realized
ground state has chi_b(C2z) = +1: at the linear level its boson is genuinely
trivial, and the Class II sector remains open.

**Layer 2 - projective reps (symmetry fractionalization).** Ask what
operator implements a space-group element on a SINGLE spinon. A linear
answer satisfies U(g1)U(g2) = U(g1 g2). The realized state instead gives a
projective answer: U(g1)U(g2) = omega(g1,g2) U(g1 g2) with non-removable
signs omega = +-1 - the parton represents a Z2 EXTENSION of the space group,
i.e. it carries no linear rep at all. This is consistent because physical
operators are parton bilinears (omega^2 = 1), and the electron - a
boson-fermion bound state - requires exactly

    omega_b . omega_f = trivial :

the electron's linear symmetry class factorizes into two CANCELING
NONTRIVIAL projective classes. This is the precise sense in which the
electron's symmetry is split between the partons (Wen's PSG; the
Essin-Hermele fractionalization classes). In the realized pi-flux state both
partons carry the same nontrivial class: screw/I/C2z/T linear; C2x and both
glides dressed by the gauge sign (-1)^{2z}; gauge-invariant fingerprint: a
single spinon transported around a mixed z-xy plaquette acquires -1 where an
electron acquires +1. No electron band structure can reproduce that.

**The hierarchy:**

| level | invariant | realized state | meaning |
|---|---|---|---|
| linear | chi_b(C2z) (screw^2-protected) | +1 (Class I) | boson carries a distinct sharp linear quantum number; unrealized (= the Class II ansatz) |
| projective | PSG class / pi fluxes | NONTRIVIAL | partons carry canceling non-linear symmetry classes (symmetry fractionalization) |

The two layers have distinct physical discriminators. Linear (Class I vs
II): the chargon-condensation transition out of the topological phase - a
Class I chargon condenses symmetrically (the SC found below U* is exactly
this), a Class II chargon cannot (forced 2D Gamma-multiplets => nematic
transition). Projective: the symmetry quantum numbers of the deconfined
excitations themselves (spinon/vison Wilson loops = -1, gauge-dressed glide
action).

Summary: the energetics declined the linear splitting but spontaneously
selected a state whose symmetry data splits at the projective level - which
is the more general (and arguably more fundamental) sense of fractionalizing
the electron's symmetry, and is what made the full gap, and hence the
topological state, possible. The strongest version of the original
criterion - a boson with the protected linear -1 - is the screw-twisted
Class II ansatz, the natural next target.
