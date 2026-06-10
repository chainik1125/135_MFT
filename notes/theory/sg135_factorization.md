# Symmetry factorization of the SG 135 slave-boson mean field

**Space group:** $P4_2/mbc\,1'$ (No. 135, $D_{4h}^{13}$), with spinful time reversal.
**Model:** Wieder–Kim–Rappe–Kane double-Dirac tight-binding model [PRL **116**, 186402 (2016); arXiv:1512.00074], as used in the HK paper (`sg135_hk (2)/main.tex`, Sec. "The Double-Dirac Spin Liquid in Space Group P4₂/mbc1′").
**Parton decomposition:** $c^\dagger_{i\sigma} = f^\dagger_{i\sigma} h_i + \sigma\, d^\dagger_i f_{i,-\sigma}$, gauge charges $(f,h,d) = (+1,+1,-1)$, on-site constraint $Q_i = n^f_i + n^h_i - n^d_i = 1$. Filling: 4 electrons/cell (half filling), hence $\nu_f = 4$/cell and integer chargon filling.

**Status of sources.** The Bilbao Crystallographic Server (cryst.ehu.es / cryst.ehu.eus) currently sits behind a Cloudflare Turnstile JS challenge, so the live WYCKPOS/BANDREP/Corepresentations pages could not be scripted. The WYCKPOS table below was retrieved from a Wayback-archived copy of the Bilbao output for `nph-wp-list?gnum=135` and is verbatim ITA data. All representation-theoretic statements (site coreps, induced band reps, degeneracies, parities) were *computed ab initio* by explicit induction (method in Appendix A; scripts `/tmp/sg135_induction.py`, `/tmp/sg135_pairing.py`) and cross-checked against Wieder *et al.* (single 8-dim corep at $A$) and Watanabe–Po–Vishwanath–Zaletel (8Z band-insulator rule). Where Bilbao label conventions ($\bar\Gamma_i$ numbering) could not be re-verified live, labels are defined self-containedly by their characters.

---

## 0. Executive answers

1. **Wyckoff position.** The four sites $(0,0,0)$, $(\tfrac12,\tfrac12,0)$, $(0,0,\tfrac12)$, $(\tfrac12,\tfrac12,\tfrac12)$ are a single orbit: Wyckoff **4a**, site-symmetry group $G_w = 2/m..$ ($C_{2h}$, twofold axis $\parallel \hat z$, mirror $m_z$). Every 4a site is an inversion center.
2. **Site factorization.** $\rho_e = \bar E_g$ (s-orbital ⊗ spin-1/2). Exactly **four** linear factorizations $\rho_e = \rho_b\otimes\rho_f$: $(A_g,\bar E_g)$, $(B_g,\bar E_g)$, $(A_u,\bar E_u)$, $(B_u,\bar E_u)$. No complex/T-paired single-valued options exist for $2/m$. After pairing (IGG $=\mathbb Z_2$), gauge twists collapse these to **two physically distinct classes**: Class I $\{(A_g,\bar E_g)\simeq(A_u,\bar E_u)\}$ and Class II $\{(B_g,\bar E_g)\simeq(B_u,\bar E_u)\}$, distinguished by the gauge-invariant chargon eigenvalue $\chi_b(C_{2z}) = \chi_b\big((\{C_{4z}|00\tfrac12\})^2\big) = \pm1$.
3. **Induced band reps.** Both spinon EBRs ($\bar E_g{\uparrow}G$, $\bar E_u{\uparrow}G$, 8 bands each) are **connected through a single 8-dim corep at $A$** — the $8\mathbb Z$ rule holds for *every* factorization; no choice of $\rho_b$ evades it. All four boson EBRs (4 bands) are likewise connected (4-dim corep at $A$), but bosons at integer filling per site always admit a symmetric gapped Mott (atomic) state. **The constraint is evaded by fractionalization itself — boson statistics for the charge sector and pairing ($U(1)\to\mathbb Z_2$) for the spinon sector — not by a nontrivial $\rho_b$.**
4. **Mean-field pairing.** The hopping-form-factor singlet pairing transforms in the **identity corep of $P4_2/mbc \times \mathcal T$**, channel by channel, with *no* compensating gauge transformation needed (trivial PSG phases). Caveat: all four form factors **vanish at $A$**; the BdG gap at $A$ is controlled solely by $\mu_f$ (or by an additional on-site singlet $\Delta_0$, also symmetry-allowed).
5. **Synthesis.** A converged symmetric gapped mean field at $\nu=4$ is a 3+1d **$\mathbb Z_2$ fractionalized ("Z2 quantum spin liquid / fractionalized Mott") insulator** — precisely the loophole WPVZ left open between their interacting bound ($\nu \in 2\mathbb Z$, $\nu\ge4$) and the band-theory bound ($\nu\in 8\mathbb Z$). Our ansatz realizes Class I (trivial $\rho_b$). Class II would require a flux-twisted ansatz and is physically distinguished by the chargon spectrum at $A$ and by symmetry breaking at the chargon-condensation transition.

---

## 1. Wyckoff position and site-symmetry group

### 1.1 Bilbao WYCKPOS data for SG 135 ($P4_2/mbc$, origin at center $2/m$ on the $4_2$ axis)

| Mult. | Letter | Site symmetry | Orbit representatives |
|---|---|---|---|
| 16 | i | $1$ | $(x,y,z)$, … |
| 8 | h | $m..$ | $(x,y,0)$, … |
| 8 | g | $..2$ | $(x,x{+}\tfrac12,\tfrac14)$, … |
| 8 | f | $2..$ | $(0,\tfrac12,z)$, … |
| 8 | e | $2..$ | $(0,0,z)$, … |
| 4 | d | $2.22$ | $(0,\tfrac12,\tfrac14)$, $(\tfrac12,0,\tfrac34)$, $(0,\tfrac12,\tfrac34)$, $(\tfrac12,0,\tfrac14)$ |
| 4 | c | $2/m..$ | $(0,\tfrac12,0)$, $(\tfrac12,0,\tfrac12)$, $(\tfrac12,0,0)$, $(0,\tfrac12,\tfrac12)$ |
| 4 | b | $\bar4..$ | $(0,0,\tfrac14)$, $(0,0,\tfrac34)$, $(\tfrac12,\tfrac12,\tfrac34)$, $(\tfrac12,\tfrac12,\tfrac14)$ |
| **4** | **a** | $\mathbf{2/m..}$ | $\mathbf{(0,0,0),\ (0,0,\tfrac12),\ (\tfrac12,\tfrac12,0),\ (\tfrac12,\tfrac12,\tfrac12)}$ |

The model's four sublattices — $\tau=\pm$ at $(0,0,0)$, $(\tfrac12,\tfrac12,0)$ and $\mu=\pm$ stacking $z=0$, $z=\tfrac12$ — are exactly the **4a orbit** (the HK paper's statement "Together these four sites form the $4a$ Wyckoff position" is confirmed). Site ordering used throughout: $1=(0,0,0)$, $2=(\tfrac12,\tfrac12,0)$, $3=(0,0,\tfrac12)$, $4=(\tfrac12,\tfrac12,\tfrac12)$; i.e. $(\mu,\tau) = (++),(+-),(-+),(--)$, so $\mu^x$ swaps $1\leftrightarrow3$, $2\leftrightarrow4$ and $\tau^x$ swaps $1\leftrightarrow2$, $3\leftrightarrow4$.

### 1.2 Site-symmetry group of 4a

Space-group elements fixing $q_1=(0,0,0)$ exactly (zero residual translation):

$$G_w \;=\; \{\,\{E|0\},\ \{C_{2z}|001\}\!\sim\!C_{2z},\ \{I|000\},\ \{m_z|001\}\!\sim\!m_z\,\}\;\cong\; 2/m \;(C_{2h}),$$

with $C_{2z} = (\{C_{4z}|00\tfrac12\})^2$ modulo the lattice translation $(0,0,1)$, and $m_z = I\cdot C_{2z}$. Order 4, index 4 in $D_{4h}$ ⇒ orbit length $16/4=4$. ✓ Note $C_{2z}$ is *not* an independent generator of $G$: it is the square of the $4_2$ screw. This single fact drives the Class I/II distinction in §2.4.

Coset representatives mapping $q_1$ to the orbit (used for induction): $g_1=\{E|0\}$, $g_2=\{C_{2x}|\tfrac12\tfrac12 0\}$, $g_3=\{C_{4z}|00\tfrac12\}$, $g_4=\{E|100\}\,g_3\,g_2$ (so that $g_\alpha q_1 = q_\alpha$ exactly).

---

## 2. Site-level factorization of the electron corep

### 2.1 Coreps of $2/m\,1'$ at 4a

**Single-valued** (boson candidates; $\mathcal T^2=+1$). All four irreps of $C_{2h}$ are real 1D ⇒ Wigner type (a): each corep = irrep, no doubling, **no complex/T-paired options exist**.

| $\rho_b$ | $E$ | $C_{2z}$ | $I$ | $m_z$ | orbital analogue |
|---|---|---|---|---|---|
| $A_g$ | 1 | $+1$ | $+1$ | $+1$ | $s$ |
| $B_g$ | 1 | $-1$ | $+1$ | $-1$ | $d_{xz}$-like |
| $A_u$ | 1 | $+1$ | $-1$ | $-1$ | $p_z$ |
| $B_u$ | 1 | $-1$ | $-1$ | $+1$ | $p_x$-like |

**Double-valued** (spinon candidates; $\mathcal T^2=-1$). The double group of $C_{2h}$ has four 1D spinor irreps, complex in pairs ($\chi(C_{2z}) = \mp i$). With $\bar E$ the $2\pi$ rotation ($\chi(\bar E) = -1$ for all four):

| irrep | $C_{2z}$ | $I$ | $m_z$ | $\mathcal T$ partner |
|---|---|---|---|---|
| $^1\!\bar E_g$ | $-i$ | $+1$ | $-i$ | $^2\!\bar E_g$ |
| $^2\!\bar E_g$ | $+i$ | $+1$ | $+i$ | $^1\!\bar E_g$ |
| $^1\!\bar E_u$ | $-i$ | $-1$ | $+i$ | $^2\!\bar E_u$ |
| $^2\!\bar E_u$ | $+i$ | $-1$ | $-i$ | $^1\!\bar E_u$ |

Time reversal conjugates characters, exchanging the superscript-1/2 partners within each parity sector ⇒ two 2D **physical coreps** (Wigner pairing of complex conjugates):

$$\bar E_g = {}^1\!\bar E_g \oplus {}^2\!\bar E_g,\qquad \bar E_u = {}^1\!\bar E_u \oplus {}^2\!\bar E_u,$$

with corep characters $\chi(E)=2$, $\chi(C_{2z})=0$, $\chi(I)=\pm2$, $\chi(m_z)=0$. Both carry the spin-1/2 factor system: $\rho(C_{2z})^2=-1$, $\rho(m_z)^2=-1$, $\rho(I)^2=+1$, $\mathcal T^2=-1$. (Numerically: $\rho(C_{2z}) = -i\sigma^z$ has eigenvalues $\mp i$ exchanged by $\mathcal T = i\sigma^y K$. ✓)

**Electron at 4a:** $s$-like orbital ⊗ spin-1/2, $\rho(I) = +1$ ⇒ $\boxed{\rho_e = \bar E_g}$.

### 2.2 The four factorizations $\rho_e = \rho_b \otimes \rho_f$

Because each $\rho_b$ is real 1D with $\rho_b^{\otimes2} = A_g$, the spinon corep is fixed: $\rho_f = \rho_b \otimes \bar E_g$. Tensoring with a $g$ character permutes the $^{1,2}$ components ($B_g\otimes{}^1\!\bar E_g = {}^2\!\bar E_g$), tensoring with a $u$ character flips the parity:

| # | $\rho_b$ (holon $h$, doublon $d$; $\mathcal T_b^2=+1$) | $\rho_f$ (spinon; $\mathcal T_f^2=-1$) | $\rho_f(C_{2z})$ | $\rho_f(I)$ | comment |
|---|---|---|---|---|---|
| 1 | $A_g$ | $\bar E_g$ | $-i\sigma^z$ | $+1$ | trivial assignment |
| 2 | $B_g$ | $B_g\otimes\bar E_g \cong \bar E_g$ | $+i\sigma^z$ | $+1$ | boson $C_{2z}$-odd |
| 3 | $A_u$ | $\bar E_u$ | $-i\sigma^z$ | $-1$ | boson parity-odd |
| 4 | $B_u$ | $\bar E_u$ | $+i\sigma^z$ | $-1$ | both |

Bookkeeping check (TR): $\mathcal T h \mathcal T^{-1} = h$, $\mathcal T d \mathcal T^{-1} = d$ ($\mathcal T_b^2 = +1$), $\mathcal T f^\dagger_\sigma \mathcal T^{-1} = \sigma f^\dagger_{-\sigma}$ ($\mathcal T_f^2=-1$); substituting into $c^\dagger_\sigma = f^\dagger_\sigma h + \sigma d^\dagger f_{-\sigma}$ reproduces $\mathcal T c^\dagger_\sigma \mathcal T^{-1} = \sigma c^\dagger_{-\sigma}$. ✓ The factor systems multiply correctly: $\omega_b \cdot \omega_f = \omega_{\rm spin}$ with $\omega_b$ trivial in all four rows.

**Projective options.** One may also let $\rho_b$ be *projective* (nontrivial $\omega_b \in H^2(C_{2h}\times \mathbb Z_2^{\mathcal T},U(1))$, e.g. a Kramers boson $\mathcal T_b^2=-1$, or $\{\rho_b(C_{2z}),\rho_b(I)\}=0$), compensated by $\omega_f = \omega_{\rm spin}\,\omega_b^{-1}$. These are *not* realizable by an on-site scalar holon/doublon: they correspond to distinct symmetry-fractionalization (PSG) classes of the $\mathbb Z_2$ phase that would require intrinsically different mean fields (flux patterns), not different 1D site labels. They are the natural home of "exotic" candidate states but are outside the present decoupling; see §5.3.

### 2.3 What is *not* gauge-invariant here

The pair $(\rho_b,\rho_f)$ can be twisted by attaching a gauge rotation to each symmetry: $g \mapsto g\,e^{i\theta_g Q}$ relabels $(\rho_b,\rho_f)\to(\lambda\rho_b,\lambda\rho_f)$ with $\lambda(g) = e^{i\theta_g}$ a character into the invariant gauge group (IGG), while $\rho_e = \rho_b\otimes\rho_f$ stays fixed.

- **U(1) mean field (no pairing):** IGG $= U(1)$. Both $\theta(I)=\pi$ (giving $\lambda|_{G_w} = A_u$) and $\theta(C_{4z})=\pi/2$ (giving $\lambda(C_{2z}) = e^{2i\theta} = -1$, i.e. $\lambda|_{G_w} = B_g$ or $B_u$) are consistent twists ⇒ *all four* factorizations are gauge-relabelings of one another.
- **Paired ($\mathbb Z_2$) mean field:** IGG $=\mathbb Z_2$, so $\lambda: G\to\{\pm1\}$. Since $C_{2z} = (\text{screw})^2$, every $\mathbb Z_2$ character obeys $\lambda(C_{2z}) = \lambda(C_{4z})^2 = +1$: restricted to $G_w$, the available twists are only $\{A_g, A_u\}$.

### 2.4 Gauge-invariant classification of the linear factorizations

$$\text{Class I} := \{(A_g,\bar E_g)\simeq(A_u,\bar E_u)\},\qquad \text{Class II} := \{(B_g,\bar E_g)\simeq(B_u,\bar E_u)\}.$$

The $g/u$ (parity) split is pure gauge in either phase. The invariant datum separating I from II in the $\mathbb Z_2$ phase is

$$\chi_b\!\left(\{C_{2z}|001\}\right) \;=\; \chi_b\!\left((\{C_{4z}|00\tfrac12\})^2\right) \;=\; \pm 1 ,$$

the *linear* $C_{2z}$ eigenvalue of the chargon — protected because no $\mathbb Z_2$ twist can flip the square of an element. (Caveat: a further relabeling $e\to e\times m$, fusing the chargon with a vison, can shift this if the vison itself carries $C_{2z}$ quantum numbers; comparing candidate states requires the full anyon data, §5.2. Within a fixed parton scheme, Class I vs II is sharp.)

---

## 3. Induced band representations

Method: explicit Frobenius induction in Bloch form, $[U^{(\rho)}(g,k)]_{\beta\alpha} = e^{-i(\mathbf{W}k)\cdot \mathbf R_{g\alpha}}\,\chi_\rho(h(g,\alpha))$ with stabilizer elements $h = g_\beta^{-1}\{E|-\mathbf R\}\,g\,g_\alpha \in G_w$ computed exactly; degeneracies and per-multiplet characters extracted by symmetrizing random Hermitian matrices over the little group + TR (Appendix A). Representation property verified to machine precision.

### 3.1 Spinon sector: double-valued EBRs (8 bands), with $\mathcal T$, $\mathcal T^2=-1$

$\bar E_g{\uparrow}G$ ($\rho_f$ row 1–2) and $\bar E_u{\uparrow}G$ (rows 3–4):

| TRIM | degeneracies | characters per multiplet, $\bar E_g{\uparrow}G$ | $\bar E_u{\uparrow}G$ |
|---|---|---|---|
| $\Gamma$ | $2+2+2+2$ | $\chi(C_4^{\rm screw}) = +\sqrt2,+\sqrt2,-\sqrt2,-\sqrt2$; $\chi(I)=+2$ each | same $C_4$; $\chi(I) = -2$ each |
| $Z$ | $4+4$ | all of $\chi(I),\chi(C_{2z}),\chi(C_4),\chi(C_{2x}),\chi(m_z) = 0$ | same |
| $M$ | $4+4$ | $\chi(I) = +4$ each, others $0$ | $\chi(I) = -4$ each |
| $A$ | $\mathbf{8}$ | single irreducible corep; all listed characters $0$ | same |
| $R$ | $4+4$ | all $0$ | same |
| $X$ | $4+4$ | all $0$ | same |

In $D_{4h}$ double-group labels at $\Gamma$: $\bar E_g{\uparrow}G|_\Gamma = 2\bar E_{1g}\oplus 2\bar E_{2g}$ (i.e. $2\bar\Gamma_6^+\oplus2\bar\Gamma_7^+$), and $\bar E_u{\uparrow}G|_\Gamma = 2\bar\Gamma_6^-\oplus2\bar\Gamma_7^-$.

**Consequences.**
- At $A$ there is a single 8-dimensional double-valued corep — this *is* Wieder et al.'s "single 8DIR at the $A$ point"; both spinon EBRs are **connected** (indecomposable): any energetically isolated set of bands must contain a multiple of 8 bands. This reproduces the **$8\mathbb Z$ band-insulator rule** of WPVZ for SG 135 + TRS + SOC.
- Therefore at spinon filling $\nu_f = 4$/cell, **no number-conserving symmetric spinon insulator exists for *any* factorization** — the $g/u$ choice only flips inversion characters at $\Gamma$ and $M$ and cannot unstick the $A$-point octet. The mean-field double-Dirac point at $A$ is filling-enforced.
- The only symmetric escape is **pairing**: the WPVZ/connectivity argument uses $U(1)$ charge conservation; a BdG Hamiltonian (spinon parity $\mathbb Z_2$ only) is not constrained by it. This is the load-bearing step — see §4 for whether the specific pairing is symmetric, and §3.3 for the $A$-point caveat.

### 3.2 Boson sector: single-valued EBRs (4 bands), with $\mathcal T$, $\mathcal T^2=+1$

| TRIM | $A_g{\uparrow}G$ | $B_g{\uparrow}G$ | $A_u{\uparrow}G$ | $B_u{\uparrow}G$ |
|---|---|---|---|---|
| $\Gamma$ | $A_{1g}\oplus A_{2g}\oplus B_{1g}\oplus B_{2g}$ (four 1D) | $E_g\oplus E_g$ (two 2D) | $A_{1u}\oplus A_{2u}\oplus B_{1u}\oplus B_{2u}$ | $E_u\oplus E_u$ |
| $Z$ | $2+2$, $\chi(C_{2z})=+2$ | $2+2$, $\chi(C_{2z})=-2$, $\chi(C_{2x})=\pm2$ | $2+2$, $\chi(C_{2z})=+2$ | $2+2$, $\chi(C_{2z})=-2$ |
| $M$ | $2+2$, $\chi(I)=\chi(m_z)=+2$ | $2+2$, $\chi(I)=+2$, $\chi(m_z)=-2$ | $2+2$, $\chi(I)=\chi(m_z)=-2$ | $2+2$, $\chi(I)=-2$, $\chi(m_z)=+2$ |
| $A$ | $\mathbf 4$, $\chi(C_{2z})=+4$ | $\mathbf 4$, $\chi(C_{2z})=-4$ | $\mathbf 4$, $\chi(C_{2z})=+4$ | $\mathbf 4$, $\chi(C_{2z})=-4$ |
| $R$ | $2+2$, $\chi(m_z)=\pm2$ | $2+2$, $\chi(m_z)=\pm2$ | $2+2$, $\chi(m_z)=\pm2$ | $2+2$, $\chi(m_z)=\pm2$ |
| $X$ | $2+2$, $\chi(m_z)=+2$ both | $2+2$, $\chi(m_z)=-2$ both | $2+2$, $\chi(m_z)=-2$ both | $2+2$, $\chi(m_z)=+2$ both |

(Unlisted characters vanish. Inversion characters at $Z,A,R,X$ vanish for all four.)

**Consequences.**
- All four boson EBRs are **connected**, glued by a single 4-dim corep at $A$ where $C_{2z}$ acts as $\chi_b(C_{2z})\cdot\mathbb 1$. The single-valued analogue of the $8\mathbb Z$ rule for SG 135 + TRS is therefore $4\mathbb Z$ — a *spinless fermion* band insulator needs filling $4\mathbb Z$ (satisfied at 4, by filling the whole EBR).
- **For bosons this connectivity is not an obstruction.** A symmetric gapped chargon sector at integer filling per 4a site is simply the atomic (site-localized) insulator built from $\rho_b{\uparrow}G$ — it exists for **every** $\rho_b$, trivial or not. (Bosonic LSM only requires integer filling per cell, with the WPVZ nonsymmorphic refinement $\nu_b \in 2\mathbb Z$; $\nu_b = 4$ ✓.) In the slave-boson variables: holons/doublons gapped with $\langle n_h\rangle = \langle n_d\rangle$, net boson charge integer per site, constraint $Q_i = 1$.
- **Answer to the key question:** the minimal-connectivity ($8\mathbb Z$) constraint is *not* evaded by choosing a nontrivial $\rho_b$ — it is evaded (i) in the charge sector by bosonic statistics (Mott/atomic state regardless of $\rho_b$) and (ii) in the spin sector by pairing. A nontrivial $\rho_b$ is **not load-bearing for gappability**; it *is* physically meaningful for which symmetry quantum numbers the gapped sectors carry and for the phase transitions out of the spin liquid (§3.4, §5).

### 3.3 The $A$-point caveat for the paired spinon sector

Every one of the model's form factors (hopping *and* the hopping-form-factor pairings, §4) vanishes at $A=(\pi,\pi,\pi)$: $\cos\frac{k_x}{2}\cos\frac{k_y}{2} = \cos\frac{k_z}{2} = (\cos k_x - \cos k_y) = \sin\frac{k_x}{2}\sin\frac{k_y}{2}\cos\frac{k_z}{2} = 0$, and the SOC factors too: $H(A) = 0$, $\Delta(A) = 0$ (verified numerically to $10^{-16}$). Hence at $A$ the BdG spectrum is exactly $\pm\mu_f$ (8-fold each):

- with $\mu_f = 0$ and only the four bond-pairing channels, **the BdG spectrum remains gapless at $A$ no matter how large the $\Delta_i$ are** (verified on a $16^3$ grid: min gap $=0$);
- $\mu_f \neq 0$ gaps $A$ (BdG levels $\pm|\mu_f|$); an **on-site singlet** $\Delta_0\,\mathbb 1_{4}\otimes i\sigma^y$ is also fully symmetry-allowed (identity corep, verified) and gaps $A$ even at $\mu_f = 0$.

*Practical check for the numerics:* a converged "fully gapped" solution must have $\mu_f\neq0$ and/or a nonzero on-site singlet amplitude; otherwise the gap claim fails exactly at the double-Dirac momentum.

### 3.4 Where a nontrivial $\rho_b$ *would* matter

The chargon dispersion built on $\rho_b{\uparrow}G$ pins observable quantum numbers:

- $A_g$ chargon ($\Gamma$ content $A_{1g}\oplus A_{2g}\oplus B_{1g}\oplus B_{2g}$): a band minimum at $\Gamma$ in $A_{1g}$ condenses into a fully symmetric state.
- $B_g$ chargon ($\Gamma$ content $E_g\oplus E_g$): *every* level at $\Gamma$ is a 2D irrep ⇒ condensation at $\Gamma$ necessarily breaks the point group (two-component order parameter) — a nematic descendant.
- For all four EBRs the $A$-point quadruplet is irreducible: a chargon minimum at $A$ (possible at strong frustration since the bare boson hopping inherits the form factors and also vanishes at $A$) condenses with broken translation symmetry (unit-cell doubling along all three axes).

---

## 4. Symmetry of the mean-field spinon pairing

### 4.1 Setup and conventions

Hopping channels (HK paper Eq. (135 Hamiltonian), spin-independent part):

$$\mathcal H^1(k) = t_{xy}\,g_{xy}\,\tau^x + t_z\,g_z\,\mu^x + t'_1\,g_1\,\mu^z + t'_2\,g_2\,\mu^y\tau^y,$$
$$g_{xy} = \cos\tfrac{k_x}{2}\cos\tfrac{k_y}{2},\quad g_z = \cos\tfrac{k_z}{2},\quad g_1 = \cos k_x - \cos k_y,\quad g_2 = \sin\tfrac{k_x}{2}\sin\tfrac{k_y}{2}\cos\tfrac{k_z}{2}.$$

Pairing ansatz (spin singlet, same form factors): $\hat\Delta = \sum_k f^\dagger_{k}\,\Delta(k)\,(f^\dagger_{-k})^T + \text{h.c.}$ with

$$\Delta(k) \;=\; \Big[\textstyle\sum_i t_i\Delta_i\, g_i(k)\, M_i\Big]\otimes (i\sigma^y),\qquad M_i \in \{\tau^x,\ \mu^x,\ \mu^z,\ \mu^y\tau^y\},\ \ \Delta_i \in \mathbb R .$$

Generator reps (HK paper Table; periodic Bloch convention, constant matrices up to global $e^{-i(\mathbf Wk)\cdot \mathbf w}$ phases which cancel between $k$ and $-k$ in any pairing bilinear):

$$U(\{C_{4z}|00\tfrac12\}) = \mu^x e^{i\pi\sigma^z/4},\quad U(\{C_{2x}|\tfrac12\tfrac12 0\}) = i\tau^x\sigma^x,\quad U(\{I|000\}) = \mathbb 1,\quad \mathcal T = i\sigma^y K.$$

> **Convention note (bug-relevant):** the matrix $\mu^x e^{+i\pi\sigma^z/4}$ represents the screw with the spatial action $(x,y,z)\to(y,-x,z+\tfrac12)$, i.e. $\{C^-_{4z}|00\tfrac12\}$; pairing it with $(x,y,z)\to(-y,x,z+\tfrac12)$ requires $e^{-i\pi\sigma^z/4}$. With consistent sense, the full $\mathcal H^0_{135}$ (including all three SOC terms as printed in the HK paper, with the corrected $\cos\frac{k_x}{2}\sin\frac{k_y}{2}$ coefficient) is invariant under all 16 coset operations and TR to machine precision; with mismatched sense the SOC terms appear (spuriously) to break the screw.

Glides used below: $b$-glide $\{m_x|\tfrac12\tfrac12 0\}$ / $\{m_y|\tfrac12\tfrac12 0\}$ (plane $\perp[100]$ resp. $[010]$), $c$-glide $\{m_{110}|\tfrac12\tfrac12\tfrac12\}$, with $U(m_y) = U(I)U(C_{2z})U(C_{2x})$, $U(m_{110}) = U(I)\,U(C_{4z})U(C_{2x})$.

### 4.2 Channel-by-channel transformation (analytic)

Invariance condition for a hopping channel: $U\,[g_iM_i](k)\,U^\dagger = [g_iM_i](\mathbf Wk)$; for the singlet pairing: $U\,[g_iM_i\otimes i\sigma^y](k)\,U^{T} = [g_iM_i\otimes i\sigma^y](\mathbf Wk)$. Since every $U$ above satisfies $U(-k)^* = U(k)$ (real orbital permutations; SU(2) part handled by $u\,(i\sigma^y)\,u^T = i\sigma^y$ for all $u\in SU(2)$), the singlet pairing condition reduces channel-by-channel to the hopping condition. Sign bookkeeping:

| channel | screw $C_{4z}$: $g_i\to$, $M_i\to$ | $C_{2x}$: $g_i\to$, $M_i\to$ | $b$-glide: $g_i\to$, $M_i\to$ | $c$-glide: $g_i\to$, $M_i\to$ | $I$ | net |
|---|---|---|---|---|---|---|
| $g_{xy}\,\tau^x$ | $+,\ +$ | $+,\ +$ | $+,\ +$ | $+,\ +$ | $+$ | **inv.** |
| $g_z\,\mu^x$ | $+,\ +$ | $+,\ +$ | $+,\ +$ | $+,\ +$ | $+$ | **inv.** |
| $g_1\,\mu^z$ | $-,\ -$ | $+,\ +$ | $+,\ +$ | $-,\ -$ | $+$ | **inv.** |
| $g_2\,\mu^y\tau^y$ | $-,\ -$ | $-,\ -$ | $-,\ -$ | $+,\ +$ | $+$ | **inv.** |

(E.g. screw: $g_1(\mathbf W^{-1} k) = \cos k_y - \cos k_x = -g_1$ while $\mu^x\mu^z\mu^x = -\mu^z$; the $d$-wave-like sign of $(\cos k_x - \cos k_y)$ is compensated by the sublattice-odd matrix $\mu^z$ — the composite is fully symmetric. Same mechanism for $g_2$ with $\mu^y\tau^y$.)

Time reversal: all $g_i$ are real and **even**, all $M_i$ real symmetric, the spin structure is the singlet ⇒ $(\mathbb 1\otimes i\sigma^y)\,\Delta(-k)^*\,(\mathbb 1\otimes i\sigma^y)^T = \Delta(k)$. Fermion antisymmetry $\Delta(k) = -\Delta(-k)^T$ holds ($M_i^T = M_i$, $g_i$ even). **Numerical verification:** every channel, every generator (screw, $C_{2x}$, both glides, $C_{2z}$, $I$), TR, and antisymmetry hold to $\lesssim10^{-16}$; the on-site singlet $\Delta_0\,\mathbb 1\otimes i\sigma^y$ is likewise invariant.

### 4.3 Conclusion and gauge-invariant statement

**The hopping-form-factor singlet pairing transforms in the identity (trivial) corep of $P4_2/mbc\times\mathcal T$. No channel breaks any space-group or time-reversal symmetry. The paired spinon mean field is fully space-group symmetric.**

Gauge bookkeeping: $\Delta$ carries gauge charge 2, so physical symmetry only requires $U_g\,\Delta(k)\,U_g^T = e^{i\phi_g}\,\Delta(\mathbf Wk)$ with $\{e^{i\phi_g}\}\subset$ IGG (PSG invariance). Here $\phi_g \equiv 0$ for all generators: the PSG is the **trivial (zero-flux) extension**, IGG $=\mathbb Z_2 = \{\pm1\} = \{e^{i\pi Q}\}$. Gauge-invariant content: (i) the PSG equivalence class (trivial); (ii) reality of all $\Delta_i$ relative to the $t_i$ (a global phase of $\Delta$ is gauge, but *relative* phases among channels are physical — TR pins them real; a converged solution with relatively complex $\Delta_i$ would signal spontaneous TR breaking); (iii) gauge fluxes through Wilson loops of the combined $(t,\Delta)$ ansatz (all trivial here).

Together with §2.4: this ansatz realizes **Class I** — the boson sector inherits the trivial rep $\rho_b = A_g$ (equivalently $A_u$ in another gauge). A Class II state would need the spinon ansatz to be invariant only *up to* a nontrivial $\mathbb Z_2$ gauge transformation on the screw (e.g. a $\mu$-staggered sign pattern $\Lambda_i = (-1)^{2z_i}$), i.e. a genuinely different (π-flux-like) mean field.

---

## 5. Synthesis: the LSM-mandated $\mathbb Z_2$ fractionalized insulator

### 5.1 Placement relative to Watanabe–Po–Vishwanath–Zaletel

For SG 135 + TRS + SOC:

- **Band theory:** insulators only at $\nu \in 8\mathbb Z$ (single 8-dim corep at $A$; §3.1; Wieder et al.).
- **WPVZ interacting bound:** symmetric short-range-entangled insulators need $\nu$ even and $\ge 4$ in nonsymmorphic groups; SG 135 is one of the ten groups they single out where "a sym-SRE insulator is possible at $\nu = 4$, whereas a band insulator at the same filling is impossible," leaving open whether a stronger constraint exists or an interaction-enabled example exists.
- **This construction:** if the slave-boson mean field converges to a fully gapped, symmetric solution at $\nu=4$ (chargons Mott-gapped, spinons BdG-gapped, trivial PSG), the resulting state is a symmetric gapped insulator that is **long-range entangled**: a deconfined $\mathbb Z_2$ gauge theory in 3+1d. It does *not* contradict the $8\mathbb Z$ rule (it is no band insulator) and does not realize the open "SRE at $\nu=4$" option — it occupies the topologically ordered branch. Ground-state degeneracy $2^3 = 8$ on $T^3$; pointlike charges $e$ (chargon: gapped holon/doublon, electric charge $\pm1$, spin-0, $\mathcal T^2=+1$) and $\varepsilon = e\times m$-side fermion $f$ (BdG spinon: neutral, Kramers, spin-1/2), plus **vison loop** excitations (the $\mathbb Z_2$ flux lines of the broken-down $U(1)$ gauge field; energetically set by the pairing scale). The electron is the gauge-neutral composite $c = b^\dagger f$ — a gapped charge gap (chargon) and spin gap (BdG) with no symmetry breaking: exactly the "featureless-looking but topologically ordered" resolution of the filling-4 LSM constraint.

The HK solvable model in the same space group (HK paper) reaches a symmetric gapped/spin-liquid regime through infinite-range interactions; the slave-boson state is the candidate for the same LSM physics with *local* Hubbard interactions.

### 5.2 Quantum numbers carried by the boson sector; trivial vs nontrivial factorizations

| | Class I (realized by our ansatz) | Class II (hypothetical, flux-twisted) |
|---|---|---|
| chargon site rep | $A_g$ ($\simeq A_u$ by gauge) | $B_g$ ($\simeq B_u$) |
| invariant label | $\chi_b((4_2)^2) = +1$ | $\chi_b((4_2)^2) = -1$ |
| chargon levels at $\Gamma$ | four 1D irreps incl. $A_{1g}$ | only 2D $E_g$ doublets |
| chargon quadruplet at $A$ | $C_{2z} = +\mathbb 1$ | $C_{2z} = -\mathbb 1$ |
| condensation at $\Gamma$ (gap closing → confinement) | symmetric superconductor (condensing the charged $b$ breaks only $U(1)$) → reconnects to FL/band-like physics | forced point-group breaking: two-component $E_g$ condensate ⇒ nematic superconductor; symmetric confinement impossible at $\Gamma$ |
| spinon partner | $\bar E_g{\uparrow}G$ (parities all $+$ at $\Gamma, M$) | $\bar E_g{\uparrow}G$ with $C_{2z}$ components swapped |

**Physical discriminators.**
1. **Chargon spectroscopy:** the charge-gap minimum and its multiplet structure (1D vs forced-2D at $\Gamma$; the sign of $C_{2z}$ on the $A$-quadruplet) appear in the single-electron spectral function as the threshold of the $b\otimes f$ two-particle continuum and in the optical charge gap.
2. **Nature of the insulator–superconductor (Higgs/confinement) transition:** Class I admits a single-component, symmetry-preserving chargon condensate; Class II cannot condense at $\Gamma$ without nematicity — the proximate phase diagram differs qualitatively.
3. **Vison symmetry fractionalization:** a vison loop linking a 4a site braids with the unit gauge charge there ($\pi$ phase). In Class II the additional $C_{2z}$-odd character of that background charge twists the projective action of the screw on vison configurations (the vison sector "sees" $\chi_b(C_{2z})$ through the $e$–$m$ braiding constraint $\omega_e\,\omega_m\,\omega_\varepsilon = \omega_c$); measurable in principle via symmetry quantum numbers of flux-loop excitations at high-symmetry axes, or in the entanglement spectrum on $C_{2z}$-symmetric cuts.
4. **Surface:** both classes can terminate in gapped symmetric surfaces with surface topological order (3+1d $\mathbb Z_2$); the spinon BdG sector additionally has inversion-indicator data (all-$+$ vs all-$-$ parities at $\Gamma$, $M$ are gauge-relabelable, but the *relative* BdG occupied-parity counts after convergence determine possible higher-order Majorana features of the spinon superconductor — to be evaluated from converged numerics, not fixed by symmetry alone).

### 5.3 Beyond linear factorizations

Genuinely distinct $\mathbb Z_2$-fractionalized insulators at $\nu = 4$ are classified not by $(\rho_b,\rho_f)$ labels but by symmetry-fractionalization classes (projective $\omega_b,\omega_f,\omega_{\rm vison}$ subject to $\omega_b\omega_f = \omega_{\rm spin}$ and anomaly matching). Examples not realized by the present decoupling: Kramers chargon ($\mathcal T_b^2 = -1$), chargon with $\rho_b(I)^2 = -1$, visons carrying Kramers or fractional screw quantum numbers. Each would require a different PSG (nontrivial flux pattern through symmetric loops). The present mean field — trivial PSG, Class I — is the "minimal" symmetric $\mathbb Z_2$ fractionalized insulator compatible with the SG 135 LSM constraint.

### 5.4 Checklist for the in-progress numerics

1. **Gap at $A$:** confirm $\mu_f \neq 0$ or on-site $\Delta_0 \neq 0$ in the converged solution (§3.3); otherwise the BdG sector is gapless at $A$ by construction.
2. **Reality:** check the converged $\Delta_i/t_i$ are relatively real (TR; §4.3); a relative phase ⇒ chiral state.
3. **Boson sector:** verify the chargon Green's function is gapped with the $\Gamma$-multiplet structure of $A_g{\uparrow}G$ (four 1D levels), confirming Class I.
4. **Constraint:** $\langle Q_i\rangle = 1$ with $\langle n_h\rangle = \langle n_d\rangle$ at half filling (particle-hole symmetric boson sector ⟺ $\mu_f$ shift consistency; recall the model is PH-symmetric only at $t_2' = 0$).
5. **Degeneracy diagnostics:** $8$-fold topological GSD on $T^3$ (or entanglement signatures) distinguishes the $\mathbb Z_2$ insulator from any SRE scenario.

---

## Appendix A: computational method and checks

- **Space-group data:** 16 coset representatives $(W,w)$ of $P4_2/mbc$ taken from the Bilbao WYCKPOS/GENPOS output (archived copy, identical to ITA); closure and $(C_{4z}|00\tfrac12)^2 = \{C_{2z}|001\}$ verified.
- **Induction:** for $g = \{W|w\}$ and site $\alpha$: $W q_\alpha + w = q_\beta + \mathbf R_{g\alpha}$; stabilizer $h = g_\beta^{-1}\{E|-\mathbf R\}g\,g_\alpha \in G_w$ (residual translation verified to vanish exactly); Bloch matrices $[U^{(\rho)}(g,k)]_{\beta\alpha} = e^{-i(Wk)\cdot\mathbf R_{g\alpha}}\chi_\rho(h)$ in the $e^{ik\cdot\mathbf R}$ convention; spinor case $U_f = U^{(\rho_{\rm orb})}\otimes S(W)$ with $S$ the SU(2) lift of the proper part of $W$. Projective representation property verified numerically at all TRIMs.
- **Degeneracies/characters:** random Hermitian matrices symmetrized over the little group and the appropriate antiunitary ($\mathbb 1\otimes i\sigma^y K$ or $K$); eigenvalue clustering gives the forced multiplet structure; per-multiplet traces of $U(I)$, $U(C_{2z})$, $U(C_{4z})$, $U(C_{2x})$, $U(m_z)$ give the characters quoted in §3 (stable across random seeds).
- **Pairing checks:** all entries of §4.2 verified to $\le 10^{-16}$ at random generic $k$; $H(A) = \Delta(A) = 0$; BdG spectra at $A$ equal $\pm\mu_f$; $16^3$ BZ scans for minimal BdG gap with/without $\mu_f$, $\Delta_0$.
- **Stabilizer table for reference** (screw, sites $1\to3$, $2\to4$, $3\to1$, $4\to2$ with $h = E, E, C_{2z}, C_{2z}$; $C_{2x}$: $1\leftrightarrow2$ ($h=E$), $3\leftrightarrow4$ ($h=C_{2z}$); $b$-glide $m_y$: $1\leftrightarrow2$ ($h = m_z$), $3\leftrightarrow4$ ($h = I$); $c$-glide $m_{110}$: $1\leftrightarrow4$, $2\leftrightarrow3$ (all $h = I$); inversion: diagonal, $h = I$). E.g. for the $B_g$ boson this gives $U(\text{screw}) = -i\mu^y\otimes\tau^0$, $U(C_{2x}) = \mu^z\otimes\tau^x$, $U(m_y) = -\mu^z\otimes\tau^x$, $U(I) = \mathbb 1$ at $\Gamma$ — squaring the screw indeed yields $\chi_{B_g}(C_{2z}) = -1$.

## Appendix B: references

- Bilbao Crystallographic Server, WYCKPOS for SG 135 (Aroyo *et al.*, Z. Krist. **221**, 15 (2006)); data cross-checked against ITA Vol. A.
- B. J. Wieder, Y. Kim, A. M. Rappe, C. L. Kane, "Double Dirac semimetals in three dimensions," PRL **116**, 186402 (2016); arXiv:1512.00074. (Model; single 8-dim corep at $A$; "the WPVZ bound of 4 disagrees with band theory" for SG 135.)
- H. Watanabe, H. C. Po, A. Vishwanath, M. P. Zaletel, "Filling constraints for spin-orbit coupled insulators in symmorphic and nonsymmorphic crystals," PNAS **112**, 14551 (2015); arXiv:1505.04193. (Interacting bound; the ten-group disclaimer incl. SG 135.)
- HK paper: `notes/theory/sg135_hk (2)/main.tex` (model conventions, Table of generator reps, misprint correction footnote).
- B. Bradlyn *et al.*, "Topological quantum chemistry," Nature **547**, 298 (2017); L. Elcoro *et al.*, J. Appl. Cryst. **50**, 1457 (2017) (double-group corep conventions).
- X.-G. Wen, PRB **65**, 165113 (2002) (PSG); A. M. Essin, M. Hermele, PRB **87**, 104406 (2013) (symmetry fractionalization).
