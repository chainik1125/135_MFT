# SG 130 control: can the SG 135 slave-boson construction be transplanted?

**Space group:** $P4/ncc\,1'$ (No. 130, $D_{4h}^{8}$), spinful TRS.
**Model:** Wieder–Kim–Rappe–Kane SG 130 double-Dirac tight-binding model [PRL **116**, 186402 (2016)], in the form used (and misprint-corrected) in the HK paper `sg135_hk (2)/main.tex`, Eq. (130noninteractingH):
$$\mathcal H^0_{130} = t_{xy}\tau^x c_xc_y + t_z\mu^x c_z + \lambda_1\mu^y\tau^z\sigma^z c_z + \lambda_2\tau^z(\sigma^x\sin k_y - \sigma^y\sin k_x) + \lambda_3\mu^z\tau^x(\sigma^x s_xc_y + \sigma^y c_xs_y),$$
with $c_\alpha=\cos\frac{k_\alpha}{2}$, $s_\alpha=\sin\frac{k_\alpha}{2}$; generators $\{C_{4z}|000\}\to e^{i\pi\sigma^z/4}$, $\{C_{2x}|\tfrac12\tfrac12 0\}\to i\tau^x\sigma^x$, $\{I|\tfrac12\tfrac12\tfrac12\}\to\mu^x\tau^x$, $\mathcal T = i\sigma^y K$. Parameters as in the HK paper: $t_{xy}=1$, $t_z=0.5$, $\lambda_{1,2,3}=0.3$.

**Status of sources.** The WPVZ bounds below were read directly from the **local copies** of Watanabe–Po–Vishwanath–Zaletel, PNAS **112**, 14551 (2015) and its SI (`notes/theory/Watanabe Bound/`), so Task 1 is *verified at the source*, not from memory. Everything group-theoretic (Wyckoff orbits, site-symmetry groups, coreps, pairing channels, BdG spectra) was computed ab initio in Python (scripts `/tmp/sg130_analysis.py`, `/tmp/sg130_part2.py`–`part5.py`), mirroring the SG 135 methodology (`sg135_factorization.md`, Appendix A). The Bilbao server was not used. Model invariance under all 16 coset representatives + TR was verified to $\lesssim10^{-10}$; all channel/corep statements hold to $\lesssim10^{-15}$.

---

## 0. Executive answers

1. **WPVZ bounds (verified from the paper).** Table S1: **SG 130: $\nu \in 8\mathbb Z$**; **SG 135: $\nu\in4\mathbb Z$** (dagger: "$\nu = 8n-4$ is prohibited for the noninteracting case; there is no known interacting model of sym-SRE at these fillings either"). Main text: the bounds are *provably tight* for all but ten NS groups — **73, 106, 110, 133, 135, 142, 206, 220, 228, 230**. SG 130 is **not** among them: its interacting (Bieberbach) bound coincides with band theory, $\nu_{\rm SRE}=\nu_{\rm band}=8$ (130 inherits the $8\mathbb Z$ bound from elementary NS group No. 33, $Pna2_1$, the "2nd amphidicosm" box of Fig. S1). So at $\nu=4$ a **symmetric short-range-entangled insulator is rigorously impossible in SG 130 even with interactions**, while in SG 135 it is merely unrealizable in band theory. **Crucial scope caveat:** the WPVZ theorem constrains only sym-**SRE** states (unique gapped ground state); topologically ordered symmetric insulators are *not* constrained — WPVZ list TO/symmetry-fractionalization constraints as an open question in their conclusion. "Even interacting symmetric insulators are forbidden in SG 130 at $\nu=4$" is therefore correct **only with "SRE" inserted**.

2. **Wyckoff position (computed).** The model's four sites $(0,0,0)$, $(\tfrac12,\tfrac12,0)$, $(0,0,\tfrac12)$, $(\tfrac12,\tfrac12,\tfrac12)$ are a single multiplicity-4 orbit with site-symmetry group $G_w = \{E, C_{4z}, C_{2z}, C_{4z}^{-1}\} \cong C_4$ ("$4..$") — **Wyckoff 4c** of $P4/ncc$ (the Cu site of Bi$_2$CuO$_4$). Sharp contrasts with SG 135's 4a ($2/m$, $C_{2h}$): (i) the SG 130 sites sit **on the 4-fold axis** and are **not inversion centers** (inversion centers in SG 130 form a multiplicity-**8** orbit with site symmetry $\bar 1$ only; no multiplicity-4 position contains any improper element); (ii) $U(C_{4z})$ acts **trivially on the sublattice index** ($\rho(C_4)=e^{i\pi\sigma^z/4}\otimes\mathbb 1_{\rm orb}$); (iii) the 4c position has a free $z$ parameter.

3. **Key question.** With pairing restricted to the model's hopping-form-factor (spin-independent $t$-bond) channels, the **answer is NO**: SG 130 has exactly **two** strictly symmetric singlet channels, $\Delta_{xy}\,\tau^x c_xc_y$ and $\Delta_z\,\mu^x c_z$, whose matrices **commute** (no SG 135-style anticommuting pair exists) and which **both vanish identically on the BZ-edge lines** $\{k_x{=}\pi,k_z{=}\pi\}\cup\{k_y{=}\pi,k_z{=}\pi\}$ ($A$–$R$ and its $C_4$ partner). On those lines the normal bands form symmetry-forced 4-fold quartets sweeping continuously through $[0, 0.375]$ down to $0$ at $A$, while the filling constraint $\nu_f=4$ pins $\mu$ to $\approx 0.003$ — so the BdG spectrum has **exact, filling-enforced nodal points on the zone-edge lines for every parameter choice**. $\lambda$ (= $\mu_f$) cannot lift them: it gaps $A$ itself (BdG $=\pm\mu$ there) but merely slides the node along the line; $|\mu|>0.375$ would remove it but forces $\nu_f \in [3.07, 4.92]\ne 4$ endpoints, i.e. is filling-incompatible. **This is the sharp obstruction the project predicted, and it is where the SG 135 construction breaks.**

4. **But the obstruction is range-limited, not absolute.** The strictly symmetric **sublattice-diagonal** singlets — on-site $\Delta_0\mathbb 1$ (suppressed by repulsive $U$), the in-plane NNN extended-$s$ $(\cos k_x+\cos k_y)\mathbb 1$ on the $\lambda_2$ SOC bond, and $\cos k_z\,\mathbb 1$ — are alive at $A$ and on the nodal lines. Numerically, $\Delta_{s}( \cos k_x + \cos k_y)\mathbb 1\otimes i\sigma^y$ alone yields a **fully gapped, fully symmetric BdG spinon sector at exactly $\nu_f=4$** (min gap $0.059$ at $\Delta_s=1$, $\mu^*=0$; symmetric to $10^{-15}$). So symmetry does **not** forbid a gapped symmetric $\mathbb Z_2$ ansatz in SG 130 — consistent with WPVZ, whose $8\mathbb Z$ bound binds only SRE states and is evaded by topological order in SG 130 exactly as the $4\mathbb Z$ window is occupied by it in SG 135 (cf. the HK paper's SG 130 model: gapped, nondegenerate, symmetric at $\nu=4$, hence necessarily long-range entangled).

5. **Verdict (confidence: high for items verified, stated inline).** The control behaves as predicted **at the mirrored level of the construction**: the literal SG 135 recipe (pairing in the hopping channels + chemical potential, trivial PSG) produces a *symmetric but nodal* — not gapped — spinon sector in SG 130, with the failure localized exactly at the glide-protected zone-edge quartets. It does **not** behave as predicted at the level of "no symmetric gapped slave-boson state exists at all": longer-range (still on-model-bond) channels gap it, and no LSM-type theorem can forbid this because the resulting state is topologically ordered, outside WPVZ's jurisdiction in *both* space groups. The meaningful 130-vs-135 asymmetry is (i) rigorous at the SRE level ($8\mathbb Z$ vs $4\mathbb Z$), and (ii) *quantitative* at the parton mean-field level: in SG 135 the leading (largest-$J$) channels gap the spinons; in SG 130 they provably cannot, and the gap must come from parametrically weaker channels ($J' \sim \lambda_2^2/U$ on the NNN bond) — predicting a fragile, easily nodal spin liquid in SG 130 rather than a forbidden one.

---

## 1. WPVZ bounds for SG 130 (Task 1) — read from the local PDFs

From the PNAS main text (p. 14555): *"Except for 10 NS space groups, we can prove the Bieberbach bound is necessary and sufficient by constructing a noninteracting band structure at the conjectured filling. For 10 NS groups (nos. 73, 106, 110, 133, 135, 142, 206, 220, 228 and 230), however, our method here indicates a sym-SRE insulator is possible at $\nu=4$, whereas one can in fact show a band insulator at this filling is impossible."*

| | band-theory minimum | WPVZ interacting (sym-SRE) bound | status at $\nu=4$ |
|---|---|---|---|
| SG 130 | $8\mathbb Z$ (single 8-dim corep at $A$; §3) | $\nu\in 8\mathbb Z$ (Table S1; tight) | **sym-SRE rigorously forbidden**; TO unconstrained |
| SG 135 | $8\mathbb Z$ | $\nu\in 4\mathbb Z$ (Table S1, dagger; one of the ten) | sym-SRE open; TO unconstrained |

Mechanism of the stronger SG 130 bound: per Fig. S1 of the SI, SG 130 contains the elementary NS group No. 33 ($Pna2_1$) as a $t$-subgroup; putting the system on the associated flat manifold (2nd amphidicosm) forces $\nu\in8\mathbb Z$ for any sym-SRE phase. SG 135's maximal elementary subgroups (No. 77 $P4_2$, No. 9 $Cc$ boxes) only give $4\mathbb Z$.

**Confidence: verified** (read directly from the local copies of the paper and SI; Table S1 row "130 → $8n$", "135 → $4n^\dagger$").

Scope: the theorem rules out a *unique* symmetric gapped ground state below the bound. A $\mathbb Z_2$ topologically ordered insulator (GSD $2^3$ on $T^3$) is exempt in both groups; WPVZ end the paper asking *"what are the constraints on topological order and symmetry fractionalization for NS lattices"* — explicitly open.

---

## 2. Wyckoff position and what the site symmetry destroys (Task 2)

### 2.1 Computed Wyckoff data (group generated ab initio from the HK generator table; 16 cosets, closure verified)

| orbit representative (model setting) | mult. | site symmetry | ITA letter |
|---|---|---|---|
| $(0,0,z)$, incl. model sites at $z=0,\tfrac12$ | 4 | $4..$ ($C_4$) | **4c** |
| $(0,\tfrac12,\tfrac14)$-type | 4 | $\bar 4..$ ($S_4$) | 4b |
| $(0,\tfrac12,0)$-type | 4 | $222$-type ($D_2$) | 4a |
| $(\tfrac14,\tfrac14,\tfrac14)$-type (inversion centers) | **8** | $\bar 1$ | 8d |
| $(\tfrac12,0,z)$ / $(\tfrac14,\tfrac14,0)$-type | 8 | $2..$ / $..2$ | 8e / 8f |

The model's four sublattices are the **4c orbit** (with the free parameter at $z=0$); the site-symmetry group is the *proper cyclic* group $C_4$ generated by the bare rotation $\{C_{4z}|000\}$ itself. Compare SG 135's 4a: $2/m = \{E, C_{2z}, I, m_z\}$, every site an inversion center, and $C_{2z}=(4_2\text{ screw})^2$.

Two structural consequences, both verified numerically:

- **(a) No sublattice action of $C_4$.** Since the sites sit on the 4-fold axis, the induced Bloch matrix is $U(C_{4z},k) = \mathbb 1_{\rm orb}\otimes e^{\mp i\pi\sigma^z/4}$ (trivial permutation). A spin-singlet pairing channel $F(k)\,M\otimes i\sigma^y$ therefore transforms with $F(C_4k)$ alone — **no orbital matrix can compensate a $C_4$-odd form factor**. All $d$-wave-like channels ($\cos k_x-\cos k_y$, $\sin\frac{k_x}{2}\sin\frac{k_y}{2}$, …) are strictly forbidden. In SG 135 the screw rep $\mu^x e^{i\pi\sigma^z/4}$ *does* act on sublattices, which is precisely what allowed the compensated channels $g_1\mu^z$ and $g_2\mu^y\tau^y$ — the channels that stay alive on the zone-edge lines there.

- **(b) Site-level factorization is unique up to gauge.** $\rho_e$ at 4c is the 2-dim spinor corep of $C_4\times\mathcal T$ ($C_4$ eigenvalues $e^{\pm i\pi/4}$, $\mathcal T^2=-1$). Single-valued options for $\rho_b$ ($\mathcal T_b^2=+1$): $A$ ($\chi(C_4)=+1$) and $B$ ($\chi(C_4)=-1$) (the complex pair $E$ would force a 1-dim Kramers spinon — impossible). Both give $\chi_b(C_{2z}) = \chi_b(C_4)^2=+1$, and $\lambda(C_4)=-1$ is a legal uniform $\mathbb Z_2$ gauge twist on charge-1 fields (no element-square constraint pins it, unlike $C_{2z}=({\rm screw})^2$ in SG 135). Hence $A\simeq B$: **SG 130 has a single gauge class of linear factorizations — there is no Class I/II distinction**. The SG 135 invariant $\chi_b((4_2)^2)=\pm1$ has no SG 130 analogue.

---

## 3. Induced band representation (8 bands, $s\otimes$spin-$\tfrac12$ at 4c), with $\mathcal T$

Computed by the same symmetrized-random-Hamiltonian method as the SG 135 note (fold-back phases $V(G)=\mathrm{diag}(e^{iG\cdot q_\alpha})$ verified against the model; invariance errors $\le10^{-15}$):

| momentum | degeneracies | | momentum | degeneracies |
|---|---|---|---|---|
| $\Gamma$ | $2+2+2+2$ | | line $A$–$R$ $(\pi,t,\pi)$ | $4+4$ (whole line) |
| $Z$ | $4+4$ | | line $A$–$M$ $(\pi,\pi,t)$ | $4+4$ (whole line) |
| $X$ | $4+4$ | | line $X$–$R$ $(\pi,0,t)$ | $4+4$ (whole line) |
| $M$ | $4+4$ | | lines $Z$–$R$, $X$–$M$, $\Gamma$–$Z$, $\Gamma$–$M$, $Z$–$A$ | $2$-fold |
| $R$ | $4+4$ | | planes $k_z{=}\pi$, $k_x{=}\pi$ generic | $2$-fold |
| $A$ | $\mathbf 8$ (single corep) | | generic $k$ | $2$-fold ($I\times\mathcal T$) |

- The **single 8-dim corep at $A$** reproduces Wieder et al. and the $8\mathbb Z$ band bound (the EBR is connected). Same as SG 135.
- **Difference from SG 135:** entire **4-fold-stuck lines** $A$–$R$, $A$–$M$, $X$–$R$ (SG 135 has the 8-fold $A$ point with 4+4 at $X,R,Z,M$ but the model's relevant lines carry surviving pairing channels there). In the SG 130 model the $A$–$R$ quartets are $\pm\sqrt{\lambda_2^2\sin^2k_y+\lambda_3^2\cos^2(k_y/2)}$, range $[0,0.375]$; on $A$–$M$, $\pm\cos\frac{k_z}{2}\sqrt{t_z^2+\lambda_1^2}$.
- Chargon (single-valued) EBR from 4c: $\Gamma$: four 1-dim; $A$ and $R$: single **4-dim** coreps; $Z,X,M$: $2+2$. Connected, but as in SG 135 this is no obstruction for bosons: the atomic Mott state exists at integer filling per site. The normal-state model is particle–hole symmetric in spectrum (verified), so $\nu_f=4 \Rightarrow \mu^*\simeq 0$.

---

## 4. Pairing analysis (Task 3)

### 4.1 Complete channel enumeration (singlet, fermion-antisymmetric, TRS-even), by bond shell

Projected ab initio onto the identity corep and onto all $\mathbb Z_2$ sign classes $\lambda=(\lambda_{C_4},\lambda_{C_{2x}},\lambda_I)$; every entry verified to $\lesssim 4\times10^{-15}$:

| shell (bond) | strict identity corep $(+,+,+)$ | sign-twisted classes (need $\pi$-flux saddle, see 4.4) |
|---|---|---|
| $S_0$ on-site | $\Delta_0\,\mathbb 1$ | … |
| $S_1$ $(\tfrac12,\tfrac12,0)$ — the $t_{xy}$ bond | $\tau^x c_xc_y$ | $\mu^z\tau^x c_xc_y$ $(+,+,-)$; $\tau^x s_xs_y$ $(-,-,+)$; $\mu^z\tau^x s_xs_y$ $(-,-,-)$ |
| $S_2$ $(0,0,\tfrac12)$ — the $t_z$ bond | $\mu^x c_z$ | $\mu^y\tau^z s_z$ $(+,+,-)$; $\mu^y s_z$ $(+,-,+)$; $\mu^x\tau^z c_z$ $(+,-,-)$ |
| $S_3$ $(1,0,0)$ — the $\lambda_2$ bond | $(\cos k_x{+}\cos k_y)\,\mathbb 1$ | $(\cos k_x{-}\cos k_y)\,\mathbb 1$ $(-,+,+)$; $\mu^z\tau^z(\cos k_x{+}\cos k_y)$ $(+,-,+)$; … |
| $S_4$ $(0,0,1)$ | $\cos k_z\,\mathbb 1$ | $\mu^z\tau^z\cos k_z$ $(+,-,+)$ |
| $S_5$ $(\tfrac12,\tfrac12,\tfrac12)$ | two channels ($\mu\tau$-off-diagonal, $ccc$/$ssc$-type mixtures) | … |

**Strict mirror of the SG 135 ansatz** (= singlets with the model's spin-independent hopping form factors): exactly the two channels
$$\Delta(k) = \big[\Delta_{xy}\,\tau^x\, c_xc_y + \Delta_z\,\mu^x\, c_z\big]\otimes i\sigma^y,\qquad \Delta_{xy},\Delta_z\in\mathbb R .$$
$[\tau^x,\mu^x]=0$: **the SG 135 anticommuting-pair mechanism is structurally absent.**

### 4.2 The obstruction (strict mirror): filling-enforced BdG nodes

All verified numerically; the logic is exact, not numerical:

1. On the zone-edge cross $L = \{(\pi,k_y,\pi)\}\cup\{(k_x,\pi,\pi)\}$: $c_xc_y = 0$ and $c_z=0$, so $\Delta(k)\equiv 0$ **identically on $L$** ($\max_L\|\Delta\| = 10^{-15}$). This extends to shell $S_5$ and (checked channel-by-channel) to every strictly symmetric singlet built on *sublattice-off-diagonal* (half-integer-offset) bonds: all of them also vanish on $A$–$R$ and $A$–$M$.
2. On $L$ the BdG spectrum is therefore exactly $\pm(E_{\rm quartet}(k)-\mu)$, $\pm(E_{\rm quartet}(k)+\mu)$ with $E_{\rm quartet}$ sweeping $[0, 0.375]$ continuously down to $0$ at $A$.
3. Filling: $\langle n_f\rangle(\mu)$ is monotone with $\langle n\rangle(0)=4.0$ (PH-symmetric spectrum), $\langle n\rangle(\pm0.375)=\{3.07, 4.92\}$ at $(\Delta_{xy},\Delta_z)=(2.0,1.5)$ — so $\nu_f=4$ forces $|\mu^*|\ll 0.375$ for every pairing strength tested ($\mu^*\in[0.0015,0.003]$ for $\Delta$ up to $(3.0,2.0)$).
4. Hence $E_{\rm quartet}(k)-\mu^*$ changes sign on $L$ (explicitly: at $k_y^*\approx 3.13$ for $\mu^*=0.003$) $\Rightarrow$ **exact, 4-fold-degenerate BdG zeros pinned on the zone-edge lines for every $(\Delta_{xy},\Delta_z,\mu)$ consistent with $\nu_f=4$.**

Sharper than "form factors vanish at $A$": at $A$ itself the strict-mirror BdG *is* gapped ($\pm\mu$ octets, verified), exactly as in SG 135. What $\lambda\equiv\mu_f$ cannot do here is gap the *line*, because the protected quartets disperse through every energy between $0$ and $0.375$ — the node just slides to $E_{\rm quartet}(k^*)=\mu$. This is the precise analogue of a symmetry-enforced nodal structure: a **filling-enforced BdG node** protected by the SG 130 glide/screw quartets plus the absence (by §2a) of any compensated channel that survives on $L$.

### 4.3 The rescue (still strictly symmetric, longer range)

Sublattice-diagonal singlets are immune to the mechanism of 4.2:

- $\Delta_s(\cos k_x + \cos k_y)\,\mathbb 1\otimes i\sigma^y$ ($S_3$; the bond carrying the $\lambda_2$ SOC hopping, so generated at $O(\lambda_2^2/U)$ in any honest exchange decoupling): equals $-2\Delta_s\ne0$ **at $A$** and $(\cos k_y - 1)\ne 0$ on $A$–$R$ away from $R$; orbital-identity structure ⇒ full intraband (Kramers-pair) matrix elements.
- Numerics: $\Delta_s = 1.0$ alone, $\mu^*=0$, $\langle n\rangle = 4.0000$: **min BdG gap $= 0.0586$** (24³ grid + 30 Nelder–Mead refinements + dense HS-line scans); mixed $(\Delta_{xy},\Delta_z,\Delta_s) = (0.2,0.15,0.6)$: $\mu^*=-0.071$, min gap $0.0036 > 0$ (limited by a near-zero of the *normal* bands at $(2.67,0,\pi)$, where the pairing is alive). Full symmetry of the gapped ansatz verified to $2\times10^{-15}$.
- Caveat (real and instructive): $t$-bond-dominant mixtures, e.g. $(1.0,0.8,0.5)$ or $(0.4,0.3,1.2)$, develop **accidental BdG zeros at generic $k$** (sign changes of the projected gap) — the gapped region is an open but not large set around extended-$s$-dominant ansätze. Self-consistency must land there for the state to exist.
- On-site $\Delta_0\,\mathbb 1\otimes i\sigma^y$ is symmetric in *any* space group and gaps everything ($E\ge|\Delta_0|$); it is excluded only physically (costs $+U$). This is the general reason no LSM-type theorem can ever forbid a symmetric gapped *BdG* sector: the constraint must come from which channels the interaction actually generates.

### 4.4 Sign-twisted classes are unavailable here ($\mathbb Z_2$-gauge lemma)

For pairing (gauge charge 2), a *uniform* $\mathbb Z_2$ gauge transformation acts trivially ($(\pm1)^2=1$), so a constant sign $\lambda(g)=-1$ can only be realized by a **site-dependent** $G_g(r)=\pm1$ with $G_g(i)G_g(j)=+1$ on every bond of the spinon *hopping* ansatz. The model's $t_{xy}$+$t_z$ hopping graph is connected ⇒ $G_g$ uniform ⇒ $\lambda\equiv+1$. Hence the tantalizing twisted channels (e.g. $\mu^y s_z$, which is nonzero **at $A$** and on all of $L$) require a different, $\pi$-flux-like spinon hopping saddle — a genuinely different mean field with non-Wieder spinon bands, outside this control comparison (the analogue of "Class II would require a flux-twisted ansatz" in SG 135, except that here even the zero-flux pairing classes are emptier).

---

## 5. Verdict (Task 4)

**Does the SG 130 control behave as the project predicts?** Split verdict, each part with high confidence on the computations and explicit confidence flags on interpretation:

1. **WPVZ reading — verified, with one correction of scope.** The interacting WPVZ bound for SG 130 is $\nu\in8\mathbb Z$ (tight; verified from Table S1 and the ten-group list), vs $4\mathbb Z$ for SG 135. At $\nu=4$, SG 130 forbids any symmetric **SRE** insulator even with interactions. It does **not** forbid a symmetric topologically ordered insulator — no LSM-type argument does, in either group. The project's control statement should be phrased at the SRE level. *(Confidence: verified for the bounds; the SRE-only scope is explicit in WPVZ.)*

2. **The mirrored construction does break in SG 130, at a sharp, identifiable point.** With the model's hopping-form-factor singlet channels (the honest analogue of the SG 135 ansatz), the spinon BdG sector is symmetric but **forced nodal**: $\Delta\equiv0$ on the glide-protected zone-edge lines, where filling pins $\mu$ inside the quartet bandwidth ⇒ exact nodes for all parameters. Breaking points, in order: (i) 4c site symmetry $C_4$ acts trivially on sublattices ⇒ no compensated ($d$-wave-like) channels exist; (ii) only two strict channels survive on the $t$-bonds, with commuting matrices; (iii) the 4-fold-stuck $A$–$R$ quartets sweep through $E=0$, so $\lambda$/$\mu_f$ cannot gap the line, only relocate the node. *(Confidence: high — machine-precision group theory + exact-node argument.)*

3. **But SG 130 does admit a symmetric, fully gapped $\mathbb Z_2$ spinon ansatz at $\nu_f=4$** once the strictly symmetric extended-$s$ singlet on the $\lambda_2$ bond (or the $(0,0,1)$ bond, or on-site) is included — channels that any complete decoupling generates, just at parametrically weaker coupling ($\lambda_2^2/U$). If the self-consistency favors it, the resulting state is the same kind of $\mathbb Z_2$ fractionalized insulator as in SG 135 and **does not contradict WPVZ** (it is LRE; GSD 8 on $T^3$), exactly parallel to the HK paper's conclusion that its gapped symmetric SG 130 ground state must be long-range entangled. *(Confidence: high for existence of the gapped symmetric mean field; whether self-consistency selects it is an open numerical question — the gap region excludes $t$-bond-dominant mixtures, which renode.)*

4. **Net assessment for the project.** Use SG 130 as a *graded* control: (a) the strict transplant fails — a clean, publishable, symmetry-protected mechanism (filling-enforced BdG nodal lines from the 4c/$C_4$ + glide structure); (b) the WPVZ asymmetry ($8\mathbb Z$ vs $4\mathbb Z$ for SRE) is real and verified; (c) the strong claim "no symmetric gapped slave-boson state in SG 130 at $\nu=4$" is **false as group theory** and should not be made — the correct contrast is *which interactions/channels must do the gapping and how robust the gap is* (leading-$J$ channels in 135; subleading $\lambda_2^2/U$ channels with a fragile, accident-prone gap in 130), plus the rigorous SRE-level distinction.

### Checklist for SG 130 numerics (if the control is run)
1. With $t$-bond decoupling only: converged solutions must show BdG nodes on $\{k_x{=}\pi\ {\rm or}\ k_y{=}\pi\}\cap\{k_z{=}\pi\}$; any "full gap" claim there indicates a symmetry bug.
2. A full gap requires a nonzero same-sublattice singlet ($\Delta_s$, $\Delta_{z2}$, or $\Delta_0$); check the converged $\mu^*$ stays small ($\nu_f=4$) and the gap survives Nelder–Mead refinement around grid minima (accidental generic-$k$ zeros occur for $t$-bond-dominant amplitude ratios).
3. Reality of all amplitudes (TRS); single gauge class — no Class I/II diagnostics exist in SG 130 (§2b), so chargon spectroscopy distinguishes nothing beyond the trivial class.

---

## Appendix: computational record

- Group: generated from $\{C_{4z}|000\}$, $\{C_{2x}|\tfrac12\tfrac12 0\}$, $\{I|\tfrac12\tfrac12\tfrac12\}$; 16 cosets, closure exact; contains $\{m_x|00\tfrac12\}$-type $c$-glides, the $n$-glide $\perp c$, $2_1$ screws along $\langle100\rangle$, $\bar4$ at $(0,\tfrac12,\tfrac14)$ — i.e. $P4/ncc$ in the setting with origin on the 4-fold axis (origin choice 1-like; ITA origin 2 differs by a shift to the inversion center at 8d).
- Model invariance: all 16 cosets + TR at random $k$, error $<10^{-10}$, for the HK-printed $\mathcal H^0_{130}$ with the $\lambda_1\sigma^z$ correction (both $C_4$ senses pass for this model; $e^{i\pi\sigma^z/4}$ convention per the SG 135 note's caveat).
- Coreps: Bloch induction $U(g,k)=P(g)\otimes S(g)$ up to a global phase (convention I, $e^{ik\cdot(R+q_\alpha)}$); fold-back $V(G)={\rm diag}(e^{iG\cdot q_\alpha})$ verified against $H(k+G)=V^\dagger H V$; degeneracies from doubly-symmetrized random Hermitians, invariance error $\le10^{-15}$, stable across seeds.
- Pairing: projector in directed-bond coefficient space (group action = bond permutation in convention I; Pauli constraint $c_{(\beta,\alpha,-d)}=c_{(\alpha,\beta,d)}$; TR = reality); SVD null spaces; channel identification by least-squares fit against a dictionary of lattice harmonics; all symmetry errors $\lesssim 4\times10^{-15}$.
- BdG: $H_{\rm BdG}=\begin{pmatrix} H(k)-\mu & \Delta(k)\\ \Delta^\dagger(k) & -(H(-k)-\mu)^T\end{pmatrix}$; fillings by $T=0$ occupation sums (12³–16³ grids, bisection in $\mu$); gaps on 24³–40³ grids + 801-point HS-line scans + multi-start Nelder–Mead refinement.
- Scripts: `/tmp/sg130_analysis.py` (group, Wyckoff, model checks), `/tmp/sg130_part2.py` (coreps), `/tmp/sg130_part3.py` (channels, classes, BdG scans), `/tmp/sg130_part4.py` ($S_5$, exact nodes, chargon EBR), `/tmp/sg130_part5.py` (gap robustness).
- Sources: WPVZ PNAS **112**, 14551 (2015) + SI (local PDFs, `notes/theory/Watanabe Bound/`); Wieder–Kim–Rappe–Kane PRL **116**, 186402 (2016) via the HK paper's Eq. (130noninteractingH) and generator table (`sg135_hk (2)/main.tex`, lines 899–931, 1260–1290); `sg135_factorization.md` for the mirrored methodology.
