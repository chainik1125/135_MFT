"""Real-space referee for the SG135 mean-field bookkeeping.

Step 1 (map): construct the real-space hopping list whose Bloch transform
equals H(k) = sum_i t_i g_i(k) M_i exactly (random-k check). This fixes the
per-bond amplitudes and phases with no convention ambiguity.

Step 2 (kernel check): fermion MF sector with chi_b coefficients on an L^3
lattice (periodic): E_f^realspace == E_f^kspace at the same parameters.

Step 3 (E_c referee): at a parameter point, compute per-bond expectations
< chi^f_bond > = sum_s <f+_is f_js> from the real-space correlation matrix,
form the decoupling constant  E_c^hop = sum_bonds 2 t_b Re[<chi^f_b>] chi_b_b
(per cell), and compare with C * t_i * chi_f_i * chi_b_i for C = 4 vs 8.
The channel OP chi_f_i implied by k-space stationarity is
   chi_f_i = -(1/(C t_i)) dE_f/dchi_b_i ,
so equivalently we directly report:  -dE_f/dchi_b_i  vs  sum_{bonds in i} 2 t_b <chi^f_b> s_b-pattern.
Whatever matches defines the correct C_i (including possible cancellations
in the sign-alternating channels 1 and 2).
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "common"))
import numpy as np
import jax
import jax.numpy as jnp
from sg135_solver import gfuncs, TS_DEFAULT, GAMMAS, fermion_blocks, make_kgrid
from bdg import fermion_e0

# ---------------- real-space model ----------------
# sublattice order: (mu,tau) = (A,A),(A,B),(B,A),(B,B) ; positions
POS = np.array([[0, 0, 0], [0.5, 0.5, 0], [0, 0, 0.5], [0.5, 0.5, 0.5]])
# index: mu = idx//2 (0=A,1=B), tau = idx%2


def bond_list():
    """(alpha, beta, displacement, amplitude) with H_realspace summing
    t_ab f+_a(r) f_b(r+d) + h.c.  Amplitudes chosen so the Bloch sum matches
    t_i g_i M_i; verified numerically in check_bloch()."""
    bonds = []
    txy, tz, t1, t2 = TS_DEFAULT[:4]
    # channel xy: tau^x within same mu: (mu,A)<->(mu,B), 4 in-plane diagonals
    for mu in (0, 1):
        a, b = 2 * mu + 0, 2 * mu + 1
        for sx in (0.5, -0.5):
            for sy in (0.5, -0.5):
                bonds.append((a, b, np.array([sx, sy, 0.0]), txy / 4))
    # channel z: mu^x same tau: (A,tau)<->(B,tau), +-z/2
    for tau in (0, 1):
        a, b = tau, 2 + tau
        for sz in (0.5, -0.5):
            bonds.append((a, b, np.array([0.0, 0.0, sz]), tz / 2))
    # channel 1: mu^z diagonal: same site-type, +-x (plus) and +-y (minus),
    # sign + for mu=A, - for mu=B
    for idx in range(4):
        smu = 1.0 if idx < 2 else -1.0
        # only +x, +y: the h.c. term supplies the -x, -y partners
        for d, s in (([1, 0, 0], 1.0), ([0, 1, 0], -1.0)):
            bonds.append((idx, idx, np.array(d, float), smu * s * t1 / 2))
    # channel 2: tau^x mu^y: (A,A)<->(B,B)-type pairs? No: tau^x flips tau,
    # mu^y flips mu with +-i: bonds between (A,tau) and (B,tau') with tau'!=tau,
    # displacement (+-1/2,+-1/2,+-1/2), amplitude from
    # g2 = sin(kx/2)sin(ky/2)cos(kz/2) = (1/8) sum_d sgn(dx)sgn(dy) e^{ik.d} * (-1)...
    # mu^y matrix element: (mu=A->B) = +i... handled with explicit factor:
    for tau_a in (0, 1):
        a = 0 * 2 + tau_a          # mu=A, tau_a
        b = 1 * 2 + (1 - tau_a)    # mu=B, tau_b != tau_a
        for sx in (0.5, -0.5):
            for sy in (0.5, -0.5):
                for sz in (0.5, -0.5):
                    # Gamma_2 = mu^y tau^y: [mu^y]_AB [tau^y]_{tau_a,1-tau_a} =
                    # (-i)(-i) = -1 for tau_a=A, (-i)(+i) = +1 for tau_a=B;
                    # with g2 expansion coefficient -(1/8) sx sy:
                    sgn_tau = 1.0 if tau_a == 0 else -1.0
                    amp = sgn_tau * (t2 / 8) * np.sign(sx) * np.sign(sy)
                    bonds.append((a, b, np.array([sx, sy, sz]), amp))
    return bonds


def hk_from_bonds(bonds, k):
    H = np.zeros((4, 4), complex)
    for a, b, d, t in bonds:
        H[a, b] += t * np.exp(1j * (k @ d))
        H[b, a] += np.conj(t * np.exp(1j * (k @ d)))
    return H / 1.0


def hk_reference(k):
    gxy, gz, g1, g2 = [np.asarray(g) for g in gfuncs(*[jnp.asarray(np.array([kk])) for kk in k])]
    M = (TS_DEFAULT[0] * gxy[0] * np.asarray(GAMMAS[0]) + TS_DEFAULT[1] * gz[0] * np.asarray(GAMMAS[1])
         + TS_DEFAULT[2] * g1[0] * np.asarray(GAMMAS[2]) + TS_DEFAULT[3] * g2[0] * np.asarray(GAMMAS[3]))
    return M


def check_bloch():
    bonds = bond_list()
    rng = np.random.default_rng(3)
    worst = 0.0
    for _ in range(12):
        k = rng.uniform(-np.pi, np.pi, 3)
        d = np.abs(hk_from_bonds(bonds, k) - hk_reference(k)).max()
        worst = max(worst, d)
    print(f"step1 bloch match: worst |H_real(k) - H_ref(k)| = {worst:.2e}")
    return worst < 1e-12


# ---------------- real-space MF fermion sector ----------------
def realspace_fermion(chi_b, Delta_b, lam, L=6):
    """Build the spinon MF Hamiltonian on L^3 cells: hopping t_b*chi_b_i per bond
    (channel i), singlet pairing t_b*Delta_b_i per bond, chemical -lam.
    Returns (E0 per cell, correlation function on one xy-bond and one z-bond,
    and channel-summed bond expectations)."""
    bonds = bond_list()
    txy, tz, t1, t2 = TS_DEFAULT[:4]
    chan_of = []
    for n, (a, b, d, t) in enumerate(bonds):
        if abs(d[2]) == 0 and abs(d[0]) == 0.5:
            chan_of.append(0)
        elif abs(d[2]) == 0.5 and d[0] == 0 and d[1] == 0:
            chan_of.append(1)
        elif np.linalg.norm(d) == 1.0:
            chan_of.append(2)
        else:
            chan_of.append(3)
    N = L**3
    nsite = 4 * N
    # site index: ((cx*L+cy)*L+cz)*4 + alpha
    def sid(c, alpha):
        return ((c[0] % L * L + c[1] % L) * L + c[2] % L) * 4 + alpha

    cells = [(x, y, z) for x in range(L) for y in range(L) for z in range(L)]
    Hhop = np.zeros((nsite, nsite), complex)   # spin-diagonal hopping (same for both spins)
    Hpair = np.zeros((nsite, nsite), complex)  # singlet pairing amplitude on bonds
    for c in cells:
        for n, (a, b, d, t) in enumerate(bonds):
            ch = chan_of[n]
            # bond from site (c,alpha=a) to site at displacement d: target cell
            ra = POS[a] + np.array(c)
            rb = ra + d
            cell_b = np.round(rb - POS[b]).astype(int)
            ia, ib = sid(c, a), sid(cell_b, b)
            Hhop[ia, ib] += t * chi_b[ch]
            Hhop[ib, ia] += np.conj(t * chi_b[ch])
            Hpair[ia, ib] += t * Delta_b[ch]
            Hpair[ib, ia] += np.conj(t * Delta_b[ch])  # Delta(k)=Delta(-k) real channel
    Hhop -= lam * np.eye(nsite)
    # BdG for singlet pairing: spinor (f_up, f+_down): [[Hhop, Hpair],[Hpair^+, -Hhop^T]]
    K = np.block([[Hhop, Hpair], [Hpair.conj().T, -Hhop.T]])
    ev, V = np.linalg.eigh(K)
    # E0 (both spins): E = sum_{ev<0} ev + Tr Hhop  ... derive: H = sum_ks ...
    # For the spinor above, H_phys = sum_n ev_n theta(-ev_n) + Tr[Hhop] (down-spin constant)
    occ = ev < 0
    E0 = ev[occ].sum() + np.trace(Hhop).real
    # correlation <f+_a f_b> for spin up: from V: f_up sector = first nsite rows;
    # ground state: filled negative modes: C = V_occ V_occ^dagger restricted
    Vocc = V[:, occ]
    Cmat = (Vocc @ Vocc.conj().T)[:nsite, :nsite]  # <f_i f+_j> for up spin
    corr_up = np.eye(nsite) - Cmat.T               # <f+_i f_j>
    return E0 / N, corr_up, bonds, chan_of, sid


def main():
    ok = check_bloch()
    if not ok:
        print("BLOCH MISMATCH - fix bond_list before anything else")
        return
    # step 2: energy comparison, hopping-only MF point
    chi_b = np.array([0.4, 0.3, 0.15, 0.1])
    Delta_b = np.array([0.3, 0.25, 0.1, 0.08])
    lam = -0.4
    L = 6
    E0_real, corr, bonds, chan_of, sid = realspace_fermion(chi_b, Delta_b, lam, L)
    # k-space at same point
    p = np.zeros(18)
    p[0:4] = chi_b
    p[8:12] = Delta_b
    p[16] = lam
    kx, ky, kz = make_kgrid(L)  # SAME grid density as L^3 lattice -> exact match
    # shift: finite lattice k-grid = unshifted Gamma-centered; rebuild grid unshifted:
    kk = 2 * np.pi * np.arange(L) / L
    KX, KY, KZ = np.meshgrid(kk, kk, kk, indexing="ij")
    kxj, kyj, kzj = map(jnp.asarray, (KX.ravel(), KY.ravel(), KZ.ravel()))
    xi, dl = fermion_blocks(jnp.asarray(p), kxj, kyj, kzj, TS_DEFAULT)
    E0_k = float(jnp.mean(fermion_e0(xi, dl)))
    print(f"step2 E_f: realspace {E0_real:.8f}  kspace {E0_k:.8f}  diff {E0_real-E0_k:+.2e}")

    # step 3: channel bond-expectation sums vs autodiff dE_f/dchi_b_i
    c0 = (0, 0, 0)
    chan_sums = np.zeros(4, complex)
    for n, (a, b, d, t) in enumerate(bonds):
        cell_b = np.round(POS[a] + d - POS[b]).astype(int)
        ia, ib = sid(c0, a), sid(tuple(cell_b), b)
        chif_bond = 2.0 * corr[ia, ib]  # both spins
        chan_sums[chan_of[n]] += 2.0 * t * chif_bond  # 2 from h.c.
    def ef_of_p(q):
        xi2, dl2 = fermion_blocks(q, kxj, kyj, kzj, TS_DEFAULT)
        return jnp.mean(fermion_e0(xi2, dl2))
    g = np.asarray(jax.grad(ef_of_p)(jnp.asarray(p)))
    for i, name in enumerate(("xy", "z", "1", "2")):
        # k-space: dE_f/dchi_b_i ; real-space prediction: sum_bonds 2 t_b <chi^f_b>
        print(f"step3 ch{name}: dEf/dchi_b = {g[i]:+.6f}   bond-sum = {np.real(chan_sums[i]):+.6f}"
              f"   ratio = {g[i]/np.real(chan_sums[i]) if abs(chan_sums[i])>1e-12 else float('nan'):+.3f}")
    print("interpretation: dEf/dchi_b_i == bond-sum  =>  E_c cross-term must be"
          " -bond-sum*chi_f... i.e. C_i t_i = |bond-sum/chi-product|; see notes.")


if __name__ == "__main__":
    main()
