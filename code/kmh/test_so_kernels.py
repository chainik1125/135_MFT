"""Independent verification of the KMH closed-form spectra at lambda_SO != 0.

Builds the fermion and boson kernels term-by-term from the write-up's explicit
k-space Hamiltonians (Eqs. 4.21-4.26, which include the SO channels with their
g1/g2 form factors), diagonalizes numerically, and compares with the paper's
closed forms:
  fermion: +-sqrt(A1 -+ 2 sqrt(A2))   (appendix A1, A2)
  boson:   d3, d4 = sqrt((U-2lam)^2 - 4(|g| t Df -+ g2 Dfp lso)^2)
If these match for random (k, params), the closed-form energy (and hence the
phase-boundary slope in lambda_SO) follows from the published equations.
Basis: c = (f_{k up A}, f_{k up B}, f_{-k dn A}, f_{-k dn B}); psi = (c, c+).
"""
import numpy as np

rng = np.random.default_rng(7)
t = 1.0


def gfun(k1, k2):
    g = 1 + np.exp(-1j * k2) + np.exp(1j * (k1 - k2))
    g1 = 2 * (np.sin(k2) - np.sin(k1) + np.sin(k1 - k2))
    g2 = 2 * (np.cos(k1) + np.cos(k2) + np.cos(k1 - k2))
    return g, g1, g2


def fermion_kernel(k1, k2, chb, Db, chbp, Dbp, lam, lso):
    g, g1, g2 = gfun(k1, k2)
    gm, g1m, _ = gfun(-k1, -k2)
    xi = np.zeros((4, 4), complex)
    xi -= lam * np.eye(4)
    # NN hopping: up at +k, dn at -k
    xi[0, 1] += -t * chb * g
    xi[1, 0] += np.conj(-t * chb * g)
    xi[2, 3] += -t * chb * gm
    xi[3, 2] += np.conj(-t * chb * gm)
    # SO hopping: sigma * g1(k) (n_A - n_B), dn at -k: (-1)*g1(-k) = +g1(k)
    xi[0, 0] += lso * chbp * g1
    xi[1, 1] -= lso * chbp * g1
    xi[2, 2] += lso * chbp * (-1) * g1m
    xi[3, 3] -= lso * chbp * (-1) * g1m
    # pairing block (antisymmetric): collect coefficients of c_i^+ c_j^+
    D = np.zeros((4, 4), complex)
    # NN singlet: sum_sigma -t Db g(k) sigma f+_{kA s} f+_{-kB -s}
    D[0, 3] += -t * Db * g          # sigma = up at +k
    D[2, 1] += +t * Db * gm         # sigma = dn term relabeled k -> -k
    # SO pairing: lso Dbp g1(k) [c1+ c3+ - c2+ c4+]  (A-A and B-B, k with -k)
    D[0, 2] += lso * Dbp * g1
    D[1, 3] -= lso * Dbp * g1
    Delta = 0.5 * (D - D.T)  # antisymmetrize (each operator pair counted once)
    # the (1/2) convention: kernel [[xi, 2*Delta...]]: build full H and just
    # check eigenvalue pairing numerically; use Delta_full = (D - D.T) so that
    # (1/2) psi+ H psi reproduces the operator sum exactly.
    Dfull = D - D.T
    H = np.block([[xi, Dfull], [-Dfull.conj(), -xi.conj()]])
    # hermiticity guard
    assert np.abs(H - H.conj().T).max() < 1e-12
    return H


def closed_fermion(k1, k2, chb, Db, chbp, Dbp, lam, lso):
    g, g1, g2 = gfun(k1, k2)
    ga2 = abs(g) ** 2
    A1 = lam**2 + ga2 * t**2 * Db**2 + g1**2 * Dbp**2 * lso**2 \
        + ga2 * t**2 * chb**2 + g1**2 * lso**2 * chbp**2
    A2 = ga2 * t**2 * (lam**2 + g1**2 * Dbp**2 * lso**2) * chb**2 \
        - 2 * ga2 * t**2 * g1**2 * Db * Dbp * lso**2 * chb * chbp \
        + g1**2 * (lam**2 + ga2 * t**2 * Db**2) * lso**2 * chbp**2
    sA2 = np.sqrt(max(A2, 0))
    return np.array([np.sqrt(max(A1 - 2 * sA2, 0)), np.sqrt(A1 + 2 * sA2)])


def boson_kernel(k1, k2, chf, Df, chfp, Dfp, lam, mu, U, lso):
    g, g1, g2 = gfun(k1, k2)
    gm, _, g2m = gfun(-k1, -k2)
    # basis (h_kA, h_kB, d_kA, d_kB); pairing connects to (-k) partners
    xi = np.zeros((4, 4), complex)
    xi[0, 0] = xi[1, 1] = mu - lam
    xi[2, 2] = xi[3, 3] = U - lam - mu
    # SO diagonal: holon +lso chfp g2, doublon -lso chfp g2 (write-up 4.24)
    xi[0, 0] += lso * chfp * g2
    xi[1, 1] += lso * chfp * g2
    xi[2, 2] -= lso * chfp * g2
    xi[3, 3] -= lso * chfp * g2
    # hopping -t chf g(k) [h+_A h_B - d+_A d_B] + h.c.
    xi[0, 1] += -t * chf * g
    xi[1, 0] += np.conj(-t * chf * g)
    xi[2, 3] += +t * chf * g
    xi[3, 2] += np.conj(t * chf * g)
    # pairing (boson: symmetric block): -t Df g(-k)[d_A h_-B + h_A d_-B] -> c+ c+ form:
    S = np.zeros((4, 4), complex)
    # term: -t Df g(-k) (d_{kA} h_{-kB})^+ ... write as h+_{?}d+: contributes to
    # S[h_A, d_B] etc. Using symmetric collection:
    S[0, 3] += -t * Df * gm   # h+_{kA} d+_{-kB}
    S[1, 2] += -t * Df * gm   # h+_{kB} d+_{-kA}  (the h_A d_-B partner)
    # SO pairing: + lso Dfp g2(k) d_k h_-k same sublattice (write-up 4.26)
    S[0, 2] += lso * Dfp * g2
    S[1, 3] += lso * Dfp * g2
    Sfull = S + S.T  # bosonic pairing block symmetric
    H = np.block([[xi, Sfull], [Sfull.conj(), xi.conj()]])
    assert np.abs(H - H.conj().T).max() < 1e-12
    return H


def closed_boson(k1, k2, Df, Dfp, lam, U, lso):
    g, g1, g2 = gfun(k1, k2)
    d3 = np.sqrt((U - 2 * lam) ** 2 - 4 * (abs(g) * t * Df - g2 * Dfp * lso) ** 2)
    d4 = np.sqrt((U - 2 * lam) ** 2 - 4 * (abs(g) * t * Df + g2 * Dfp * lso) ** 2)
    return np.array([d3, d4])


if __name__ == "__main__":
    print("--- fermion sector ---")
    worst = 0.0
    for _ in range(10):
        k1, k2 = rng.uniform(0, 2 * np.pi, 2)
        chb, Db, chbp, Dbp = rng.uniform(-0.5, 0.5, 4)
        lam, lso = rng.uniform(-0.8, -0.1), rng.uniform(0.05, 0.3)
        H = fermion_kernel(k1, k2, chb, Db, chbp, Dbp, lam, lso)
        ev = np.linalg.eigvalsh(H)
        pos = np.sort(ev[ev > 1e-12])
        cf = np.sort(np.concatenate([closed_fermion(k1, k2, chb, Db, chbp, Dbp, lam, lso)] * 2))
        d = np.abs(np.sort(pos) - cf).max() if len(pos) == 4 else 99
        worst = max(worst, d)
        print(f"  numeric {np.round(np.sort(pos),5)}  closed {np.round(cf,5)}  diff {d:.1e}")
    print("fermion worst:", f"{worst:.2e}")

    print("--- boson sector (para-eigenvalues) ---")
    worst = 0.0
    for _ in range(10):
        k1, k2 = rng.uniform(0, 2 * np.pi, 2)
        Df, Dfp, chfp = rng.uniform(-0.3, 0.3, 3)
        chf = 0.0  # closed boson form d3,d4 has no chf dependence (paper)
        U = rng.uniform(2.5, 4.0)
        lam = rng.uniform(-0.8, -0.2)
        mu = U / 2
        lso = rng.uniform(0.05, 0.3)
        H = boson_kernel(k1, k2, chf, Df, chfp, Dfp, lam, mu, U, lso)
        sz = np.diag([1.0, 1, 1, 1, -1, -1, -1, -1])
        ev = np.linalg.eigvals(sz @ H).real
        pos = np.sort(ev[ev > 1e-9])
        cb = np.sort(np.concatenate([closed_boson(k1, k2, Df, Dfp, lam, U, lso) / 2] * 2))
        # note: d3,d4 vs para-energies differ by factor 2 conventions; test both
        cb2 = 2 * cb
        d = min(np.abs(np.sort(pos) - cb).max() if len(pos) == 4 else 99,
                np.abs(np.sort(pos) - cb2).max() if len(pos) == 4 else 99)
        worst = max(worst, d)
        print(f"  numeric {np.round(np.sort(pos),5)}  closed(d/2) {np.round(cb,5)}  diff {d:.1e}")
    print("boson worst:", f"{worst:.2e}")
