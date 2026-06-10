"""Cross-validate generic BdG machinery (bdg.py) against the KMH closed-form
energies (kmh_solver.e_total) at lambda_SO = 0, no condensates.

The two implementations are independent: closed forms transcribed from the
paper appendix vs. numerical eigh/Colpa on kernels transcribed from the
write-up's Pauli decompositions (Eqs. 4.47, 4.71). Agreement as a FUNCTION of
random (chi_b, chi_f, Delta_b, Delta_f, lam, U) validates zero-point and
Tr-xi conventions before they are reused for SG135.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "kmh"))
import numpy as np
import jax.numpy as jnp
from bdg import fermion_e0, boson_e0
from kmh_solver import e_total, make_kgrid

I2 = np.eye(2); X = np.array([[0,1],[1,0]],complex)
Y = np.array([[0,-1j],[1j,0]]); Z = np.diag([1.,-1.]).astype(complex)
kron3 = lambda a,b,c: np.kron(a, np.kron(b, c))


def kmh_numeric(chb, chf, Db, Df, lam, U, k1, k2, t=1.0):
    g = 1.0 + np.exp(-1j*k2) + np.exp(1j*(k1-k2))
    nk = len(k1)
    # fermion kernel 8x8 (Nambu(lambda) x spin-block(sigma) x sublattice(tau)), Eq 4.71
    chi = t*chb*g; dlt = t*Db*g
    Hf = (-lam*kron3(Z,I2,I2)[None]
          - np.real(chi)[:,None,None]*kron3(Z,I2,X)[None]
          + np.imag(chi)[:,None,None]*kron3(I2,Z,Y)[None]
          + np.real(dlt)[:,None,None]*kron3(Y,Y,X)[None]
          + np.imag(dlt)[:,None,None]*kron3(X,X,Y)[None])
    ev = np.linalg.eigvalsh(Hf)
    e0f = -0.5*np.sum(np.where(ev>0, ev, 0.), axis=-1) + 0.5*(-4*lam)
    # boson kernel 8x8 (Nambu(lambda) x holon-doublon(nu) x sublattice(tau)), Eq 4.47
    D = (U-2*lam)/2.; D0 = U/2.  # mu=0 at half filling for this test (D0=(U-2mu)/2)
    chib = t*chf*g; dltb = t*Df*g
    Hb = (D*kron3(I2,I2,I2)[None] + D0*kron3(I2,Z,I2)[None]
          - np.real(chib)[:,None,None]*kron3(I2,Z,X)[None]
          + np.imag(chib)[:,None,None]*kron3(Z,I2,Y)[None]
          - np.real(dltb)[:,None,None]*kron3(X,X,X)[None]
          - np.imag(dltb)[:,None,None]*kron3(Y,Y,Y)[None])
    xi_b = jnp.asarray(Hb[:, :4, :4]); del_b = jnp.asarray(Hb[:, :4, 4:])
    e0b, wmin = boson_e0(xi_b, del_b)
    ec = 6*t*(chb*chf + Db*Df) + 2*lam   # per cell, x=0, lso=0, no condensates
    return float(np.mean(e0f) + np.mean(np.asarray(e0b)) + ec), float(np.min(np.asarray(wmin)))


if __name__ == "__main__":
    k1, k2 = make_kgrid(60)
    k1n, k2n = np.asarray(k1), np.asarray(k2)
    rng = np.random.default_rng(0)
    print(f"{'closed':>12} {'numeric':>12} {'diff':>10} {'minH_b':>8}")
    ok = True
    for trial in range(8):
        U = rng.uniform(1.0, 3.0); lam = rng.uniform(-0.8, -0.05)
        chb, chf = rng.uniform(-0.5, 0.5, 2)
        Db, Df = rng.uniform(0.05, 0.4, 2)
        # keep boson spectrum real: need (U-2lam)^2 > 4 max|t g Df|^2 = 36 Df^2
        if (U-2*lam)**2 <= 4*(3*Df)**2 + 0.1: continue
        p = jnp.array([chb, chf, Db, Df, 0,0,0,0, 0,0,0,0, lam, 0.0])
        ec_closed = float(e_total(p, U, 0.0, k1, k2))
        en, wmin = kmh_numeric(chb, chf, Db, Df, lam, U, k1n, k2n)
        d = en - ec_closed
        flag = "" if abs(d) < 1e-8 else "  <-- MISMATCH"
        if abs(d) > 1e-8: ok = False
        print(f"{ec_closed:12.6f} {en:12.6f} {d:10.2e} {wmin:8.4f}{flag}")
    print("CONVENTIONS", "VALIDATED" if ok else "MISMATCH - diagnose constants/factors")
