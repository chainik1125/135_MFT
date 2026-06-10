"""Slave-boson mean-field theory of the Kane-Mele-Hubbard model at half filling.

Reference: Wen, Kargarian, Vaezi, Fiete, PRB 84, 235149 (2011) [arXiv:1110.3328],
"Doping the Kane-Mele-Hubbard model: A Slave-Boson Approach".

Strategy: the appendix self-consistency equations are exactly the stationarity
conditions of the ground-state energy E_g = E_f + E_b + E_c (Eqs. A1-A3 + d_i).
We transcribe ONLY the compact energy expressions and obtain the residual
system by automatic differentiation (jax.grad), eliminating transcription
errors in the 13 long coupled equations. Solutions are found with
scipy.optimize.least_squares from multiple physical seeds (SL-like, SC-like),
and the dimerized (DM) phase is treated as the decoupled-dimer limit of the
same theory. The phase at each (U, lambda_SO) is the stationary solution of
lowest E_g.

Order of variables p (14, all real; t == 1):
  0 chi_b   1 chi_f   2 Delta_b  3 Delta_f
  4 chi_bp  5 chi_fp  6 Delta_bp 7 Delta_fp   (primes = NNN / spin-orbit OPs)
  8 hA  9 hB  10 dA  11 dB                    (k=0 condensate amplitudes)
  12 lam  13 mu                               (Lagrange multipliers)

Conventions (paper): g(k) = 1 + e^{-i k2} + e^{i(k1-k2)},
g1 = 2[sin k2 - sin k1 + sin(k1-k2)], g2 = 2[cos k1 + cos k2 + cos(k1-k2)],
with k_i = k . a_i; BZ average = uniform grid over (k1,k2) in [0,2pi)^2.
"""
import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import least_squares

jax.config.update("jax_enable_x64", True)

EPS = 1e-12


def make_kgrid(nk=120):
    """Uniform BZ grid in (k1,k2), shifted half-step to avoid Dirac points."""
    s = 2 * np.pi / nk
    k = np.arange(nk) * s + 0.5 * s
    k1, k2 = np.meshgrid(k, k, indexing="ij")
    return jnp.asarray(k1.ravel()), jnp.asarray(k2.ravel())


def _ssqrt(x):
    """Safe sqrt: clips tiny negatives from roundoff; large negatives are
    handled separately via domain penalties."""
    return jnp.sqrt(jnp.clip(x, EPS))


def e_total(p, U, lso, k1, k2, x=0.0, t=1.0):
    """Ground-state energy per unit cell, Eqs. (A?) of arXiv:1110.3328 appendix."""
    chb, chf, Db, Df, chbp, chfp, Dbp, Dfp, hA, hB, dA, dB, lam, mu = p
    N = k1.shape[0]

    g = 1.0 + jnp.exp(-1j * k2) + jnp.exp(1j * (k1 - k2))
    g2abs = jnp.real(g * jnp.conj(g))           # |g|^2
    gR = jnp.real(g)
    g1 = 2.0 * (jnp.sin(k2) - jnp.sin(k1) + jnp.sin(k1 - k2))
    g2f = 2.0 * (jnp.cos(k1) + jnp.cos(k2) + jnp.cos(k1 - k2))

    # --- fermionic (spinon) energy ---
    A1 = (lam**2 + g2abs * t**2 * Db**2 + g1**2 * Dbp**2 * lso**2
          + g2abs * t**2 * chb**2 + g1**2 * lso**2 * chbp**2)
    A2 = (g2abs * t**2 * (lam**2 + g1**2 * Dbp**2 * lso**2) * chb**2
          - 2.0 * g2abs * t**2 * g1**2 * Db * Dbp * lso**2 * chb * chbp
          + g1**2 * (lam**2 + g2abs * t**2 * Db**2) * lso**2 * chbp**2)
    sA2 = _ssqrt(A2)
    Ef = -(1.0 / N) * jnp.sum(_ssqrt(A1 - 2.0 * sA2) + _ssqrt(A1 + 2.0 * sA2))

    # --- bosonic (chargeon) energy ---
    # |g t Df -+ g2 Dfp lso|^2 with complex g, real g2:
    m_minus = t**2 * Df**2 * g2abs - 2.0 * t * Df * g2f * Dfp * lso * gR + g2f**2 * Dfp**2 * lso**2
    m_plus = t**2 * Df**2 * g2abs + 2.0 * t * Df * g2f * Dfp * lso * gR + g2f**2 * Dfp**2 * lso**2
    U2l = (U - 2.0 * lam) ** 2
    Eb = -U + (0.5 / N) * jnp.sum(_ssqrt(U2l - 4.0 * m_minus) + _ssqrt(U2l - 4.0 * m_plus))

    # --- constant / condensate energy ---
    Ec = (2.0 * lam + 2.0 * x * mu
          - 6.0 * (dB * hA + dA * hB) * t * Df
          + 12.0 * (dA * hA + dB * hB) * Dfp * lso
          + 6.0 * (dA * dB - hA * hB) * t * chf
          + 6.0 * t * (Db * Df + chb * chf)
          + (hA**2 + hB**2) * (-lam + mu + 6.0 * lso * chfp)
          - (dA**2 + dB**2) * (lam - U + mu + 6.0 * lso * chfp)
          - 12.0 * lso * (Dbp * Dfp + chbp * chfp))
    return Ef + Eb + Ec


grad_e = jax.jit(jax.grad(e_total, argnums=0), static_argnames=())


def domain_penalty(p, U, lso, k1, k2, t=1.0):
    """Boson spectrum must be real: (U-2lam)^2 >= 4|g t Df -+ g2 Dfp lso|^2
    everywhere; fermion A1 - 2 sqrt(A2) >= 0. Returns >=0 violations."""
    chb, chf, Db, Df, chbp, chfp, Dbp, Dfp, hA, hB, dA, dB, lam, mu = p
    g = 1.0 + jnp.exp(-1j * k2) + jnp.exp(1j * (k1 - k2))
    g2abs = jnp.real(g * jnp.conj(g))
    gR = jnp.real(g)
    g2f = 2.0 * (jnp.cos(k1) + jnp.cos(k2) + jnp.cos(k1 - k2))
    m_minus = t**2 * Df**2 * g2abs - 2.0 * t * Df * g2f * Dfp * lso * gR + g2f**2 * Dfp**2 * lso**2
    m_plus = t**2 * Df**2 * g2abs + 2.0 * t * Df * g2f * Dfp * lso * gR + g2f**2 * Dfp**2 * lso**2
    U2l = (U - 2.0 * lam) ** 2
    v1 = jnp.max(jnp.clip(4.0 * m_minus - U2l, 0.0))
    v2 = jnp.max(jnp.clip(4.0 * m_plus - U2l, 0.0))
    return jnp.array([v1, v2])


domain_penalty = jax.jit(domain_penalty)


def residuals(p, U, lso, k1, k2, x=0.0, w_pen=10.0):
    r = grad_e(jnp.asarray(p), U, lso, k1, k2, x)
    pen = w_pen * domain_penalty(jnp.asarray(p), U, lso, k1, k2)
    return np.concatenate([np.asarray(r), np.asarray(pen)])


SEEDS = {
    # chb  chf   Db    Df   chbp chfp  Dbp   Dfp   hA    hB    dA    dB   lam   mu
    "SL":   [0.0, 0.0, 0.50, 0.40, 0.0, 0.0, 0.02, 0.02, 0.0, 0.0, 0.0, 0.0, -0.30, 0.0],
    "SC":   [0.0, 0.0, 0.40, 0.60, 0.0, 0.0, 0.05, 0.05, 0.55, -0.55, 0.55, -0.55, -0.20, 0.0],
    "SC2":  [0.5, 0.5, 0.0, 0.0, 0.05, 0.05, 0.0, 0.0, 0.55, 0.55, 0.55, 0.55, -0.20, 0.0],
    "SCb":  [0.0, 0.0, 0.55, 0.50, 0.0, 0.0, 0.10, 0.10, 0.40, -0.40, 0.40, -0.40, -0.50, 0.0],
}


def solve_point(U, lso, k1, k2, seeds=None, tol=1e-9):
    """Try all seeds; return list of converged distinct solutions with energies."""
    out = []
    for name, s0 in (seeds or SEEDS).items():
        try:
            sol = least_squares(residuals, np.array(s0, float), args=(U, lso, k1, k2),
                                method="trf", xtol=1e-14, ftol=1e-14, gtol=1e-14,
                                max_nfev=400)
        except Exception:
            continue
        rn = np.linalg.norm(sol.fun)
        if rn < 1e-6:
            p = sol.x
            E = float(e_total(jnp.asarray(p), U, lso, k1, k2))
            out.append({"seed": name, "p": p, "E": E, "rnorm": rn})
    return out


# ---------------- dimerized (DM) phase: decoupled-dimer limit ----------------
def e_dimer(q, U, t=1.0):
    """Slave-boson MFT energy per unit cell in the extreme dimer limit:
    one active NN bond per unit cell (z_eff=1, |g|->1), all NNN OPs zero,
    no condensates (atomic-like insulator). q = [Db, Df, lam]."""
    Db, Df, lam = q
    Ef = -2.0 * _ssqrt(lam**2 + t**2 * Db**2)
    Eb = -U + _ssqrt((U - 2.0 * lam) ** 2 - 4.0 * t**2 * Df**2)
    Ec = 2.0 * lam + 2.0 * t * Db * Df
    return Ef + Eb + Ec


grad_dimer = jax.jit(jax.grad(e_dimer, argnums=0))


def solve_dimer(U, t=1.0):
    best = None
    for s0 in ([0.5, 0.4, -0.3], [0.8, 0.2, -0.6], [0.3, 0.45, -0.1]):
        def res(q):
            r = np.asarray(grad_dimer(jnp.asarray(q), U))
            pen = max(0.0, 4.0 * t**2 * q[1] ** 2 - (U - 2.0 * q[2]) ** 2)
            return np.concatenate([r, [10.0 * pen]])
        sol = least_squares(res, np.array(s0, float), method="trf",
                            xtol=1e-14, ftol=1e-14, max_nfev=400)
        if np.linalg.norm(sol.fun) < 1e-7:
            E = float(e_dimer(jnp.asarray(sol.x), U))
            if best is None or E < best["E"]:
                best = {"q": sol.x, "E": E}
    return best


# ---------------- classification ----------------
def classify(p, cond_tol=2e-2, op_tol=2e-2):
    chb, chf, Db, Df, chbp, chfp, Dbp, Dfp, hA, hB, dA, dB, lam, mu = p
    condensed = max(abs(hA), abs(hB), abs(dA), abs(dB)) > cond_tol
    if condensed:
        return "SC"
    if abs(Df) > op_tol or abs(Db) > op_tol:
        return "SL"
    return "trivial"


def observables(p, U, lso, k1, k2):
    """Single-particle gap at Dirac point (lso=0 SL formula) and double occupancy."""
    chb, chf, Db, Df, chbp, chfp, Dbp, Dfp, hA, hB, dA, dB, lam, mu = p
    gap = abs(lam) + np.sqrt(max((U / 2 - lam) ** 2 - 9 * Df**2, 0.0))
    dU = jax.grad(e_total, argnums=1)(jnp.asarray(p), U, lso, k1, k2)
    docc = 0.5 * float(dU)  # per site
    return gap, docc
