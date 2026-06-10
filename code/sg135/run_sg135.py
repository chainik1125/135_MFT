"""SG135 slave-boson runner.

Stage A (validation):
  A1: non-interacting limit - fermion kernel with chi_b_i=1, Delta=0, lam=0
      must reproduce the double-Dirac band structure (8-fold at A=(pi,pi,pi);
      band path comparison vs Wieder et al. Fig 1d pattern).
  A2: atomic limit - all OPs zero, lam=mu_L=0 -> E_g = 0.
Stage B: uncondensed (gapped-boson) solutions vs U with continuation; report
  order parameters, total energy vs the E=0 atomic Mott state, boson stability
  margin (BEC onset), and the fermion spectral gap.
"""
import sys, time
import numpy as np
import jax
import jax.numpy as jnp
from scipy.optimize import least_squares
from sg135_solver import (make_kgrid, e_total, fermion_blocks, boson_blocks,
                          min_boson_eig, TS_DEFAULT, GAMMAS, kron, I2, X, Y, Z)
sys.path.insert(0, "../common")

t0 = time.time()
log = lambda *a: (print(f"[{time.time()-t0:7.1f}s]", *a), sys.stdout.flush())

NK = int(sys.argv[1]) if len(sys.argv) > 1 else 16
kx, ky, kz = make_kgrid(NK)
ZERO_C8 = jnp.zeros(8)


# ---------------- Stage A1: band structure check ----------------
def band_check():
    pts = {"G": (0, 0, 0), "X": (0, np.pi, 0), "M": (np.pi, np.pi, 0),
           "Z": (0, 0, np.pi), "R": (0, np.pi, np.pi), "A": (np.pi, np.pi, np.pi)}
    p = np.zeros(18)
    p[0:4] = 1.0  # chi_b_i = 1 (free-fermion limit of the spinon sector)
    p = jnp.asarray(p)
    log("A1 band check (chi_b=1, Delta=0, lam=0); 8 Nambu-doubled levels per k:")
    for name, kpt in pts.items():
        kxa = jnp.array([kpt[0]]); kya = jnp.array([kpt[1]]); kza = jnp.array([kpt[2]])
        xi, dl = fermion_blocks(p, kxa, kya, kza, TS_DEFAULT)
        H = jnp.concatenate([jnp.concatenate([xi, dl], axis=-1),
                             jnp.concatenate([-jnp.conj(dl), -jnp.conj(xi)], axis=-1)], axis=-2)
        ev = np.sort(np.asarray(jnp.linalg.eigvalsh(H))[0])
        # count degeneracy of the level nearest zero
        uniq, counts = np.unique(np.round(ev, 6), return_counts=True)
        log(f"  {name}: levels {dict(zip(uniq, counts))}")
    log("  expect at A: one 16-fold (8-fold double Dirac x Nambu doubling) at E=0")


# ---------------- Stage A2: atomic limit ----------------
def atomic_check(U=2.0):
    p = jnp.zeros(18)
    E = float(e_total(p, ZERO_C8, U, kx, ky, kz))
    log(f"A2 atomic limit E_g(all zero, U={U}) = {E:.2e} (expect 0)")


# ---------------- Stage B: uncondensed solve ----------------
def _resid_jax(p, U):
    r = jax.grad(e_total, argnums=0)(p, ZERO_C8, U, kx, ky, kz)
    wmin = min_boson_eig(p, U, kx, ky, kz)
    pen = 50.0 * jnp.clip(-wmin, 0.0)
    # mu_L is a flat direction without condensates at x=0; pin via tiny tether
    return jnp.concatenate([r, jnp.array([pen]), 1e-3 * p[jnp.array([17])]])


_resid_jit = jax.jit(_resid_jax)
_resid_jac = jax.jit(jax.jacfwd(_resid_jax, argnums=0))


def residuals(p, U):
    return np.asarray(_resid_jit(jnp.asarray(p), U))


def residuals_jac(p, U):
    return np.asarray(_resid_jac(jnp.asarray(p), U))


def solve_uncondensed(U, seeds):
    out = []
    for s0 in seeds:
        try:
            sol = least_squares(residuals, np.array(s0), jac=residuals_jac, args=(U,),
                                method="trf", xtol=1e-13, ftol=1e-13, gtol=1e-13,
                                max_nfev=600)
        except Exception:
            continue
        if np.linalg.norm(sol.fun) < 1e-5:
            E = float(e_total(jnp.asarray(sol.x), ZERO_C8, U, kx, ky, kz))
            wmin = float(min_boson_eig(jnp.asarray(sol.x), U, kx, ky, kz))
            out.append({"p": sol.x, "E": E, "wmin": wmin,
                        "rnorm": float(np.linalg.norm(sol.fun))})
    out.sort(key=lambda s: s["E"])
    return out


def seed_bank(U):
    seeds = []
    rng = np.random.default_rng(42)
    base = np.zeros(18)
    base[16] = -0.5
    # SL-like: pairing in all channels
    s = base.copy(); s[8:12] = [0.5, 0.4, 0.15, 0.15]; s[12:16] = [0.35, 0.3, 0.1, 0.1]
    seeds.append(s)
    # pairing dominated by xy / z channels only
    s = base.copy(); s[8] = 0.6; s[12] = 0.4; seeds.append(s)
    s = base.copy(); s[9] = 0.6; s[13] = 0.4; seeds.append(s)
    # hopping-dominated (chi-type, FL*-like)
    s = base.copy(); s[0:4] = [0.5, 0.4, 0.1, 0.1]; s[4:8] = [0.4, 0.3, 0.1, 0.1]
    seeds.append(s)
    # mixed random
    for _ in range(2):
        s = base.copy()
        s[0:16] = rng.uniform(-0.1, 0.5, 16) * (np.abs(rng.uniform(size=16)) > 0.3)
        seeds.append(s)
    return seeds


if __name__ == "__main__":
    band_check()
    atomic_check()
    if len(sys.argv) > 2 and sys.argv[2] == "solve":
        log(f"Stage B: U sweep, nk={NK}")
        prev = []
        for U in np.arange(4.0, 0.399, -0.2):
            sols = solve_uncondensed(U, [s["p"] for s in prev[:2]] + seed_bank(U))
            if sols:
                b = sols[0]
                p = b["p"]
                log(f"U={U:.1f}: E={b['E']:+.5f} wminHb={b['wmin']:+.4f} rn={b['rnorm']:.1e} "
                    f"Db={np.round(p[8:12],3)} Df={np.round(p[12:16],3)} "
                    f"chib={np.round(p[0:4],3)} chif={np.round(p[4:8],3)} lam={p[16]:+.3f}")
                prev = sols
            else:
                log(f"U={U:.1f}: NO uncondensed solution converged")
                prev = []
        np.save("sg135_last.npy", np.array([s["p"] for s in prev]) if prev else np.zeros(0))
