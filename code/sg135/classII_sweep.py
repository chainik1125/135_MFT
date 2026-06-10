"""Production confirmation: gapped PSG-twisted branch vs nodal branch vs SC,
U in [0.4, 6], at nk given on the command line. Saves classII_sweep.npz."""
import sys, time
import numpy as np
import jax
import jax.numpy as jnp
from scipy.optimize import least_squares
from sg135_solver import make_kgrid, e_total, min_boson_eig

NK = int(sys.argv[1]) if len(sys.argv) > 1 else 14
kx, ky, kz = make_kgrid(NK)
Z8 = jnp.zeros(8)
t0 = time.time()
log = lambda *a: (print(f"[{time.time()-t0:7.1f}s]", *a), sys.stdout.flush())


def make_funcs(U):
    def resid(p):
        r = jax.grad(e_total, argnums=0)(p, Z8, U, kx, ky, kz)
        w = min_boson_eig(p, U, kx, ky, kz)
        return jnp.concatenate([r, jnp.array([50.0 * jnp.clip(-w, 0.0)]),
                                1e-3 * jnp.array([p[17] - U / 2])])
    rj = jax.jit(resid)
    jj = jax.jit(jax.jacfwd(resid))
    return rj, jj


def solve(rj, jj, U, seed):
    rf = lambda p: np.nan_to_num(np.asarray(rj(jnp.asarray(p))), nan=1e3)
    jf = lambda p: np.nan_to_num(np.asarray(jj(jnp.asarray(p))), nan=0.0)
    sol = least_squares(rf, np.array(seed), jac=jf, method="trf",
                        xtol=1e-13, ftol=1e-13, max_nfev=500)
    rn = np.linalg.norm(sol.fun)
    p = sol.x
    if rn > 1e-5 or np.abs(np.concatenate([p[:16], p[18:]])).max() < 0.02:
        return None
    w = float(min_boson_eig(jnp.asarray(p), U, kx, ky, kz))
    if w < -1e-6:
        return None
    return float(e_total(jnp.asarray(p), Z8, U, kx, ky, kz)), p, w


sG = np.zeros(22); sG[8] = 0.25; sG[12] = 0.35; sG[20] = 0.4; sG[21] = 0.55; sG[16] = -0.03
sZ = np.zeros(22); sZ[9] = 0.45; sZ[13] = 0.55; sZ[16] = -0.03
rows = []
for U in np.arange(0.4, 6.001, 0.2):
    rj, jj = make_funcs(float(U))
    sG[17] = sZ[17] = U / 2
    rG = solve(rj, jj, U, sG)
    rZ = solve(rj, jj, U, sZ)
    eG = rG[0] if rG else np.nan
    eZ = rZ[0] if rZ else np.nan
    if rG: sG = rG[1]
    if rZ: sZ = rZ[1]
    gw = rG[2] if rG else np.nan
    log(f"U={U:.1f}: E_gapped={eG:+.5f} (wminB={gw:+.3f})  E_nodal={eZ:+.5f}  "
        f"winner={'GAPPED' if (not np.isnan(eG) and (np.isnan(eZ) or eG < eZ)) else 'nodal/none'}")
    rows.append((U, eG, eZ))
    np.savez("classII_sweep.npz", rows=np.array(rows))
log("done")
