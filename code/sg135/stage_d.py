"""Stage D: hunt for parameters where the FULLY-GAPPED mixed-channel Z2 state
is the GLOBAL slave-boson minimum (user directive: relax assumptions / tune the
microscopic Hamiltonian to reach the topological state).

Strategy: at fixed U, sweep t_z downward from 0.5 (the z-only nodal state's
advantage should shrink as the z-chains weaken); at each (U, t_z) converge
three branches by continuation:
  (a) z-only nodal, (b) xy-only nodal, (c) mixed xy+z (gapped candidate),
plus the condensed (SC) state, and compare energies. Also report the mixed
state's minimal BdG gap and boson gap so 'gapped' is verified at the winner.

Usage: python stage_d.py <U> [nk]
"""
import sys, time
import numpy as np
import jax
import jax.numpy as jnp
from scipy.optimize import least_squares
from sg135_solver import make_kgrid, e_total, min_boson_eig, fermion_blocks

U = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
NK = int(sys.argv[2]) if len(sys.argv) > 2 else 12
kx, ky, kz = make_kgrid(NK)
Z8 = jnp.zeros(8)
t0 = time.time()
log = lambda *a: (print(f"[{time.time()-t0:6.1f}s]", *a), sys.stdout.flush())


def resid(p, U, ts):
    r = jax.grad(e_total, argnums=0)(p, Z8, U, kx, ky, kz, ts)
    wmin = min_boson_eig(p, U, kx, ky, kz, ts)
    pen = 50.0 * jnp.clip(-wmin, 0.0)
    return jnp.concatenate([r, jnp.array([pen]), 1e-3 * jnp.array([p[17] - U / 2])])


resid_j = jax.jit(resid, static_argnames=("ts",))
jac_j = jax.jit(jax.jacfwd(resid, argnums=0), static_argnames=("ts",))


def solve(seed, ts):
    rf = lambda p: np.nan_to_num(np.asarray(resid_j(jnp.asarray(p), U, ts)), nan=1e3)
    jf = lambda p: np.nan_to_num(np.asarray(jac_j(jnp.asarray(p), U, ts)), nan=0.0)
    sol = least_squares(rf, np.array(seed), jac=jf, method="trf",
                        xtol=1e-13, ftol=1e-13, max_nfev=500)
    rn = np.linalg.norm(sol.fun)
    if rn > 1e-5 or np.abs(sol.x[:16]).max() < 0.02:
        return None
    p = sol.x
    wmin = float(min_boson_eig(jnp.asarray(p), U, kx, ky, kz, ts))
    if wmin < -1e-6:
        return None
    E = float(e_total(jnp.asarray(p), Z8, U, kx, ky, kz, ts))
    return {"p": p, "E": E, "wmin": wmin, "rnorm": rn}


def fermion_min_gap(p, ts, nk=16):
    ka, kb, kc = make_kgrid(nk)
    xi, dl = fermion_blocks(jnp.asarray(p), ka, kb, kc, ts)
    top = np.concatenate([np.asarray(xi), np.asarray(dl)], axis=-1)
    bot = np.concatenate([-np.asarray(dl).conj(), -np.asarray(xi).conj()], axis=-1)
    ev = np.linalg.eigvalsh(np.concatenate([top, bot], axis=-2))
    return float(np.abs(ev).min())


def seed_vec(dxy, dz, lam):
    s = np.zeros(18)
    s[8], s[9] = 0.7 * dxy, 0.7 * dz
    s[12], s[13] = dxy, dz
    s[16], s[17] = lam, U / 2
    return s


if __name__ == "__main__":
    branches = {"z-only": seed_vec(0.0, 0.55, -0.02),
                "xy-only": seed_vec(0.45, 0.0, -0.02),
                "mixed": seed_vec(0.35, 0.30, -0.02)}
    log(f"U={U}, t_z sweep")
    for tz in (0.5, 0.4, 0.3, 0.25, 0.2, 0.15, 0.1):
        ts = (1.0, tz, 0.3, 0.3)
        row = {}
        for name in list(branches):
            r = solve(branches[name]["p"] if isinstance(branches[name], dict) else branches[name], ts)
            if r is not None:
                # classify actual structure
                dxy, dz = abs(r["p"][12]), abs(r["p"][13])
                kind = "mixed" if min(dxy, dz) > 0.03 else ("z" if dz > dxy else "xy")
                row[name] = (r, kind)
                branches[name] = r  # continuation
        msg = f"tz={tz:.2f}: "
        best = None
        for name, (r, kind) in row.items():
            msg += f"{name}->{kind} E={r['E']:+.5f}  "
            if best is None or r["E"] < best[1]["E"]:
                best = (name, r, kind)
        if best and best[2] == "mixed":
            g = fermion_min_gap(best[1]["p"], ts)
            msg += f"| GLOBAL=MIXED gap={g:.4f} wminB={best[1]['wmin']:+.4f} <== TOPOLOGICAL CANDIDATE"
            np.save(f"sol_mixed_U{U}_tz{tz}.npy", best[1]["p"])
        elif best:
            msg += f"| global={best[2]}"
        log(msg)
