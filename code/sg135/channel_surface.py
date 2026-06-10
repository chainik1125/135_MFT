"""Channel-competition energy surface for the SG135 slave-boson mean field.

For fixed spinon pairings (Delta_f_xy, Delta_f_z) on a 2D grid, stationarize
the remaining parameters (Delta_b_xy, Delta_b_z, lam; mu tethered to U/2;
chi and channels 1,2 fixed to zero - they vanish in all converged solutions),
then plot E(Delta_f_xy, Delta_f_z). Two valleys along the axes separated by a
ridge demonstrate that channel mixing is energetically disfavored - i.e. the
fully-gapped multi-channel Z2 state is not a self-consistent minimum.

Usage: python channel_surface.py <U> [nk] [ngrid]
"""
import sys, time
import numpy as np
import jax
import jax.numpy as jnp
from scipy.optimize import least_squares
from sg135_solver import make_kgrid, e_total, min_boson_eig

U = float(sys.argv[1]) if len(sys.argv) > 1 else 4.0
NK = int(sys.argv[2]) if len(sys.argv) > 2 else 10
NG = int(sys.argv[3]) if len(sys.argv) > 3 else 21
kx, ky, kz = make_kgrid(NK)
Z8 = jnp.zeros(8)

IDX_FREE = jnp.array([8, 9, 16])  # Delta_b_xy, Delta_b_z, lam


def build_p(free, dxy, dz):
    p = jnp.zeros(18)
    p = p.at[12].set(dxy).at[13].set(dz)
    p = p.at[8].set(free[0]).at[9].set(free[1]).at[16].set(free[2])
    p = p.at[17].set(U / 2.0)
    return p


def resid(free, dxy, dz):
    p = build_p(jnp.asarray(free), dxy, dz)
    g = jax.grad(e_total, argnums=0)(p, Z8, U, kx, ky, kz)
    wmin = min_boson_eig(p, U, kx, ky, kz)
    pen = 50.0 * jnp.clip(-wmin, 0.0)
    return jnp.concatenate([g[jnp.array([8, 9, 16])], jnp.array([pen])])


resid_j = jax.jit(resid)
jac_j = jax.jit(jax.jacfwd(resid, argnums=0))

t0 = time.time()
grid = np.linspace(0.0, 0.75, NG)
E = np.full((NG, NG), np.nan)
free0 = np.array([0.05, 0.05, -0.05])
for i, dxy in enumerate(grid):
    fseed = free0.copy()
    for j, dz in enumerate(grid):
        rf = lambda f: np.nan_to_num(np.asarray(resid_j(jnp.asarray(f), dxy, dz)),
                                     nan=1e3, posinf=1e3, neginf=-1e3)
        jf = lambda f: np.nan_to_num(np.asarray(jac_j(jnp.asarray(f), dxy, dz)),
                                     nan=0.0, posinf=0.0, neginf=0.0)
        sol = least_squares(rf, fseed, jac=jf,
                            method="trf", xtol=1e-12, ftol=1e-12, max_nfev=200)
        if np.linalg.norm(sol.fun) < 1e-6:
            E[i, j] = float(e_total(build_p(jnp.asarray(sol.x), dxy, dz), Z8, U, kx, ky, kz))
            fseed = sol.x
    print(f"row {i+1}/{NG} done [{time.time()-t0:.0f}s]", flush=True)
np.savez("channel_surface.npz", grid=grid, E=E, U=U)
print("saved channel_surface.npz")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(6.4, 5.4))
vmax = np.nanpercentile(E, 95)
pc = ax.pcolormesh(grid, grid, E.T, shading="nearest", cmap="viridis")
fig.colorbar(pc, label="$E_g$ per cell $/t_{xy}$")
cs = ax.contour(grid, grid, E.T, levels=12, colors="w", linewidths=0.5)
ax.set_xlabel(r"$\Delta_f^{xy}$")
ax.set_ylabel(r"$\Delta_f^{z}$")
ax.set_title(f"SG135 MF energy surface, U={U}: channel competition")
fig.tight_layout()
fig.savefig("channel_surface.png", dpi=160)
print("saved channel_surface.png")
