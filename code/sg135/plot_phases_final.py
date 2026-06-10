"""Final SG135 phase-energy figure.

Curves: gapped Z2 (PSG-twisted) and nodal z-state from classII_sweep.npz
(nk=12); SC points from condensed solves; z-dimer VBS curve from the
decoupled-dimer formula (t_b = 0.25); atomic Mott baseline 0.
Usage: python plot_phases_final.py classII_sweep.npz
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "kmh"))
import numpy as np
import jax.numpy as jnp
from scipy.optimize import least_squares
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from kmh_solver import e_dimer, grad_dimer

d = np.load(sys.argv[1] if len(sys.argv) > 1 else "classII_sweep.npz")
rows = d["rows"]  # (U, E_gapped, E_nodal)


def e_vbs(U, tb=0.25):
    best = None
    for s0 in ([0.5, 0.4, -0.2], [0.9, 0.2, -0.5], [0.3, 0.45, -0.05]):
        def res(q):
            r = np.asarray(grad_dimer(jnp.asarray(q), U, tb))
            pen = max(0.0, 4 * tb**2 * q[1]**2 - (U - 2 * q[2])**2)
            return np.concatenate([r, [10 * pen]])
        sol = least_squares(res, np.array(s0), method="trf", xtol=1e-14, ftol=1e-14,
                            max_nfev=400)
        if np.linalg.norm(sol.fun) < 1e-7:
            E = float(e_dimer(jnp.asarray(sol.x), U, tb))
            best = E if best is None else min(best, E)
    return 2 * best if best is not None else np.nan


U = rows[:, 0]
fig, ax = plt.subplots(figsize=(7.0, 5.2))
ax.plot(U, rows[:, 1], "o-", color="#2e8b57", lw=2, ms=4,
        label="gapped Z$_2$ insulator (PSG-twisted) — symmetric")
ax.plot(U, rows[:, 2], "s-", color="#27408b", lw=1.6, ms=4,
        label="nodal z-paired liquid — symmetric")
vbs = np.array([e_vbs(u) for u in U])
ax.plot(U, vbs, "^--", color="#777777", lw=1.4, ms=4,
        label="z-dimer VBS — breaks translation (MFT over-favors)")
# transition region recomputed at nk=12 (transition_nk12.npy: U, E_gapped, E_SC)
try:
    tr = np.load("transition_nk12.npy")
    ax.plot(tr[:, 0], tr[:, 2], "*-", color="#b22222", ms=11, lw=1.2,
            label="SC (chargon condensate)")
    sel = np.isfinite(tr[:, 1]) & (tr[:, 0] < 1.0)
    ax.plot(tr[sel, 0], tr[sel, 1], "o-", color="#2e8b57", ms=4, lw=2)
except FileNotFoundError:
    pass
ax.axhline(0, color="k", lw=1, ls=":", label="atomic Mott")
ax.set_xlabel("U / $t_{xy}$", fontsize=12)
ax.set_ylabel("$E_g$ per cell / $t_{xy}$", fontsize=12)
ax.set_title(r"SG135 slave-boson states at $\nu=4$ (half filling), nk=12")
ax.legend(fontsize=8.5, loc="lower right")
ax.set_xlim(0.3, 6.05)
fig.tight_layout()
fig.savefig("sg135_phases.png", dpi=160)
print("saved sg135_phases.png")
