"""KMH gate figure: our slave-boson phase boundaries vs Wen et al. Fig. 1.

Boundary values hand-extracted from the bisection log (kmh_bounds.log):
the SC upper edge is SC->SL for lso <= 0.10 and SC->DM beyond the triple
point; spurious low-U cells where the SC branch failed to converge (isolated
'DM' at U=1.0-1.3 for lso >= 0.1) are excluded. SL-DM line = 1.982 flat.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

lsos_sc = [0.0, 0.05, 0.10, 0.15, 0.20]
uc_sc = [1.584, 1.624, 1.780, 1.982, 2.013]   # SC upper edge (bisections)
lsos_sl = [0.0, 0.05, 0.10]
uc_sl = [1.982, 1.982, 1.982]                  # SL-DM (flat)

fig, ax = plt.subplots(figsize=(6.2, 5.0))
ax.plot([0.0, 0.10, 0.22, 0.30], [1.57, 1.93, 2.35, 2.72], "k--", lw=2,
        label="paper: SC upper boundary")
ax.plot([0.0, 0.10], [1.93, 1.93], "k-.", lw=2,
        label="paper: spin liquid / dimer boundary")
ax.plot(lsos_sc, uc_sc, "o-", color="#27408b", lw=2,
        label="this work: SC upper boundary")
ax.plot(lsos_sl, uc_sl, "s-", color="#b22222", lw=2,
        label="this work: SL/DM boundary (ends at triple point)")
ax.annotate("SC = superconductor", (0.055, 1.32), fontsize=11, fontweight="bold")
ax.annotate("SL = spin liquid", (0.012, 1.80), fontsize=9, fontweight="bold")
ax.annotate("DM = dimer", (0.13, 2.45), fontsize=11, fontweight="bold")
ax.set_xlabel(r"$\lambda_{SO}/t$", fontsize=12)
ax.set_ylabel(r"$U/t$", fontsize=12)
ax.set_ylim(1.2, 3.0)
ax.set_xlim(-0.005, 0.31)
ax.legend(fontsize=9, loc="upper left")
ax.set_title("KMH slave-boson phase boundaries: this work vs Wen et al. (2011)")
fig.tight_layout()
fig.savefig("kmh_boundaries.png", dpi=160)
print("saved kmh_boundaries.png")
