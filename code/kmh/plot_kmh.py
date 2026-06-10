"""Phase diagram figure: our slave-boson KMH reproduction vs paper boundaries.

Reads kmh_sweep.npz (phase codes: 0=SC 1=SL 2=DM 3=trivial 4=none) and draws
the (lambda_SO/t, U/t) map with the Wen et al. PRB 84, 235149 Fig. 1 phase
boundaries overlaid as dashed reference lines (digitized anchor points).
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

d = np.load("kmh_sweep.npz")
Us, lsos, phase = d["Us"], d["lsos"], d["phase"]

fig, ax = plt.subplots(figsize=(6.4, 5.2))
cmap = ListedColormap(["#4878cf", "#6acc65", "#d65f5f", "#cccccc", "#ffffff"])
ax.pcolormesh(lsos, Us, phase.T, cmap=cmap, vmin=-0.5, vmax=4.5, shading="nearest")

# paper Fig. 1 boundaries (digitized anchors from phase_diagram_at_half_filling.pdf)
ax.plot([0.0, 0.10], [1.93, 1.93], "k--", lw=1.8)            # SL-DM (horizontal)
ax.plot([0.0, 0.10, 0.22, 0.30], [1.57, 1.93, 2.35, 2.72],   # SC-SL / SC-DM rising
        "k--", lw=1.8, label="Wen et al. (2011) boundaries")

for lab, x, y, c in [("SC", 0.15, 1.0, "w"), ("SL", 0.012, 1.75, "k"), ("DM", 0.08, 2.6, "w")]:
    ax.text(x, y, lab, fontsize=16, fontweight="bold", color=c)
ax.set_xlabel(r"$\lambda_{SO}/t$", fontsize=13)
ax.set_ylabel(r"$U/t$", fontsize=13)
ax.set_xlim(0, 0.30)
ax.set_ylim(Us.min(), 3.0)
ax.legend(loc="upper left", fontsize=9)
ax.set_title("KMH slave-boson phase diagram: this work (color) vs paper (dashed)")
fig.tight_layout()
fig.savefig("kmh_phase_diagram.png", dpi=160)
print("saved kmh_phase_diagram.png")
