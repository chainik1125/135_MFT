"""KMH gate figure: our slave-boson phase boundaries vs Wen et al. Fig. 1."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

with open("kmh_boundaries.json") as f:
    d = json.load(f)

fig, ax = plt.subplots(figsize=(6.2, 5.0))
# paper boundaries (digitized anchors)
ax.plot([0.0, 0.10, 0.22, 0.30], [1.57, 1.93, 2.35, 2.72], "k--", lw=2,
        label="paper: SC boundary")
ax.plot([0.0, 0.10], [1.93, 1.93], "k-.", lw=2, label="paper: SL–DM")
lsos = np.array(d["lsos"], float)
lo = np.array([np.nan if v is None else v for v in d["Uc_low"]], float)
hi = np.array([np.nan if v is None else v for v in d["Uc_high"]], float)
ax.plot(lsos, lo, "o-", color="#27408b", lw=2, label="this work: SC boundary")
ax.plot(lsos, hi, "s-", color="#b22222", lw=2, label="this work: SL–DM")
ax.set_xlabel(r"$\lambda_{SO}/t$", fontsize=12)
ax.set_ylabel(r"$U/t$", fontsize=12)
ax.set_ylim(1.3, 3.0)
ax.legend(fontsize=9, loc="upper left")
ax.set_title("KMH slave-boson phase boundaries: reproduction vs paper")
fig.tight_layout()
fig.savefig("kmh_boundaries.png", dpi=160)
print("saved kmh_boundaries.png")
