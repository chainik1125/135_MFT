"""Plot the SG135 mean-field Bogoliubov band structure (fermion + boson)
from sg135_bands.npz (produced by verify_sg135.py)."""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

d = np.load(sys.argv[1] if len(sys.argv) > 1 else "sg135_bands.npz", allow_pickle=True)
ks, ticks, ef, eb, U = d["ks"], d["ticks"], d["ef"], d["eb"], float(d["U"])
labels = ["$\\Gamma$", "X", "M", "$\\Gamma$", "Z", "R", "A", "Z", "X", "R", "M", "A"]
xt = list(ticks[:-1]) + [len(ks) - 1]

fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True,
                         gridspec_kw={"height_ratios": [1.2, 1]})
x = np.arange(len(ks))
for b in range(ef.shape[1]):
    axes[0].plot(x, ef[:, b], lw=0.8, color="#27408b")
axes[0].axhline(0, color="k", lw=0.5, ls=":")
axes[0].set_ylabel("spinon BdG energy $/t_{xy}$")
axes[0].set_title(f"SG135 slave-boson mean field, U={U}: quasiparticle bands")

if eb is not None and eb.size:
    for b in range(eb.shape[1]):
        axes[1].plot(x, eb[:, b], lw=0.8, color="#8b2740")
    axes[1].axhline(0, color="k", lw=0.5, ls=":")
    axes[1].set_ylabel("chargon (boson) energy $/t_{xy}$")

for ax in axes:
    for t in ticks:
        ax.axvline(t, color="0.8", lw=0.5)
    ax.set_xticks(xt[:len(labels)])
    ax.set_xticklabels(labels[:len(xt)])
ax.set_xlim(0, len(ks) - 1)
fig.tight_layout()
out = "sg135_mf_bands.png"
fig.savefig(out, dpi=160)
print("saved", out)
print("min |fermion band| over path:", np.abs(ef).min().round(5))
if eb is not None and eb.size:
    pos = np.where(eb > 1e-9, eb, np.inf)
    print("min positive boson band over path:", pos.min().round(5))
