"""SG135 phase-energy plot: parse sg135_run.log (Stage B uncondensed + Stage C
condensed rows) and plot E(U) for: z-nodal uncondensed, condensed (SC), atomic
Mott baseline E=0, plus the mixed gapped branch where available (stage_d files
or explicit points)."""
import re, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

log = open(sys.argv[1] if len(sys.argv) > 1 else "sg135_run.log").read()
unc, cond = {}, {}
for m in re.finditer(r"U=(\d+\.\d+): E=([+-]\d+\.\d+) wminHb", log):
    unc[float(m.group(1))] = float(m.group(2))
for m in re.finditer(r"U=(\d+\.\d+): COND E=([+-]\d+\.\d+) \|c\|max=(\d+\.\d+)", log):
    U, E, c = float(m.group(1)), float(m.group(2)), float(m.group(3))
    if c > 0.02:
        cond[U] = E

fig, ax = plt.subplots(figsize=(6.6, 5.0))
if unc:
    u = sorted(unc)
    ax.plot(u, [unc[x] for x in u], "o-", color="#27408b", lw=2,
            label="uncondensed (nodal z-paired liquid)")
if cond:
    u = sorted(cond)
    ax.plot(u, [cond[x] for x in u], "s-", color="#b22222", lw=2,
            label="condensed (SC)")
# mixed gapped points (manual: from local/stage_d runs; (U, E) pairs)
mixed_pts = [(1.0, -0.19390)]
if len(sys.argv) > 2:
    d = np.load(sys.argv[2])
    mixed_pts = [tuple(r) for r in d]
ax.plot(*zip(*mixed_pts), "D", color="#2e8b57", ms=8,
        label="mixed-channel gapped Z$_2$ (metastable)")
ax.axhline(0, color="k", lw=1, ls=":", label="atomic Mott baseline")
ax.set_xlabel("U / $t_{xy}$", fontsize=12)
ax.set_ylabel("$E_g$ per cell / $t_{xy}$", fontsize=12)
ax.set_title("SG135 slave-boson phases at half filling ($t_z$=0.5, $t_1$=$t_2$=0.3)")
ax.legend(fontsize=9)
fig.tight_layout()
fig.savefig("sg135_phases.png", dpi=160)
print("saved sg135_phases.png; uncond pts:", len(unc), "cond pts:", len(cond))
