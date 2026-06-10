"""Trace the KMH phase boundaries U_c1(lso) (SC->SL or SC->DM) and U_c2(lso)
(SL->DM) by bisection at a set of lambda_SO values. Much cheaper than a full
grid sweep and directly comparable to the paper's Fig. 1 lines.

Output: kmh_boundaries.json
"""
import sys, json, time
import numpy as np
from kmh_solver import make_kgrid
from run_kmh import ground_state, phase_at, bisect_boundary
import run_kmh

t0 = time.time()
log = lambda *a: (print(f"[{time.time()-t0:7.1f}s]", *a), sys.stdout.flush())

if __name__ == "__main__":
    res = {"lsos": [], "Uc_low": [], "Uc_high": [], "phases": []}
    row_cache = {}  # U -> solutions from previous lso row (lso-continuation seeds)
    for lso in (0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30):
        # coarse scan with continuation (in U and in lso) to locate transitions
        prev, rows = (), []
        new_cache = {}
        for U in np.arange(1.0, 3.2001, 0.1):
            uk = round(float(U), 2)
            lso_seeds = row_cache.get(uk, ())
            # perturb prior-row seeds into the primed directions (both sign branches)
            pert = []
            for s in lso_seeds:
                for eps in (0.08, -0.08):
                    q = np.array(s, float)
                    q[6:8] = np.where(np.abs(q[6:8]) < 0.02, eps, q[6:8])
                    pert.append(q)
            ph, cs = phase_at(U, lso, tuple(prev) + tuple(pert))
            rows.append((U, ph, cs))
            if cs:
                prev = tuple(c[2]["p"] for c in cs if "p" in c[2])[:3]
                new_cache[uk] = [c[2]["p"] for c in cs if "p" in c[2]][:2]
        row_cache = new_cache
        seq = [(U, ph) for U, ph, _ in rows]
        log(f"lso={lso:.2f} sequence: " + "".join(p[1][0] if p[1] != 'none' else '?' for p in seq))
        bounds = []
        for i in range(len(rows) - 1):
            if rows[i][1] != rows[i+1][1] and rows[i][1] != "none" and rows[i+1][1] != "none":
                Uc = bisect_boundary(rows[i][0], rows[i+1][0], lso, rows[i][1],
                                     seeds_lo=[c[2]["p"] for c in rows[i][2] if "p" in c[2]][:2],
                                     seeds_hi=[c[2]["p"] for c in rows[i+1][2] if "p" in c[2]][:2])
                bounds.append((rows[i][1], rows[i+1][1], round(float(Uc), 4)))
                log(f"   {rows[i][1]}->{rows[i+1][1]} at U_c={Uc:.3f}")
        res["lsos"].append(lso)
        res["phases"].append([p[1] for p in seq])
        res["Uc_low"].append(bounds[0][2] if bounds else None)
        res["Uc_high"].append(bounds[1][2] if len(bounds) > 1 else (bounds[0][2] if bounds and bounds[0][1] == "DM" else None))
        with open("kmh_boundaries.json", "w") as f:
            json.dump(res, f, indent=1)
    log("saved kmh_boundaries.json")
    log("PAPER: Uc_low(0)=1.57, Uc_low(0.1)=1.93, Uc_low(0.22)=2.35, Uc_low(0.3)=2.72; Uc_high(0..0.1)=1.93")
