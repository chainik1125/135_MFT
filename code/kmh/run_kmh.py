"""Runner: validate the KMH slave-boson solver against the paper, then sweep.

Validation targets (Wen et al. PRB 84, 235149):
  V1: U=1.8t, lso=0      -> SL ground state, single-particle gap 0.57t
  V2: U=1.9t, lso=0.02t  -> double occupancy ~ 0.23
  V3: lso=0              -> U_c1 (SC->SL) ~ 1.5t, U_c2 (SL->DM) ~ 1.9t
Then: full (lambda_SO, U) phase diagram sweep, results to npz + png.
"""
import sys, json, time
import numpy as np
import jax.numpy as jnp
from kmh_solver import (make_kgrid, solve_point, solve_dimer, classify,
                        observables, e_total, SEEDS)

t0 = time.time()
k1, k2 = make_kgrid(int(sys.argv[1]) if len(sys.argv) > 1 else 120)
log = lambda *a: (print(f"[{time.time()-t0:7.1f}s]", *a), sys.stdout.flush())


def ground_state(U, lso, extra_seeds=()):
    sols = solve_point(U, lso, k1, k2, extra_seeds=extra_seeds)
    dm = solve_dimer(U)
    cands = [(s["E"], classify(s["p"]), s) for s in sols]
    if dm is not None:
        cands.append((dm["E"], "DM", dm))
    if not cands:
        return None
    cands.sort(key=lambda c: c[0])
    return cands


def phase_at(U, lso, extra_seeds=()):
    cs = ground_state(U, lso, extra_seeds)
    if cs is None:
        return "none", None
    return cs[0][1], cs


def bisect_boundary(U_lo, U_hi, lso, ph_lo, tol=2e-3, seeds_lo=(), seeds_hi=()):
    """Bisect the ground-state phase change between U_lo (phase ph_lo) and U_hi."""
    lo, hi = U_lo, U_hi
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        ph, cs = phase_at(mid, lso, tuple(seeds_lo) + tuple(seeds_hi))
        if ph == ph_lo:
            lo = mid
            if cs:
                seeds_lo = [c[2]["p"] for c in cs if "p" in c[2]][:2]
        else:
            hi = mid
            if cs:
                seeds_hi = [c[2]["p"] for c in cs if "p" in c[2]][:2]
    return 0.5 * (lo + hi)


if __name__ == "__main__":
    # ---------- V1 ----------
    log("V1: U=1.8, lso=0")
    cs = ground_state(1.8, 0.0)
    for E, ph, s in cs:
        log(f"   {ph:8s} E={E:+.6f}" + (f" p={np.round(s['p'],4)}" if 'p' in s else f" q={np.round(s['q'],4)}"))
    E, ph, s = cs[0]
    if ph == "SL":
        gap, docc = observables(s["p"], 1.8, 0.0, k1, k2)
        log(f"   -> SL gap = {gap:.4f} t (paper: 0.57t), Docc = {docc:.4f}")
    else:
        log(f"   -> WARNING ground state is {ph}, not SL")

    # ---------- V2 ----------
    log("V2: U=1.9, lso=0.02")
    cs = ground_state(1.9, 0.02)
    E, ph, s = cs[0]
    if "p" in s:
        gap, docc = observables(s["p"], 1.9, 0.02, k1, k2)
        log(f"   {ph} E={E:+.6f}  Docc = {docc:.4f} (paper: ~0.23)")
    else:
        log(f"   ground state DM E={E:+.6f}")

    # ---------- V3: scan U at lso=0 with continuation, then bisect ----------
    log("V3: U scan at lso=0")
    rows, prev = [], ()
    for U in np.arange(1.0, 2.4001, 0.05):
        cs = ground_state(U, 0.0, extra_seeds=prev)
        E, ph, s = cs[0]
        rows.append((U, ph, E, cs))
        prev = tuple(c[2]["p"] for c in cs if "p" in c[2])[:3]
        log(f"   U={U:.2f}  {ph:8s} E={E:+.6f}")
    for i in range(len(rows) - 1):
        if rows[i][1] != rows[i+1][1]:
            Uc = bisect_boundary(rows[i][0], rows[i+1][0], 0.0, rows[i][1],
                                 seeds_lo=[c[2]["p"] for c in rows[i][3] if "p" in c[2]][:2],
                                 seeds_hi=[c[2]["p"] for c in rows[i+1][3] if "p" in c[2]][:2])
            log(f"   BOUNDARY {rows[i][1]}->{rows[i+1][1]} at U_c = {Uc:.3f}"
                f"  (paper: SC->SL 1.5, SL->DM 1.9)")

    if len(sys.argv) > 2 and sys.argv[2] == "sweep":
        log("FULL SWEEP")
        Us = np.arange(0.2, 3.001, 0.05)
        lsos = np.arange(0.0, 0.3001, 0.02)
        phase = np.empty((len(lsos), len(Us)), dtype=object)
        energy = np.full((len(lsos), len(Us)), np.nan)
        for i, lso in enumerate(lsos):
            prev = ()
            for j, U in enumerate(Us):
                cs = ground_state(U, lso, extra_seeds=prev)
                if cs is None:
                    phase[i, j] = "none"
                    continue
                E, ph, s = cs[0]
                phase[i, j], energy[i, j] = ph, E
                prev = tuple(c[2]["p"] for c in cs if "p" in c[2])[:3]
            log(f"  lso={lso:.2f} done: " + "".join({"SC":"S","SL":"L","DM":"D","trivial":".","none":"?"}[p] for p in phase[i]))
        np.savez("kmh_sweep.npz", Us=Us, lsos=lsos,
                 phase=np.vectorize({"SC":0,"SL":1,"DM":2,"trivial":3,"none":4}.get)(phase),
                 energy=energy)
        log("saved kmh_sweep.npz")
