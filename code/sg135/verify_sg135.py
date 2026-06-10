"""Stretch-goal verification for a converged SG135 slave-boson mean field:
  (a) explicit symmetry check of the MF BdG Hamiltonian at all HSPs (+ random k)
      under the P4_2/mbc generators and time reversal,
  (b) Bogoliubov band structure along the full HSP path (fermion + boson),
  (c) gap / non-degeneracy audit: min fermion BdG gap, min boson para-gap,
      no condensate, boson Gamma-multiplet structure (Class I check).

Generator reps (notes/theory/sg135_factorization.md Sec. 4.1; basis ordering
here is spin (x) mu (x) tau for the 8-orbital space):
  U(screw 4_2) = exp(-i pi sz/4) (x) mu^x (x) tau^0 , k -> (-ky, kx, kz)
     [sense fixed numerically; the report's +i pi/4 matrix is the C4z^- sense]
  U(C2x|1/2 1/2 0) = i sx (x) mu^0 (x) tau^x       , k -> (kx, -ky, -kz)
  U(I) = 1                                          , k -> -k
  T = (i sy (x) 1_4) K                              , k -> -k
Checks: U xi(k) U^+ = xi(Wk) and U Delta(k) U^T = Delta(Wk) (singlet, real
channels -> trivial PSG phase); TR: (isy) xi(-k)^* (isy)^+ = xi(k), and
(isy) Delta(-k)^* (isy)^T = Delta(k).
Usage: python verify_sg135.py <solution.npy> [nk_gap]
  solution.npy: length-18 p vector or length-26 (p + condensates).
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "common"))
import numpy as np
import jax.numpy as jnp
from sg135_solver import fermion_blocks, boson_blocks, TS_DEFAULT, make_kgrid
from bdg import boson_e0

I2 = np.eye(2)
X = np.array([[0, 1], [1, 0]], complex)
Y = np.array([[0, -1j], [1j, 0]])
Z = np.diag([1.0, -1.0]).astype(complex)


def kron3(a, b, c):
    return np.kron(a, np.kron(b, c))


ROT4 = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], float)   # k -> (-ky, kx, kz)
GENS = {
    "screw_42": (kron3(np.diag([np.exp(-1j*np.pi/4), np.exp(1j*np.pi/4)]), X, I2), ROT4),
    "C2x": (1j * kron3(X, I2, X), np.diag([1.0, -1.0, -1.0])),
    "inversion": (np.eye(8, dtype=complex), -np.eye(3)),
}
# derived: C2z = screw^2 (k -> (-kx,-ky,kz)); glide_b = I*C2z*C2x etc. -- products
GENS["C2z"] = (GENS["screw_42"][0] @ GENS["screw_42"][0], ROT4 @ ROT4)
GENS["glide_b(mx)"] = (GENS["inversion"][0] @ GENS["C2z"][0] @ GENS["C2x"][0],
                       (-np.eye(3)) @ (ROT4 @ ROT4) @ GENS["C2x"][1])
GENS["glide_c(m110)"] = (GENS["inversion"][0] @ GENS["screw_42"][0] @ GENS["C2x"][0],
                         (-np.eye(3)) @ ROT4 @ GENS["C2x"][1])

HSP = {"G": (0, 0, 0), "X": (0, np.pi, 0), "M": (np.pi, np.pi, 0),
       "Z": (0, 0, np.pi), "R": (0, np.pi, np.pi), "A": (np.pi, np.pi, np.pi)}


def blocks_at(p, k):
    kx, ky, kz = (jnp.array([k[0]]), jnp.array([k[1]]), jnp.array([k[2]]))
    xi, dl = fermion_blocks(jnp.asarray(p), kx, ky, kz, TS_DEFAULT)
    return np.asarray(xi[0]), np.asarray(dl[0])


GAUGE_CANDS = {"1": np.eye(8, dtype=complex),
               "tau_z": kron3(I2, I2, Z),
               "mu_z": kron3(I2, Z, I2),
               "mu_z*tau_z": kron3(I2, Z, Z)}


def check_symmetries(p, seed=0, nrand=16, tol=5e-7):
    """Returns dict name -> (worst deviation, gauge) over HSPs + random k.
    Each generator is checked with every IGG=Z2 gauge candidate G (PSG check:
    (GU) H(k) (GU)^+ = H(Wk)); the best gauge is reported. '1' = plain."""
    rng = np.random.default_rng(seed)
    ks = list(HSP.values()) + [rng.uniform(-np.pi, np.pi, 3) for _ in range(nrand)]
    out = {}
    for name, (U, W) in GENS.items():
        best = (np.inf, None)
        for gname, G in GAUGE_CANDS.items():
            UG = G @ U
            worst = 0.0
            for k in ks:
                k = np.asarray(k, float)
                xi, dl = blocks_at(p, k)
                xiW, dlW = blocks_at(p, W @ k)
                worst = max(worst, np.abs(UG @ xi @ UG.conj().T - xiW).max(),
                            np.abs(UG @ dl @ UG.T - dlW).max())
            if worst < best[0]:
                best = (worst, gname)
        out[name] = best
    # time reversal (antiunitary): isy xi(-k)* isy^+ = xi(k); same for Delta
    Tm = kron3(1j * Y, I2, I2)
    worst = 0.0
    for k in ks:
        k = np.asarray(k, float)
        xi, dl = blocks_at(p, k)
        xim, dlm = blocks_at(p, -k)
        worst = max(worst, np.abs(Tm @ xim.conj() @ Tm.conj().T - xi).max(),
                    np.abs(Tm @ dlm.conj() @ Tm.T - dl).max())
    out["time_reversal"] = (worst, "1")
    # fermion antisymmetry of the pairing block: Delta(k) = -Delta(-k)^T
    worst = 0.0
    for k in ks:
        _, dl = blocks_at(p, np.asarray(k, float))
        _, dlm = blocks_at(p, -np.asarray(k, float))
        worst = max(worst, np.abs(dl + dlm.T).max())
    out["pairing_antisym"] = (worst, "1")
    return out


def kpath(npts=60):
    path = [("G", "X"), ("X", "M"), ("M", "G"), ("G", "Z"), ("Z", "R"),
            ("R", "A"), ("A", "Z"), ("X", "R"), ("M", "A")]
    ks, labels, ticks = [], [], [0]
    for a, b in path:
        ka, kb = np.array(HSP[a], float), np.array(HSP[b], float)
        seg = [ka + (kb - ka) * t for t in np.linspace(0, 1, npts, endpoint=False)]
        ks.extend(seg)
        ticks.append(len(ks))
        labels.append((a, b))
    return np.array(ks), labels, ticks


def bands(p, U=None):
    ks, labels, ticks = kpath()
    kx, ky, kz = map(jnp.asarray, (ks[:, 0], ks[:, 1], ks[:, 2]))
    xi, dl = fermion_blocks(jnp.asarray(p), kx, ky, kz, TS_DEFAULT)
    Hf = np.block([[np.asarray(xi), np.asarray(dl)],
                   [-np.asarray(dl).conj(), -np.asarray(xi).conj()]]
                  ) if False else None
    top = np.concatenate([np.asarray(xi), np.asarray(dl)], axis=-1)
    bot = np.concatenate([-np.asarray(dl).conj(), -np.asarray(xi).conj()], axis=-1)
    Hf = np.concatenate([top, bot], axis=-2)
    ef = np.linalg.eigvalsh(Hf)
    eb = None
    if U is not None:
        xb, db = boson_blocks(jnp.asarray(p), U, kx, ky, kz, TS_DEFAULT)
        n = 8
        H = np.concatenate([np.concatenate([np.asarray(xb), np.asarray(db)], axis=-1),
                            np.concatenate([np.asarray(db).conj(), np.asarray(xb).conj()], axis=-1)], axis=-2)
        sz = np.concatenate([np.ones(n), -np.ones(n)])
        evs = []
        for Hk in H:
            ev = np.linalg.eigvals(np.diag(sz) @ Hk)
            evs.append(np.sort(ev.real))
        eb = np.array(evs)
    return ks, ticks, ef, eb


def gap_audit(p, U, nk=20):
    kx, ky, kz = make_kgrid(nk)
    xi, dl = fermion_blocks(jnp.asarray(p), kx, ky, kz, TS_DEFAULT)
    top = np.concatenate([np.asarray(xi), np.asarray(dl)], axis=-1)
    bot = np.concatenate([-np.asarray(dl).conj(), -np.asarray(xi).conj()], axis=-1)
    Hf = np.concatenate([top, bot], axis=-2)
    ef = np.linalg.eigvalsh(Hf)
    fgap = np.abs(ef).min()
    # explicit A-point gap
    xiA, dlA = blocks_at(p, np.array([np.pi, np.pi, np.pi]))
    HA = np.block([[xiA, dlA], [-dlA.conj(), -xiA.conj()]])
    fgapA = np.abs(np.linalg.eigvalsh(HA)).min()
    xb, db = boson_blocks(jnp.asarray(p), U, kx, ky, kz, TS_DEFAULT)
    e0b, wmin = boson_e0(jnp.asarray(np.asarray(xb)), jnp.asarray(np.asarray(db)))
    return float(fgap), float(fgapA), float(np.min(np.asarray(wmin)))


ORB = {  # orbital (mu x tau) parts of the generators, 4x4
    "screw_42": np.kron(X, np.eye(2)),
    "C2x": np.kron(np.eye(2), X),
    "inversion": np.eye(4, dtype=complex),
}
ORB["C2z"] = ORB["screw_42"] @ ORB["screw_42"]
ORB["glide_b(mx)"] = ORB["inversion"] @ ORB["C2z"] @ ORB["C2x"]
ORB["glide_c(m110)"] = ORB["inversion"] @ ORB["screw_42"] @ ORB["C2x"]
# bosons (holon/doublon) are spinless: their rep is 1_nu (x) orbital part
GENS_BOSON = {name: (np.kron(np.eye(2, dtype=complex), ORB[name]), GENS[name][1])
              for name in ORB}


def check_boson_symmetries(p, U, seed=0, nrand=12):
    """PSG check for the chargon (boson) kernel: xi transforms with
    U H U^+; the pairing block (zeta=+1) with U D U^T. Note the boson rep
    carries NO spin factor (1 in the holon-doublon space x orbital part)."""
    from sg135_solver import boson_blocks as bb
    rng = np.random.default_rng(seed)
    ks = list(HSP.values()) + [rng.uniform(-np.pi, np.pi, 3) for _ in range(nrand)]
    out = {}
    for name, (U8, W) in GENS_BOSON.items():
        best = (np.inf, None)
        for gname, G in GAUGE_CANDS.items():
            UG = G @ U8
            worst = 0.0
            for k in ks:
                k = np.asarray(k, float)
                kx, ky, kz = (jnp.array([k[0]]), jnp.array([k[1]]), jnp.array([k[2]]))
                xi, dl = bb(jnp.asarray(p), U, kx, ky, kz, TS_DEFAULT)
                kW = W @ k
                kxw, kyw, kzw = (jnp.array([kW[0]]), jnp.array([kW[1]]), jnp.array([kW[2]]))
                xiW, dlW = bb(jnp.asarray(p), U, kxw, kyw, kzw, TS_DEFAULT)
                worst = max(worst,
                            np.abs(UG @ np.asarray(xi[0]) @ UG.conj().T - np.asarray(xiW[0])).max(),
                            np.abs(UG @ np.asarray(dl[0]) @ UG.T - np.asarray(dlW[0])).max())
            if worst < best[0]:
                best = (worst, gname)
        out[name] = best
    return out


def hessian_audit(p, U, nk=8):
    """Min eigenvalue of the Hessian of E_g in the order-parameter subspace
    (multiplier directions excluded: constrained stationarity is a saddle in
    lam/mu by construction). Negative => not even a local minimum."""
    import jax
    from sg135_solver import make_kgrid as mkg, e_total
    kx, ky, kz = mkg(nk)
    op_idx = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 18, 19, 20, 21])

    def grad_ops(v):
        q = jnp.asarray(p).at[jnp.asarray(op_idx)].set(v)
        g = jax.grad(e_total, argnums=0)(q, jnp.zeros(8), U, kx, ky, kz)
        return g[jnp.asarray(op_idx)]

    H = np.asarray(jax.jacfwd(grad_ops)(jnp.asarray(p)[jnp.asarray(op_idx)]))
    H = np.nan_to_num(0.5 * (H + H.T), nan=0.0)
    ev = np.linalg.eigvalsh(H)
    return ev


if __name__ == "__main__":
    # clean verification: disable the 1e-7 degeneracy-splitting regulator so
    # symmetry deviations are measured at machine precision
    import sg135_solver
    sg135_solver._BREAK8 = jnp.zeros((8, 8))
    sol = np.load(sys.argv[1], allow_pickle=True)
    arr = np.atleast_2d(np.asarray(sol, float))
    q = arr[0]
    # layouts: 18 = channels 0-3; 22 = + Class II z-channel (idx 18-21);
    # 26 = 18 + condensates; 30 = 22 + condensates
    if q.size == 18 or q.size == 22:
        p, c8 = q, np.zeros(8)
    elif q.size == 26:
        p, c8 = q[:18], q[18:]
    else:
        p, c8 = q[:22], q[22:]
    Uval = float(sys.argv[2]) if len(sys.argv) > 2 else None
    print("== spinon-sector PSG check (worst |GU H (GU)^+ - H(Wk)|; best Z2 gauge) ==")
    for name, (dev, g) in check_symmetries(p).items():
        print(f"  {name:14s}: {dev:.2e}  gauge={g:10s} {'OK' if dev < 1e-8 else '<-- VIOLATION'}")
    if Uval is not None:
        print("== chargon-sector PSG check ==")
        for name, (dev, g) in check_boson_symmetries(p, Uval).items():
            print(f"  {name:14s}: {dev:.2e}  gauge={g:10s} {'OK' if dev < 1e-8 else '<-- VIOLATION'}")
        fgap, fgapA, wmin = gap_audit(p, Uval)
        lam = float(p[16])
        ks, ticks, ef, eb = bands(p, Uval)
        ebpos = np.where(eb > 1e-9, eb, np.inf) if eb is not None else None
        chargon_gap = float(ebpos.min()) if ebpos is not None else float("nan")
        print(f"== gap audit ==  grid min BdG gap = {fgap:.4f}; analytic min = |lam| = "
              f"{abs(lam):.4f} (attained on the lines (kx or ky = pi) & kz = pi, incl. A)")
        print(f"   chargon EXCITATION gap = {chargon_gap:.4f} (kernel stability margin = {wmin:+.4f})"
              f"   condensate max = {np.abs(c8).max():.3f}")
        hev = hessian_audit(p, Uval)
        print(f"== Hessian audit (OP subspace, 20 dims) ==  min eig = {hev[0]:+.5f}; "
              f"eigs<0: {(hev < -1e-8).sum()}")
        np.savez("sg135_bands.npz", ks=ks, ticks=ticks, ef=ef, eb=eb, p=p, U=Uval)
        print("saved sg135_bands.npz")
