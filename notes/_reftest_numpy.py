"""Referee re-implementation (numpy only) of sg135_solver fermion/boson blocks
to audit verify_sg135.py claims. Not part of the deliverable."""
import numpy as np

I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], complex)
Y = np.array([[0, -1j], [1j, 0]], complex)
Z = np.diag([1.0, -1.0]).astype(complex)

def kron(*ms):
    out = ms[0]
    for m in ms[1:]:
        out = np.kron(out, m)
    return out

GAMMAS = (kron(I2, X), kron(X, I2), kron(Z, I2), kron(Y, Y), kron(X, Z))
TS = (1.0, 0.5, 0.3, 0.3, 0.5)
IDX_CHB = (0, 1, 2, 3, 18); IDX_CHF = (4, 5, 6, 7, 19)
IDX_DB = (8, 9, 10, 11, 20); IDX_DF = (12, 13, 14, 15, 21)
BREAK8 = np.diag(np.linspace(0.0, 1.0, 8)) * 1e-7

def gfuncs(k):
    kx, ky, kz = k
    return (np.cos(kx/2)*np.cos(ky/2), np.cos(kz/2),
            np.cos(kx)-np.cos(ky), np.sin(kx/2)*np.sin(ky/2)*np.cos(kz/2),
            np.cos(kz/2))

def fermion_blocks(p, k, brk=True):
    gs = gfuncs(k)
    xi = -p[16]*np.eye(8, dtype=complex) + (BREAK8 if brk else 0)
    delta = np.zeros((8, 8), complex)
    for i in range(5):
        xi = xi + TS[i]*p[IDX_CHB[i]]*gs[i]*kron(I2, GAMMAS[i])
        delta = delta + 1j*TS[i]*p[IDX_DB[i]]*gs[i]*kron(Y, GAMMAS[i])
    return xi, delta

def boson_blocks(p, U, k, brk=True):
    gs = gfuncs(k)
    D = (U-2*p[16])/2.0; D0 = (U-2*p[17])/2.0
    xi = D*np.eye(8, dtype=complex) + D0*kron(Z, np.eye(4)) + (BREAK8 if brk else 0)
    delta = np.zeros((8, 8), complex)
    for i in range(5):
        xi = xi + TS[i]*p[IDX_CHF[i]]*gs[i]*kron(Z, GAMMAS[i])
        delta = delta + TS[i]*p[IDX_DF[i]]*gs[i]*kron(X, GAMMAS[i])
    return xi, delta

ROT4 = np.array([[0,-1,0],[1,0,0],[0,0,1]], float)
def kron3(a,b,c): return np.kron(a, np.kron(b,c))
GENS = {
 "screw_42": (kron3(np.diag([np.exp(-1j*np.pi/4), np.exp(1j*np.pi/4)]), X, I2), ROT4),
 "C2x": (1j*kron3(X, I2, X), np.diag([1.,-1.,-1.])),
 "inversion": (np.eye(8, dtype=complex), -np.eye(3)),
}
GENS["C2z"] = (GENS["screw_42"][0]@GENS["screw_42"][0], ROT4@ROT4)
GENS["glide_b"] = (GENS["inversion"][0]@GENS["C2z"][0]@GENS["C2x"][0],
                   (-np.eye(3))@(ROT4@ROT4)@GENS["C2x"][1])
GENS["glide_c"] = (GENS["inversion"][0]@GENS["screw_42"][0]@GENS["C2x"][0],
                   (-np.eye(3))@ROT4@GENS["C2x"][1])
HSP = {"G":(0,0,0),"X":(0,np.pi,0),"M":(np.pi,np.pi,0),
       "Z":(0,0,np.pi),"R":(0,np.pi,np.pi),"A":(np.pi,np.pi,np.pi)}
GAUGE = {"1":np.eye(8,dtype=complex), "tau_z":kron3(I2,I2,Z),
         "mu_z":kron3(I2,Z,I2), "mu_z*tau_z":kron3(I2,Z,Z)}
# boson orbital space: N(2) x mu x tau; spatial U acts trivially on N
def bgen(U8):
    # extract orbital 4x4 part of fermion U (spin x mu x tau) is not generic;
    # build boson reps directly: orbital part for screw = mu^x tau^0, C2x = mu^0 tau^x, I = 1
    pass
BOS_GENS = {
 "screw_42": (kron3(I2, X, I2), ROT4),               # N (x) mu^x tau^0
 "C2x": (kron3(I2, I2, X), np.diag([1.,-1.,-1.])),   # N (x) tau^x
 "inversion": (np.eye(8, dtype=complex), -np.eye(3)),
}
BOS_GENS["C2z"] = (BOS_GENS["screw_42"][0]@BOS_GENS["screw_42"][0], ROT4@ROT4)
BOS_GENS["glide_b"] = (BOS_GENS["inversion"][0]@BOS_GENS["C2z"][0]@BOS_GENS["C2x"][0],
                       (-np.eye(3))@(ROT4@ROT4)@BOS_GENS["C2x"][1])
BOS_GENS["glide_c"] = (BOS_GENS["inversion"][0]@BOS_GENS["screw_42"][0]@BOS_GENS["C2x"][0],
                       (-np.eye(3))@ROT4@BOS_GENS["C2x"][1])
BOS_GAUGE = {"1":np.eye(8,dtype=complex), "tau_z":kron3(I2,I2,Z),
             "mu_z":kron3(I2,Z,I2), "mu_z*tau_z":kron3(I2,Z,Z)}

def check(p, blocks, gens, gauge, brk, U=None):
    rng = np.random.default_rng(0)
    ks = list(HSP.values()) + [rng.uniform(-np.pi, np.pi, 3) for _ in range(16)]
    out = {}
    for name,(Um,W) in gens.items():
        best = (np.inf, None)
        for gname,G in gauge.items():
            UG = G@Um; worst = 0.0
            for k in ks:
                k = np.asarray(k, float)
                if U is None:
                    xi, dl = blocks(p, k, brk)
                    xiW, dlW = blocks(p, W@k, brk)
                else:
                    xi, dl = blocks(p, U, k, brk)
                    xiW, dlW = blocks(p, U, W@k, brk)
                worst = max(worst, abs(UG@xi@UG.conj().T-xiW).max(),
                            abs(UG@dl@UG.T-dlW).max())
            if worst < best[0]:
                best = (worst, gname)
        out[name] = best
    return out

if __name__ == "__main__":
    p = np.load("/Users/dmitrymanning-coe/Documents/Research/Barry Bradlyn/135_MFT/code/sg135/sol_classII_U1.npy")
    print("== FERMION symmetry check, WITH _BREAK8 (as shipped) ==")
    for n,(d,g) in check(p, fermion_blocks, GENS, GAUGE, True).items():
        print(f"  {n:10s}: {d:.2e} gauge={g}  {'OK(<5e-7)' if d<5e-7 else 'VIOLATION'}")
    print("== FERMION symmetry check, WITHOUT _BREAK8 ==")
    for n,(d,g) in check(p, fermion_blocks, GENS, GAUGE, False).items():
        print(f"  {n:10s}: {d:.2e} gauge={g}")
    print("== BOSON (chargon) symmetry check, WITHOUT _BREAK8 ==")
    for n,(d,g) in check(p, boson_blocks, BOS_GENS, BOS_GAUGE, False, U=1.0).items():
        print(f"  {n:10s}: {d:.2e} gauge={g}")
    # gap audits at several grids; their shifted grid convention
    def min_gap(p, nk, brk=False):
        s = 2*np.pi/nk
        kk = np.arange(nk)*s + 0.5*s - np.pi
        mn = 1e9; argk = None
        for kx in kk:
            for ky in kk:
                for kz in kk:
                    xi, dl = fermion_blocks(p, (kx,ky,kz), brk)
                    H = np.block([[xi, dl], [-dl.conj(), -xi.conj()]])
                    ev = np.linalg.eigvalsh(H)
                    g = np.abs(ev).min()
                    if g < mn:
                        mn = g; argk = (kx,ky,kz)
        return mn, argk
    for nk in (8, 12, 20, 32):
        mn, argk = min_gap(p, nk)
        print(f"min |E| over shifted nk={nk} grid: {mn:.5f} at k={np.round(argk,3)}")
    kA = (np.pi, np.pi, np.pi)
    xi, dl = fermion_blocks(p, kA, False)
    H = np.block([[xi, dl], [-dl.conj(), -xi.conj()]])
    print("gap at A:", np.abs(np.linalg.eigvalsh(H)).min(), " (lam =", p[16], ")")
    # degeneracy at a random k without BREAK8
    rng = np.random.default_rng(7)
    k = rng.uniform(-np.pi, np.pi, 3)
    xi, dl = fermion_blocks(p, k, False)
    H = np.block([[xi, dl], [-dl.conj(), -xi.conj()]])
    ev = np.sort(np.linalg.eigvalsh(H))
    print("BdG eigenvalues at random k (no BREAK8):", np.round(ev, 6))
