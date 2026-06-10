"""Real-space referee for the STAGGERED (pi-flux) z-channel (channel 4).

Mirrors test_realspace.py for the load-bearing channel of the headline state:
1. Bloch map: the tau-staggered z-bond list (per-bond amplitude +-t_z/2,
   sign by tau column) reproduces t_z g_z mu^x tau^z exactly.
2. Hopping-only mean field on a 6^3 lattice: E0 real-space == k-space.
3. Bond expectations: dE_f/dchi_b4 equals the bond sum exactly and the
   measured decoupling constant is C_4 = 4 t_z.
Run: python test_realspace_ch4.py   (expect 'C_4 = +4.0000')
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "common"))
import numpy as np
import jax
import jax.numpy as jnp
import test_realspace as T
import sg135_solver
from sg135_solver import GAMMAS, TS_DEFAULT, fermion_blocks

sg135_solver._BREAK8 = jnp.zeros((8, 8))

bonds4 = []
tz = TS_DEFAULT[1]
for tau in (0, 1):
    a, b = tau, 2 + tau
    s_tau = 1.0 if tau == 0 else -1.0
    for sz in (0.5, -0.5):
        bonds4.append((a, b, np.array([0.0, 0.0, sz]), s_tau * tz / 2))

# 1) Bloch check
rng = np.random.default_rng(11)
worst = 0.0
for _ in range(10):
    k = rng.uniform(-np.pi, np.pi, 3)
    worst = max(worst, np.abs(T.hk_from_bonds(bonds4, k)
                              - tz * np.cos(k[2] / 2) * np.asarray(GAMMAS[4])).max())
print(f"ch4 bloch match: {worst:.2e}")

# 2-3) lattice referee
chi_b5 = np.array([0.3, 0.25, 0.1, 0.08, 0.2])
p = np.zeros(22); p[0:4] = chi_b5[:4]; p[18] = chi_b5[4]; p[16] = -0.4
L = 6
allbonds = T.bond_list() + bonds4
ch_ids = []
for a, b, d, t in T.bond_list():
    if abs(d[2]) == 0 and abs(d[0]) == 0.5: ch_ids.append(0)
    elif abs(d[2]) == 0.5 and d[0] == 0 and d[1] == 0: ch_ids.append(1)
    elif np.linalg.norm(d) == 1.0: ch_ids.append(2)
    else: ch_ids.append(3)
ch_ids += [4] * len(bonds4)
N = L**3
nsite = 4 * N
sid = lambda c, alpha: ((c[0] % L * L + c[1] % L) * L + c[2] % L) * 4 + alpha
Hhop = np.zeros((nsite, nsite), complex)
for c in [(x, y, z) for x in range(L) for y in range(L) for z in range(L)]:
    for n, (a, b, d, t) in enumerate(allbonds):
        cell_b = np.round(T.POS[a] + np.array(c) + d - T.POS[b]).astype(int)
        ia, ib = sid(c, a), sid(tuple(cell_b), b)
        Hhop[ia, ib] += t * chi_b5[ch_ids[n]]
        Hhop[ib, ia] += np.conj(t * chi_b5[ch_ids[n]])
Hhop -= p[16] * np.eye(nsite)
ev, V = np.linalg.eigh(Hhop)
E0_real = 2 * ev[ev < 0].sum() / N
kk = 2 * np.pi * np.arange(L) / L
KX, KY, KZ = np.meshgrid(kk, kk, kk, indexing="ij")
kxj, kyj, kzj = map(jnp.asarray, (KX.ravel(), KY.ravel(), KZ.ravel()))
xi, _ = fermion_blocks(jnp.asarray(p), kxj, kyj, kzj, TS_DEFAULT)
evk = np.linalg.eigvalsh(np.asarray(xi))
E0_k = float(np.where(evk < 0, evk, 0).sum() / N)
print(f"E0 realspace={E0_real:.10f} kspace={E0_k:.10f} diff={E0_real-E0_k:+.2e}")
occ = ev < 0
Vo = V[:, occ]
corr = (Vo @ Vo.conj().T).T
per, bsum = [], 0.0
for a, b, d, t in bonds4:
    cell_b = np.round(T.POS[a] + d - T.POS[b]).astype(int)
    chif_b = 2 * np.real(corr[sid((0, 0, 0), a), sid(tuple(cell_b), b)])
    per.append(chif_b * np.sign(t))
    bsum += 2 * t * chif_b
chif4 = float(np.mean(per))


def ef_of(q):
    xi2, _ = fermion_blocks(q, kxj, kyj, kzj, TS_DEFAULT)
    evj = jnp.linalg.eigvalsh(xi2)
    return jnp.sum(jnp.where(evj < 0, evj, 0)) / N


g = float(jax.grad(ef_of)(jnp.asarray(p))[18])
print(f"dEf/dchi_b4 = {g:+.6f}  bond_sum = {bsum:+.6f}  ratio = {g/bsum:+.4f}"
      f"  C_4 = {bsum/(tz*chif4):+.4f}  (expect +4.0000)")
