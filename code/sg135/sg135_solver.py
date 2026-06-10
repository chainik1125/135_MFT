"""Slave-boson mean-field theory for the SG135 double-Dirac tight-binding model
(Wieder-Kim-Rappe-Kane PRL 116, 186402) with Hubbard U, at half filling.

Electron Bloch Hamiltonian (4 sublattices = mu (z-offset) x tau (xy-offset);
spin degenerate; write-up Eqs. 5.1-5.2):
    H(k) = t_xy [tau^x mu^0] gxy + t_z [tau^0 mu^x] gz
         + t1   [tau^0 mu^z] g1  + t2 [tau^x mu^y] g2
    gxy = cos(kx/2)cos(ky/2), gz = cos(kz/2),
    g1 = cos kx - cos ky,     g2 = sin(kx/2)sin(ky/2)cos(kz/2).

Slave boson c+_s = f+_s h + s d+ f_{-s} per site; uniform (homogeneous,
"isotropic" per channel) mean fields: for each hopping channel i in
{xy, z, 1, 2}: chi_b_i, chi_f_i, Delta_b_i, Delta_f_i; multipliers lam, mu_L;
k=0 condensates h_a, d_a on the 4 sublattices (reduced by symmetry patterns).

Kernels (validated conventions; see code/common/bdg.py + test_bdg_kmh.py):
  fermion 16x16: Nambu(L) x spin-block(S) x mu(M) x tau(T)
    H_f = -lam L^z S^0 M^0 T^0
          + sum_i [ chi_i(k) L^z S^0 - Delta_i(k) L^y S^y ] (x) Gamma_i
    with chi_i = t_i chi_b_i g_i(k), Delta_i = t_i Delta_b_i g_i(k),
    Gamma_xy = M^0 T^x, Gamma_z = M^x T^0, Gamma_1 = M^z T^0, Gamma_2 = M^y T^y
    (write-up Eqs. 5.40-5.42; note Gamma_2 = mu^y tau^y in the sin-convention).
  boson 16x16: Nambu(L) x holon-doublon(N) x mu x tau
    xi_b   = D N^0 M^0 T^0 + D0 N^z M^0 T^0 + sum_i t_i chi_f_i g_i(k) N^z Gamma_i
    delta_b= sum_i t_i Delta_f_i g_i(k) N^x Gamma_i
    D = (U - 2 lam)/2,  D0 = (U - 2 mu_L)/2   (write-up Eqs. 5.16-5.18).

Constant energy (per cell, 4 sites): bond-counted decoupling constant
(KMH-validated rule E_c_hop = 2 t_bond (bonds/cell) (chi_b chi_f + D_b D_f);
all four channels give 4 sites/cell x n_i bonds /2 x 2 t_i/n_i = 4 t_i):
    E_c = sum_i C_i t_i (chi_b_i chi_f_i + Delta_b_i Delta_f_i)
        + 4 (lam + x mu_L)
        + (U - lam - mu_L) sum_a d_a^2 + (mu_L - lam) sum_a h_a^2
        + condensate-bilinear terms from the k=0 part of H_b (computed
          directly by sandwiching the k=0 boson kernel with the condensate
          vector - no hand-derived formula needed).
The factor C_i = 4 (bond-counted) vs the write-up's 8: settled by the
Hellmann-Feynman check in checks() (docc two ways) - run check_ec_factor().
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "common"))
import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import least_squares
from bdg import fermion_e0, boson_e0

jax.config.update("jax_enable_x64", True)

I2 = jnp.eye(2, dtype=jnp.complex128)
X = jnp.array([[0, 1], [1, 0]], dtype=jnp.complex128)
Y = jnp.array([[0, -1j], [1j, 0]], dtype=jnp.complex128)
Z = jnp.array([[1, 0], [0, -1]], dtype=jnp.complex128)


def kron(*ms):
    out = ms[0]
    for m in ms[1:]:
        out = jnp.kron(out, m)
    return out


# Gamma_i in (mu x tau) space, 4x4
GAMMAS = (kron(I2, X), kron(X, I2), kron(Z, I2), kron(Y, Y))
TS_DEFAULT = (1.0, 0.5, 0.3, 0.3)  # t_xy, t_z, t1, t2 (write-up Fig. 15 values)
# Decoupling constants per channel: C_i = 2 * sum_{bonds/cell} (Bloch weight)
#   xy: 8 bonds x 1/4; z: 4 x 1/2; ch1: 8 x 1/2; ch2: 16 x 1/8
# VALIDATED numerically in test_realspace.py (real-space bond expectations);
# the write-up's uniform 2 z_i = 8 (Eq. 5.53) is incorrect for xy/z/ch2.
C_VEC = (4.0, 4.0, 8.0, 4.0)


def gfuncs(kx, ky, kz):
    gxy = jnp.cos(kx / 2) * jnp.cos(ky / 2)
    gz = jnp.cos(kz / 2)
    g1 = jnp.cos(kx) - jnp.cos(ky)
    g2 = jnp.sin(kx / 2) * jnp.sin(ky / 2) * jnp.cos(kz / 2)
    return (gxy, gz, g1, g2)


def make_kgrid(nk=24):
    s = 2 * np.pi / nk
    k = np.arange(nk) * s + 0.5 * s - np.pi
    kx, ky, kz = np.meshgrid(k, k, k, indexing="ij")
    return tuple(map(jnp.asarray, (kx.ravel(), ky.ravel(), kz.ravel())))


# parameter vector p (18):
#  0-3  chi_b_i   4-7  chi_f_i   8-11 Delta_b_i   12-15 Delta_f_i
#  16 lam  17 mu_L
# condensates handled separately (cond vector c8: h on 4 sublattices, d on 4)
# tiny deterministic diagonal splitting: lifts exact degeneracies so that the
# forward-mode eigh JVP (1/(li-lj) eigenvector tangents) stays finite. Energy
# error O(1e-7), far below solver tolerances.
_BREAK8 = jnp.diag(jnp.linspace(0.0, 1.0, 8)) * 1e-7


def fermion_blocks(p, kx, ky, kz, ts):
    gs = gfuncs(kx, ky, kz)
    nk = kx.shape[0]
    xi = (-p[16] * jnp.eye(8, dtype=jnp.complex128) + _BREAK8)[None, :, :].repeat(nk, axis=0)
    # xi: spin-block(S) x (mu tau): S^0 (x) sum_i chi_i G_i  -> 8x8
    delta = jnp.zeros((nk, 8, 8), dtype=jnp.complex128)
    for i in range(4):
        xi = xi + (ts[i] * p[0 + i]) * gs[i][:, None, None] * kron(I2, GAMMAS[i])
        # pairing block: from H_f = ... -Delta_i L^y S^y Gamma_i:
        # L^y top-right = -i, so Delta-block = (-i)(-1)Delta_i S^y G_i = i Delta_i S^y G_i
        delta = delta + (1j * ts[i] * p[8 + i]) * gs[i][:, None, None] * kron(Y, GAMMAS[i])
    return xi, delta


def boson_blocks(p, U, kx, ky, kz, ts):
    gs = gfuncs(kx, ky, kz)
    nk = kx.shape[0]
    D = (U - 2 * p[16]) / 2.0
    D0 = (U - 2 * p[17]) / 2.0
    xi = (D * jnp.eye(8, dtype=jnp.complex128)
          + D0 * kron(Z, jnp.eye(4, dtype=jnp.complex128)) + _BREAK8)[None].repeat(nk, axis=0)
    delta = jnp.zeros((nk, 8, 8), dtype=jnp.complex128)
    for i in range(4):
        xi = xi + (ts[i] * p[4 + i]) * gs[i][:, None, None] * kron(Z, GAMMAS[i])
        delta = delta + (ts[i] * p[12 + i]) * gs[i][:, None, None] * kron(X, GAMMAS[i])
    return xi, delta


def cond_energy(p, c8, U, ts, x=0.0):
    """k=0 condensate contribution: sandwich the k=0 boson kernel structure.
    c8 = (h_AA, h_AB, h_BA, h_BB, d_AA, d_AB, d_BA, d_BB) real amplitudes
    (sublattice order: (mu,tau) = AA, AB, BA, BB)."""
    h = c8[:4]
    d = c8[4:]
    gs0 = (1.0, 1.0, 0.0, 0.0)  # g_i(k=0)
    E = (U - p[16] - p[17]) * jnp.sum(d**2) + (p[17] - p[16]) * jnp.sum(h**2)
    # k=0 part of H_b with operators -> condensate amplitudes, in THIS module's
    # sign labeling (kernels carry +t chi g, +t Delta g; cf. KMH Eq. 4.93 which
    # uses the opposite labeling - signs here must match boson_blocks):
    #   E += sum_i t_i g_i(0) [ chi_f_i (h G_i h - d G_i d) + 2 Delta_f_i h G_i d ]
    for i in range(2):  # g1(0) = g2(0) = 0
        G = jnp.real(GAMMAS[i])
        coef = ts[i] * gs0[i]
        E = E + coef * p[4 + i] * (h @ G @ h - d @ G @ d)
        E = E + 2.0 * coef * p[12 + i] * (h @ G @ d)
    return E


def e_total(p, c8, U, kx, ky, kz, ts=TS_DEFAULT, x=0.0):
    xf, df = fermion_blocks(p, kx, ky, kz, ts)
    e0f = jnp.mean(fermion_e0(xf, df))
    xb, db = boson_blocks(p, U, kx, ky, kz, ts)
    e0b_k, wmin = boson_e0(xb, db)
    e0b = jnp.mean(e0b_k)
    ec = 4.0 * (p[16] + x * p[17])
    for i in range(4):
        ec = ec + C_VEC[i] * ts[i] * (p[0 + i] * p[4 + i] + p[8 + i] * p[12 + i])
    ec = ec + cond_energy(p, c8, U, ts, x)
    return e0f + e0b + ec


def min_boson_eig(p, U, kx, ky, kz, ts=TS_DEFAULT):
    xb, db = boson_blocks(p, U, kx, ky, kz, ts)
    _, wmin = boson_e0(xb, db)
    return jnp.min(wmin)
