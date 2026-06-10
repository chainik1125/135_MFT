"""Generic numerical Bogoliubov machinery for slave-boson mean-field theories.

Fermions: kernel H = [[xi, Delta], [-Delta.conj(), -xi.conj()]] (write-up Eq. 5.37,
basis psi = (c, c^dag)). Ground-state energy per k:
    E0_f(k) = -1/2 sum_n eps_n(k) + 1/2 Tr xi(k),
with eps_n the positive eigenvalues of the Hermitian kernel. [Write-up Eq. 2.18,
zeta = -1.]

Bosons: kernel H = [[xi, Delta], [Delta.conj(), xi.conj()]] (Eq. 2.10, zeta=+1),
para-diagonalized following Colpa (Physica A 134, 377 (1986)): if H(k) > 0,
Cholesky H = K^dag K, then the eigenvalues of K Sigma_z K^dag come in pairs
+-eps_n; the quasiparticle energies are the positive ones and
    E0_b(k) = 1/2 sum_n eps_n(k) - 1/2 Tr xi(k)
(zero-point of the normal-ordered form; verified against the KMH closed form).
When H is not positive definite (boson condensation boundary) we shift by a tiny
epsilon for differentiability; physical solutions keep the spectrum >= 0 and
condensate amplitudes enter through the k=0 constant term of the energy.

Both functions are jax-differentiable (eigh has well-defined gradients for
non-degenerate spectra; degeneracies are handled by tiny symmetric breaking).
"""
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


def fermion_e0(xi, delta):
    """xi: (..., n, n) Hermitian; delta: (..., n, n) with Delta = -Delta^T pairing
    block as in write-up Eq. (5.37). Returns E0 per k, shape (...,)."""
    H = jnp.block if False else None  # placeholder to keep jit-friendly
    top = jnp.concatenate([xi, delta], axis=-1)
    bot = jnp.concatenate([-jnp.conj(delta), -jnp.conj(xi)], axis=-1)
    K = jnp.concatenate([top, bot], axis=-2)
    evals = jnp.linalg.eigvalsh(K)
    pos = jnp.sum(jnp.where(evals > 0, evals, 0.0), axis=-1)
    return -0.5 * pos + 0.5 * jnp.real(jnp.trace(xi, axis1=-2, axis2=-1))


def boson_e0(xi, delta, eps=1e-10):
    """Colpa para-diagonalization. xi Hermitian, delta symmetric (zeta=+1).
    Returns (E0 per k, min eigenvalue of H for domain checks)."""
    top = jnp.concatenate([xi, delta], axis=-1)
    bot = jnp.concatenate([jnp.conj(delta), jnp.conj(xi)], axis=-1)
    H = jnp.concatenate([top, bot], axis=-2)
    n2 = H.shape[-1]
    n = n2 // 2
    # deterministic diagonal splitting: keeps the eigh JVP finite at exact
    # degeneracies (1e-7, far below physical scales and solver tolerances)
    brk = jnp.diag(jnp.linspace(0.0, 1.0, n2)) * 1e-7
    wmin = jnp.linalg.eigvalsh(H + brk)[..., 0]
    shift = jax.lax.stop_gradient(jnp.abs(jnp.minimum(wmin, 0.0)))  # only active when unstable
    Hs = H + (shift[..., None, None] + eps) * jnp.eye(n2)
    K = jnp.linalg.cholesky(Hs)            # Hs = K K^dag (lower)
    sz = jnp.concatenate([jnp.ones(n), -jnp.ones(n)])
    M = jnp.swapaxes(jnp.conj(K), -1, -2) * sz  # K^dag Sigma_z  (right-multiply diag)
    W = M @ K + brk                          # K^dag Sigma_z K, Hermitian
    ev = jnp.linalg.eigvalsh(W)              # pairs -+eps_n
    pos = jnp.sum(jnp.where(ev > 0, ev, 0.0), axis=-1)
    return 0.5 * pos - 0.5 * jnp.real(jnp.trace(xi, axis1=-2, axis2=-1)), wmin
