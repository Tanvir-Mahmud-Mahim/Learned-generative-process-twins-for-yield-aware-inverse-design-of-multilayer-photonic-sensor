"""Differentiable transfer-matrix method (TMM) in JAX.

Normal-incidence characteristic-matrix formulation for a non-magnetic
stratified stack (air / N variable-index SiNx layers / fused-silica substrate).
Fully vectorized over wavelengths and batch of designs; exact gradients via
reverse-mode autodiff (the discrete adjoint of the transfer recursion).

Validated against the open-source `tmm` package (S. Byrnes,
arXiv:1603.02720) in experiments/exp1_validate.py.
"""

import jax
import jax.numpy as jnp
import numpy as np

from . import materials

jax.config.update("jax_enable_x64", True)
import os as _os
_cache = _os.path.expanduser("~/.jaxcache")
_os.makedirs(_cache, exist_ok=True)
jax.config.update("jax_compilation_cache_dir", _cache)
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.5)

# ----- fixed spectral grid (visible band, sensor front-end regime) -----
LAM_UM = np.linspace(0.40, 0.80, 161)          # 161 wavelength points
DISP = np.asarray(materials.dispersion_shape(LAM_UM))   # (L,)
N_SUB = np.asarray(materials.n_sio2(LAM_UM))            # (L,)
N_INC = 1.0                                             # air

# design box (Yesilyurt et al. 2023: t in [20,120] nm, n(lam0) in [1.6, 2.4])
N_LAYERS = 20
T_LO, T_HI = 0.020, 0.120     # um
N_LO, N_HI = 1.60, 2.40


def transmittance(d_um, n0):
    """Intensity transmittance T(lam) of the stack.

    Parameters
    ----------
    d_um : (N,) physical thicknesses in um
    n0   : (N,) layer indices at the reference wavelength lam0

    Returns
    -------
    T : (L,) transmittance on LAM_UM
    """
    lam = jnp.asarray(LAM_UM)                     # (L,)
    n = n0[:, None] * jnp.asarray(DISP)[None, :]  # (N, L) layer index vs lam
    delta = 2.0 * jnp.pi * n * d_um[:, None] / lam[None, :]   # (N, L)
    eta = n                                       # normal incidence, s=p

    cosd = jnp.cos(delta)
    sind = jnp.sin(delta)
    # characteristic matrices per layer: [[cosd, i sind/eta], [i eta sind, cosd]]
    m00 = cosd + 0.0j
    m01 = 1j * sind / eta
    m10 = 1j * eta * sind
    m11 = cosd + 0.0j

    def matmul_layer(carry, mats):
        a00, a01, a10, a11 = carry
        b00, b01, b10, b11 = mats
        return (a00 * b00 + a01 * b10,
                a00 * b01 + a01 * b11,
                a10 * b00 + a11 * b10,
                a10 * b01 + a11 * b11), None

    eye = (jnp.ones_like(m00[0]), jnp.zeros_like(m00[0]),
           jnp.zeros_like(m00[0]), jnp.ones_like(m00[0]))
    (M00, M01, M10, M11), _ = jax.lax.scan(
        matmul_layer, eye, (m00, m01, m10, m11))

    eta_sub = jnp.asarray(N_SUB) + 0.0j
    B = M00 + M01 * eta_sub
    C = M10 + M11 * eta_sub
    denom = N_INC * B + C
    t = 2.0 * N_INC / denom
    T = jnp.real(eta_sub) / N_INC * jnp.abs(t) ** 2
    return T


transmittance_batch = jax.jit(jax.vmap(transmittance, in_axes=(0, 0)))
transmittance_jit = jax.jit(transmittance)


# ----- sensor-front-end figures of merit -----
def band_masks(center_um, half_um=0.015, guard_um=0.030):
    """Stop band [c-h, c+h]; pass bands outside [c-h-g, c+h+g]."""
    lam = LAM_UM
    stop = (lam >= center_um - half_um) & (lam <= center_um + half_um)
    outside = (lam <= center_um - half_um - guard_um) | (lam >= center_um + half_um + guard_um)
    return jnp.asarray(stop, dtype=jnp.float64), jnp.asarray(outside, dtype=jnp.float64)


def notch_merit(d_um, n0, stop_mask, pass_mask):
    """Merit J in [0,1] for a fluorescence-rejection notch sensor front-end:
    J = 0.5*(mean passband T) + 0.5*(mean stopband (1-T))."""
    T = transmittance(d_um, n0)
    jp = jnp.sum(T * pass_mask) / jnp.sum(pass_mask)
    js = jnp.sum((1.0 - T) * stop_mask) / jnp.sum(stop_mask)
    return 0.5 * jp + 0.5 * js


notch_merit_jit = jax.jit(notch_merit)
notch_merit_batch = jax.jit(jax.vmap(notch_merit, in_axes=(0, 0, None, None)))
grad_notch_merit = jax.jit(jax.grad(notch_merit, argnums=(0, 1)))


def clip_design(d_um, n0):
    return jnp.clip(d_um, T_LO, T_HI), jnp.clip(n0, N_LO, N_HI)
