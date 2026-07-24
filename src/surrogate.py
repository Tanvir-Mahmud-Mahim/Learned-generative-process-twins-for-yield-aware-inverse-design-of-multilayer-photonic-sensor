"""Transfer-matrix-informed neural surrogate (the PINN pillar).

The stratified-medium analog of a physics-informed network: instead of a PDE
residual at collocation points, the surrogate predicts the complex interface
amplitudes (B_j, C_j) of the characteristic-matrix recursion at collocation
wavelengths, and the physics loss penalizes violation of the exact interface
transfer relation
        [B_j, C_j]^T = M_j(lam) [B_{j+1}, C_{j+1}]^T ,
with the substrate boundary condition  [B_N, C_N] = [1, eta_sub]  imposed
*by construction* (hard encoding, cf. hPINN).  The transmittance follows from
(B_0, C_0) exactly as in the analytic TMM.  A sparse data loss anchors the
surrogate to reference spectra; the recursion residual regularizes it between
and beyond training designs.

Inputs are lifted by random Fourier features (Tancik et al., NeurIPS 2020).
"""

import jax
import jax.numpy as jnp
import numpy as np
import optax

from . import tmm_jax

jax.config.update("jax_enable_x64", True)

N = tmm_jax.N_LAYERS
HID = (256, 256)
FF_DIM = 64
FF_SIGMA = 2.0


def init_surrogate(key):
    k1, k2 = jax.random.split(key)
    B = jax.random.normal(k1, (2 * N + 1, FF_DIM)) * FF_SIGMA
    sizes = (2 * FF_DIM,) + HID + (4 * N,)   # (Re,Im) of B_j, C_j for j=0..N-1
    params = []
    ks = jax.random.split(k2, len(sizes) - 1)
    for li, (k, (i, o)) in enumerate(zip(ks, zip(sizes[:-1], sizes[1:]))):
        scale = jnp.sqrt(1.0 / i) * (0.01 if li == len(sizes) - 2 else 1.0)
        w = jax.random.normal(k, (i, o)) * scale
        params.append((w, jnp.zeros(o)))
    return {"B": B, "mlp": params}


def _forward_amps(params, c, lam_norm, eta_sub):
    """Predict interface amplitudes for one design at one wavelength.

    Amplitudes are parameterized as multiplicative corrections around the
    empty-stack solution (B_j = 1, C_j = eta_sub), which keeps the
    transmittance denominator well conditioned at initialization.
    """
    x = jnp.concatenate([c, lam_norm[None]])
    ff = 2 * jnp.pi * (x @ params["B"])
    h = jnp.concatenate([jnp.sin(ff), jnp.cos(ff)])
    for w, b in params["mlp"][:-1]:
        h = jnp.tanh(h @ w + b)
    w, b = params["mlp"][-1]
    out = h @ w + b
    Bj = 1.0 + out[0:N] + 1j * out[N:2 * N]
    Cj = eta_sub * (1.0 + out[2 * N:3 * N] + 1j * out[3 * N:4 * N])
    return Bj, Cj


def _layer_matrices(d, n0, lam_um):
    disp = jnp.interp(lam_um, jnp.asarray(tmm_jax.LAM_UM), jnp.asarray(tmm_jax.DISP))
    n = n0 * disp
    delta = 2 * jnp.pi * n * d / lam_um
    eta = n
    return jnp.cos(delta), jnp.sin(delta), eta


def physics_residual(params, c, d, n0, lam_um, lam_norm):
    """Squared recursion residual + hard substrate boundary, one (design, lam)."""
    eta_sub = jnp.interp(lam_um, jnp.asarray(tmm_jax.LAM_UM),
                         jnp.asarray(tmm_jax.N_SUB)) + 0.0j
    Bj, Cj = _forward_amps(params, c, lam_norm, eta_sub)
    cosd, sind, eta = _layer_matrices(d, n0, lam_um)
    # substrate boundary hard-encoded as the (N)-th amplitude pair
    Bnext = jnp.concatenate([Bj[1:], jnp.array([1.0 + 0.0j])])
    Cnext = jnp.concatenate([Cj[1:], jnp.array([eta_sub])])
    pB = cosd * Bnext + 1j * sind / eta * Cnext
    pC = 1j * eta * sind * Bnext + cosd * Cnext
    r = jnp.abs(Bj - pB) ** 2 + jnp.abs(Cj - pC) ** 2
    return jnp.sum(r)


def predicted_T(params, c, lam_um, lam_norm):
    eta_sub = jnp.interp(lam_um, jnp.asarray(tmm_jax.LAM_UM),
                         jnp.asarray(tmm_jax.N_SUB)) + 0.0j
    Bj, Cj = _forward_amps(params, c, lam_norm, eta_sub)
    denom = Bj[0] + Cj[0]
    t = 2.0 / denom
    return jnp.real(eta_sub) * jnp.abs(t) ** 2


def norm_recipe(d, n):
    dn = (d - tmm_jax.T_LO) / (tmm_jax.T_HI - tmm_jax.T_LO) * 2 - 1
    nn = (n - tmm_jax.N_LO) / (tmm_jax.N_HI - tmm_jax.N_LO) * 2 - 1
    return jnp.concatenate([dn, nn], axis=-1)


def lam_to_norm(lam_um):
    lo, hi = tmm_jax.LAM_UM[0], tmm_jax.LAM_UM[-1]
    return (lam_um - lo) / (hi - lo) * 2 - 1


_v_predT = jax.vmap(predicted_T, in_axes=(None, None, 0, 0))
_v_res = jax.vmap(physics_residual, in_axes=(None, None, None, None, 0, 0))


def loss_fn(params, batch, phys_batch, lam_w):
    """batch: (C, D, Nn, LAM, LNORM, Tref) supervised tuples.
    phys_batch: (Cp, Dp, Np, LAMP, LNORMP) collocation designs sampled
    uniformly from the design box (label-free) -- the mechanism by which the
    recursion residual regularizes the surrogate at UNSEEN designs."""
    C, D, Nn, LAM, LNORM, Tref = batch

    def per_design(c, d, n0, lam, lnorm, tref):
        tp = _v_predT(params, c, lam, lnorm)
        return jnp.mean((tp - tref) ** 2)

    data = jnp.mean(jax.vmap(per_design)(C, D, Nn, LAM, LNORM, Tref))

    if lam_w > 0.0:
        Cp, Dp, Np, LAMP, LNORMP = phys_batch

        def per_colloc(c, d, n0, lam, lnorm):
            return jnp.mean(_v_res(params, c, d, n0, lam, lnorm))

        phys = jnp.mean(jax.vmap(per_colloc)(Cp, Dp, Np, LAMP, LNORMP))
    else:
        phys = 0.0
    return data + lam_w * phys, (data, phys)


def train_surrogate(key, designs_d, designs_n, Tref, steps=3000, batch=16,
                    n_lam=24, lam_w=0.1, lr=2e-3, lam_mask=None,
                    resume=None, max_seconds=None):
    """Tref: (R, L) reference spectra on tmm_jax.LAM_UM.

    lam_mask: optional (R, L) boolean array; if given, only True entries are
    ever sampled as supervision (row-split protocol).
    """
    import time as _time
    t_start = _time.time()
    opt = optax.adam(lr)
    if resume is None:
        params = init_surrogate(key)
        st = opt.init(params)
        t0 = 0
        hist = []
    else:
        params, st, t0, hist = (resume["params"], resume["st"],
                                resume["t"], resume["hist"])
    R, L = Tref.shape
    C = np.asarray(norm_recipe(jnp.asarray(designs_d), jnp.asarray(designs_n)))
    LAMG = np.asarray(tmm_jax.LAM_UM)

    @jax.jit
    def step(params, st, batch, phys_batch):
        (l, aux), g = jax.value_and_grad(loss_fn, has_aux=True)(
            params, batch, phys_batch, lam_w)
        up, st = opt.update(g, st)
        return optax.apply_updates(params, up), st, l, aux

    rng = np.random.default_rng(t0)
    if lam_mask is not None:
        allowed = [np.where(lam_mask[r])[0] for r in range(R)]
    for t in range(t0, steps):
        ridx = rng.integers(0, R, batch)
        if lam_mask is None:
            lidx = rng.integers(0, L, (batch, n_lam))
        else:
            lidx = np.stack([rng.choice(allowed[r], n_lam) for r in ridx])
        lam = LAMG[lidx]
        b = (jnp.asarray(C[ridx]), jnp.asarray(designs_d[ridx]),
             jnp.asarray(designs_n[ridx]), jnp.asarray(lam),
             jnp.asarray(lam_to_norm(lam)),
             jnp.asarray(np.take_along_axis(Tref[ridx], lidx, axis=1)))
        # label-free collocation designs sampled uniformly from the box
        dp_ = rng.uniform(tmm_jax.T_LO, tmm_jax.T_HI, (batch, tmm_jax.N_LAYERS))
        np_ = rng.uniform(tmm_jax.N_LO, tmm_jax.N_HI, (batch, tmm_jax.N_LAYERS))
        cp_ = np.asarray(norm_recipe(jnp.asarray(dp_), jnp.asarray(np_)))
        lamp = LAMG[rng.integers(0, L, (batch, n_lam))]
        pb = (jnp.asarray(cp_), jnp.asarray(dp_), jnp.asarray(np_),
              jnp.asarray(lamp), jnp.asarray(lam_to_norm(lamp)))
        params, st, l, aux = step(params, st, b, pb)
        if t % 200 == 0:
            hist.append((t, float(l), float(aux[0]), float(aux[1])))
        if max_seconds is not None and _time.time() - t_start > max_seconds:
            return None, {"params": params, "st": st, "t": t + 1,
                          "hist": hist, "done": False}
    return params, {"params": params, "hist": hist, "done": True, "t": steps,
                    "st": st}


def predict_spectrum(params, d, n0):
    c = norm_recipe(jnp.asarray(d), jnp.asarray(n0))
    lam = jnp.asarray(tmm_jax.LAM_UM)
    return np.asarray(_v_predT(params, c, lam, lam_to_norm(lam)))


def r2_score(y, yhat):
    ss = np.sum((y - yhat) ** 2)
    st = np.sum((y - y.mean()) ** 2)
    return 1.0 - ss / st
