"""Material dispersion models grounded in the refractiveindex.info open database (CC0).

Sources
-------
- Si3N4: K. Luke et al., Opt. Lett. 40, 4823 (2015). Sellmeier ("formula 1")
  coefficients from refractiveindex.info: 0 3.0249 0.1353406 40314 1239.842,
  valid 0.310-5.504 um.
- SiO2 (fused silica substrate): I. H. Malitson, JOSA 55, 1205 (1965).
  Coefficients: 0 0.6961663 0.0684043 0.4079426 0.1162414 0.8974794 9.896161,
  valid 0.21-6.7 um.

Variable-index SiNx layers (achievable range n(lam0) in [1.6, 2.4] per
Yesilyurt et al., Nanophotonics 12, 993 (2023), HDPCVD-grown SiNx) are modeled
by scaling the Si3N4 Sellmeier dispersion shape to the prescribed index at the
reference wavelength lam0.  This preserves a physically credible dispersion
slope while allowing a continuously variable index, matching the
single-material variable-index platform of the Purdue work.
"""

import numpy as np

LAM0_UM = 0.550  # reference wavelength for layer index specification (um)

# refractiveindex.info "formula 1" (Sellmeier): n^2 - 1 = sum_i c1_i lam^2/(lam^2 - c2_i^2)
SI3N4_LUKE = [(3.0249, 0.1353406), (40314.0, 1239.842)]
SIO2_MALITSON = [(0.6961663, 0.0684043), (0.4079426, 0.1162414), (0.8974794, 9.896161)]


def sellmeier(lam_um, terms):
    lam2 = np.asarray(lam_um, dtype=np.float64) ** 2
    n2 = 1.0 + sum(c1 * lam2 / (lam2 - c2 ** 2) for c1, c2 in terms)
    return np.sqrt(n2)


def n_si3n4(lam_um):
    return sellmeier(lam_um, SI3N4_LUKE)


def n_sio2(lam_um):
    return sellmeier(lam_um, SIO2_MALITSON)


def dispersion_shape(lam_um):
    """Si3N4 dispersion shape normalized to 1 at LAM0_UM (dimensionless)."""
    return n_si3n4(lam_um) / n_si3n4(LAM0_UM)


def layer_index(n_at_lam0, lam_um):
    """Index of a variable-index SiNx layer with prescribed n at LAM0_UM.

    n_i(lam) = n_i(lam0) * S(lam),  S = n_Si3N4(lam)/n_Si3N4(lam0).
    """
    n_at_lam0 = np.asarray(n_at_lam0, dtype=np.float64)
    S = dispersion_shape(lam_um)
    if n_at_lam0.ndim == 0:
        return n_at_lam0 * S
    return n_at_lam0[..., None] * S[None, :]
