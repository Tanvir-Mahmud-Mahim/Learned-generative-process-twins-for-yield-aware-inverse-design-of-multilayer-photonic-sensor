# FabGAN-ID: Learned Generative Process Twins for Yield-Aware Inverse Design of Multilayer Photonic Sensor Front-Ends

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![JAX](https://img.shields.io/badge/JAX-CPU-orange)](https://github.com/jax-ml/jax)

Official implementation and benchmark for the article *"FabGAN-ID: Learned
Generative Process Twins in a Fully Differentiable Fabrication Loop for
Yield-Aware Inverse Design of Multilayer Photonic Sensor Front-Ends"*
(submitted to IEEE Sensors Journal).

**The idea in one line:** stop *assuming* a fabrication-error model during
robust inverse design — *learn* it from historical (recipe, outcome) traces
with a conditional GAN, and optimize manufacturing-yield tail statistics by
backpropagating **through the learned process model and the exact physics at
once**.

> *Learn what cannot be simulated; differentiate what can.*

## Highlights

- **Learned generative process twin** — a conditional, moment-matched
  WGAN-GP trained on only 400 historical deposition traces cuts the
  estimation error of the yield-deciding tail statistics
  (P5 floor, CVaR at 5%) by **29–35%** vs. diagonal- and full-covariance
  Gaussian models fitted to identical data.
- **Fully differentiable fabrication loop** — an exact transfer-matrix
  solver (validated to 5e-15 against the open `tmm` reference; adjoint
  gradients to a 1e-10 median relative error) composed with the
  differentiable generator enables *pathwise* CVaR robustification:
  **+7.2% CVaR yield floor** on a held-out true process, dominating every
  Gaussian-twin alternative.
- **Analytic policy gradients** train specification-conditioned correction
  policies where model-free soft actor-critic diverges outright.
- **Open benchmark with a hidden ground-truth process** — 300 variable-index
  designs (48,300 exact spectra) + 400 process traces, so competing process
  models can be scored against a distribution they have never seen.

## Repository layout

```
src/                 Core library (JAX)
  materials.py         Sellmeier dispersion (refractiveindex.info, CC0)
  tmm_jax.py           Differentiable transfer-matrix solver + merit functions
  process.py           Hidden ground-truth deposition process (evaluation only)
  fabgan.py            Conditional WGAN-GP twin (+ tail-calibrated variant)
                       and Gaussian baselines
  adjoint.py           Probe-seeded adjoint engine; pathwise CVaR robustification
  surrogate.py         Recursion-informed neural surrogate (protocol study)
  rl_loop.py           Chain-GCN SAC baseline + analytic-policy-gradient policy
experiments/         E1–E7b: all studies (chunked/resumable — rerun each
                     script until it prints DONE)
make_figures/        Publication figures and pipeline-generated tables
data/                Released benchmark (designs, spectra, process traces)
```

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

CPU-only; no GPU required. The full study reproduces in ~45 CPU-minutes on
two cores.

## Quick start

```python
import numpy as np, jax.numpy as jnp
from src import tmm_jax, fabgan

# exact spectrum of a random 20-layer variable-index stack
d = np.random.uniform(0.02, 0.12, 20)     # thicknesses [um]
n = np.random.uniform(1.6, 2.4, 20)       # indices at 550 nm
T = tmm_jax.transmittance_jit(jnp.asarray(d), jnp.asarray(n))
```

## Reproducing the paper

Run the experiments in order; each is **chunked and resumable** — invoke it
repeatedly until it prints `DONE` (state is checkpointed between calls):

```bash
for e in exp1_validate exp2_dataset exp3_fabgan exp3b_lowdata exp4_inverse \
         exp5_yield exp6_policy exp6b_apg exp6c_dense exp7_surrogate exp7b_physcomb; do
  until python experiments/$e.py | grep -q DONE; do :; done
done
python make_figures/figures.py        # all publication figures
python make_figures/fig_diagrams.py   # framework / abstract schematics
python make_figures/fig_sensor2.py    # sensor-context figure
python make_figures/make_tables.py    # every number in the paper, from results
```

All randomness is seeded; every quantitative value in the article is emitted
by `make_tables.py` from the produced result files.

| Experiment | Question it answers |
|---|---|
| E1 | Solver/adjoint validation (vs. `tmm`, vs. finite differences) |
| E3 / E3b | Twin fidelity vs. Gaussian fits; trace-scarcity study |
| E4 | Nominal inverse design, query efficiency |
| E5 | **Does twin fidelity convert to true-process yield?** |
| E6 / E6b / E6c | SAC vs. analytic policy gradients; zero-shot transfer |
| E7 / E7b | Surrogate protocol study (why the loop uses exact physics) |

## The benchmark dataset

`data/` ships the released benchmark (see `data/README.md` for the full
schema): `benchmark_designs.npz` (300 recipes, 48,300 exact spectra) and
`process_traces.npz` (400 recipe/outcome pairs from the *held-out*
ground-truth process). Treat the generating process as hidden: learn only
from the traces, score against fresh draws. The dataset is also archived on
Zenodo (DOI in the article's Data Availability statement).

## Citation

If you use this code or the benchmark, please cite the article (see
`CITATION.cff`):

```bibtex
@article{mahim2026fabganid,
  title   = {FabGAN-ID: Learned Generative Process Twins in a Fully
             Differentiable Fabrication Loop for Yield-Aware Inverse Design
             of Multilayer Photonic Sensor Front-Ends},
  author  = {Mahim, Tanvir M. and Rahman, M. Mosaddequr and Mohsin, Abu S. M.},
  journal = {IEEE Sensors Journal (submitted)},
  year    = {2026}
}
```

## License

Licensed under the **Apache License, Version 2.0** — see [LICENSE](LICENSE)
and [NOTICE](NOTICE). Material dispersion data derive from the public-domain
(CC0) refractiveindex.info database.
