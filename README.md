# FabGAN-ID

**Learning the Fabrication Process for Yield-Aware Inverse Design of Multilayer Photonic Sensor Filters**

Official implementation and open benchmark for the FabGAN-ID study (submitted to *IEEE Sensors Journal*).

The idea in one line: stop *assuming* a fabrication-error model during robust inverse design; learn it from historical (recipe, outcome) traces with a conditional GAN, and optimize manufacturing-yield tail statistics by backpropagating through the learned process model and the exact physics at once.

> Learn what cannot be simulated; differentiate what can.

This repository contains code and the open benchmark only. The manuscript is not distributed here.

## Highlights
- **Learned generative process twin** — a conditional, moment-matched Wasserstein GAN with gradient penalty (WGAN-GP), trained on only 400 historical deposition traces, cuts the estimation error of the yield-deciding tail statistics (the fifth percentile P5 and the conditional value-at-risk CVaR at 5%) by **34–38%** against the strongest of eight density baselines fitted to identical data: diagonal and full-covariance Gaussians, a five-component Gaussian mixture, a nonparametric bootstrap, Gaussian and t copulas, a Silverman-bandwidth kernel density estimate, and a RealNVP normalizing flow.
- **Fully differentiable fabrication loop** — an exact transfer-matrix solver (validated to 5e-15 against the open `tmm` reference; adjoint gradients to a 1e-10 median relative error, with the twin's design gradients cross-validated against true-process finite differences at Pearson r = 0.91) composed with the differentiable generator enables pathwise CVaR robustification. On a held-out true process the FabGAN-robustified design lifts the CVaR-5% lower-tail merit by **+7.2%** (relative) over the nominal optimum, **matches an independent true-process oracle** (zero regret), dominates every Gaussian-twin alternative, and raises the simulated pass/fail filter yield from **0.16% to 63.7%** (8/5000 vs. 3186/5000 draws; the advantage holds across a 4x3 pass/fail-threshold sweep and across six retrained hidden-process variants in six of seven processes).
- **Analytic policy gradients** train specification-conditioned correction policies in an exploratory equal-wall-clock comparison in which model-free soft actor-critic (SAC) did not converge.
- **Open benchmark with a hidden ground-truth process** — 300 variable-index designs (48,300 wavelength–transmittance samples) plus 400 process traces, so competing process models can be scored against a distribution they have never seen. This is a fully simulation-based study; the ground-truth process is a held-out simulator, not a wafer.

## Repository layout
```
src/                    core library (JAX)
  tmm_jax.py            differentiable transfer-matrix solver + merit functions
  process.py            hidden ground-truth deposition process (evaluation only)
  fabgan.py             conditional WGAN-GP twin (+ tail-calibrated variant) and Gaussian baselines
  adjoint.py            probe-seeded adjoint engine; pathwise CVaR robustification
  surrogate.py          recursion-informed neural surrogate (protocol study)
experiments/            E1–E7b plus exp_rev_* revision studies (chunked/resumable
                        — rerun each script until it prints DONE)
make_figures/           publication figures and pipeline-generated tables
data/                   released benchmark (designs, spectra, process traces)
results/                all experiment outputs (JSON/NPY/PKL)
```

## Installation
```
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```
CPU-only; no GPU required.

## Reproducing the study
Run the experiments in order; each is chunked and resumable — invoke it repeatedly until it prints `DONE` (state is checkpointed between calls):
```
for e in exp1_validate exp2_dataset exp3_fabgan exp3b_lowdata exp4_inverse \
         exp5_yield exp6_policy exp6b_apg exp6c_dense exp7_surrogate exp7b_physcomb; do
  until python3 experiments/$e.py | grep -q DONE; do :; done
done
# additional revision studies
for e in exp_rev_baselines exp_rev_oracle exp_rev_grad exp_rev_latent exp_rev_sensor; do
  until python3 experiments/$e.py | grep -q DONE; do :; done
done
# second-round revision studies (exp_r2_seedsbase runs to completion in one call)
for e in exp_r2_density exp_r2_seeds exp_r2_yield exp_r2_grad exp_r2_bandpass exp_r2_families; do
  until python3 experiments/$e.py | grep -q DONE; do :; done
done
python3 experiments/exp_r2_seedsbase.py
python3 make_figures/figures.py       # all publication figures
python3 make_figures/make_tables.py   # every number, emitted from the result files
```
All randomness is seeded.

| Experiment | Question it answers |
|---|---|
| E1 | Solver/adjoint validation (vs. `tmm`, vs. finite differences) |
| E3 / E3b | Twin fidelity vs. Gaussian fits; trace-scarcity study |
| E4 | Nominal inverse design, query efficiency |
| E5 | Does twin fidelity convert to true-process yield? |
| E6 / E6b / E6c | SAC vs. analytic policy gradients; zero-shot transfer |
| E7 / E7b | Surrogate protocol study (why the loop uses exact physics) |
| exp_rev_baselines | Gaussian-mixture and nonparametric-bootstrap fidelity baselines |
| exp_rev_oracle | True-process oracle, mean-only baseline, and regret |
| exp_rev_grad | Design-gradient validation vs. true-process finite differences |
| exp_rev_latent | Latent-dimension ablation |
| exp_rev_sensor | Device-level sensor metrics and hard pass/fail filter yield |
| exp_r2_density | Copula, kernel-density, and RealNVP-flow baselines |
| exp_r2_seeds / exp_r2_seedsbase | Training-seed/dataset variability, learning curves, like-for-like baselines |
| exp_r2_yield | Yield counts, Wilson CIs, pass/fail-threshold sweep |
| exp_r2_grad | Finite-K CVaR gradient bias/variance study |
| exp_r2_bandpass | Second objective: 600-nm band-pass transfer |
| exp_r2_families | Process-family robustness sweep (7 hidden processes) |

## The benchmark dataset
`data/` ships the released benchmark (see `data/README.md` for the full schema): `benchmark_designs.npz` (300 recipes, 48,300 wavelength–transmittance samples) and `process_traces.npz` (400 recipe/outcome pairs from the held-out ground-truth process). Treat the generating process as hidden: learn only from the traces, and score against fresh draws. The dataset is also archived on Zenodo (DOI in the article's Data Availability statement).

## Data provenance
- Material dispersion: refractiveindex.info database (CC0 / public domain).
- Variable-index SiNx platform (n in [1.6, 2.4], t in [20, 120] nm): Yesilyurt et al., *Nanophotonics* 12, 993 (2023).
- Benchmark designs, spectra, and process traces: generated by this pipeline (exp2) and released for public use.

## License
Licensed under the Apache License, Version 2.0 — see `LICENSE` and `NOTICE`. Material dispersion data derive from the public-domain (CC0) refractiveindex.info database.
