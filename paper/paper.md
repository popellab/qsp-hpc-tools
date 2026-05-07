---
title: 'QSP HPC Tools: A Python package for cached, high-performance QSP simulation workflows'
tags:
  - Python
  - MATLAB
  - quantitative systems pharmacology
  - high-performance computing
  - simulation-based inference
  - SLURM
authors:
  - name: Joel Eliason
    orcid: 0000-0003-2227-8727
    affiliation: 1
  - name: Aleksander S. Popel
    affiliation: 1
affiliations:
  - name: Department of Biomedical Engineering, Johns Hopkins University, Baltimore, MD, USA
    index: 1
date: 6 May 2026
bibliography: paper.bib
---

# Summary

Quantitative systems pharmacology (QSP) models are mechanistic, ordinary differential equation (ODE)-based representations of disease biology and drug action used throughout pharmaceutical development [@Gadkar2016]. A critical step in QSP workflows is generating large ensembles of simulations—often called virtual patients—by sampling parameter sets from prior distributions and evaluating model outputs under one or more therapeutic scenarios [@Allen2016; @Rieger2018; @Craig2023]. When these simulations feed into modern inference methods such as simulation-based inference (SBI), the computational cost scales rapidly: thousands to tens of thousands of forward simulations are required, each involving stiff ODE integration in MATLAB over clinically relevant timescales [@Cranmer2020].

`qsp-hpc-tools` is a Python package that automates the submission, caching, and retrieval of QSP simulations on SLURM high-performance computing (HPC) clusters. It exposes two interchangeable callable simulators—`QSPSimulator` for MATLAB/SimBiology models and `CppSimulator` for a faster generated C++ backend—both of which abstract the full simulation lifecycle behind a shared three-tier caching strategy, so that researchers can iterate on inference workflows without redundant computation.

# Statement of need

QSP models are predominantly implemented in MATLAB using SimBiology or custom ODE solvers, while the most active developments in Bayesian inference and SBI are in the Python ecosystem [@Goncalves2020; @Tejero-Cantero2020]. Bridging these two environments on HPC infrastructure currently requires ad hoc scripts for SSH-based job submission, SLURM array task management, result collection, and file format conversion—effort that is duplicated across research groups and is a recurring source of bugs.

General-purpose workflow managers such as Snakemake and Nextflow [@Koster2012; @DiTommaso2017] handle HPC job orchestration well, but they are designed around static, DAG-based pipelines. QSP model calibration is iterative: priors are revised, test statistics are added or removed, and scenario definitions evolve rapidly during model development. Each such change should invalidate exactly the affected cached results—no more, no less—while preserving everything else. Achieving this with a general workflow manager requires manually encoding invalidation logic that `qsp-hpc-tools` derives automatically from content hashes. Similarly, MATLAB's SimBiology provides built-in parameter scanning but lacks HPC job management, Python interoperability, or a caching layer. Probabilistic programming frameworks like Stan with Torsten [@Margossian2022] offer integrated Bayesian inference for pharmacometrics, but require rewriting models in Stan rather than working with existing MATLAB codebases.

`qsp-hpc-tools` occupies a different niche: it wraps existing MATLAB QSP models as a Python-callable simulator with content-aware caching, multi-scenario support, and transparent HPC orchestration—without requiring researchers to rewrite their models or manage job submission scripts. It is currently used in production for pancreatic cancer QSP model calibration, where it manages multi-scenario simulation campaigns of thousands of virtual patients across SLURM clusters, feeding directly into neural posterior estimation pipelines.

# Design and key features

## Three-tier caching

When simulations are requested, both simulators check three levels before submitting new HPC jobs:

1. **Local pool**: Previously downloaded simulation results stored in scenario-specific directories.
2. **HPC test statistics**: Pre-computed summary statistics on the cluster that can be downloaded and used without transferring full simulation timecourses.
3. **HPC full simulations**: Complete simulation outputs on the cluster available for download.

Only when all three tiers miss does the system submit new SLURM array jobs. This strategy avoids redundant computation during the rapid iteration typical of SBI workflow development.

For models with an expensive pre-treatment burn-in phase, the C++ backend additionally caches the post-evolve ODE state in an LMDB-packed `evolve_cache` keyed by parameter set and model configuration, allowing multiple downstream therapeutic scenarios to skip the shared burn-in integration. On models where burn-in dominates wall time, this delivers near-linear speedups in the number of scenarios sharing a configuration.

## Content-based cache invalidation

Pool directories are keyed by a configuration hash computed over the semantically meaningful components of the simulation setup: prior distributions, calibration target definitions (including observable code, units, and species), model script content, model version, and scenario name. Calibration targets are defined as structured YAML files—compatible with the MAPLE framework for LLM-assisted literature extraction—that specify observable functions, physical constants with units, and empirical data with uncertainty bounds. Cosmetic changes—such as renaming a parameter description—do not invalidate the cache, while changes to the actual simulation semantics automatically trigger re-computation. Hash normalization ensures that equivalent configurations produce identical hashes regardless of file ordering or whitespace.

## Multi-scenario support

QSP workflows frequently evaluate the same parameter sets under multiple therapeutic protocols (e.g., treatment vs. control). `qsp-hpc-tools` supports this natively: a shared `cache_sampling_seed` ensures identical parameter draws across scenarios, while each scenario maintains its own simulation pool. Joint NaN filtering across scenarios is handled downstream, ensuring that only parameter sets valid under all conditions are retained for inference.

## HPC integration

`HPCJobManager` handles SSH-based codebase synchronization, SLURM array job submission, progress monitoring, and result collection. Workers on the cluster execute simulations in parallel and write results in Parquet format via a Python bridge, enabling efficient cross-language data transfer. Configuration is managed through a single global credentials file with an interactive setup wizard accessible via the `qsp-hpc setup` CLI command.

## Pluggable backends: MATLAB and generated C++

The package supports two interchangeable simulation backends that share the cache, scheduler, and result-loading machinery. `QSPSimulator` wraps existing MATLAB/SimBiology models, requiring only a model script and a structure file describing species, units, and initial conditions. `CppSimulator` instead drives a generated C++ ODE simulator (`qsp_sim`) built from an SBML model description by the companion `qsp-codegen` package [@qsp_codegen], which is pip-installed on the cluster and invoked automatically by `ensure_cpp_binary` to drive the in-repo CMake build. The C++ backend yields roughly 25–87× per-simulation speedups over the MATLAB path on representative pancreatic cancer models while preserving the same calibration target interface, so existing MAPLE-style YAMLs and downstream inference code work unchanged.

## Posterior-predictive simulation and prior restriction

Beyond drawing fresh parameter sets from the prior, both simulators expose `simulate_with_parameters(theta, backend="local"|"hpc")`, which evaluates the model at user-supplied parameter arrays such as posterior draws produced by a downstream inference package. Results are cached under a hash of the input theta array, so repeated posterior-predictive checks against the same posterior samples reuse prior work. To further reduce wasted simulation effort, `get_theta_pool` supports an optional `RestrictionClassifier` from the inference layer that rejection-samples the prior, discarding parameter draws predicted to fail the simulator before they are ever submitted.

# Typical usage

```python
from qsp_hpc import QSPSimulator

# Initialize the MATLAB-backed simulator for a treatment scenario
sim = QSPSimulator(
    priors_csv="priors.csv",
    calibration_targets="calibration_targets/",  # directory of MAPLE YAMLs
    model_script="pdac_qsp_model",
    model_version="v2",
    scenario="treatment",
    seed=42,
)

# Request 2000 simulations; cached results are returned when available
theta, x = sim(2000)

# Re-use the same simulator for posterior-predictive checks at user-supplied
# parameter draws (typically posteriors from qsp-inference)
x_posterior = sim.simulate_with_parameters(posterior_samples)
```

The C++ backend is a drop-in replacement for the simulation step:

```python
from qsp_hpc import CppSimulator
from qsp_hpc.batch.hpc_job_manager import HPCJobManager

sim = CppSimulator(
    priors_csv="priors.csv",
    binary_path="/path/to/qsp_sim",
    template_xml="/path/to/param_all.xml",
    scenario_yaml="scenarios/treatment.yaml",
    calibration_targets="calibration_targets/treatment/",
    job_manager=HPCJobManager(),
)
theta, test_stats = sim.run_hpc(2000)
```

# Acknowledgements

This work was supported by the National Institutes of Health. The author thanks the Maryland Advanced Research Computing Center (MARCC) for HPC resources.

# References
