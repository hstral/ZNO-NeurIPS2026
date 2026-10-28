# ZNO: Stable Rational Neural Operators in the Z-Domain

Code for **"ZNO: Stable Rational Neural Operators in the Z-Domain for
Discrete-Time Dynamics"** by Xianli Zhu and Jia Yin (NeurIPS 2026).

A ZNO layer takes a hidden sequence with `w` channels and projects it onto
`r ≪ w` latent channels. It filters each latent channel with its own bank of
stable discrete-time poles (an IIR filter) plus an optional short FIR branch,
and mixes the filtered streams back to `w` channels, in parallel with a
pointwise skip. Each layer is therefore a low-rank MIMO rational transfer
matrix,

```
H(z) = A · diag(G_1(z), …, G_r(z)) · B,    G_a(z) = Σ_k c_ak / (1 − p_ak z⁻¹) + FIR_a(z⁻¹),
```

Poles are mapped into the unit disk (`|p| = ρ_max · sigmoid(ρ̃)`), so every
layer is stable by construction. The layer is evaluated exactly by a causal
recurrence whose cost is linear in the sequence length, with fused GPU kernels.

This repository contains:

- **Models.** ZNO (`benchmark_1d/models.py`), FNO, official S4D, LNO, and the
  stabilized LNO (`benchmark_discrete/models.py`), and an adapter around the
  **official S5** JAX/Flax core (`s5_baseline/`).
- **GPU kernels for the ZNO recurrence.** The Triton kernel of the paper's runs
  (`triton`, default), a faster chunked-scan Triton kernel (`triton_fast`), and a
  fused CUDA/C++ extension (`cuda`).
- **Benchmarks.** Generators for the synthetic discrete-time tasks (resonant
  ARMA, sixth-order IIR cascade, nonlinear NARX, near-unit-circle difficulty
  bins), loaders for the public nonlinear system-identification benchmarks
  (Silverbox, Wiener–Hammerstein, CED, Cascaded Tanks, EMPS), SysIdentPy
  NARMAX baselines, and the 1D LNO ODE benchmark.
- **The full experimental pipeline.** Every study of the paper is defined as data
  in `experiments/protocols.py` and turned into explicit run manifests. The
  completed runs are collected into one table, from which `paper/` regenerates
  every table and figure of the paper.

## Installation

```bash
conda env create -f environment.yml && conda activate zno     # or: pip install -r requirements.txt
python -m pytest tests                                         # CPU; GPU and JAX tests skip when unavailable
python scripts/run_smoke_test.py --device cpu                  # one epoch of every model on tiny data
```

Python 3.10+ and PyTorch 2.1+ are required; the GPU kernels need Triton and a
GPU of compute capability 7.0 or higher. The official S5 baseline needs a
separate JAX environment and the upstream code at a pinned commit:

```bash
conda env create -f environment-s5.yml && conda activate zno-s5
bash scripts/fetch_official_s5.sh        # clones lindermanlab/S5 into third_party/S5
python -m pytest tests/test_s5_adapter.py
```

## Training a model

```bash
python benchmark_discrete/generate_suite.py --suite resonant_arma_near_unit_circle_v4 --output-dir data/controlled
python run_discrete_operator.py --case resonant_arma_near_unit_circle_v4 --model zno \
  --data-root data/controlled --output-dir runs/example --run-name zno-seed0 --seed 0 \
  --width 20 --depth 4 --rank 8 --num-poles 40 --fir-order 4 --zno-kernel-backend triton_fast
```

The run directory receives `summary.json` (best-validation and test relative L2
errors, parameter counts, configuration, training time), the best-validation
checkpoint, and the test predictions. `--model` also accepts `fno`,
`s4d_official`, `lno`, and `lno_stable`.

**Kernel backends** (`--zno-kernel-backend`). `triton` is the kernel of the
paper's runs. `triton_fast` processes conjugate pole pairs once, scans time in
tiles with `tl.associative_scan`, and recomputes states in the backward pass; its
outputs and gradients equal `triton` up to float32 rounding, and it trained ZNO
1.5–1.7× faster on A30 GPUs with the same mean test errors. `cuda` is a fused
CUDA/C++ extension. On a CPU, the PyTorch reference scan is used.

## Reproducing the paper

The repository contains no run outputs; the tables and figures are regenerated
from your own runs:

```bash
bash scripts/prepare_data.sh                          # datasets -> data/
python -m experiments.build all                       # run manifests -> runs/manifests/
python -m experiments.manifest run --manifest runs/manifests/matched.json --all   # train, per study
python -m experiments.collect                         # results/runs.csv
python paper/make_tables.py                           # paper/tables/*.tex
python paper/make_figures.py --checkpoints            # paper/figs/*.pdf
```

[REPRODUCTION.md](REPRODUCTION.md) describes each step, maps every study to the
tables and figures it produces, and documents the selection rules and the
compute (1,656 GPU runs; about 170 GPU-hours on A30s). Without training,
`python -m experiments.verify_parameter_counts` rebuilds every configuration on
the CPU and checks its parameter count against the runs behind the paper.

## Repository layout

```
benchmark_1d/          ZNO layer and GPU kernels, LNO/FNO, 1D ODE benchmark
benchmark_discrete/    task generators, public loaders, models, trainer, SysIdentPy baselines
s5_baseline/           official-S5 regression adapter, optimizer recipe, trainer
experiments/           protocols, manifest builder and runner, selection, collection, diagnostics
paper/                 table and figure scripts (read results/runs.csv)
scripts/               data preparation, S5 checkout, smoke test
tools/                 parameter counting, kernel benchmark and backend comparison
tests/                 unit, kernel, protocol, and pipeline tests
run_discrete_operator.py, run_1d_operator.py, run_external_sysid_baseline.py
1D_*/Data/data.mat     1D ODE benchmark data from the LNO release
```

## Conventions

- **Metric.** The relative L2 error is computed per example and averaged over
  examples.
- **Uncertainty.** Mean ± population standard deviation over seeds.
- **Parameter counts.** Real degrees of freedom: complex scalars count twice.
  Summaries record both `parameter_count` (real) and `parameter_count_elements`
  (`sum(p.numel())`).
- **Objectives.** FNO, LNO, S4D, and S5 minimize the task loss. ZNO's default
  objective adds a pole-safety term (λ = 1e-3) and a suffix term (λ = 1e-2);
  `--lambda-pole 0 --lambda-roll 0` gives the task loss, as in Table 1.
- **Selection.** Configurations are selected by validation error only, on
  test-free copies of the data.

## Citation

```bibtex
@inproceedings{zhu2026zno,
  title     = {{ZNO}: Stable Rational Neural Operators in the {Z}-Domain for Discrete-Time Dynamics},
  author    = {Zhu, Xianli and Yin, Jia},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```

## License

The original code is released under the MIT License (`LICENSE`). Third-party
code, data, and benchmarks keep their own licenses; see
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
