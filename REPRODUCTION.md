# Reproducing the paper

Every table, figure, and reported number of the paper is produced from training
runs by the commands below. The repository ships no run outputs: you generate
the data, train the runs, collect them into one table, and regenerate the
tables and figures from that table.

| Step | Command | Output |
| --- | --- | --- |
| 1. Data | `bash scripts/prepare_data.sh` | `data/` |
| 2. Manifests | `python -m experiments.build all` | `runs/manifests/<study>.json` |
| 3. Training | `python -m experiments.manifest run --manifest runs/manifests/<study>.json --all` | `runs/<study>/…/summary.json` |
| 4. Collection | `python -m experiments.collect` | `results/runs.csv` |
| 5. Tables | `python paper/make_tables.py` | `paper/tables/*.tex` |
| 6. Figures | `python paper/make_figures.py --checkpoints` | `paper/figs/*.pdf` |

Every configuration is defined once, in
[`experiments/protocols.py`](experiments/protocols.py) and the two files in
`experiments/configs/` (the capacity-study topologies and the public candidate
grids). Commands are run from the repository root.

**Compute.** `experiments.build all` writes 1,676 runs: 1,656 GPU runs and 20
CPU fits of the classical baselines. On an NVIDIA A30, a GPU run takes about
5 minutes (median; 90% finish within 10 minutes, the longest about an hour),
and the 1,650 GPU runs counted in Table 8 took about 170 GPU-hours. All runs are
independent, so they can be dispatched as array jobs.

## 1. Environments

```bash
# PyTorch: ZNO, FNO, official S4D, LNO variants, SysIdentPy baselines, tables and figures
conda env create -f environment.yml && conda activate zno

# JAX: official S5 core, in a separate environment
conda env create -f environment-s5.yml && conda activate zno-s5
bash scripts/fetch_official_s5.sh        # clones lindermanlab/S5 at the pinned commit into third_party/S5
```

Check the installation (CPU, about a minute):

```bash
python -m pytest tests                   # GPU and JAX tests are skipped when unavailable
python scripts/run_smoke_test.py --device cpu   # one epoch of every model on tiny data
python -m experiments.verify_parameter_counts   # rebuilds every configuration; compares parameter counts
```

### ZNO kernels

On a GPU, the ZNO recurrence runs in a fused kernel chosen with
`--zno-kernel-backend` (or `python -m experiments.build all --zno-backend …`):

| Backend | Description |
| --- | --- |
| `triton` (default) | The kernel of the paper's runs. |
| `triton_fast` | Chunked parallel scan: conjugate pole pairs share one lane, time is processed in tiles of 32 steps with `tl.associative_scan`, and the backward pass recomputes states instead of storing them. Outputs and gradients equal `triton` up to float32 rounding (`tests/test_triton_fast.py`). On A30 GPUs, ZNO training in the Table 1 configuration was 1.5–1.7× faster with the same mean test errors; because rounding differences grow over 600 epochs, individual ARMA and IIR seeds reach different, statistically equivalent optima (per-seed test errors within a few percent). |
| `cuda` | Fused CUDA/C++ extension, compiled on first use. |

On a CPU, ZNO uses the PyTorch reference scan. Triton needs a GPU of compute
capability 7.0 or higher. To compare the backends on your GPU:

```bash
python -m pytest tests/test_triton_fast.py
python tools/benchmark_zno_kernels.py
python tools/compare_zno_backends.py --data-root data/controlled   # 5 training epochs per backend
```

## 2. Data

```bash
bash scripts/prepare_data.sh             # all steps; or name steps, e.g. controlled views s5-views
```

| Step | Content | Location |
| --- | --- | --- |
| `controlled` | resonant ARMA, sixth-order IIR cascade, nonlinear NARX: 1024/256/256 trajectories of length 2048 | `data/controlled/` |
| `difficulty` | five near-unit-circle pole-radius bins of the resonant ARMA task | `data/difficulty/` |
| `extrapolation` | the same tasks and bins with test sequences of length 4096 and 8192 | `data/extrapolation/T*/`, `data/difficulty_extrapolation/T*/` |
| `public` | Silverbox, Wiener–Hammerstein, EMPS, Cascaded Tanks, CED, downloaded through `nonlinear-benchmarks`; window settings in `PUBLIC_DATA_BUILD` | `data/public/` |
| `views` | test-free copies for the validation-only selection runs: the official test arrays are not copied | `data/selection_views/` |
| `s5-views` | the same arrays exported for the JAX pipeline, with array hashes | `data/s5_views/` |

The generators are seeded, so the synthetic datasets are the paper's. The NARX
arrays can differ in the last float32 bit across CPUs (vectorized
transcendental functions), which has no effect at the reported precision. The
1D ODE benchmark uses the data of the LNO release, bundled in
`1D_*/Data/data.mat`. The step takes a few minutes, plus the downloads.

## 3. Manifests

```bash
python -m experiments.build --list       # studies and run counts
python -m experiments.build all          # writes runs/manifests/<study>.json
```

A manifest is a reviewable list of runs. Each entry holds the exact command
(`argv`), the study, task, model, configuration tag, seed, whether the run may
read the official test set, and the path of the `summary.json` it writes. The
studies and the paper items they produce:

| Study | Runs | Paper |
| --- | ---: | --- |
| `matched` | 75 | Table 1, Figure 3a, Figure 8; 8.5k rows of Tables 12–13 |
| `zno-default` | 15 | Table 9; ZNO at 8.5k in Tables 12–13 and Figure 3b–d; first row of Table 11 |
| `suffix-control` | 60 | Table 10 |
| `iir-optimizer` | 20 | Table 11 |
| `capacity` | 180 | Tables 12–13, Figure 3b–d |
| `tuning-selection` | 132 | Table 15; validation column of Table 14 |
| `tuning-confirm` | 45 | Table 14 |
| `lno-reference` | 15 | Table 16 |
| `lno-unconstrained` | 6 | Appendix H (unmodified LNO) |
| `fixed` | 45 | Table 7 configurations: $T=2048$ of Table 17 and Figure 5, Figures 9–10 |
| `difficulty` | 75 | Figure 4 |
| `extrapolation` | 90 | Table 17, Figure 5 |
| `difficulty-extrapolation` | 150 | Appendix I |
| `public-selection` | 339 | Table 20 (selection) |
| `public` | 120 | Tables 2, 20, 22 (seeds 3–7); Appendix K (seeds 0–4) |
| `public-s5-selection` | 75 | Table 19 |
| `public-s5` | 25 | S5 column of Tables 2, 20, 22 (written to `runs/public`) |
| `classical` | 20 | Tables 2 and 21 (CPU) |
| `emps-interventions` | 20 | Table 23 |
| `tanks-interventions` | 15 | Table 24 |
| `ablation` | 100 | Table 25, Figure 6 |
| `one-d` | 54 | Figure 7 |

Options: `--zno-backend triton_fast` selects the faster kernel;
`--data-dir` and `--runs-dir` move the data and run roots; `--extra-args
'--epochs 1'` appends arguments to every command, e.g. for a dry run of a
whole manifest.

## 4. Training

```bash
python -m experiments.manifest run --manifest runs/manifests/matched.json --all --skip-existing
python -m experiments.manifest run --manifest runs/manifests/matched.json --index 7   # one entry
python -m experiments.manifest status --manifest runs/manifests/matched.json
```

On a cluster, dispatch one `--index` per array task. Entries with runner `s5`
need the JAX environment and all others the PyTorch environment; split mixed
manifests (`matched`, `capacity`, `public-s5-selection`, `public-s5`) with
`--runner s5` and `--exclude-runner s5`:

```bash
# in the PyTorch environment
python -m experiments.manifest run --manifest runs/manifests/capacity.json --all --exclude-runner s5 --skip-existing
# in the JAX environment
python -m experiments.manifest run --manifest runs/manifests/capacity.json --all --runner s5 --skip-existing
```

The `classical` runs (SysIdentPy) run on a CPU. Every PyTorch run writes
`summary.json` (best-validation and test errors, real and element parameter
counts, the full model configuration, training time, pole statistics), the
best-validation checkpoint `model_best.pt`, and `predictions_test.npz`.

GPU kernels are not bit-for-bit deterministic across hardware and library
versions, so retrained runs reproduce the reported means and standard
deviations up to seed-level noise.

## 5. Tables, figures, and reported numbers

```bash
python -m experiments.collect                 # results/runs.csv, one row per run
python paper/make_tables.py                   # paper/tables/*.tex
python paper/make_figures.py --checkpoints    # paper/figs/*.pdf
```

`experiments.collect` stops if a run is missing (`--allow-missing` collects a
partial set). `paper/make_tables.py` writes the 19 tables of the paper under the
names its LaTeX source uses (`tab_matched_symmetric.tex` for Table 1, …;
`tab_suffix_control`, `tab_iir_optimizer`, `tab_extrapolation`, and
`tab_ablation` for Tables 10, 11, 17, and 25), and prints the numbers quoted in
the text: the long-horizon ratios of Table 17 and Appendix I and the error
range of the unmodified LNO. Values are mean ± population standard deviation
over the stated seeds; parameter counts are real degrees of freedom. Training
times (Table 8, Figure 8) depend on the hardware.

`paper/make_figures.py` draws the schematics and the result figures from
`results/runs.csv`; with `--checkpoints`, it also draws the pole maps (Figure 9)
and prediction traces (Figure 10) from `runs/fixed` and writes the per-layer pole
statistics quoted in Appendix O to `results/pole_map_stats.csv`. Every script
can also be run alone; see `paper/figures/`.

## 6. Validation-only selection

The tuning study (Tables 14–15), the public benchmarks (Tables 19–20), and the
classical baselines select configurations by validation error only. Selection
runs read the test-free views, and their summaries carry no test error.

The reported studies (`tuning-confirm`, `public`, `public-s5`) are built with the
paper's selections, which are recorded in `experiments/protocols.py`
(`TUNING_SELECTED`, `PUBLIC_SELECTED`, `S5_PUBLIC_SELECTED`), so all studies can
run at once. `paper/make_tables.py` recomputes every selection from your
selection runs and prints a note when one differs from the selection the
reported runs use. To follow your own selections, select and rebuild:

```bash
python -m experiments.select --manifest runs/manifests/tuning-selection.json --output-dir runs/selection/tuning
python -m experiments.select --manifest runs/manifests/public-selection.json --output-dir runs/selection/public \
  --tie-break params
python -m experiments.select --manifest runs/manifests/public-s5-selection.json --output-dir runs/selection/public-s5 \
  --tie-break params
python -m experiments.build tuning-confirm --selected runs/selection/tuning/selected.json
python -m experiments.build public --selected runs/selection/public/selected.json
python -m experiments.build public-s5 --selected runs/selection/public-s5/selected.json
```

Rules: the lowest mean best-validation error over seeds 0–2. In the tuning
study, an exact tie keeps the candidate listed first; for S5, ties go to the
smaller state size. Each classical family is tuned on the validation split by
`run_external_sysid_baseline.py`, and the family with the lowest validation
error is reported (an exact tie goes to the alphabetically first family name,
which decides Wiener–Hammerstein, where both AOLS families reach identical
errors). The tuning-study confirmation seeds 0–4 overlap the selection seeds
0–2; the public benchmarks report the disjoint seeds 3–7.

## 7. Diagnostics (Appendix K)

These use the checkpoints of seeds 0–4 of `runs/public` (`--seeds` changes this):

```bash
python -m experiments.diagnostics.checkpoint_poles --data-root data/public \
  --checkpoint-root runs/public --output runs/diagnostics/checkpoint_poles.json
python -m experiments.diagnostics.signal_shift --data-root data/public --output-dir runs/diagnostics/shift
python -m experiments.diagnostics.export_predictions --data-root data/public \
  --checkpoint-root runs/public --output-dir runs/diagnostics/predictions
python -m experiments.diagnostics.prefix_sensitivity --prediction-root runs/diagnostics/predictions \
  --output-dir runs/diagnostics/prefix
# Supervised adaptation on official EMPS test labels: a diagnostic, never a benchmark result
python -m experiments.diagnostics.emps_adaptation --data-root data/public \
  --checkpoint-root runs/public --output-dir runs/diagnostics/emps_adaptation
```

## Protocol details

- **Metric.** Relative L2 error per example, averaged over examples; the test
  error is evaluated once, with the best-validation checkpoint.
- **Objectives.** FNO, LNO, S4D, and S5 minimize the task loss. ZNO's default
  objective adds a pole-safety term (λ = 1e-3, safe radius 0.95) and a suffix
  term (λ = 1e-2, context 512 steps); the 8.5k comparison of Table 1 switches
  both off (`--lambda-pole 0 --lambda-roll 0`), so every model minimizes the same
  task loss.
- **Recipes.** 600 epochs, batch size 32, and per-model learning rates (Table 4).
  The IIR-optimizer variants of Table 11 train for 800 epochs; the 1D benchmark
  follows the LNO release (Table 26). Official S5 uses learning rate 1e-3 on the
  synthetic tasks and 2e-3 on the public benchmarks.
- **Fixed configurations.** The difficulty sweep, the long-horizon evaluation,
  and the pole maps and traces use the configurations of Table 7; the difficulty
  sweep uses the resonant-ARMA configurations in every bin. Each long-horizon
  run retrains its configuration at $T=2048$ with the same seed and tests at
  the longer length.
- **Parameter counts.** Real degrees of freedom (complex scalars count twice).
  Summaries also record `parameter_count_elements` (`sum(p.numel())`), the
  convention of Table 26.
