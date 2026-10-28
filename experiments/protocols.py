"""Single source of truth for every configuration reported in the paper.

Each study of the paper (Appendix D, Table 5, "Studies in this paper") is expressed
here as data: tasks, seeds, model arguments for ``run_discrete_operator.py`` /
``run_1d_operator.py`` (PyTorch models) and ``python -m s5_baseline.train``
(official S5 core), candidate grids, and selected configurations.
``experiments/build.py`` turns these definitions into run manifests.

Conventions
-----------
* Model arguments are appended to the runner defaults of
  ``benchmark_discrete/configs.py`` (FNO: Adam lr 2e-3, StepLR(100, 0.5);
  S4D: lr 5e-4, clip 1.0, StepLR(150, 0.5); ZNO and LNO: lr 2e-3,
  StepLR(150, 0.5); weight decay 1e-4, 600 epochs, batch 32, best-validation
  checkpoint).
* ZNO's *default objective* is the runner default: task loss plus the
  pole-safety term (lambda 1e-3) and the suffix term (lambda 1e-2, context T/4).
  ``ZNO_TASK_LOSS`` switches both off. FNO, LNO, S4D, and S5 always use the task
  loss, except in the suffix-loss control.
* ZNO kernel-backend flags are not part of these arguments; ``experiments.build``
  appends ``--zno-kernel-backend`` (default ``triton``, the kernel of the paper's
  runs; ``triton_fast`` is the faster, numerically equivalent scan).
* Parameter counts are real degrees of freedom (complex scalars count twice).
"""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = Path(__file__).resolve().parent / "configs"

# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------
CONTROLLED_CASES = {
    "resonant_arma": "resonant_arma_near_unit_circle_v4",
    "iir_cascade": "high_order_iir_cascade_v5",
    "nonlinear_narx": "narx_softplus",
}
INPUT_CHANNELS = {"resonant_arma": 7, "iir_cascade": 16, "nonlinear_narx": 1}
PUBLIC_CASES = {
    "silverbox": "silverbox_public_v1",
    "wiener_hammerstein": "wiener_hammer_public_v1",
    "ced": "ced_public_v1",
    "cascaded_tanks": "cascaded_tanks_public_v1",
    "emps": "emps_public_v1",
}
DIFFICULTY_CASES = {
    "r0900_0930": "resonant_arma_lm_r0900_0930_v1",
    "r0930_0960": "resonant_arma_lm_r0930_0960_v1",
    "r0960_0980": "resonant_arma_lm_r0960_0980_v1",
    "r0980_0990": "resonant_arma_lm_r0980_0990_v1",
    "r0990_0995": "resonant_arma_lm_r0990_0995_v1",
}
ONE_D_CASES = ("pendulum_c0", "pendulum_c05", "duffing_c0", "duffing_c05", "lorenz_rho5", "lorenz_rho10")
EXTRAPOLATION_LENGTHS = (4096, 8192)
CASE_TO_TASK = {
    **{case: task for task, case in CONTROLLED_CASES.items()},
    **{case: task for task, case in PUBLIC_CASES.items()},
    **{case: task for task, case in DIFFICULTY_CASES.items()},
    **{case: case for case in ONE_D_CASES},
}

# Arguments for benchmark_discrete/build_external_benchmark.py (public protocol).
PUBLIC_DATA_BUILD = {
    "silverbox_public_v1": ("--sequence-length", "2048", "--train-size", "512",
                            "--val-size", "128", "--test-size", "256"),
    "wiener_hammer_public_v1": ("--sequence-length", "2048", "--train-size", "512",
                                "--val-size", "128", "--test-size", "256"),
    "emps_public_v1": ("--sequence-length", "2048", "--train-size", "512",
                       "--val-size", "128", "--test-size", "256",
                       "--train-stride", "128", "--val-stride", "128", "--test-stride", "128"),
    "cascaded_tanks_public_v1": ("--sequence-length", "128", "--train-size", "128",
                                 "--val-size", "16", "--test-size", "64",
                                 "--train-stride", "8", "--val-stride", "8", "--test-stride", "8"),
    "ced_public_v1": ("--sequence-length", "48", "--train-size", "128",
                      "--val-size", "16", "--test-size", "24",
                      "--train-stride", "2", "--val-stride", "2", "--test-stride", "2"),
}

# ---------------------------------------------------------------------------
# Seeds
# ---------------------------------------------------------------------------
SEEDS = (0, 1, 2, 3, 4)
SELECTION_SEEDS = (0, 1, 2)
ONE_D_SEEDS = (0, 1, 2)
# FNO/S4D/ZNO on the public benchmarks: seeds 0-7 are trained; the paper reports
# seeds 3-7 and the EMPS/Cascaded Tanks diagnostics use seeds 0-4.
PUBLIC_TRAIN_SEEDS = (0, 1, 2, 3, 4, 5, 6, 7)
PUBLIC_REPORT_SEEDS = (3, 4, 5, 6, 7)
DIAGNOSTIC_SEEDS = (0, 1, 2, 3, 4)

# ---------------------------------------------------------------------------
# Objectives
# ---------------------------------------------------------------------------
ZNO_TASK_LOSS = ("--lambda-roll", "0", "--lambda-pole", "0", "--rollout-context-steps", "512")
ZNO_DEFAULT_OBJECTIVE = ("--lambda-roll", "1e-2", "--lambda-pole", "1e-3", "--rollout-context-steps", "512")
SUFFIX_FOR_BASELINES = ("--lambda-roll", "1e-2", "--rollout-context-steps", "512", "--rollout-on-all-models")

# ---------------------------------------------------------------------------
# 8.5k comparison (Tables 1 and 6; Figure 8, timing)
# ---------------------------------------------------------------------------
MATCHED = {
    "fno": ("--width", "4", "--depth", "4", "--modes", "126"),
    "lno_stable": ("--width", "14", "--depth", "1", "--modes", "8"),
    "s4d_official": ("--width", "20", "--depth", "4", "--state-dim", "12"),
    "zno": ("--width", "20", "--depth", "4", "--rank", "8", "--num-poles", "40", "--fir-order", "4"),
}
# Official S5: width 20, four layers, base state size 40 (20 stored conjugate
# modes), learning rate 1e-3, grouped Adam/AdamW, warm-up + cosine, no clipping.
S5_MATCHED = ("--d-model", "20", "--n-layers", "4", "--ssm-size-base", "40",
              "--learning-rate", "1e-3", "--weight-decay", "0.05")

# ---------------------------------------------------------------------------
# Objective and optimization controls (Appendix E, Tables 9-11)
# ---------------------------------------------------------------------------
SUFFIX_CONTROL = {
    # model, tag: arguments. "ZNO no suffix/pole" is the ZNO run of the 8.5k comparison.
    ("fno", "suffix"): ("--width", "4", "--depth", "4", "--modes", "256", *SUFFIX_FOR_BASELINES),
    ("s4d_official", "suffix"): (*MATCHED["s4d_official"], *SUFFIX_FOR_BASELINES),
    ("zno", "base"): (*MATCHED["zno"], *ZNO_DEFAULT_OBJECTIVE),
    ("zno", "no-suffix"): (*MATCHED["zno"], "--lambda-roll", "0", "--lambda-pole", "1e-3",
                           "--rollout-context-steps", "512"),
}
# IIR cascade, 8.5k ZNO, default objective. These runs trained for 800 epochs.
IIR_OPTIMIZER = {
    "lr1e-3-clip0.25": ("--learning-rate", "1e-3", "--grad-clip", "0.25", "--epochs", "800"),
    "lr1e-3-clip0.5": ("--learning-rate", "1e-3", "--grad-clip", "0.5", "--epochs", "800"),
    "lr8e-4-clip0.25-safe0.90": ("--learning-rate", "8e-4", "--grad-clip", "0.25", "--epochs", "800",
                                 "--safe-radius", "0.90"),
    "lr8e-4-clip0.5": ("--learning-rate", "8e-4", "--grad-clip", "0.5", "--epochs", "800"),
}

# ---------------------------------------------------------------------------
# Capacity study (Appendix F, Tables 12-13)
# ---------------------------------------------------------------------------
CAPACITY_CONFIG_FILE = CONFIG_DIR / "capacity_study.json"


def capacity_configs() -> dict:
    return json.loads(CAPACITY_CONFIG_FILE.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Tuning study (Appendix G, Tables 14-15): 44 candidates, 132 selection runs
# ---------------------------------------------------------------------------
FNO_TUNING_CANDIDATES = {
    "baseline": (),  # runner default: width 2, 128 modes
    "w4": ("--width", "4"),
    "w4-m256": ("--width", "4", "--modes", "256"),
    "w8-m256": ("--width", "8", "--modes", "256"),
}
S4D_TUNING_CANDIDATES = {
    "sd12": ("--state-dim", "12"),
    "sd16": ("--state-dim", "16"),
    "sd24": ("--state-dim", "24"),
    "sd32": ("--state-dim", "32"),
    "sd24-lr3e4": ("--state-dim", "24", "--learning-rate", "3e-4"),
    "sd32-lr3e4": ("--state-dim", "32", "--learning-rate", "3e-4"),
}
ZNO_TUNING_CANDIDATES = {  # runner default: width 20, rank 8, F=4, K per case
    "resonant_arma_near_unit_circle_v4": {
        "baseline": (),
        "p64": ("--num-poles", "64"),
        "rank12": ("--rank", "12"),
        "p64-r12": ("--num-poles", "64", "--rank", "12"),
        "p64-r12-fir0": ("--num-poles", "64", "--rank", "12", "--fir-order", "0"),
    },
    "high_order_iir_cascade_v5": {
        "baseline": (),
        "p80": ("--num-poles", "80"),
        "rank12": ("--rank", "12"),
        "p80-r12": ("--num-poles", "80", "--rank", "12"),
        "p80-r12-safe090": ("--num-poles", "80", "--rank", "12", "--safe-radius", "0.90"),
    },
    "narx_softplus": {
        "baseline": (),
        "rank12": ("--rank", "12"),
        "p32": ("--num-poles", "32"),
        "p32-r12": ("--num-poles", "32", "--rank", "12"),
    },
}


def tuning_candidates(case: str, model: str) -> dict[str, tuple[str, ...]]:
    if model == "fno":
        return FNO_TUNING_CANDIDATES
    if model == "s4d_official":
        return S4D_TUNING_CANDIDATES
    if model == "zno":
        return ZNO_TUNING_CANDIDATES[case]
    raise ValueError(model)


# ---------------------------------------------------------------------------
# Tuning-study selection reported in the paper (Table 14): lowest mean
# best-validation error over seeds 0-2. ``experiments.select`` recomputes it from
# your own selection runs.
# ---------------------------------------------------------------------------
TUNING_SELECTED = {
    "resonant_arma_near_unit_circle_v4": {"fno": "w8-m256", "s4d_official": "sd24", "zno": "p64-r12"},
    "high_order_iir_cascade_v5": {"fno": "w8-m256", "s4d_official": "sd16", "zno": "p80-r12"},
    "narx_softplus": {"fno": "w4-m256", "s4d_official": "sd32", "zno": "rank12"},
}

# ---------------------------------------------------------------------------
# Fixed configurations (Table 7): difficulty sweep, long-horizon evaluation,
# pole maps and traces. ZNO uses its default objective.
# ---------------------------------------------------------------------------
FIXED = {
    "resonant_arma_near_unit_circle_v4": {
        "fno": ("--width", "8", "--depth", "4", "--modes", "256"),
        "s4d_official": ("--width", "20", "--depth", "4", "--state-dim", "24"),
        "zno": ("--width", "20", "--depth", "4", "--rank", "12", "--num-poles", "64", "--fir-order", "0"),
    },
    "high_order_iir_cascade_v5": {
        "fno": ("--width", "8", "--depth", "4", "--modes", "256"),
        "s4d_official": ("--width", "20", "--depth", "4", "--state-dim", "16"),
        "zno": ("--width", "20", "--depth", "4", "--rank", "12", "--num-poles", "80", "--fir-order", "4"),
    },
    "narx_softplus": {
        "fno": ("--width", "4", "--depth", "4", "--modes", "256"),
        "s4d_official": ("--width", "20", "--depth", "4", "--state-dim", "16"),
        "zno": ("--width", "20", "--depth", "4", "--rank", "8", "--num-poles", "16", "--fir-order", "4"),
    },
}

# ---------------------------------------------------------------------------
# Stabilized LNO (Appendix H, Table 16): the 8.5k setting is part of the 8.5k comparison.
# ---------------------------------------------------------------------------
LNO_REFERENCE = ("--width", "12", "--depth", "1", "--modes", "32")
# Unmodified reference LNO at the same topology (ARMA and NARX, seeds 0-2):
# trains to unbounded errors, which motivates the stabilized variant.
LNO_UNCONSTRAINED_CASES = ("resonant_arma_near_unit_circle_v4", "narx_softplus")

# ---------------------------------------------------------------------------
# Public benchmarks (Appendix J, Tables 2 and 18-22)
# ---------------------------------------------------------------------------
PUBLIC_GRID_FILE = CONFIG_DIR / "public_candidate_grids.json"


def public_candidates(case: str, model: str) -> dict[str, tuple[str, ...]]:
    grids = json.loads(PUBLIC_GRID_FILE.read_text(encoding="utf-8"))
    return {tag: tuple(args.split()) for tag, args in grids[case][model].items()}


# Configurations selected by validation over seeds 0-2 (Table 20).
PUBLIC_SELECTED = {
    "silverbox_public_v1": {"fno": "w8m384", "s4d_official": "sd24", "zno": "p80-r12-f8"},
    "wiener_hammer_public_v1": {"fno": "w8m384", "s4d_official": "sd32", "zno": "p64-r16-f4"},
    "ced_public_v1": {"fno": "w6m24", "s4d_official": "sd8", "zno": "p24-r8-f4"},
    "cascaded_tanks_public_v1": {"fno": "w6m32", "s4d_official": "sd12", "zno": "p32-r12-f4"},
    "emps_public_v1": {"fno": "w8m256", "s4d_official": "sd12", "zno": "p32-r8-f4"},
}
# Official S5 on the public benchmarks: width 20, four layers, base state size
# selected from these candidates (ties go to the smaller size), learning rate 2e-3.
S5_PUBLIC_STATE_SIZES = (16, 24, 32, 48, 64)
S5_PUBLIC_SELECTED = {
    "silverbox_public_v1": 64,
    "wiener_hammer_public_v1": 48,
    "ced_public_v1": 32,
    "cascaded_tanks_public_v1": 16,
    "emps_public_v1": 16,
}
S5_PUBLIC_RECIPE = ("--d-model", "20", "--n-layers", "4",
                    "--learning-rate", "2e-3", "--weight-decay", "0.05")
CLASSICAL_FAMILIES = ("sysid_aols_poly", "sysid_aols_fourier",
                      "sysid_frols_poly", "sysid_frols_fourier")

# ---------------------------------------------------------------------------
# Diagnostics (Appendix K, Tables 23-24): ZNO default objective unless the variant changes it.
# ---------------------------------------------------------------------------
EMPS_INTERVENTIONS = {
    "hard0999-penalty": ("--num-poles", "32", "--rank", "8", "--fir-order", "4",
                         "--radius-limit", "0.999", "--safe-radius", "0.95",
                         "--lambda-pole", "1e-3"),
    "hard105-penalty": ("--num-poles", "32", "--rank", "8", "--fir-order", "4",
                        "--radius-limit", "1.05", "--safe-radius", "0.95",
                        "--lambda-pole", "1e-3"),
    "unbounded-penalty": ("--num-poles", "32", "--rank", "8", "--fir-order", "4",
                          "--zno-pole-parameterization", "z_unbounded",
                          "--safe-radius", "0.95", "--lambda-pole", "1e-3"),
    "unbounded-no-penalty": ("--num-poles", "32", "--rank", "8", "--fir-order", "4",
                             "--zno-pole-parameterization", "z_unbounded",
                             "--safe-radius", "0.95", "--lambda-pole", "0"),
}
TANKS_INTERVENTIONS = {
    "safe099": ("--num-poles", "32", "--rank", "12", "--fir-order", "4",
                "--radius-limit", "0.999", "--safe-radius", "0.99", "--lambda-pole", "1e-3"),
    "no-pole-penalty": ("--num-poles", "32", "--rank", "12", "--fir-order", "4",
                        "--radius-limit", "0.999", "--safe-radius", "0.95", "--lambda-pole", "0"),
    "high-radius-init-no-penalty": ("--num-poles", "32", "--rank", "12", "--fir-order", "4",
                                    "--radius-limit", "0.999", "--safe-radius", "0.95",
                                    "--zno-pole-init-radius-low", "0.90",
                                    "--zno-pole-init-radius-high", "0.995",
                                    "--lambda-pole", "0"),
}

# ---------------------------------------------------------------------------
# Internal ZNO ablation (Appendix L, Table 25): one knob at a time from the anchor
# (width 20, depth 4, r=8, F=4, rho_safe=0.95; K=32 on ARMA, K=48 on IIR).
# ---------------------------------------------------------------------------
ABLATION_CASES = {
    "resonant_arma_near_unit_circle_v4": {"anchor": 32, "low": 16, "high": 64},
    "high_order_iir_cascade_v5": {"anchor": 48, "low": 24, "high": 80},
}


def ablation_variants(case: str) -> dict[str, tuple[str, ...]]:
    poles = ABLATION_CASES[case]
    anchor = ("--width", "20", "--depth", "4", "--rank", "8", "--fir-order", "4")
    k = ("--num-poles", str(poles["anchor"]))
    return {
        "anchor": (*anchor, *k),
        "rank4": (*anchor, *k, "--rank", "4"),
        "rank12": (*anchor, *k, "--rank", "12"),
        "poles-low": (*anchor, "--num-poles", str(poles["low"])),
        "poles-high": (*anchor, "--num-poles", str(poles["high"])),
        "fir0": (*anchor, *k, "--fir-order", "0"),
        "fir2": (*anchor, *k, "--fir-order", "2"),
        "fir8": (*anchor, *k, "--fir-order", "8"),
        "safe0.90": (*anchor, *k, "--safe-radius", "0.90"),
        "safe0.99": (*anchor, *k, "--safe-radius", "0.99"),
    }


# ---------------------------------------------------------------------------
# 1D LNO ODE benchmark (Appendix M, Table 26): the runner defaults of benchmark_1d/configs.py
# are the matched-budget configurations; FNO and LNO use the task loss, ZNO its
# default objective.
# ---------------------------------------------------------------------------
ONE_D_MODELS = ("lno", "fno", "zno")
