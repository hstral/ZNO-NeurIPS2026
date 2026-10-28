"""Rebuild every configuration of the experiment manifests on CPU and compare its
parameter count with the run behind the paper.

The entries of each study are built exactly as ``python -m experiments.build``
writes them. PyTorch models are instantiated from the entry's training command
through the runner's own argument parser; official S5 counts use the closed form
of the adapter, which ``tests/test_s5_adapter.py`` checks against the Flax
parameter tree when JAX is available. The expected counts
(``experiments/configs/paper_parameter_counts.json``) are those of the runs that
produced the paper, as real degrees of freedom (complex scalars count twice) and
stored tensor elements; the counts printed in the paper's tables are a subset.
SysIdentPy baselines are skipped, since their term counts result from the fits.

No data, GPU, or training are needed. Usage::

    python -m experiments.verify_parameter_counts [--study matched ...]
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import json

from . import build as B
from . import protocols as P

COUNTS_FILE = P.CONFIG_DIR / "paper_parameter_counts.json"


def input_channels(case: str) -> int:
    if case in P.DIFFICULTY_CASES.values():
        return P.INPUT_CHANNELS["resonant_arma"]
    return P.INPUT_CHANNELS.get(P.CASE_TO_TASK[case], 1)  # public benchmarks: one input


def torch_model_config(entry: dict):
    """Model configuration and run arguments of a PyTorch entry, as its runner builds them."""
    one_d = entry["runner"] == "one_d"
    runner = importlib.import_module("run_1d_operator" if one_d else "run_discrete_operator")
    train = importlib.import_module(f"{'benchmark_1d' if one_d else 'benchmark_discrete'}.train")
    kwargs = runner.experiment_kwargs(entry["argv"][1:])
    accepted = set(inspect.signature(train._override_config).parameters) - {"base"}
    config = train._override_config(kwargs["case"].models[kwargs["model_name"]],
                                    **{k: v for k, v in kwargs.items() if k in accepted})
    return config, kwargs


def torch_counts(entry: dict) -> tuple[int, int]:
    import torch

    from benchmark_1d.parameters import parameter_counts

    one_d = entry["runner"] == "one_d"
    models = importlib.import_module(f"{'benchmark_1d' if one_d else 'benchmark_discrete'}.models")
    config, kwargs = torch_model_config(entry)
    if one_d:
        model = models.build_model(model_name=entry["model"], config=config, grid=torch.linspace(0, 1, 16),
                                   use_grid=kwargs["case"].use_grid)
    else:
        model = models.build_model(model_name=entry["model"], config=config,
                                   grid=torch.arange(16, dtype=torch.float32), use_grid=False,
                                   input_channels=input_channels(entry["case"]))
    counts = parameter_counts(model)
    return counts["parameter_count"], counts["parameter_count_elements"]


def s5_counts(entry: dict) -> tuple[int, int]:
    from s5_baseline.closed_form import s5_real_parameters

    argv = entry["argv"]

    def flag(name):
        return int(argv[argv.index(name) + 1])

    real = s5_real_parameters(input_channels=input_channels(entry["case"]), d_model=flag("--d-model"),
                              n_layers=flag("--n-layers"), ssm_size_base=flag("--ssm-size-base"))
    return real, real


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--study", action="append", choices=list(B.STUDIES), help="check only this study (repeatable)")
    args = parser.parse_args()
    expected = json.loads(COUNTS_FILE.read_text(encoding="utf-8"))
    ctx = argparse.Namespace(layout=B.Layout("data", "runs"), zno_backend="triton", device="cpu",
                             selected=None, extra_args=[])
    cache, total, failures = {}, 0, []
    for study in args.study or list(B.STUDIES):
        runs = configs = 0
        for entry in B.build(study, ctx):
            if entry["runner"] == "classical":
                continue
            key = (entry["runner"], entry["case"], entry["model"], tuple(entry["model_args"]))
            if key not in cache:
                cache[key] = s5_counts(entry) if entry["runner"] == "s5" else torch_counts(entry)
                configs += 1
            # Entries record their study (the S5 runs built by public-s5 belong to "public").
            want = expected.get(entry["study"], {}).get(entry["case"], {}).get(entry["model"], {}).get(entry["tag"])
            label = f"{entry['study']}/{entry['case']}/{entry['model']}/{entry['tag']}"
            if want is None:
                failures.append(f"{label}: no expected count")
            elif list(cache[key]) != want:
                failures.append(f"{label}: computed {list(cache[key])}, expected {want}")
            runs += 1
        total += runs
        print(f"{study:26s} {runs:5d} runs checked")
    for failure in dict.fromkeys(failures):
        print("MISMATCH", failure)
    print(f"{total} runs, {len(cache)} distinct configurations checked; {len(set(failures))} mismatches")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
