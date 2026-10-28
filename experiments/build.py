"""Build run manifests for the studies reported in the paper.

Every study uses a conventional layout: datasets under ``--data-dir`` (default
``data/``, see REPRODUCTION.md) and runs under ``--runs-dir`` (default ``runs/``),
with one run directory per study. Manifests are written to
``<runs-dir>/manifests/<study>.json``.

Examples::

    python -m experiments.build matched            # Table 1 (75 runs)
    python -m experiments.build all                # every study: 1,656 GPU runs and 20 CPU fits
    python -m experiments.build matched --zno-backend triton_fast
    python -m experiments.build --list

All configurations are defined in ``experiments/protocols.py``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import protocols as P
from .manifest import classical_entry, discrete_entry, one_d_entry, s5_entry, write_manifest


class Layout:
    """Conventional data and run directories."""

    def __init__(self, data_dir: str, runs_dir: str):
        self.data = Path(data_dir)
        self.runs = Path(runs_dir)

    controlled = property(lambda self: self.data / "controlled")
    difficulty = property(lambda self: self.data / "difficulty")
    public = property(lambda self: self.data / "public")
    selection_controlled = property(lambda self: self.data / "selection_views" / "controlled")
    selection_public = property(lambda self: self.data / "selection_views" / "public")
    s5_controlled = property(lambda self: self.data / "s5_views" / "controlled")
    s5_public = property(lambda self: self.data / "s5_views" / "public")

    def extrapolation(self, length: int) -> Path:
        return self.data / "extrapolation" / f"T{length}"

    def difficulty_extrapolation(self, length: int) -> Path:
        return self.data / "difficulty_extrapolation" / f"T{length}"

    def out(self, study: str) -> Path:
        return self.runs / study


def _discrete(ctx, study, case, model, tag, seed, data_root, model_args, test_access=True, **extra):
    if model == "zno":
        model_args = (*model_args, "--zno-kernel-backend", ctx.zno_backend)
    return discrete_entry(study=study, case=case, model=model, tag=tag, seed=seed, data_root=data_root,
                          output_root=ctx.layout.out(study), model_args=model_args, device=ctx.device,
                          test_access=test_access, **extra)


# --- studies ---------------------------------------------------------------

def matched(ctx):
    """Table 1 (8.5k comparison, task loss) and Figure 8 (timing): FNO, LNO, S4D, S5, ZNO."""
    entries = []
    for case in P.CONTROLLED_CASES.values():
        for seed in P.SEEDS:
            for model in ("fno", "lno_stable", "s4d_official", "zno"):
                model_args = P.MATCHED[model]
                if model == "zno":
                    model_args = (*model_args, *P.ZNO_TASK_LOSS)
                entries.append(_discrete(ctx, "matched", case, model, "8.5k", seed,
                                         ctx.layout.controlled, model_args))
            entries.append(s5_entry(study="matched", case=case, tag="8.5k", seed=seed,
                                    view_root=ctx.layout.s5_controlled, output_root=ctx.layout.out("matched"),
                                    model_args=P.S5_MATCHED, evaluate_test=True, run_kind="matched"))
    return entries


def zno_default(ctx):
    """Table 9 and the 8.5k ZNO entries of the capacity study: ZNO with its default objective."""
    return [_discrete(ctx, "zno-default", case, "zno", "8.5k", seed, ctx.layout.controlled, P.MATCHED["zno"])
            for case in P.CONTROLLED_CASES.values() for seed in P.SEEDS]


def suffix_control(ctx):
    """Table 10: suffix loss for FNO and S4D, ZNO without its auxiliary terms."""
    return [_discrete(ctx, "suffix-control", case, model, tag, seed, ctx.layout.controlled, model_args)
            for case in P.CONTROLLED_CASES.values()
            for (model, tag), model_args in P.SUFFIX_CONTROL.items()
            for seed in P.SEEDS]


def iir_optimizer(ctx):
    """Table 11: ZNO on the IIR cascade with modified learning rate and clipping (800 epochs)."""
    case = P.CONTROLLED_CASES["iir_cascade"]
    return [_discrete(ctx, "iir-optimizer", case, "zno", tag, seed, ctx.layout.controlled,
                      (*P.MATCHED["zno"], *variant))
            for tag, variant in P.IIR_OPTIMIZER.items() for seed in P.SEEDS]


def capacity(ctx):
    """Tables 12-13, Figure 3b-d: 4k, 16k, and 32k operating points (8.5k reuses matched/zno-default)."""
    config = P.capacity_configs()
    entries = []
    for budget, payload in config["budgets"].items():
        for task, meta in config["tasks"].items():
            for seed in config["seeds"]:
                for model, spec in payload["models"].items():
                    if model == "s5_official":
                        entries.append(s5_entry(
                            study="capacity", case=meta["case"], tag=budget, seed=seed,
                            view_root=ctx.layout.s5_controlled, output_root=ctx.layout.out("capacity"),
                            model_args=tuple(spec["args"]), evaluate_test=True,
                            run_kind=f"capacity_{budget}", budget=budget))
                    else:
                        entries.append(_discrete(ctx, "capacity", meta["case"], model, budget, seed,
                                                 ctx.layout.controlled, tuple(spec["args"]), budget=budget))
    return entries


def tuning_selection(ctx):
    """Table 15: 44 candidates x seeds 0-2 on test-free selection views."""
    return [_discrete(ctx, "tuning-selection", case, model, tag, seed, ctx.layout.selection_controlled,
                      model_args, test_access=False)
            for case in P.CONTROLLED_CASES.values()
            for model in ("fno", "s4d_official", "zno")
            for tag, model_args in P.tuning_candidates(case, model).items()
            for seed in P.SELECTION_SEEDS]


def _chosen(ctx, default: dict) -> dict:
    """The paper's selection ``{(case, model): tag}``, overridden by the ``--selected``
    files (written by ``python -m experiments.select``) for the pairs they contain."""
    chosen = dict(default)
    for path in ctx.selected or []:
        for row in json.loads(Path(path).read_text(encoding="utf-8"))["selected"]:
            if (row["case"], row["model"]) in chosen:
                chosen[(row["case"], row["model"])] = row["tag"]
    return chosen


def tuning_confirm(ctx):
    """Table 14: selected configurations x seeds 0-4 (paper selection, or --selected)."""
    chosen = _chosen(ctx, {(case, model): tag for case, models in P.TUNING_SELECTED.items()
                           for model, tag in models.items()})
    return [_discrete(ctx, "tuning-confirm", case, model, tag, seed, ctx.layout.controlled,
                      P.tuning_candidates(case, model)[tag])
            for (case, model), tag in sorted(chosen.items()) for seed in P.SEEDS]


def lno_reference(ctx):
    """Table 16, second column: stabilized LNO with width 12 and 32 modes."""
    return [_discrete(ctx, "lno-reference", case, "lno_stable", "w12-m32", seed, ctx.layout.controlled,
                      P.LNO_REFERENCE)
            for case in P.CONTROLLED_CASES.values() for seed in P.SEEDS]


def lno_unconstrained(ctx):
    """Appendix H: the unmodified reference LNO at the same topology (diverges)."""
    return [_discrete(ctx, "lno-unconstrained", case, "lno", "w12-m32", seed, ctx.layout.controlled,
                      P.LNO_REFERENCE)
            for case in P.LNO_UNCONSTRAINED_CASES for seed in P.SELECTION_SEEDS]


def fixed(ctx):
    """Table 7 configurations at T=2048: long-horizon baseline, pole maps, traces."""
    return [_discrete(ctx, "fixed", case, model, "fixed", seed, ctx.layout.controlled, model_args, length=2048)
            for case, models in P.FIXED.items() for model, model_args in models.items() for seed in P.SEEDS]


def difficulty(ctx):
    """Figure 4: near-unit-circle difficulty bins with the resonant-ARMA fixed configurations."""
    arma = P.FIXED[P.CONTROLLED_CASES["resonant_arma"]]
    return [_discrete(ctx, "difficulty", case, model, "fixed", seed, ctx.layout.difficulty, arma[model], length=2048)
            for case in P.DIFFICULTY_CASES.values() for model in ("fno", "s4d_official", "zno") for seed in P.SEEDS]


def extrapolation(ctx):
    """Table 17, Figure 5: retrain the fixed configurations, test at T=4096 and 8192."""
    return [_discrete(ctx, "extrapolation", case, model, f"T{length}", seed, ctx.layout.extrapolation(length),
                      model_args, length=length)
            for length in P.EXTRAPOLATION_LENGTHS
            for case, models in P.FIXED.items() for model, model_args in models.items() for seed in P.SEEDS]


def difficulty_extrapolation(ctx):
    """Appendix I: the difficulty bins tested at T=4096 and 8192."""
    arma = P.FIXED[P.CONTROLLED_CASES["resonant_arma"]]
    return [_discrete(ctx, "difficulty-extrapolation", case, model, f"T{length}", seed,
                      ctx.layout.difficulty_extrapolation(length), arma[model], length=length)
            for length in P.EXTRAPOLATION_LENGTHS
            for case in P.DIFFICULTY_CASES.values() for model in ("fno", "s4d_official", "zno") for seed in P.SEEDS]


def public_selection(ctx):
    """Table 20: public FNO/S4D/ZNO grids (113 candidates) x seeds 0-2 on test-free views."""
    return [_discrete(ctx, "public-selection", case, model, tag, seed, ctx.layout.selection_public,
                      model_args, test_access=False)
            for case in P.PUBLIC_CASES.values()
            for model in ("fno", "s4d_official", "zno")
            for tag, model_args in P.public_candidates(case, model).items()
            for seed in P.SELECTION_SEEDS]


def public(ctx):
    """Table 2: selected FNO/S4D/ZNO configurations x seeds 0-7 (reported: 3-7; diagnostics: 0-4)."""
    chosen = _chosen(ctx, {(case, model): tag for case, models in P.PUBLIC_SELECTED.items()
                           for model, tag in models.items()})
    return [_discrete(ctx, "public", case, model, tag, seed, ctx.layout.public,
                      P.public_candidates(case, model)[tag])
            for (case, model), tag in chosen.items() for seed in P.PUBLIC_TRAIN_SEEDS]


def public_s5_selection(ctx):
    """Table 19: S5 base state sizes x seeds 0-2 (validation only)."""
    return [s5_entry(study="public-s5-selection", case=case, tag=f"P{size}", seed=seed,
                     view_root=ctx.layout.s5_public, output_root=ctx.layout.out("public-s5-selection"),
                     model_args=(*P.S5_PUBLIC_RECIPE, "--ssm-size-base", str(size)),
                     evaluate_test=False, run_kind="public_capacity_selection")
            for case in P.PUBLIC_CASES.values() for size in P.S5_PUBLIC_STATE_SIZES for seed in P.SELECTION_SEEDS]


def public_s5(ctx):
    """Table 2: selected S5 state sizes x seeds 3-7 (paper selection, or --selected)."""
    chosen = _chosen(ctx, {(case, "s5_official"): f"P{size}" for case, size in P.S5_PUBLIC_SELECTED.items()})
    sizes = {case: int(tag[1:]) for (case, _), tag in chosen.items()}
    return [s5_entry(study="public", case=case, tag=f"P{size}", seed=seed, view_root=ctx.layout.s5_public,
                     output_root=ctx.layout.out("public"),
                     model_args=(*P.S5_PUBLIC_RECIPE, "--ssm-size-base", str(size)),
                     evaluate_test=True, run_kind="public_final")
            for case, size in sizes.items() for seed in P.PUBLIC_REPORT_SEEDS]


def classical(ctx):
    """Table 21: four SysIdentPy NARMAX families per public benchmark (CPU)."""
    return [classical_entry(study="classical", case=case, family=family, data_root=ctx.layout.public,
                            output_root=ctx.layout.out("classical"))
            for case in P.PUBLIC_CASES.values() for family in P.CLASSICAL_FAMILIES]


def _interventions(ctx, study, case, variants):
    return [_discrete(ctx, study, case, "zno", tag, seed, ctx.layout.public, variant)
            for tag, variant in variants.items() for seed in P.DIAGNOSTIC_SEEDS]


def emps_interventions(ctx):
    """Table 23: EMPS radius-map variants (seeds 0-4)."""
    return _interventions(ctx, "emps-interventions", P.PUBLIC_CASES["emps"], P.EMPS_INTERVENTIONS)


def tanks_interventions(ctx):
    """Table 24: Cascaded Tanks initialization variants (seeds 0-4)."""
    return _interventions(ctx, "tanks-interventions", P.PUBLIC_CASES["cascaded_tanks"], P.TANKS_INTERVENTIONS)


def ablation(ctx):
    """Table 25, Figure 6: one knob at a time from the ZNO anchor."""
    return [_discrete(ctx, "ablation", case, "zno", tag, seed, ctx.layout.controlled, model_args)
            for case in P.ABLATION_CASES for tag, model_args in P.ablation_variants(case).items()
            for seed in P.SEEDS]


def one_d(ctx):
    """Figure 7: 1D LNO ODE benchmark with the matched-budget runner defaults (seeds 0-2)."""
    entries = []
    for case in P.ONE_D_CASES:
        for model in P.ONE_D_MODELS:
            model_args = ("--zno-kernel-backend", ctx.zno_backend) if model == "zno" else ()
            for seed in P.ONE_D_SEEDS:
                entries.append(one_d_entry(study="one-d", case=case, model=model, tag="matched", seed=seed,
                                           output_root=ctx.layout.out("one-d"), model_args=model_args,
                                           device=ctx.device))
    return entries


STUDIES = {
    "matched": matched,
    "zno-default": zno_default,
    "suffix-control": suffix_control,
    "iir-optimizer": iir_optimizer,
    "capacity": capacity,
    "tuning-selection": tuning_selection,
    "tuning-confirm": tuning_confirm,
    "lno-reference": lno_reference,
    "lno-unconstrained": lno_unconstrained,
    "fixed": fixed,
    "difficulty": difficulty,
    "extrapolation": extrapolation,
    "difficulty-extrapolation": difficulty_extrapolation,
    "public-selection": public_selection,
    "public": public,
    "public-s5-selection": public_s5_selection,
    "public-s5": public_s5,
    "classical": classical,
    "emps-interventions": emps_interventions,
    "tanks-interventions": tanks_interventions,
    "ablation": ablation,
    "one-d": one_d,
}


def build(study: str, ctx) -> list[dict]:
    entries = STUDIES[study](ctx)
    if ctx.extra_args:
        for entry in entries:
            if entry["runner"] != "classical":
                entry["argv"] = [*entry["argv"], *ctx.extra_args]
                entry["extra_args"] = list(ctx.extra_args)
    return entries


def main() -> None:
    listing = "\n".join(f"  {name:26s} {fn.__doc__.strip().splitlines()[0]}" for name, fn in STUDIES.items())
    parser = argparse.ArgumentParser(description=__doc__ + "\nStudies:\n" + listing,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("study", nargs="?", choices=["all", *STUDIES])
    parser.add_argument("--list", action="store_true", help="list the studies and their run counts")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--manifest", type=Path, help="manifest path (default: <runs-dir>/manifests/<study>.json)")
    parser.add_argument("--zno-backend", choices=["triton", "triton_fast", "cuda"], default="triton",
                        help="ZNO kernel backend (default: triton, the kernel of the paper's runs)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--selected", action="append",
                        help="selected.json written by `python -m experiments.select` (repeatable); "
                             "overrides the paper's selection in tuning-confirm, public, and public-s5")
    parser.add_argument("--extra-args", default="",
                        help="arguments appended to every training command, e.g. '--epochs 1' for a dry run")
    args = parser.parse_args()
    ctx = argparse.Namespace(layout=Layout(args.data_dir, args.runs_dir), zno_backend=args.zno_backend,
                             device=args.device, selected=args.selected, extra_args=args.extra_args.split())
    if args.list or not args.study:
        total = 0
        for name in STUDIES:
            entries = build(name, ctx)
            total += len(entries)
            print(f"{name:26s} {len(entries):5d} runs   {STUDIES[name].__doc__.strip().splitlines()[0]}")
        print(f"{'total':26s} {total:5d} runs")
        return
    names = list(STUDIES) if args.study == "all" else [args.study]
    for name in names:
        manifest = args.manifest if (args.manifest and len(names) == 1) else Path(args.runs_dir) / "manifests" / f"{name}.json"
        write_manifest(manifest, build(name, ctx))


if __name__ == "__main__":
    main()
