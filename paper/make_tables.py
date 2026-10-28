"""Generate the paper's result tables (LaTeX) from ``results/runs.csv``.

``results/runs.csv`` is written by ``python -m experiments.collect`` from your own
runs. Every number of the tables is computed from it: values are mean +/-
population standard deviation (ddof=0) over the stated seeds, parameter counts are
real degrees of freedom (complex scalars count twice), and the configurations
marked as selected are recomputed from the validation-only selection runs.

Files are written to ``paper/tables/`` under the names the paper's LaTeX source
inputs (``tab_matched_symmetric.tex``, ``tab_public.tex``, ...). The four
tables the paper typesets inline are written as ``tab_suffix_control.tex``
(Table 10), ``tab_iir_optimizer.tex`` (Table 11), ``tab_extrapolation.tex``
(Table 17), and ``tab_ablation.tex`` (Table 25). Training times in
``tab_compute.tex`` depend on the hardware.

Usage::

    python paper/make_tables.py [--results results/runs.csv] [--out paper/tables] [--only tab_public ...]
"""

from __future__ import annotations

import argparse
from pathlib import Path

from runs_table import (PUBLIC_REPORT_SEEDS, REPO, SEEDS, SELECTION_SEEDS, TASKS, Cell, P, best_of, cfg_int,
                        load, select_min)

TASK_LABEL = {"resonant_arma": "Resonant ARMA", "iir_cascade": "IIR cascade", "nonlinear_narx": "Nonlinear NARX"}
SHORT = {"resonant_arma": "ARMA", "iir_cascade": "IIR", "nonlinear_narx": "NARX"}
PUBLIC = ["silverbox", "wiener_hammerstein", "ced", "cascaded_tanks", "emps"]
PUBLIC_LABEL = {"silverbox": "Silverbox", "wiener_hammerstein": "Wiener--Hammerstein", "ced": "CED",
                "cascaded_tanks": "Cascaded Tanks", "emps": "EMPS"}
# Decimal places per task; NARX and Wiener--Hammerstein need more digits to show
# the small differences and standard deviations.
DIGITS = {"resonant_arma": 3, "iir_cascade": 3, "nonlinear_narx": 5, "silverbox": 3,
          "wiener_hammerstein": 4, "ced": 3, "cascaded_tanks": 3, "emps": 3}
KEY = {"FNO": "fno", "LNO": "lno_stable", "S4D": "s4d_official", "S5": "s5_official", "ZNO": "zno"}
HEAD = {"FNO": "FNO", "LNO": "Stabilized LNO", "S4D": "S4D", "S5": "S5", "ZNO": "ZNO (ours)"}
MODELS = ["FNO", "S4D", "S5", "ZNO"]
MAIN_MODELS = ["FNO", "LNO", "S4D", "S5", "ZNO"]
BUDGETS = ["4k", "8.5k", "16k", "32k"]
BUDGET_TARGET = {"4k": 4000, "8.5k": 8500, "16k": 16000, "32k": 32000}
CLASSICAL_LABEL = {"sysid_aols_poly": "AOLS polynomial", "sysid_aols_fourier": "AOLS Fourier",
                   "sysid_frols_poly": "FROLS polynomial", "sysid_frols_fourier": "FROLS Fourier"}


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------
def num(n):
    """Integer with thin-space thousands separator."""
    return f"{int(n):,}".replace(",", r"\,")


def fmt_sd(sd, d):
    """Standard deviation at ``d`` decimals. A nonzero value that would round to zero is
    shown with one more decimal, and as at least one unit of it (3.5e-5 -> 0.0001)."""
    text = f"{sd:.{d}f}"
    if sd > 0 and float(text) == 0.0:
        text = f"{max(sd, 10.0 ** -(d + 1)):.{d + 1}f}"
    return text


def ms(mean, sd, d, bold=False):
    m = f"{mean:.{d}f}"
    if bold:
        return rf"$\mathbf{{{m}}}_{{\pm {fmt_sd(sd, d)}}}$"
    return rf"${m}_{{\pm {fmt_sd(sd, d)}}}$"


def val(x, d, bold=False):
    text = f"{x:.{d}f}"
    return rf"$\mathbf{{{text}}}$" if bold else f"${text}$"


def signed(x, d):
    return f"${'+' if x >= 0 else '-'}{abs(x):.{d}f}$"


def scount(n, suffix=""):
    return r"{\scriptsize(" + num(n) + suffix + ")}"


def close_rows(body):
    """End a block of two-line rows: drop the extra spacing after the last row."""
    body[-1] = body[-1].replace(r"\\[2pt]", r"\\")


def describe(model, cell, *, width=True, lr_note=False):
    if model == "fno":
        return rf"$w{{=}}{cfg_int(cell, 'width')}$, $m{{=}}{cfg_int(cell, 'modes')}$"
    if model == "s4d_official":
        text = (rf"$w{{=}}{cfg_int(cell, 'width')}$, " if width else "") + rf"$N{{=}}{cfg_int(cell, 'state_dim')}$"
        if lr_note and float(cell.config("learning_rate")) != 5e-4:
            text += r", lr $3{\times}10^{-4}$" if float(cell.config("learning_rate")) == 3e-4 else \
                rf", lr ${float(cell.config('learning_rate')):g}$"
        return text
    if model == "s5_official":
        return rf"$H{{=}}{cfg_int(cell, 'd_model')}$, $P{{=}}{cfg_int(cell, 'ssm_size_base')}$"
    if model == "zno":
        text = (rf"$w{{=}}{cfg_int(cell, 'width')}$, " if width else "")
        text += (rf"$r{{=}}{cfg_int(cell, 'rank')}$, $K{{=}}{cfg_int(cell, 'num_poles')}$, "
                 rf"$F{{=}}{cfg_int(cell, 'fir_order')}$")
        safe = float(cell.config("safe_radius"))
        if safe != 0.95:
            text += rf", $\rho_{{\mathrm{{safe}}}}{{=}}{safe:.2f}$"
        return text
    raise ValueError(model)


class Writer:
    def __init__(self, out: Path, results: Path, only: set[str] | None):
        self.out, self.results, self.only = out, results, only

    def wants(self, name):
        return self.only is None or name in self.only

    def __call__(self, name, body):
        self.out.mkdir(parents=True, exist_ok=True)
        header = (f"% Auto-generated by paper/make_tables.py from {self.results.name}.\n"
                  "% Do not edit by hand; rerun the script instead.\n")
        (self.out / f"{name}.tex").write_text(header + "\n".join(body) + "\n", encoding="utf-8")
        print(f"wrote {self.out / name}.tex")


def check_selection(what, recomputed, used):
    if recomputed != used:
        print(f"note: {what}: your selection runs select {recomputed!r}, the reported runs use {used!r}; "
              "rebuild the reported study with `--selected` (see REPRODUCTION.md) to follow your selection")


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------
def tab_matched(runs, write):
    """Table 1 (8.5k real parameters; ZNO on the task loss) and Table 9 (ZNO objective)."""
    def cell(task, m):
        return Cell(runs.get("matched", task=task, model=KEY[m], seeds=SEEDS))

    if write.wants("tab_matched_symmetric"):
        body = [r"\begin{tabular}{@{}lccccc@{}}", r"\toprule",
                "Task & " + " & ".join(HEAD[m] for m in MAIN_MODELS) + r" \\", r"\midrule"]
        for t in TASKS:
            cells = {m: cell(t, m) for m in MAIN_MODELS}
            best = best_of(cells)
            body.append(TASK_LABEL[t] + " & " + " & ".join(
                ms(cells[m].test, cells[m].test_sd, DIGITS[t], m == best) for m in MAIN_MODELS) + r" \\")
            body.append(" & " + " & ".join(scount(cells[m].params) for m in MAIN_MODELS) + r" \\[2pt]")
        close_rows(body)
        write("tab_matched_symmetric", body + [r"\bottomrule", r"\end{tabular}"])

    if write.wants("tab_matched_recipe"):
        body = [r"\begin{tabular}{@{}lccc@{}}", r"\toprule",
                r"Task & ZNO, task loss (Table~\ref{tab:discrete_main}) & ZNO, default objective & Difference \\",
                r"\midrule"]
        for t in TASKS:
            a = cell(t, "ZNO")
            b = Cell(runs.get("zno-default", task=t, model="zno", seeds=SEEDS))
            d = DIGITS[t]
            diff = b.test - a.test
            body.append(f"{TASK_LABEL[t]} & {ms(a.test, a.test_sd, d)} & {ms(b.test, b.test_sd, d)} & "
                        + (signed(diff, 5) if abs(diff) >= 5e-6 else r"$<10^{-5}$") + r" \\")
        write("tab_matched_recipe", body + [r"\bottomrule", r"\end{tabular}"])


def budget_cell(runs, budget, task, m):
    """Capacity-study cell; the 8.5k row is the 8.5k comparison, ZNO with its default objective."""
    if budget == "8.5k":
        if m == "ZNO":
            return Cell(runs.get("zno-default", task=task, model="zno", seeds=SEEDS))
        return Cell(runs.get("matched", task=task, model=KEY[m], seeds=SEEDS))
    return Cell(runs.get("capacity", task=task, model=KEY[m], tag=budget, seeds=SEEDS))


def tab_budget(runs, write):
    """Table 13: four real budgets x three tasks x four models."""
    body = [r"\begin{tabular}{@{}llcccc@{}}", r"\toprule",
            "Budget & Task & " + " & ".join(HEAD[m] for m in MODELS) + r" \\", r"\midrule"]
    for i, b in enumerate(BUDGETS):
        if i:
            body.append(r"\midrule")
        for t in TASKS:
            cells = {m: budget_cell(runs, b, t, m) for m in MODELS}
            best = best_of(cells)
            lead = b if t == TASKS[0] else ""
            body.append(f"{lead} & {SHORT[t]} & " + " & ".join(
                ms(cells[m].test, cells[m].test_sd, DIGITS[t], m == best) + " " + scount(cells[m].params)
                for m in MODELS) + r" \\")
    write("tab_budget", body + [r"\bottomrule", r"\end{tabular}"])


def tab_budget_configs(runs, write):
    """Table 12: topologies of the four real budgets with the range of real counts over tasks."""
    body = [r"\begin{tabular}{@{}lllll@{}}", r"\toprule", r"Budget & FNO & S4D & S5 & ZNO \\", r"\midrule"]
    for b in BUDGETS:
        topo, ranges = [], []
        for m in MODELS:
            cells = [budget_cell(runs, b, t, m) for t in TASKS]
            topo.append(describe(KEY[m], cells[0]))
            counts = sorted(c.params for c in cells)
            target = BUDGET_TARGET[b]
            if not all(abs(v - target) <= 0.05 * target for v in counts):
                print(f"note: {b} {m}: real counts {counts} are not within 5% of {target}")
            ranges.append(r"{\scriptsize " + f"{num(counts[0])}--{num(counts[-1])}" + "}")
        body.append(f"{b} & " + " & ".join(topo) + r" \\")
        body.append(" & " + " & ".join(ranges) + r" \\[2pt]")
    close_rows(body)
    write("tab_budget_configs", body + [r"\bottomrule", r"\end{tabular}"])


def public_cell(runs, task, m):
    return Cell(runs.get("public", task=task, model=KEY[m], seeds=PUBLIC_REPORT_SEEDS))


def classical_cells(runs, task):
    return {f: Cell(runs.get("classical", task=task, model=f)) for f in P.CLASSICAL_FAMILIES}


def classical_selected(cells):
    """Validation-selected family; an exact tie (on Wiener--Hammerstein both AOLS
    families reach identical errors) goes to the alphabetically first family name."""
    return min(cells, key=lambda f: (cells[f].val, f))


def tab_public(runs, write):
    """Table 2: public benchmarks (seeds 3-7) and the validation-selected classical family."""
    body = [r"\begin{tabular}{@{}lccccc@{}}", r"\toprule",
            r"Benchmark & Classical & FNO & S4D & S5 & ZNO (ours) \\", r"\midrule"]
    for t in PUBLIC:
        d = DIGITS[t]
        cells = {m: public_cell(runs, t, m) for m in MODELS}
        classical = classical_cells(runs, t)
        c = classical[classical_selected(classical)]
        best = min([*MODELS, "classical"], key=lambda k: c.test if k == "classical" else cells[k].test)
        body.append(f"{PUBLIC_LABEL[t]} & {val(c.test, d, best == 'classical')} & "
                    + " & ".join(ms(cells[m].test, cells[m].test_sd, d, m == best) for m in MODELS) + r" \\")
        body.append(" & " + scount(c.params, " terms") + " & "
                    + " & ".join(scount(cells[m].params) for m in MODELS) + r" \\[2pt]")
    close_rows(body)
    write("tab_public", body + [r"\bottomrule", r"\end{tabular}"])


def tab_public_gap(runs, write):
    """Table 22: best-validation and test errors of the reported public runs."""
    body = [r"\begin{tabular}{@{}llrrr@{}}", r"\toprule",
            r"Benchmark & Model & best-val rel.\ $\ell_2$ & test rel.\ $\ell_2$ & test $-$ val \\", r"\midrule"]
    for i, t in enumerate(PUBLIC):
        if i:
            body.append(r"\midrule")
        d = 4 if t == "wiener_hammerstein" else 3
        for j, m in enumerate(MODELS):
            c = public_cell(runs, t, m)
            lead = rf"\multirow{{4}}{{*}}{{{PUBLIC_LABEL[t]}}}" if j == 0 else ""
            body.append(f"{lead} & {m} & {val(c.val, d)} & {val(c.test, d)} & {signed(c.gap, d)}" + r" \\")
    write("tab_public_gap", body + [r"\bottomrule", r"\end{tabular}"])


def tab_public_configs(runs, write):
    """Table 20: selected public configurations with real counts; checks the selection."""
    for t in PUBLIC:
        for m in ("FNO", "S4D", "ZNO"):
            chosen, _ = select_min(runs, "public-selection", t, KEY[m])
            check_selection(f"public {t} {m}", chosen, public_cell(runs, t, m).rows[0]["tag"])
    body = [r"\begin{tabular}{@{}lllr@{}}", r"\toprule",
            r"Benchmark & Model & Selected configuration & Real params \\", r"\midrule"]
    for i, t in enumerate(PUBLIC):
        if i:
            body.append(r"\midrule")
        for j, m in enumerate(MODELS):
            c = public_cell(runs, t, m)
            lead = rf"\multirow{{4}}{{*}}{{{PUBLIC_LABEL[t]}}}" if j == 0 else ""
            body.append(f"{lead} & {m} & {describe(KEY[m], c)} & {num(c.params)}" + r" \\")
    write("tab_public_configs", body + [r"\bottomrule", r"\end{tabular}"])


def tab_s5_capacity(runs, write):
    """Table 19: S5 mean best-validation error per base state size (seeds 0-2)."""
    sizes = sorted({int(r["tag"][1:]) for r in runs.get("public-s5-selection")})
    cells = {(t, s): Cell(runs.get("public-s5-selection", task=t, tag=f"P{s}", seeds=SELECTION_SEEDS))
             for t in PUBLIC for s in sizes}
    body = [r"\begin{tabular}{@{}l" + "r" * len(sizes) + r"@{}}", r"\toprule",
            "Benchmark & " + " & ".join(f"$P{{=}}{s}$" for s in sizes) + r" \\",
            r" & " + " & ".join(scount(cells[(PUBLIC[0], s)].params) for s in sizes) + r" \\", r"\midrule"]
    for t in PUBLIC:
        d = 4 if t in ("silverbox", "wiener_hammerstein") else 3
        chosen = min(sizes, key=lambda s: (cells[(t, s)].val, s))  # ties go to the smaller size
        check_selection(f"S5 {t}", f"P{chosen}", public_cell(runs, t, "S5").rows[0]["tag"])
        body.append(f"{PUBLIC_LABEL[t]} & " + " & ".join(val(cells[(t, s)].val, d, s == chosen) for s in sizes)
                    + r" \\")
    write("tab_s5_capacity", body + [r"\bottomrule", r"\end{tabular}"])


def tab_classical(runs, write):
    """Table 21: SysIdentPy NARMAX families; the asterisk marks the validation-selected family."""
    body = [r"\begin{tabular}{@{}lllrrr@{}}", r"\toprule",
            r"Benchmark & Family & Configuration & Terms & Validation & Test \\", r"\midrule"]
    for i, t in enumerate(PUBLIC):
        if i:
            body.append(r"\midrule")
        cells = classical_cells(runs, t)
        chosen = classical_selected(cells)
        for j, f in enumerate(P.CLASSICAL_FAMILIES):
            c, sel = cells[f], f == chosen
            lead = rf"\multirow{{4}}{{*}}{{{PUBLIC_LABEL[t]}}}" if j == 0 else ""
            name = CLASSICAL_LABEL[f] + (r"$^{\ast}$" if sel else "")
            body.append(f"{lead} & {name} & " + r"\texttt{" + c.config("config") + "}"
                        + f" & {c.params} & {val(c.val, 3, sel)} & {val(c.test, 3, sel)}" + r" \\")
    write("tab_classical", body + [r"\bottomrule", r"\end{tabular}"])


def tab_tuned(runs, write):
    """Tables 14-15: tuning-study selection (seeds 0-2) and the selected configurations (seeds 0-4)."""
    selection = {(t, m): select_min(runs, "tuning-selection", t, KEY[m]) for t in TASKS for m in ("FNO", "S4D", "ZNO")}
    confirm = {(t, m): Cell(runs.get("tuning-confirm", task=t, model=KEY[m], seeds=SEEDS))
               for t in TASKS for m in ("FNO", "S4D", "ZNO")}
    for (t, m), (chosen, _) in selection.items():
        check_selection(f"tuning {t} {m}", chosen, confirm[(t, m)].rows[0]["tag"])

    if write.wants("tab_tuned_validation"):
        body = [r"\begin{tabular}{@{}lllrcc@{}}", r"\toprule",
                r"Task & Model & Selected configuration & Real params & Val.\ (seeds 0--2) & Test (seeds 0--4) \\",
                r"\midrule"]
        for i, t in enumerate(TASKS):
            if i:
                body.append(r"\midrule")
            d = DIGITS[t]
            cells = {m: confirm[(t, m)] for m in ("FNO", "S4D", "ZNO")}
            best = best_of(cells)
            for j, (m, c) in enumerate(cells.items()):
                tag = c.rows[0]["tag"]
                score = selection[(t, m)][1][tag]
                lead = rf"\multirow{{3}}{{*}}{{{TASK_LABEL[t]}}}" if j == 0 else ""
                desc = describe(KEY[m], c, width=m == "ZNO", lr_note=True)
                body.append(f"{lead} & {m} & {desc} & {num(c.params)} & {val(score, d)} & "
                            + ms(c.test, c.test_sd, d, m == best) + r" \\")
        write("tab_tuned_validation", body + [r"\bottomrule", r"\end{tabular}"])

    if write.wants("tab_tuned_grid"):
        body = [r"\begin{tabular}{@{}lllrr@{}}", r"\toprule",
                r"Task & Model & Candidate & Real params & Mean best val.\ (seeds 0--2) \\", r"\midrule"]
        for i, t in enumerate(TASKS):
            if i:
                body.append(r"\midrule")
            d = DIGITS[t]
            for m in ("FNO", "S4D", "ZNO"):
                chosen, scores = selection[(t, m)]
                cells = {tag: Cell(runs.get("tuning-selection", task=t, model=KEY[m], tag=tag, seeds=SELECTION_SEEDS))
                         for tag in scores}
                for tag in sorted(scores, key=lambda g: cells[g].params):  # stable: grid order on ties
                    desc = describe(KEY[m], cells[tag], width=False, lr_note=True)
                    body.append(f"{TASK_LABEL[t]} & {m} & {desc} & {num(cells[tag].params)} & "
                                f"{val(scores[tag], d, tag == chosen)}" + r" \\")
        write("tab_tuned_grid", body + [r"\bottomrule", r"\end{tabular}"])


def tab_lno(runs, write):
    """Table 16: stabilized LNO at about 8.5k and about 20k real parameters."""
    body = [r"\begin{tabular}{@{}lcc@{}}", r"\toprule",
            r"Task & $w{=}14$, $m{=}8$ (Table~\ref{tab:discrete_main}) & $w{=}12$, $m{=}32$ \\", r"\midrule"]
    for t in TASKS:
        d = DIGITS[t]
        a = Cell(runs.get("matched", task=t, model="lno_stable", seeds=SEEDS))
        b = Cell(runs.get("lno-reference", task=t, model="lno_stable", seeds=SEEDS))
        body.append(f"{TASK_LABEL[t]} & {ms(a.test, a.test_sd, d)} & {ms(b.test, b.test_sd, d)}" + r" \\")
        body.append(f" & {scount(a.params)} & {scount(b.params)}" + r" \\[2pt]")
    close_rows(body)
    write("tab_lno", body + [r"\bottomrule", r"\end{tabular}"])


EMPS_ROWS = [
    ("hard0999-penalty", [r"$\rho_{\max}{=}0.999$ (sigmoid)", "on"]),
    ("hard105-penalty", [r"$\rho_{\max}{=}1.05$ (sigmoid)", "on"]),
    ("unbounded-penalty", [r"none (softplus)", "on"]),
    ("unbounded-no-penalty", [r"none (softplus)", "off"]),
]
TANKS_ROWS = [
    ("safe099", [r"$\mathcal{U}[0.30,0.95]$", r"on, $\rho_{\mathrm{safe}}{=}0.99$"]),
    ("no-pole-penalty", [r"$\mathcal{U}[0.30,0.95]$", "off"]),
    ("high-radius-init-no-penalty", [r"$\mathcal{U}[0.90,0.995]$", "off"]),
]


def intervention_table(runs, write, study, rows, name, columns):
    body = [r"\begin{tabular}{@{}" + "l" * len(columns) + r"cccc@{}}", r"\toprule",
            " & ".join(columns) + r" & Validation & Test & Test $-$ val. & Max $|p|$ \\", r"\midrule"]
    for tag, desc in rows:
        c = Cell(runs.get(study, tag=tag, seeds=SEEDS))
        body.append(" & ".join(desc) + f" & {ms(c.val, c.val_sd, 3)} & {ms(c.test, c.test_sd, 3)} & "
                    f"{ms(c.gap, c.gap_sd, 3)} & ${c.max_pole:.4f}$" + r" \\")
    write(name, body + [r"\bottomrule", r"\end{tabular}"])


def tab_diagnostics(runs, write):
    """Tables 23-24: EMPS radius-map and Cascaded Tanks memory interventions (seeds 0-4)."""
    if write.wants("tab_emps_interventions"):
        intervention_table(runs, write, "emps-interventions", EMPS_ROWS, "tab_emps_interventions",
                           ["Radius map", "Pole penalty"])
    if write.wants("tab_tanks_interventions"):
        intervention_table(runs, write, "tanks-interventions", TANKS_ROWS, "tab_tanks_interventions",
                           ["Initial radii", "Pole penalty"])


def tab_suffix_control(runs, write):
    """Table 10: suffix-loss control (seeds 0-4)."""
    columns = [("FNO + suffix", "suffix-control", "fno", "suffix"),
               ("S4D + suffix", "suffix-control", "s4d_official", "suffix"),
               ("ZNO base", "suffix-control", "zno", "base"),
               ("ZNO no suffix", "suffix-control", "zno", "no-suffix"),
               ("ZNO no suffix/pole", "matched", "zno", None)]
    body = [r"\begin{tabular}{@{}lrrrrr@{}}", r"\toprule",
            "Task & " + " & ".join(c[0] for c in columns) + r" \\", r"\midrule"]
    for t in TASKS:
        cells = [Cell(runs.get(study, task=t, model=model, tag=tag, seeds=SEEDS)) for _, study, model, tag in columns]
        body.append(f"{TASK_LABEL[t]} & " + " & ".join(ms(c.test, c.test_sd, DIGITS[t]) for c in cells) + r" \\")
    write("tab_suffix_control", body + [r"\bottomrule", r"\end{tabular}"])


IIR_LABEL = {
    "lr1e-3-clip0.25": "lr1e-3, clip 0.25",
    "lr1e-3-clip0.5": "lr1e-3, clip 0.5",
    "lr8e-4-clip0.25-safe0.90": "lr8e-4, clip 0.25, safe 0.90",
    "lr8e-4-clip0.5": "lr8e-4, clip 0.5",
}


def tab_iir_optimizer(runs, write):
    """Table 11: ZNO on the IIR cascade with modified learning rate and clipping (seeds 0-4)."""
    cells = {"default": Cell(runs.get("zno-default", task="iir_cascade", model="zno", seeds=SEEDS))}
    for tag in P.IIR_OPTIMIZER:
        cells[tag] = Cell(runs.get("iir-optimizer", task="iir_cascade", model="zno", tag=tag, seeds=SEEDS))
    best = best_of(cells)
    body = [r"\begin{tabular}{@{}llr@{}}", r"\toprule",
            r"Model & Variant & IIR cascade (6th) rel.\ $\ell_2$ (5 seeds) \\", r"\midrule"]
    for tag, c in cells.items():
        model = r"ZNO (Table~\ref{tab:matched_recipe})" if tag == "default" else "ZNO"
        variant = r"recipe of Table~\ref{tab:recipes}" if tag == "default" else IIR_LABEL.get(tag, tag)
        body.append(f"{model} & {variant} & {ms(c.test, c.test_sd, 3, tag == best)}" + r" \\")
    write("tab_iir_optimizer", body + [r"\bottomrule", r"\end{tabular}"])


def tab_extrapolation(runs, write):
    """Table 17: fixed configurations trained at T=2048 and evaluated at 2048, 4096, and 8192 (seeds 0-4)."""
    lengths = [2048, *P.EXTRAPOLATION_LENGTHS]
    models = ["FNO", "S4D", "ZNO"]
    body = [r"\begin{tabular}{@{}ll" + "r" * len(lengths) + r"@{}}", r"\toprule",
            "Task & Model & " + " & ".join(f"$T={n}$" for n in lengths) + r" \\"]
    for t in TASKS:
        cells = {}
        for m in models:
            for n in lengths:
                if n == 2048:
                    rows = runs.get("fixed", task=t, model=KEY[m], seeds=SEEDS)
                else:
                    rows = runs.get("extrapolation", task=t, model=KEY[m], tag=f"T{n}", seeds=SEEDS)
                cells[(m, n)] = Cell(rows)
        best = {n: min(models, key=lambda m: cells[(m, n)].test) for n in lengths}
        body.append(r"\midrule")
        for j, m in enumerate(models):
            lead = rf"\multirow{{3}}{{*}}{{{TASK_LABEL[t]}}}" if j == 0 else ""
            body.append(f"{lead} & {m} & " + " & ".join(
                ms(cells[(m, n)].test, cells[(m, n)].test_sd, 3, best[n] == m) for n in lengths) + r" \\")
    write("tab_extrapolation", body + [r"\bottomrule", r"\end{tabular}"])


def tab_ablation(runs, write):
    """Table 25: one knob at a time away from the ZNO anchor (seeds 0-4)."""
    cases = list(P.ABLATION_CASES)
    tasks = [P.CASE_TO_TASK[c] for c in cases]
    k = {key: r"\{" + ",".join(str(P.ABLATION_CASES[c][key]) for c in cases) + r"\}" for key in ("anchor", "low", "high")}
    labels = {
        "anchor": rf"anchor ($r=8,F=4,K={k['anchor']},\rho_{{\mathrm{{safe}}}}=0.95$)",
        "rank4": r"rank $r=4$", "rank12": r"rank $r=12$",
        "poles-low": rf"poles $K={k['low']}$", "poles-high": rf"poles $K={k['high']}$",
        "fir0": r"FIR order $F=0$", "fir2": r"FIR order $F=2$", "fir8": r"FIR order $F=8$",
        "safe0.90": r"safe radius $\rho_{\mathrm{safe}}=0.90$", "safe0.99": r"safe radius $\rho_{\mathrm{safe}}=0.99$",
    }
    tags = list(P.ablation_variants(cases[0]))
    cells = {(tag, t): Cell(runs.get("ablation", task=t, model="zno", tag=tag, seeds=SEEDS)) for tag in tags for t in tasks}
    best = {t: min(tags, key=lambda g: cells[(g, t)].test) for t in tasks}
    body = [r"\begin{tabular}{@{}l" + "r" * len(tasks) + r"@{}}", r"\toprule",
            "Configuration & " + " & ".join(TASK_LABEL[t] + (" (6th)" if t == "iir_cascade" else "") for t in tasks)
            + r" \\", r"\midrule"]
    for tag in tags:
        body.append(labels.get(tag, tag) + " & " + " & ".join(
            ms(cells[(tag, t)].test, cells[(tag, t)].test_sd, 3, best[t] == tag) for t in tasks) + r" \\")
    write("tab_ablation", body + [r"\bottomrule", r"\end{tabular}"])


def long_horizon_text(runs):
    """Print the long-horizon ratios quoted in the caption of Table 17 and in Appendix I."""
    models = [("FNO", "fno"), ("S4D", "s4d_official"), ("ZNO", "zno")]

    def mean(study, task, model, tag=None):
        return Cell(runs.get(study, task=task, model=model, tag=tag, seeds=SEEDS)).test

    ratios = {m: [mean("extrapolation", t, key, "T8192") / mean("fixed", t, key) for t in TASKS] for m, key in models}
    print("Table 17, T=8192 / T=2048 on {ARMA, IIR, NARX}: "
          + "; ".join(f"{m} " + ", ".join(f"{r:.2f}" for r in ratios[m]) for m, _ in models))
    bins = list(P.DIFFICULTY_CASES)
    lengths = [f"T{n}" for n in P.EXTRAPOLATION_LENGTHS]
    base = {(b, key): mean("difficulty", b, key) for b in bins for _, key in models}
    longer = {(b, key, n): mean("difficulty-extrapolation", b, key, n) for b in bins for _, key in models for n in lengths}
    for m, key in models[1:]:
        values = [longer[(b, key, n)] / base[(b, key)] for b in bins for n in lengths]
        print(f"Appendix I, {m} T=4096/8192 relative to T=2048, all bins: {min(values):.2f} to {max(values):.2f}")
    print("Appendix I, FNO T=8192 / T=2048 per bin: "
          + ", ".join(f"{longer[(b, 'fno', 'T8192')] / base[(b, 'fno')]:.1f}" for b in bins))
    zno_lowest = all(min(models, key=lambda mk: errors[mk[1]])[0] == "ZNO"
                     for errors in [{key: base[(b, key)] for _, key in models} for b in bins]
                     + [{key: longer[(b, key, n)] for _, key in models} for b in bins for n in lengths])
    print(f"Appendix I, ZNO lowest mean error in every bin at every length: {zno_lowest}")


def _public_split(rows_filter):
    return lambda r: r["study"] == "public" and rows_filter(r)


COMPUTE_ROWS = [
    # (study, reference, run set, filter over rows of runs.csv)
    ("8.5k comparison", r"Tab.~\ref{tab:discrete_main}", "FNO, LNO, S4D, S5, ZNO", lambda r: r["study"] == "matched"),
    ("Objective controls", r"App.~\ref{app:suffix_fairness}", "ZNO, default objective",
     lambda r: r["study"] == "zno-default"),
    ("", "", "suffix-loss variants", lambda r: r["study"] == "suffix-control"),
    ("", "", "IIR learning-rate and clipping variants", lambda r: r["study"] == "iir-optimizer"),
    ("Capacity study", r"App.~\ref{app:budget}", "4k, 16k, 32k", lambda r: r["study"] == "capacity"),
    ("Tuning study", r"App.~\ref{app:tuned}", "selection, seeds 0--2", lambda r: r["study"] == "tuning-selection"),
    ("", "", "selected configurations, seeds 0--4", lambda r: r["study"] == "tuning-confirm"),
    ("Stabilized LNO", r"App.~\ref{app:lno_stabilized}", "about 20k", lambda r: r["study"] == "lno-reference"),
    ("Fixed configurations", r"Tab.~\ref{tab:fixed_configs}", "FNO, S4D, ZNO, seeds 0--4",
     lambda r: r["study"] == "fixed"),
    ("Difficulty sweep", r"Sec.~\ref{sec:experiments_difficulty}", "five radius bins",
     lambda r: r["study"] == "difficulty"),
    ("Long-horizon evaluation", r"Sec.~\ref{sec:experiments_extrap}", "synthetic tasks",
     lambda r: r["study"] == "extrapolation"),
    ("", "", "difficulty bins", lambda r: r["study"] == "difficulty-extrapolation"),
    ("Public benchmarks", r"Sec.~\ref{sec:experiments_public}", "selection, FNO/S4D/ZNO grids",
     lambda r: r["study"] == "public-selection"),
    ("", "", "selection, S5 state sizes", lambda r: r["study"] == "public-s5-selection"),
    ("", "", "FNO/S4D/ZNO selected, seeds 0--4",
     _public_split(lambda r: r["model"] != "s5_official" and int(r["seed"]) <= 4)),
    ("", "", "FNO/S4D/ZNO selected, seeds 5--7",
     _public_split(lambda r: r["model"] != "s5_official" and int(r["seed"]) >= 5)),
    ("", "", "S5 selected, seeds 3--7", _public_split(lambda r: r["model"] == "s5_official")),
    ("Diagnostics", r"App.~\ref{app:diagnostics}", "EMPS and Cascaded Tanks variants",
     lambda r: r["study"] in ("emps-interventions", "tanks-interventions")),
    ("ZNO ablation", r"App.~\ref{app:ablation}", "one knob at a time", lambda r: r["study"] == "ablation"),
    ("1D ODE benchmark", r"App.~\ref{app:1d_benchmark}", "matched-budget configurations",
     lambda r: r["study"] == "one-d"),
]


def tab_compute(runs, write):
    """Table 8: runs and cumulative epoch-loop hours per run set (hardware dependent).

    The six runs of the unmodified LNO (``lno-unconstrained``) are not counted, as in the paper.
    """
    body = [r"\begin{tabular}{@{}lllrr@{}}", r"\toprule", r"Study & Run set & Worker & Runs & Hours \\", r"\midrule"]
    total_n, total_h = 0, 0.0
    for study, ref, run_set, keep in COMPUTE_ROWS:
        rows = [r for r in runs.rows if keep(r)]
        hours = sum(float(r["train_time_seconds"] or 0) for r in rows) / 3600.0
        total_n, total_h = total_n + len(rows), total_h + hours
        label = f"{study} ({ref})" if study else ""
        body.append(f"{label} & {run_set} & GPU & {len(rows)} & {hours:.1f}" + r" \\")
    classical = [r for r in runs.rows if r["study"] == "classical"]
    classical_h = sum(float(r["train_time_seconds"] or 0) for r in classical) / 3600.0
    body += [r"\midrule", f"Total on GPUs & & GPU & {total_n} & {total_h:.1f}" + r" \\",
             r"Public SysIdentPy baselines (Sec.~\ref{sec:experiments_public}) & classical fits & CPU & "
             + f"{len(classical)} & {classical_h:.1f}" + r" \\"]
    write("tab_compute", body + [r"\bottomrule", r"\end{tabular}"])


TABLES = {
    "tab_matched_symmetric": tab_matched, "tab_matched_recipe": tab_matched,
    "tab_compute": tab_compute,
    "tab_suffix_control": tab_suffix_control, "tab_iir_optimizer": tab_iir_optimizer,
    "tab_budget_configs": tab_budget_configs, "tab_budget": tab_budget,
    "tab_tuned_validation": tab_tuned, "tab_tuned_grid": tab_tuned,
    "tab_lno": tab_lno, "tab_extrapolation": tab_extrapolation,
    "tab_s5_capacity": tab_s5_capacity, "tab_public_configs": tab_public_configs,
    "tab_classical": tab_classical, "tab_public_gap": tab_public_gap, "tab_public": tab_public,
    "tab_emps_interventions": tab_diagnostics, "tab_tanks_interventions": tab_diagnostics,
    "tab_ablation": tab_ablation,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", type=Path, default=REPO / "results" / "runs.csv")
    parser.add_argument("--out", type=Path, default=REPO / "paper" / "tables")
    parser.add_argument("--only", nargs="+", choices=sorted(TABLES), help="write only these tables")
    args = parser.parse_args()
    runs = load(args.results)
    write = Writer(args.out, args.results, set(args.only) if args.only else None)
    done = set()
    for name, fn in TABLES.items():
        if write.wants(name) and fn not in done:
            fn(runs, write)
            done.add(fn)
    if write.only is None or "tab_extrapolation" in write.only:
        long_horizon_text(runs)
    unconstrained = [float(r["test_rel_l2"]) for r in runs.rows if r["study"] == "lno-unconstrained" and r["test_rel_l2"]]
    if unconstrained:  # quoted in the text of Appendix H
        print(f"unmodified LNO (Appendix H): {len(unconstrained)} runs, test rel. l2 "
              f"{min(unconstrained):.2g} to {max(unconstrained):.2g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
