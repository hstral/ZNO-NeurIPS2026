import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from .public_benchmarks import load_public_benchmark_protocol


def _import_sysidentpy():
    from sysidentpy.basis_function import Fourier, Polynomial
    from sysidentpy.model_structure_selection import AOLS, ER, FROLS
    from sysidentpy.parameter_estimation import LeastSquares

    return {
        "AOLS": AOLS,
        "ER": ER,
        "FROLS": FROLS,
        "Fourier": Fourier,
        "Polynomial": Polynomial,
        "LeastSquares": LeastSquares,
    }


def _rel_l2(y_pred: np.ndarray, y_true: np.ndarray, *, ignore_prefix_steps: int = 0) -> float:
    if ignore_prefix_steps > 0:
        if y_pred.shape[1] <= ignore_prefix_steps:
            raise ValueError(
                f"ignore_prefix_steps={ignore_prefix_steps} must be smaller than sequence length {y_pred.shape[1]}"
            )
        y_pred = y_pred[:, ignore_prefix_steps:, :]
        y_true = y_true[:, ignore_prefix_steps:, :]
    diff = y_pred.reshape(y_pred.shape[0], -1) - y_true.reshape(y_true.shape[0], -1)
    denom = np.linalg.norm(y_true.reshape(y_true.shape[0], -1), axis=1) + 1e-8
    return float(np.mean(np.linalg.norm(diff, axis=1) / denom))


def _candidate_configs(model_name: str) -> List[Dict[str, object]]:
    if model_name == "sysid_aols_poly":
        return [
            {"tag": "yl2x2-d2-k3", "ylag": 2, "xlag": 2, "degree": 2, "k": 3},
            {"tag": "yl4x4-d2-k5", "ylag": 4, "xlag": 4, "degree": 2, "k": 5},
            {"tag": "yl6x6-d2-k8", "ylag": 6, "xlag": 6, "degree": 2, "k": 8},
            {"tag": "yl4x4-d3-k5", "ylag": 4, "xlag": 4, "degree": 3, "k": 5},
        ]
    if model_name == "sysid_frols_poly":
        return [
            {"tag": "yl2x2-d2-ni6", "ylag": 2, "xlag": 2, "degree": 2, "n_info_values": 6},
            {"tag": "yl3x3-d2-ni6", "ylag": 3, "xlag": 3, "degree": 2, "n_info_values": 6},
            {"tag": "yl4x4-d2-ni10", "ylag": 4, "xlag": 4, "degree": 2, "n_info_values": 10},
            {"tag": "yl4x4-d3-ni10", "ylag": 4, "xlag": 4, "degree": 3, "n_info_values": 10},
        ]
    if model_name == "sysid_aols_fourier":
        return [
            {"tag": "yl2x2-n3-k3", "ylag": 2, "xlag": 2, "n": 3, "k": 3},
            {"tag": "yl4x4-n3-k5", "ylag": 4, "xlag": 4, "n": 3, "k": 5},
            {"tag": "yl6x6-n3-k8", "ylag": 6, "xlag": 6, "n": 3, "k": 8},
            {"tag": "yl4x4-n5-k5", "ylag": 4, "xlag": 4, "n": 5, "k": 5},
        ]
    if model_name == "sysid_frols_fourier":
        return [
            {"tag": "yl2x2-n3-ni6", "ylag": 2, "xlag": 2, "n": 3, "n_info_values": 6},
            {"tag": "yl3x3-n3-ni6", "ylag": 3, "xlag": 3, "n": 3, "n_info_values": 6},
            {"tag": "yl4x4-n3-ni8", "ylag": 4, "xlag": 4, "n": 3, "n_info_values": 8},
            {"tag": "yl4x4-n5-ni8", "ylag": 4, "xlag": 4, "n": 5, "n_info_values": 8},
        ]
    if model_name == "sysid_er_poly":
        return [
            {"tag": "yl2x2-d2-perm10", "ylag": 2, "xlag": 2, "degree": 2, "n_perm": 10, "k": 2},
            {"tag": "yl3x3-d2-perm10", "ylag": 3, "xlag": 3, "degree": 2, "n_perm": 10, "k": 2},
        ]
    raise ValueError(f"Unsupported model: {model_name}")


def _build_model(model_name: str, config: Dict[str, object], seed: int):
    pkg = _import_sysidentpy()
    least_squares = pkg["LeastSquares"]()
    if model_name == "sysid_aols_poly":
        return pkg["AOLS"](
            ylag=int(config["ylag"]),
            xlag=int(config["xlag"]),
            k=int(config["k"]),
            L=1,
            estimator=least_squares,
            basis_function=pkg["Polynomial"](degree=int(config["degree"])),
            model_type="NARMAX",
        )
    if model_name == "sysid_frols_poly":
        return pkg["FROLS"](
            ylag=int(config["ylag"]),
            xlag=int(config["xlag"]),
            order_selection=True,
            info_criteria="aic",
            n_info_values=int(config["n_info_values"]),
            estimator=least_squares,
            basis_function=pkg["Polynomial"](degree=int(config["degree"])),
            model_type="NARMAX",
        )
    if model_name == "sysid_aols_fourier":
        return pkg["AOLS"](
            ylag=int(config["ylag"]),
            xlag=int(config["xlag"]),
            k=int(config["k"]),
            L=1,
            estimator=least_squares,
            basis_function=pkg["Fourier"](n=int(config["n"]), degree=1, ensemble=True),
            model_type="NARMAX",
        )
    if model_name == "sysid_frols_fourier":
        return pkg["FROLS"](
            ylag=int(config["ylag"]),
            xlag=int(config["xlag"]),
            order_selection=True,
            info_criteria="aic",
            n_info_values=int(config["n_info_values"]),
            estimator=least_squares,
            basis_function=pkg["Fourier"](n=int(config["n"]), degree=1, ensemble=True),
            model_type="NARMAX",
        )
    if model_name == "sysid_er_poly":
        return pkg["ER"](
            ylag=int(config["ylag"]),
            xlag=int(config["xlag"]),
            k=int(config["k"]),
            n_perm=int(config["n_perm"]),
            basis_function=pkg["Polynomial"](degree=int(config["degree"])),
            model_type="NARMAX",
            random_state=seed,
        )
    raise ValueError(f"Unsupported model: {model_name}")


def _predict_windows(
    model,
    x_windows: np.ndarray,
    y_windows: np.ndarray,
    *,
    initial_output_steps: int = 0,
) -> np.ndarray:
    preds = np.zeros_like(y_windows, dtype=np.float64)
    for index, (x_window, y_window) in enumerate(zip(x_windows, y_windows)):
        y_init = y_window
        if initial_output_steps > 0:
            keep = max(initial_output_steps, int(getattr(model, "max_lag", 0)))
            keep = min(keep, y_window.shape[0])
            y_init = y_window[:keep]
        y_hat = np.asarray(model.predict(X=x_window, y=y_init), dtype=np.float64).reshape(-1, 1)
        if y_hat.shape[0] != y_window.shape[0]:
            raise ValueError(
                f"Prediction length mismatch at window {index}: got {y_hat.shape[0]}, expected {y_window.shape[0]}"
            )
        preds[index] = y_hat
    return preds


def _parameter_count(model) -> int:
    theta = getattr(model, "theta", None)
    if theta is not None:
        return int(np.asarray(theta).size)
    n_terms = getattr(model, "n_terms", None)
    if n_terms is not None:
        return int(n_terms)
    return 0


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate SysIdentPy classical baselines on public nonlinear ID benchmarks.")
    parser.add_argument(
        "--suite",
        required=True,
        choices=[
            "silverbox_public_v1",
            "wiener_hammer_public_v1",
            "cascaded_tanks_public_v1",
            "emps_public_v1",
            "ced_public_v1",
        ],
    )
    parser.add_argument(
        "--model",
        required=True,
        choices=[
            "sysid_aols_poly",
            "sysid_frols_poly",
            "sysid_aols_fourier",
            "sysid_frols_fourier",
            "sysid_er_poly",
        ],
    )
    parser.add_argument("--run-name", default="default")
    parser.add_argument("--output-dir", default="runs/classical")
    parser.add_argument("--data-root", default="")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-val-windows", type=int)
    parser.add_argument("--max-test-windows", type=int)
    parser.add_argument("--auto-install", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parent.parent
    data_root = Path(args.data_root).resolve() if args.data_root else None
    protocol = load_public_benchmark_protocol(
        repo_root=repo_root,
        suite=args.suite,
        data_root=data_root,
        auto_install=args.auto_install,
    )
    metric_ignore_prefix_steps = int(protocol.metric_ignore_prefix_steps)

    x_val = protocol.x_val_windows
    y_val = protocol.y_val_windows
    x_test = protocol.x_test_windows
    y_test = protocol.y_test_windows
    if args.max_val_windows is not None:
        x_val = x_val[: args.max_val_windows]
        y_val = y_val[: args.max_val_windows]
    if args.max_test_windows is not None:
        x_test = x_test[: args.max_test_windows]
        y_test = y_test[: args.max_test_windows]

    run_dir = Path(args.output_dir).resolve() / args.suite / args.model / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    selection_rows: List[Dict[str, object]] = []
    best_model = None
    best_config: Dict[str, object] = {}
    best_val_rel = float("inf")
    best_fit_seconds = 0.0

    total_start = time.time()
    for candidate in _candidate_configs(args.model):
        fit_start = time.time()
        model = _build_model(args.model, candidate, seed=args.seed)
        model.fit(X=protocol.train_u, y=protocol.train_y)
        val_pred = _predict_windows(
            model,
            x_val,
            y_val,
            initial_output_steps=metric_ignore_prefix_steps,
        )
        val_rel = _rel_l2(val_pred, y_val, ignore_prefix_steps=metric_ignore_prefix_steps)
        fit_seconds = time.time() - fit_start
        row = {
            "tag": candidate["tag"],
            "val_rel_l2": float(val_rel),
            "fit_seconds": float(fit_seconds),
            "parameter_count": _parameter_count(model),
            "config": candidate,
        }
        selection_rows.append(row)
        if val_rel < best_val_rel:
            best_val_rel = val_rel
            best_model = model
            best_config = dict(candidate)
            best_fit_seconds = fit_seconds

    if best_model is None:
        raise RuntimeError("No valid candidate baseline was fitted")

    test_pred = _predict_windows(
        best_model,
        x_test,
        y_test,
        initial_output_steps=metric_ignore_prefix_steps,
    )
    test_rel = _rel_l2(test_pred, y_test, ignore_prefix_steps=metric_ignore_prefix_steps)
    total_seconds = time.time() - total_start

    with open(run_dir / "selection_results.json", "w", encoding="utf-8") as file:
        json.dump(selection_rows, file, indent=2)

    np.savez_compressed(
        run_dir / "predictions_test.npz",
        x_test=x_test.squeeze(-1),
        y_test=y_test.squeeze(-1),
        y_pred=test_pred.squeeze(-1),
        test_rel_l2=test_rel,
        metric_ignore_prefix_steps=metric_ignore_prefix_steps,
    )

    final_model = getattr(best_model, "final_model", None)
    theta = getattr(best_model, "theta", None)
    if final_model is not None or theta is not None:
        np.savez_compressed(
            run_dir / "identified_model.npz",
            final_model=np.asarray(final_model) if final_model is not None else np.array([]),
            theta=np.asarray(theta) if theta is not None else np.array([]),
        )

    summary = {
        "case": args.suite,
        "model": args.model,
        "run_name": args.run_name,
        "seed": args.seed,
        "device": "cpu",
        "train_time_seconds": float(total_seconds),
        "best_fit_seconds": float(best_fit_seconds),
        "best_val_rel_l2": float(best_val_rel),
        "test_rel_l2": float(test_rel),
        "metric_ignore_prefix_steps": metric_ignore_prefix_steps,
        "parameter_count": int(_parameter_count(best_model)),
        "selected_checkpoint": "best_val_rel_l2",
        "selected_config": best_config,
        "selection_candidates": len(selection_rows),
        "baseline_family": "sysidentpy",
        "suite_dir": str(protocol.suite_dir),
        "suite_meta": protocol.meta,
        "run_dir": str(run_dir),
        "evaluation_protocol": {
            "train_stream": "official train stream, normalized with stored train mean/std",
            "validation_windows": int(x_val.shape[0]),
            "test_windows": int(x_test.shape[0]),
            "window_metric": "mean relative L2 over stored windows",
            "predict_mode": "free_run_with_true_initial_output_lags",
            "initial_output_steps_used": int(metric_ignore_prefix_steps),
        },
    }
    with open(run_dir / "summary.json", "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    print("\n=============================")
    print(f"Suite: {args.suite}")
    print(f"Model: {args.model}")
    print(f"Best config: {best_config.get('tag', 'N/A')}")
    print(f"Validation error: {best_val_rel:.6e}")
    print(f"Testing error: {test_rel:.6e}")
    print(f"Results saved to: {run_dir}")
    print("=============================\n")


if __name__ == "__main__":
    main()
