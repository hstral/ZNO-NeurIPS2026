"""Learned discrete-pole statistics of trained EMPS and Cascaded Tanks checkpoints.

ZNO poles are read directly; S4D poles are discretized as exp(dt * A). Usage::

    python -m experiments.diagnostics.checkpoint_poles --data-root data/public \\
        --checkpoint-root runs/public --output runs/diagnostics/checkpoint_poles.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from benchmark_discrete.configs import OperatorConfig
from benchmark_discrete.models import build_model

from ..protocols import DIAGNOSTIC_SEEDS
from . import seed_checkpoints


CASES = ("emps_public_v1", "cascaded_tanks_public_v1")
MODELS = ("zno", "s4d_official")


def pole_radii(model: torch.nn.Module, model_name: str) -> np.ndarray:
    radii = []
    if model_name == "zno":
        for layer in model.layers:
            poles, _ = layer.stable_poles_and_residues()
            radii.append(poles.abs().detach().cpu().numpy().reshape(-1))
    elif model_name == "s4d_official":
        for layer in model.layers:
            kernel = layer.kernel
            dt = torch.exp(kernel.log_dt)
            continuous_poles = (
                -torch.exp(kernel.log_a_real) + 1j * kernel.a_imag
            )
            discrete_poles = torch.exp(continuous_poles * dt.unsqueeze(-1))
            radii.append(
                discrete_poles.abs().detach().cpu().numpy().reshape(-1)
            )
    else:  # pragma: no cover
        raise ValueError(model_name)
    return np.concatenate(radii).astype(np.float64)


def zno_conjugate_representative_radii(model: torch.nn.Module) -> np.ndarray:
    """One radius per conjugate pair (plus the real pole, if any)."""
    values = []
    for layer in model.layers:
        poles, _ = layer.stable_poles_and_residues()
        parts = []
        if layer.num_pairs:
            parts.append(poles[..., : layer.num_pairs])
        if layer.has_real_pole:
            parts.append(poles[..., -1:])
        values.append(torch.cat(parts, dim=-1).abs().detach().cpu().numpy().reshape(-1))
    return np.concatenate(values).astype(np.float64)


def summarize(values: np.ndarray) -> dict[str, float]:
    clipped = np.clip(values, 1e-12, 1.0 - 1e-12)
    time_constants = -1.0 / np.log(clipped)
    return {
        "count": int(values.size),
        "radius_mean": float(values.mean()),
        "radius_median": float(np.median(values)),
        "radius_q95": float(np.quantile(values, 0.95)),
        "radius_max": float(values.max()),
        "fraction_radius_gt_0.95": float(np.mean(values > 0.95)),
        "fraction_radius_gt_0.98": float(np.mean(values > 0.98)),
        "fraction_radius_gt_0.99": float(np.mean(values > 0.99)),
        "fraction_radius_gt_0.995": float(np.mean(values > 0.995)),
        "time_constant_median": float(np.median(time_constants)),
        "time_constant_q95": float(np.quantile(time_constants, 0.95)),
        "time_constant_max": float(time_constants.max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True,
                        help="public benchmark data root (contains <case>/data.npz)")
    parser.add_argument("--checkpoint-root", type=Path, required=True,
                        help="run root containing <case>/<model>/<run>/model_best.pt")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DIAGNOSTIC_SEEDS),
                        help="seeds of the checkpoints to use (default: 0-4)")
    args = parser.parse_args()
    data_root = args.data_root.resolve()
    checkpoint_root = args.checkpoint_root.resolve()

    payload = {}
    for case in CASES:
        with np.load(data_root / case / "data.npz") as data:
            grid = torch.from_numpy(np.asarray(data["grid"], dtype=np.float32))
            input_channels = int(data["x_train"].shape[-1])
        for model_name in MODELS:
            rows = []
            all_radii = []
            representative_radii = []
            paths = seed_checkpoints(checkpoint_root / case / model_name, args.seeds)
            if len(paths) != len(args.seeds):
                raise RuntimeError(
                    f"Expected {len(args.seeds)} checkpoints for {case}/{model_name}, found {len(paths)}"
                )
            for path in paths:
                checkpoint = torch.load(
                    path, map_location="cpu", weights_only=False
                )
                config = OperatorConfig(**checkpoint["model_config"])
                model = build_model(
                    model_name=model_name,
                    config=config,
                    grid=grid,
                    use_grid=False,
                    input_channels=input_channels,
                    dt=1.0,
                )
                model.load_state_dict(checkpoint["model_state_dict"])
                radii = pole_radii(model, model_name)
                all_radii.append(radii)
                if model_name == "zno":
                    representative_radii.append(zno_conjugate_representative_radii(model))
                rows.append(
                    {
                        "seed": int(checkpoint["seed"]),
                        "checkpoint": str(path.relative_to(checkpoint_root)),
                        **summarize(radii),
                    }
                )
            payload[f"{case}:{model_name}"] = {
                "per_seed": rows,
                "pooled": summarize(np.concatenate(all_radii)),
                **({"pooled_one_representative_per_conjugate_pair":
                    summarize(np.concatenate(representative_radii))}
                   if representative_radii else {}),
                "mean_over_seed_summaries": {
                    key: float(np.mean([row[key] for row in rows]))
                    for key in rows[0]
                    if key not in {"seed", "checkpoint", "count"}
                },
            }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
