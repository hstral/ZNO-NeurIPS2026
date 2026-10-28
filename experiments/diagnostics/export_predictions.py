"""Export validation and test predictions of trained EMPS/Tanks checkpoints.

The predictions feed ``experiments.diagnostics.prefix_sensitivity``. Usage::

    python -m experiments.diagnostics.export_predictions --data-root data/public \\
        --checkpoint-root runs/public --output-dir runs/diagnostics/predictions
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
MODELS = ("fno", "zno", "s4d_official")


def predict(
    model: torch.nn.Module,
    x: np.ndarray,
    *,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    chunks = []
    model.eval()
    with torch.no_grad():
        for start in range(0, x.shape[0], batch_size):
            batch = torch.from_numpy(x[start : start + batch_size]).to(
                device=device, dtype=torch.float32
            )
            chunks.append(model(batch).cpu().numpy())
    return np.concatenate(chunks, axis=0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True,
                        help="run root containing <case>/<model>/<run>/model_best.pt")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DIAGNOSTIC_SEEDS),
                        help="seeds of the checkpoints to use (default: 0-4)")
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    result_root = args.checkpoint_root.resolve()
    data_root = args.data_root.resolve()
    device = torch.device(args.device)
    records = []

    for case in CASES:
        with np.load(data_root / case / "data.npz") as data:
            grid = torch.from_numpy(np.asarray(data["grid"], dtype=np.float32))
            x_val = np.asarray(data["x_val"], dtype=np.float32)
            y_val = np.asarray(data["y_val"], dtype=np.float32)
            x_test = np.asarray(data["x_test"], dtype=np.float32)
            y_test = np.asarray(data["y_test"], dtype=np.float32)
        for model_name in MODELS:
            checkpoint_paths = seed_checkpoints(result_root / case / model_name, args.seeds)
            for checkpoint_path in checkpoint_paths:
                checkpoint = torch.load(
                    checkpoint_path, map_location="cpu", weights_only=False
                )
                config = OperatorConfig(**checkpoint["model_config"])
                model = build_model(
                    model_name=model_name,
                    config=config,
                    grid=grid,
                    use_grid=False,
                    input_channels=int(x_val.shape[-1]),
                    dt=1.0,
                ).to(device)
                model.load_state_dict(checkpoint["model_state_dict"])
                batch_size = 4 if case == "emps_public_v1" else 32
                val_pred = predict(
                    model, x_val, device=device, batch_size=batch_size
                )
                test_pred = predict(
                    model, x_test, device=device, batch_size=batch_size
                )
                seed = int(checkpoint["seed"])
                destination = output_dir / case / model_name / f"seed{seed}.npz"
                destination.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    destination,
                    x_val=x_val,
                    y_val=y_val,
                    y_val_pred=val_pred,
                    x_test=x_test,
                    y_test=y_test,
                    y_test_pred=test_pred,
                )
                records.append(
                    {
                        "case": case,
                        "model": model_name,
                        "seed": seed,
                        "checkpoint": str(checkpoint_path),
                        "predictions": str(destination),
                    }
                )
                del model
                torch.cuda.empty_cache()
                print(records[-1], flush=True)

    (output_dir / "prediction_manifest.json").write_text(
        json.dumps(records, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
