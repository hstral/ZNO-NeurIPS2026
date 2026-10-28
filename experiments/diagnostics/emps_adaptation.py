"""EMPS supervised adaptation diagnostic (NOT a benchmark result).

Fine-tunes trained ZNO checkpoints on *official test labels* of test windows
0-31 and evaluates on the disjoint windows 64-177; a control continues training
on the same number of training windows. Because part of the test domain is used
for supervision, the output must never be reported as benchmark performance.

Usage::

    python -m experiments.diagnostics.emps_adaptation --data-root data/public \\
        --checkpoint-root runs/public --output-dir runs/diagnostics/emps_adaptation
"""
from __future__ import annotations

import argparse
import json
import random
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from benchmark_discrete.configs import OperatorConfig
from benchmark_discrete.models import build_model
from benchmark_discrete.train import (
    LpLoss,
    _build_optimizer_and_scheduler,
    _strip_metric_prefix,
    _training_step_loss,
)

from ..protocols import DIAGNOSTIC_SEEDS
from . import seed_checkpoints


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def evaluate(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    device: torch.device,
    prefix: int,
) -> float:
    loss = LpLoss(size_average=True)
    values = []
    model.eval()
    with torch.no_grad():
        for start in range(0, x.shape[0], 4):
            x_batch = x[start : start + 4].to(device)
            y_batch = y[start : start + 4].to(device)
            prediction = model(x_batch)
            prediction, target = _strip_metric_prefix(
                prediction, y_batch, prefix
            )
            # Use per-sequence values so batching does not change the metric.
            for index in range(prediction.shape[0]):
                values.append(
                    float(
                        loss(
                            prediction[index : index + 1],
                            target[index : index + 1],
                        ).item()
                    )
                )
    return float(np.mean(values))


def adapt(
    *,
    checkpoint: dict,
    config: OperatorConfig,
    grid: torch.Tensor,
    x_adapt: torch.Tensor,
    y_adapt: torch.Tensor,
    device: torch.device,
    seed: int,
    prefix: int,
    epochs: int,
    learning_rate: float,
) -> torch.nn.Module:
    set_seed(seed)
    model = build_model(
        model_name="zno",
        config=config,
        grid=grid,
        use_grid=False,
        input_channels=int(x_adapt.shape[-1]),
        dt=1.0,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    adaptation_config = replace(
        config,
        learning_rate=learning_rate,
        epochs=epochs,
    )
    optimizer, scheduler = _build_optimizer_and_scheduler(
        model, adaptation_config
    )
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        TensorDataset(x_adapt, y_adapt),
        batch_size=8,
        shuffle=True,
        generator=generator,
    )
    rel_loss = LpLoss(size_average=True)
    for _ in range(epochs):
        model.train()
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(x_batch)
            loss = _training_step_loss(
                model=model,
                pred=prediction,
                target=y_batch,
                rel_loss=rel_loss,
                lambda_pole=1e-3,
                lambda_roll=1e-2,
                rollout_context_steps=None,
                include_pole_penalty=True,
                include_rollout_suffix=True,
                ignore_prefix_steps=prefix,
            )
            loss.backward()
            optimizer.step()
        scheduler.step()
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True,
                        help="public data root containing emps_public_v1/data.npz")
    parser.add_argument("--checkpoint-root", type=Path, required=True,
                        help="run root containing emps_public_v1/zno/<run>/model_best.pt")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DIAGNOSTIC_SEEDS),
                        help="seeds of the checkpoints to use (default: 0-4)")
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    data_dir = args.data_root.resolve() / "emps_public_v1"
    meta = json.loads((data_dir / "meta.json").read_text(encoding="utf-8"))
    prefix = int(meta["metric_ignore_prefix_steps"])
    with np.load(data_dir / "data.npz") as raw:
        x_train = torch.from_numpy(np.asarray(raw["x_train"], dtype=np.float32))
        y_train = torch.from_numpy(np.asarray(raw["y_train"], dtype=np.float32))
        x_val = torch.from_numpy(np.asarray(raw["x_val"], dtype=np.float32))
        y_val = torch.from_numpy(np.asarray(raw["y_val"], dtype=np.float32))
        x_test = torch.from_numpy(np.asarray(raw["x_test"], dtype=np.float32))
        y_test = torch.from_numpy(np.asarray(raw["y_test"], dtype=np.float32))
        grid = torch.from_numpy(np.asarray(raw["grid"], dtype=np.float32))

    adaptation_count = 32
    holdout_start = 64
    stride = 128
    window_length = int(x_test.shape[1])
    last_adaptation_endpoint = (adaptation_count - 1) * stride + window_length - 1
    first_holdout_start = holdout_start * stride
    if first_holdout_start <= last_adaptation_endpoint:
        raise ValueError("Adaptation and holdout windows overlap")

    checkpoint_root = args.checkpoint_root.resolve() / "emps_public_v1" / "zno"
    rows = []
    for checkpoint_path in seed_checkpoints(checkpoint_root, args.seeds):
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
        seed = int(checkpoint["seed"])
        config = OperatorConfig(**checkpoint["model_config"])
        set_seed(seed)
        original = build_model(
            model_name="zno",
            config=config,
            grid=grid,
            use_grid=False,
            input_channels=1,
            dt=1.0,
        ).to(device)
        original.load_state_dict(checkpoint["model_state_dict"])
        before = {
            "val": evaluate(
                original, x_val, y_val, device=device, prefix=prefix
            ),
            "test_full": evaluate(
                original, x_test, y_test, device=device, prefix=prefix
            ),
            "test_holdout": evaluate(
                original,
                x_test[holdout_start:],
                y_test[holdout_start:],
                device=device,
                prefix=prefix,
            ),
        }
        del original
        torch.cuda.empty_cache()

        train_control = adapt(
            checkpoint=checkpoint,
            config=config,
            grid=grid,
            x_adapt=x_train[:adaptation_count],
            y_adapt=y_train[:adaptation_count],
            device=device,
            seed=seed + 10_000,
            prefix=prefix,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
        )
        train_control_metrics = {
            "val": evaluate(
                train_control, x_val, y_val, device=device, prefix=prefix
            ),
            "test_holdout": evaluate(
                train_control,
                x_test[holdout_start:],
                y_test[holdout_start:],
                device=device,
                prefix=prefix,
            ),
        }
        del train_control
        torch.cuda.empty_cache()

        test_adapted = adapt(
            checkpoint=checkpoint,
            config=config,
            grid=grid,
            x_adapt=x_test[:adaptation_count],
            y_adapt=y_test[:adaptation_count],
            device=device,
            seed=seed + 10_000,
            prefix=prefix,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
        )
        test_adapted_metrics = {
            "val": evaluate(
                test_adapted, x_val, y_val, device=device, prefix=prefix
            ),
            "test_holdout": evaluate(
                test_adapted,
                x_test[holdout_start:],
                y_test[holdout_start:],
                device=device,
                prefix=prefix,
            ),
        }
        del test_adapted
        torch.cuda.empty_cache()
        row = {
            "seed": seed,
            "before": before,
            "train_control": train_control_metrics,
            "test_excitation_adaptation": test_adapted_metrics,
        }
        rows.append(row)
        print(json.dumps(row, indent=2), flush=True)

    aggregate = {}
    for condition, metric in (
        ("before", "test_holdout"),
        ("train_control", "test_holdout"),
        ("test_excitation_adaptation", "test_holdout"),
    ):
        values = np.asarray([row[condition][metric] for row in rows])
        aggregate[condition] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)),
        }
    payload = {
        "diagnostic_only": True,
        "official_test_labels_used_for_adaptation": True,
        "adaptation_windows": [0, adaptation_count - 1],
        "holdout_windows": [holdout_start, int(x_test.shape[0] - 1)],
        "window_stride": stride,
        "window_length": window_length,
        "last_adaptation_stream_index": last_adaptation_endpoint,
        "first_holdout_stream_index": first_holdout_start,
        "adaptation_epochs": args.epochs,
        "adaptation_learning_rate": args.learning_rate,
        "rows": rows,
        "aggregate": aggregate,
    }
    (output_dir / "emps_adaptation_diagnostic.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
