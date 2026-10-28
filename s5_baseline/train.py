"""Train the official S5 core on an exported train/validation data view.

The test view is opened only when ``--evaluate-test`` is given, after training,
with the best-validation checkpoint. Usage::

    python -m s5_baseline.train --train-val-file VIEW/train_val.npz \\
        --test-file VIEW/test.npz --evaluate-test --output-dir runs/s5 \\
        --run-name arma-s5-seed0 --seed 0
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import asdict
from pathlib import Path

import flax.serialization
import jax
import jax.numpy as jnp
import numpy as np
import optax

from .data_views import iter_batches, load_test, load_train_val
from .metrics import relative_l2_jax
from .model import (
    S5RegressionConfig,
    S5SequenceRegressor,
    count_real_parameters,
    initialize_model,
)
from .official_imports import OFFICIAL_S5_COMMIT
from .optim import build_optimizer


def tree_is_finite(tree) -> jnp.ndarray:
    leaves = jax.tree_util.tree_leaves(tree)
    return jnp.all(jnp.stack([jnp.all(jnp.isfinite(leaf)) for leaf in leaves]))


def build_steps(model_train, model_eval, optimizer, ignore_prefix_steps: int):
    @jax.jit
    def train_step(params, optimizer_state, inputs, targets, dropout_key):
        def loss_fn(candidate_params):
            predictions = model_train.apply(
                {"params": candidate_params},
                inputs,
                rngs={"dropout": dropout_key},
            )
            return relative_l2_jax(
                predictions,
                targets,
                ignore_prefix_steps=ignore_prefix_steps,
            )

        loss, gradients = jax.value_and_grad(loss_fn)(params)
        updates, optimizer_state = optimizer.update(
            gradients,
            optimizer_state,
            params,
        )
        params = optax.apply_updates(params, updates)
        finite = (
            jnp.isfinite(loss)
            & tree_is_finite(gradients)
            & tree_is_finite(params)
        )
        return params, optimizer_state, loss, finite

    @jax.jit
    def eval_step(params, inputs, targets):
        predictions = model_eval.apply({"params": params}, inputs)
        relative = relative_l2_jax(
            predictions,
            targets,
            ignore_prefix_steps=ignore_prefix_steps,
        )
        scored_predictions = predictions[:, ignore_prefix_steps:, :]
        scored_targets = targets[:, ignore_prefix_steps:, :]
        mse = jnp.mean(jnp.square(scored_predictions - scored_targets))
        return relative, mse, predictions

    return train_step, eval_step


def evaluate(
    params,
    eval_step,
    inputs: np.ndarray,
    targets: np.ndarray,
    *,
    batch_size: int,
    collect_predictions: bool = False,
) -> tuple[float, float, np.ndarray | None]:
    relative_values: list[float] = []
    mse_values: list[float] = []
    predictions: list[np.ndarray] = []
    for batch_inputs, batch_targets in iter_batches(inputs, targets, batch_size):
        relative, mse, batch_predictions = eval_step(
            params,
            jnp.asarray(batch_inputs),
            jnp.asarray(batch_targets),
        )
        relative_values.append(float(relative))
        mse_values.append(float(mse))
        if collect_predictions:
            predictions.append(np.asarray(jax.device_get(batch_predictions)))
    prediction_array = np.concatenate(predictions, axis=0) if predictions else None
    return (
        float(np.mean(relative_values)),
        float(np.mean(mse_values)),
        prediction_array,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the official S5 core on a sequence-regression data view."
    )
    parser.add_argument("--train-val-file", type=Path, required=True)
    parser.add_argument("--test-file", type=Path)
    parser.add_argument("--evaluate-test", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--task-name")
    parser.add_argument("--run-kind", default="unspecified")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=600)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--grad-clip", type=float)
    parser.add_argument("--d-model", type=int, default=20)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--ssm-size-base", type=int, default=40)
    parser.add_argument("--blocks", type=int, default=1)
    parser.add_argument("--activation", default="half_glu1")
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-val-samples", type=int)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.evaluate_test and args.test_file is None:
        raise SystemExit("--evaluate-test requires --test-file")
    if not args.evaluate_test and args.test_file is not None:
        raise SystemExit("Do not provide --test-file to a selection/calibration run")

    data = load_train_val(args.train_val_file.resolve())
    if args.max_train_samples is not None:
        data = data.__class__(
            x_train=data.x_train[: args.max_train_samples],
            y_train=data.y_train[: args.max_train_samples],
            x_val=data.x_val,
            y_val=data.y_val,
            ignore_prefix_steps=data.ignore_prefix_steps,
        )
    if args.max_val_samples is not None:
        data = data.__class__(
            x_train=data.x_train,
            y_train=data.y_train,
            x_val=data.x_val[: args.max_val_samples],
            y_val=data.y_val[: args.max_val_samples],
            ignore_prefix_steps=data.ignore_prefix_steps,
        )

    config = S5RegressionConfig(
        d_model=args.d_model,
        n_layers=args.n_layers,
        ssm_size_base=args.ssm_size_base,
        blocks=args.blocks,
        activation=args.activation,
    )
    model_train, params = initialize_model(
        config,
        input_channels=int(data.x_train.shape[-1]),
        output_channels=int(data.y_train.shape[-1]),
        sequence_length=int(data.x_train.shape[1]),
        batch_size=min(args.batch_size, int(data.x_train.shape[0])),
        seed=args.seed,
    )
    model_eval = S5SequenceRegressor(
        config=config,
        output_channels=int(data.y_train.shape[-1]),
        training=False,
    )
    parameter_count = count_real_parameters(params)
    steps_per_epoch = math.ceil(data.x_train.shape[0] / args.batch_size)
    optimizer, schedule = build_optimizer(
        params,
        learning_rate=args.learning_rate,
        steps_per_epoch=steps_per_epoch,
        epochs=args.epochs,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
    )
    scheduler_name = "one_epoch_linear_warmup_then_cosine_to_1e-6"
    optimizer_name = "official_s5_grouped_adam_adamw"
    optimizer_state = optimizer.init(params)
    train_step, eval_step = build_steps(
        model_train,
        model_eval,
        optimizer,
        data.ignore_prefix_steps,
    )

    run_dir = args.output_dir.resolve() / args.run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    history_path = run_dir / "history.csv"
    rng = np.random.default_rng(args.seed)
    dropout_key = jax.random.PRNGKey(args.seed + 1)
    best_val = float("inf")
    best_val_mse = float("inf")
    best_epoch = -1
    best_params_bytes: bytes | None = None
    numerical_failure: str | None = None

    start_time = time.time()
    with history_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "epoch",
                "train_rel_l2",
                "val_rel_l2",
                "val_mse",
                "learning_rate",
            ],
        )
        writer.writeheader()
        global_step = 0
        for epoch in range(args.epochs):
            order = rng.permutation(data.x_train.shape[0])
            train_values: list[float] = []
            for batch_inputs, batch_targets in iter_batches(
                data.x_train,
                data.y_train,
                args.batch_size,
                indices=order,
            ):
                dropout_key, batch_key = jax.random.split(dropout_key)
                params, optimizer_state, loss, finite = train_step(
                    params,
                    optimizer_state,
                    jnp.asarray(batch_inputs),
                    jnp.asarray(batch_targets),
                    batch_key,
                )
                if not bool(finite):
                    numerical_failure = f"non-finite loss/gradient/parameter at epoch {epoch}"
                    break
                train_values.append(float(loss))
                global_step += 1
            if numerical_failure is not None:
                break

            val_relative, val_mse, _ = evaluate(
                params,
                eval_step,
                data.x_val,
                data.y_val,
                batch_size=args.batch_size,
            )
            if not np.isfinite(val_relative) or val_relative > 10:
                numerical_failure = (
                    f"invalid validation relative L2 {val_relative} at epoch {epoch}"
                )
                break
            if val_relative < best_val:
                best_val = val_relative
                best_val_mse = val_mse
                best_epoch = epoch
                best_params_bytes = flax.serialization.to_bytes(jax.device_get(params))

            current_lr = float(schedule(max(global_step - 1, 0)))
            writer.writerow(
                {
                    "epoch": epoch,
                    "train_rel_l2": float(np.mean(train_values)),
                    "val_rel_l2": val_relative,
                    "val_mse": val_mse,
                    "learning_rate": current_lr,
                }
            )
            handle.flush()
            print(
                f"epoch={epoch:04d} train_rel={np.mean(train_values):.6f} "
                f"val_rel={val_relative:.6f} val_mse={val_mse:.6e} "
                f"lr={current_lr:.3e}",
                flush=True,
            )

    train_time_seconds = time.time() - start_time
    if best_params_bytes is None:
        numerical_failure = numerical_failure or "no finite validation checkpoint"
        best_params = params
    else:
        best_params = flax.serialization.from_bytes(params, best_params_bytes)
        (run_dir / "model_best.msgpack").write_bytes(best_params_bytes)

    test_relative = None
    test_mse = None
    if args.evaluate_test:
        test_data = load_test(args.test_file.resolve())
        if test_data.ignore_prefix_steps != data.ignore_prefix_steps:
            raise ValueError("Train/validation and test prefix masks differ")
        test_relative, test_mse, predictions = evaluate(
            best_params,
            eval_step,
            test_data.x_test,
            test_data.y_test,
            batch_size=1,
            collect_predictions=True,
        )
        np.savez_compressed(
            run_dir / "predictions_test.npz",
            y_pred=predictions,
            y_test=test_data.y_test,
            test_rel_l2=np.asarray(test_relative),
            metric_ignore_prefix_steps=np.asarray(data.ignore_prefix_steps),
        )

    summary = {
        "run_name": args.run_name,
        "run_kind": args.run_kind,
        "task": args.task_name,
        "model": "s5_official",
        "official_s5_commit": OFFICIAL_S5_COMMIT,
        "seed": args.seed,
        "parameter_count": parameter_count,
        "parameter_count_real": parameter_count,
        "model_config": config.to_dict(),
        "training_config": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "grad_clip": args.grad_clip,
            "optimizer": optimizer_name,
            "scheduler": scheduler_name,
            "objective": "plain_per_sample_relative_l2",
        },
        "data": {
            "train_val_file": str(args.train_val_file.resolve()),
            "test_file": str(args.test_file.resolve()) if args.test_file else None,
            "train_shape": list(data.x_train.shape),
            "val_shape": list(data.x_val.shape),
            "metric_ignore_prefix_steps": data.ignore_prefix_steps,
        },
        "selected_checkpoint": "best_val_rel_l2",
        "best_epoch": best_epoch,
        "best_val_rel_l2": best_val if np.isfinite(best_val) else None,
        "best_val_mse": best_val_mse if np.isfinite(best_val_mse) else None,
        "test_rel_l2": test_relative,
        "test_mse": test_mse,
        "test_accessed": bool(args.evaluate_test),
        "numerical_failure": numerical_failure,
        "train_time_seconds": train_time_seconds,
        "jax_version": jax.__version__,
        "jax_devices": [str(device) for device in jax.devices()],
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (run_dir / "config.json").write_text(
        json.dumps(
            {
                "model": asdict(config),
                "arguments": vars(args) | {
                    "train_val_file": str(args.train_val_file),
                    "test_file": str(args.test_file) if args.test_file else None,
                    "output_dir": str(args.output_dir),
                },
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    if numerical_failure is not None:
        print(f"NUMERICAL_FAILURE: {numerical_failure}", flush=True)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
