import copy
import json
import os
import random
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import scipy.io
import torch
import torch.nn.functional as F

from .adam import Adam
from .parameters import parameter_counts
from .configs import CaseConfig, OperatorConfig
from .data import build_dataloaders, load_case_data
from .models import build_model


try:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None


class LpLoss:
    def __init__(self, p: int = 2, size_average: bool = True):
        self.p = p
        self.size_average = size_average

    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        diff_norms = torch.norm(x.reshape(batch_size, -1) - y.reshape(batch_size, -1), self.p, 1)
        y_norms = torch.norm(y.reshape(batch_size, -1), self.p, 1)
        ratio = diff_norms / y_norms
        return ratio.mean() if self.size_average else ratio.sum()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _jsonable(data: Dict[str, Any]) -> Dict[str, Any]:
    payload = {}
    for key, value in data.items():
        if isinstance(value, Path):
            payload[key] = str(value)
        else:
            payload[key] = value
    return payload


def _snapshot_state_dict(model: torch.nn.Module) -> Dict[str, Any]:
    snapshot = copy.deepcopy(model.state_dict())
    if "_metadata" in snapshot:
        del snapshot["_metadata"]
    return snapshot


def _override_config(
    base: OperatorConfig,
    *,
    width: Optional[int] = None,
    depth: Optional[int] = None,
    rank: Optional[int] = None,
    modes: Optional[int] = None,
    num_poles: Optional[int] = None,
    fir_order: Optional[int] = None,
    batch_size: Optional[int] = None,
    learning_rate: Optional[float] = None,
    epochs: Optional[int] = None,
    safe_radius: Optional[float] = None,
    fused_backward_mode: Optional[str] = None,
    kernel_backend: Optional[str] = None,
) -> OperatorConfig:
    update = {}
    if width is not None:
        update["width"] = width
    if depth is not None:
        update["depth"] = depth
    if rank is not None:
        update["rank"] = rank
    if modes is not None:
        update["modes"] = modes
    if num_poles is not None:
        update["num_poles"] = num_poles
    if fir_order is not None:
        update["fir_order"] = fir_order
    if batch_size is not None:
        update["batch_size"] = batch_size
    if learning_rate is not None:
        update["learning_rate"] = learning_rate
    if epochs is not None:
        update["epochs"] = epochs
    if safe_radius is not None:
        update["safe_radius"] = safe_radius
    if fused_backward_mode is not None:
        update["fused_backward_mode"] = fused_backward_mode
    if kernel_backend is not None:
        update["kernel_backend"] = kernel_backend
    return replace(base, **update)


def _save_plot(epochs: int, train_loss: np.ndarray, val_loss: np.ndarray, output_path: Path):
    if plt is None:
        return
    x_axis = np.arange(1, epochs + 1)
    fig = plt.figure(constrained_layout=False, figsize=(7, 7))
    ax = fig.add_subplot(111)
    ax.plot(x_axis, train_loss[:, 0], color="blue", label="Train Loss")
    ax.plot(x_axis, val_loss[:, 0], color="red", label="Validation Loss")
    ax.set_yscale("log")
    ax.set_ylabel("Loss")
    ax.set_xlabel("Epochs")
    ax.legend(loc="upper left")
    fig.savefig(output_path)
    plt.close(fig)


def _zno_step_loss(
    *,
    model: torch.nn.Module,
    pred: torch.Tensor,
    target: torch.Tensor,
    rel_loss: LpLoss,
    lambda_pole: float,
    lambda_roll: float,
    rollout_context_steps: Optional[int],
) -> torch.Tensor:
    data_loss = rel_loss(pred, target)
    loss = data_loss + lambda_pole * model.pole_penalty()
    if lambda_roll > 0:
        context_steps = rollout_context_steps
        if context_steps is None:
            context_steps = max(1, pred.size(1) // 4)
        context_steps = max(0, min(context_steps, pred.size(1) - 1))
        rollout_pred = pred[:, context_steps:, :]
        rollout_target = target[:, context_steps:, :]
        loss = loss + lambda_roll * rel_loss(rollout_pred, rollout_target)
    return loss


def _build_optimizer_and_scheduler(model: torch.nn.Module, model_config: OperatorConfig):
    optimizer = Adam(
        model.parameters(),
        lr=model_config.learning_rate,
        weight_decay=model_config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=model_config.step_size,
        gamma=model_config.gamma,
    )
    return optimizer, scheduler


def _run_zno_timing_warmup(
    *,
    model: torch.nn.Module,
    model_config: OperatorConfig,
    data,
    torch_device: torch.device,
    rel_loss: LpLoss,
    lambda_pole: float,
    lambda_roll: float,
    rollout_context_steps: Optional[int],
):
    if torch_device.type != "cuda" or data.x_train.size(0) == 0:
        optimizer, scheduler = _build_optimizer_and_scheduler(model, model_config)
        return optimizer, scheduler, {"enabled": False}

    warmup_batch_size = min(model_config.batch_size, int(data.x_train.size(0)))
    x_warm = data.x_train[:warmup_batch_size].to(torch_device)
    y_warm = data.y_train[:warmup_batch_size].to(torch_device)
    initial_state = _snapshot_state_dict(model)
    model.train()
    optimizer, _ = _build_optimizer_and_scheduler(model, model_config)
    warmup_start = time.time()
    optimizer.zero_grad(set_to_none=True)
    pred = model(x_warm)
    loss = _zno_step_loss(
        model=model,
        pred=pred,
        target=y_warm,
        rel_loss=rel_loss,
        lambda_pole=lambda_pole,
        lambda_roll=lambda_roll,
        rollout_context_steps=rollout_context_steps,
    )
    loss.backward()
    optimizer.step()
    if torch_device.type == "cuda":
        torch.cuda.synchronize(torch_device)
    warmup_seconds = time.time() - warmup_start
    model.load_state_dict(initial_state)
    optimizer, scheduler = _build_optimizer_and_scheduler(model, model_config)
    return optimizer, scheduler, {
        "enabled": True,
        "batch_size": warmup_batch_size,
        "excluded_seconds": warmup_seconds,
    }


def _build_timing_protocol(
    *,
    model_name: str,
    device: str,
    model_config: OperatorConfig,
    timing_warmup: Dict[str, Any],
) -> Dict[str, Any]:
    protocol = {
        "scope": "epoch_loop_only",
        "device": device,
        "includes": [
            "train_forward_backward_optimizer",
            "validation_forward",
            "scheduler_step",
        ],
        "excludes": [
            "data_loading_and_tensor_materialization",
            "model_build_and_optimizer_build",
            "checkpoint_serialization",
            "test_evaluation",
        ],
    }
    if model_name == "zno":
        protocol["excludes"].append("excluded_cuda_warmup_step_before_timing")
        protocol["official_zno_timing"] = {
            "kernel_backend": model_config.kernel_backend,
            "fused_backward_mode": model_config.fused_backward_mode,
            "warmup_before_timing": bool(timing_warmup.get("enabled", False)),
            "warmup_excluded_seconds": float(timing_warmup.get("excluded_seconds", 0.0)),
            "warmup_batch_size": timing_warmup.get("batch_size"),
            "precompile_requested": os.environ.get("PRECOMPILE_ZNO", "0") == "1",
            "steady_state_target": "same_hardware_same_environment",
        }
    return protocol


def run_experiment(
    *,
    repo_root: Path,
    case: CaseConfig,
    model_name: str,
    output_dir: Path,
    run_name: str,
    device: str,
    seed: int,
    width: Optional[int] = None,
    depth: Optional[int] = None,
    rank: Optional[int] = None,
    modes: Optional[int] = None,
    num_poles: Optional[int] = None,
    fir_order: Optional[int] = None,
    batch_size: Optional[int] = None,
    learning_rate: Optional[float] = None,
    epochs: Optional[int] = None,
    max_train_samples: Optional[int] = None,
    max_val_samples: Optional[int] = None,
    max_test_samples: Optional[int] = None,
    lambda_pole: float = 1e-3,
    lambda_roll: float = 1e-2,
    rollout_context_steps: Optional[int] = None,
    safe_radius: Optional[float] = None,
    fused_backward_mode: Optional[str] = None,
    kernel_backend: Optional[str] = None,
):
    repo_root = Path(repo_root).resolve()
    output_dir = Path(output_dir).resolve()
    run_dir = output_dir / case.name / model_name / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    set_seed(seed)
    model_config = _override_config(
        case.models[model_name],
        width=width,
        depth=depth,
        rank=rank,
        modes=modes,
        num_poles=num_poles,
        fir_order=fir_order,
        batch_size=batch_size,
        learning_rate=learning_rate,
        epochs=epochs,
        safe_radius=safe_radius,
        fused_backward_mode=fused_backward_mode,
        kernel_backend=kernel_backend,
    )

    data = load_case_data(
        case,
        max_train_samples=max_train_samples,
        max_val_samples=max_val_samples,
        max_test_samples=max_test_samples,
    )
    torch_device = torch.device(device)
    train_loader, val_loader, test_loader = build_dataloaders(
        data,
        model_config.batch_size,
        seed=seed,
    )
    model = build_model(
        model_name=model_name,
        config=model_config,
        grid=data.grid,
        use_grid=case.use_grid,
    ).to(torch_device)
    rel_loss = LpLoss(size_average=True)
    if model_name == "zno":
        optimizer, scheduler, timing_warmup = _run_zno_timing_warmup(
            model=model,
            model_config=model_config,
            data=data,
            torch_device=torch_device,
            rel_loss=rel_loss,
            lambda_pole=lambda_pole,
            lambda_roll=lambda_roll,
            rollout_context_steps=rollout_context_steps,
        )
    else:
        optimizer, scheduler = _build_optimizer_and_scheduler(model, model_config)
        timing_warmup = {"enabled": False}

    train_error = np.zeros((model_config.epochs, 1), dtype=np.float32)
    train_mse_history = np.zeros((model_config.epochs, 1), dtype=np.float32)
    val_error = np.zeros((model_config.epochs, 1), dtype=np.float32)
    val_mse_history = np.zeros((model_config.epochs, 1), dtype=np.float32)
    best_val_rel = float("inf")
    best_val_mse = float("inf")
    best_epoch = -1
    best_state_dict = None

    start_time = time.time()
    for epoch in range(model_config.epochs):
        model.train()
        epoch_train_rel = 0.0
        epoch_train_mse = 0.0
        num_train_batches = 0

        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(torch_device)
            y_batch = y_batch.to(torch_device)
            optimizer.zero_grad()
            pred = model(x_batch)
            mse = F.mse_loss(pred.reshape(pred.size(0), -1), y_batch.reshape(y_batch.size(0), -1))
            data_loss = rel_loss(pred, y_batch)
            loss = data_loss
            if model_name == "zno":
                loss = _zno_step_loss(
                    model=model,
                    pred=pred,
                    target=y_batch,
                    rel_loss=rel_loss,
                    lambda_pole=lambda_pole,
                    lambda_roll=lambda_roll,
                    rollout_context_steps=rollout_context_steps,
                )
            loss.backward()
            optimizer.step()

            epoch_train_mse += float(mse.item())
            epoch_train_rel += float(data_loss.item())
            num_train_batches += 1

        scheduler.step()

        model.eval()
        epoch_val_rel = 0.0
        epoch_val_mse = 0.0
        num_val_batches = 0
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch = x_batch.to(torch_device)
                y_batch = y_batch.to(torch_device)
                pred = model(x_batch)
                mse = F.mse_loss(
                    pred.reshape(pred.size(0), -1),
                    y_batch.reshape(y_batch.size(0), -1),
                )
                rel = rel_loss(pred, y_batch)
                epoch_val_mse += float(mse.item())
                epoch_val_rel += float(rel.item())
                num_val_batches += 1

        epoch_train_mse /= max(num_train_batches, 1)
        epoch_train_rel /= max(num_train_batches, 1)
        epoch_val_mse /= max(num_val_batches, 1)
        epoch_val_rel /= max(num_val_batches, 1)
        train_mse_history[epoch, 0] = epoch_train_mse
        train_error[epoch, 0] = epoch_train_rel
        val_mse_history[epoch, 0] = epoch_val_mse
        val_error[epoch, 0] = epoch_val_rel
        if epoch_val_rel < best_val_rel:
            best_val_rel = epoch_val_rel
            best_val_mse = epoch_val_mse
            best_epoch = epoch
            best_state_dict = _snapshot_state_dict(model)

        message = (
            f"Epoch: {epoch:04d}, "
            f"Train Loss: {epoch_train_mse:.3e}, "
            f"Vali Loss: {epoch_val_mse:.3e}, "
            f"Train l2: {epoch_train_rel:.4f}, "
            f"Vali l2: {epoch_val_rel:.4f}"
        )
        if model_name == "zno":
            stats = model.stability_stats()
            message += (
                f", PoleMean: {stats['mean_pole_radius']:.4f}, "
                f"PoleMax: {stats['max_pole_radius']:.4f}"
            )
        print(message)

    train_seconds = time.time() - start_time
    last_val_rel = float(val_error[-1, 0])
    last_val_mse = float(val_mse_history[-1, 0])
    last_state_dict = _snapshot_state_dict(model)
    if best_state_dict is None:
        best_state_dict = last_state_dict
        best_val_rel = last_val_rel
        best_val_mse = last_val_mse
        best_epoch = model_config.epochs - 1
    model.load_state_dict(best_state_dict)

    predictions = torch.zeros_like(data.y_test)
    test_rel = 0.0
    with torch.no_grad():
        for index, (x_batch, y_batch) in enumerate(test_loader):
            x_batch = x_batch.to(torch_device)
            y_batch = y_batch.to(torch_device)
            pred = model(x_batch)
            predictions[index] = pred.detach().cpu()
            test_rel += float(rel_loss(pred, y_batch).item())
    test_rel /= max(len(test_loader), 1)

    np.savetxt(run_dir / "epoch.txt", np.arange(model_config.epochs))
    np.savetxt(run_dir / "train_loss.txt", train_mse_history)
    np.savetxt(run_dir / "vali_loss.txt", val_mse_history)
    np.savetxt(run_dir / "train_error.txt", train_error)
    np.savetxt(run_dir / "vali_error.txt", val_error)
    _save_plot(model_config.epochs, train_mse_history, val_mse_history, run_dir / "loss_history.png")

    best_checkpoint = {
        "model_state_dict": best_state_dict,
        "case": case.to_dict(),
        "model_name": model_name,
        "model_config": asdict(model_config),
        "seed": seed,
        "selected_by": "best_val_rel_l2",
        "best_epoch": best_epoch,
        "best_val_rel_l2": best_val_rel,
        "best_val_mse": best_val_mse,
    }
    torch.save(best_checkpoint, run_dir / "model.pt")
    torch.save(best_checkpoint, run_dir / "model_best.pt")
    torch.save(
        {
            "model_state_dict": last_state_dict,
            "case": case.to_dict(),
            "model_name": model_name,
            "model_config": asdict(model_config),
            "seed": seed,
            "selected_by": "last_epoch",
            "last_epoch": model_config.epochs - 1,
            "last_val_rel_l2": last_val_rel,
            "last_val_mse": last_val_mse,
        },
        run_dir / "model_last.pt",
    )

    scipy.io.savemat(
        run_dir / "wave_states_test.mat",
        mdict={
            "test_err": test_rel,
            "x_test": data.grid.numpy(),
            "y_test": data.y_test.squeeze(-1).numpy(),
            "y_pred": predictions.squeeze(-1).numpy(),
        },
    )

    summary: Dict[str, Any] = {
        "case": case.name,
        "model": model_name,
        "run_name": run_name,
        "train_time_seconds": train_seconds,
        "test_rel_l2": test_rel,
        "seed": seed,
        "device": device,
        **parameter_counts(model),
        "run_dir": str(run_dir),
        "model_config": asdict(model_config),
        "selected_checkpoint": "best_val_rel_l2",
        "best_epoch": best_epoch,
        "best_val_rel_l2": best_val_rel,
        "best_val_mse": best_val_mse,
        "last_val_rel_l2": last_val_rel,
        "last_val_mse": last_val_mse,
        "data_shape": {
            "train": list(data.x_train.shape),
            "val": list(data.x_val.shape),
            "test": list(data.x_test.shape),
        },
        "timing_protocol": _build_timing_protocol(
            model_name=model_name,
            device=device,
            model_config=model_config,
            timing_warmup=timing_warmup,
        ),
    }
    if model_name == "zno":
        summary.update(model.stability_stats())
        summary["timing_warmup"] = timing_warmup

    with open(run_dir / "summary.json", "w", encoding="utf-8") as file:
        json.dump(_jsonable(summary), file, indent=2)

    with open(run_dir / "case_config.json", "w", encoding="utf-8") as file:
        json.dump(case.to_dict(), file, indent=2)

    print("\n=============================")
    print(f"Training time: {train_seconds:.3f}s")
    print(f"Testing error: {test_rel:.3e}")
    print(f"Results saved to: {run_dir}")
    print("=============================\n")

    return summary
