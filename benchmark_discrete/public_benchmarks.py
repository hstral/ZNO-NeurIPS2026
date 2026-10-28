import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np


def _import_nonlinear_benchmarks(auto_install: bool):
    try:
        import nonlinear_benchmarks as nlb  # type: ignore

        return nlb
    except ImportError:
        if not auto_install:
            raise
        subprocess.check_call([sys.executable, "-m", "pip", "install", "nonlinear-benchmarks"])
        import nonlinear_benchmarks as nlb  # type: ignore

        return nlb


def _to_1d_pair(dataset) -> Tuple[np.ndarray, np.ndarray]:
    if hasattr(dataset, "u") and hasattr(dataset, "y"):
        u = np.asarray(dataset.u, dtype=np.float64)
        y = np.asarray(dataset.y, dtype=np.float64)
    else:
        u, y = dataset
        u = np.asarray(u, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
    return u.reshape(-1, 1), y.reshape(-1, 1)


def _resolve_suite_dir(repo_root: Path, suite: str, data_root: Optional[Path]) -> Path:
    candidates = []
    if data_root is not None:
        candidates.append(Path(data_root).resolve() / suite)
    else:
        candidates.extend(
            [
                repo_root / "benchmark_discrete_data_external_v3" / suite,
                repo_root / "benchmark_discrete_data_external_v2" / suite,
                repo_root / "benchmark_discrete_data_external_v1" / suite,
            ]
        )
    for candidate in candidates:
        if (candidate / "data.npz").exists() and (candidate / "meta.json").exists():
            return candidate
    raise FileNotFoundError(f"Could not resolve public benchmark data for suite={suite}")


def _load_suite_streams(suite: str, auto_install: bool):
    nlb = _import_nonlinear_benchmarks(auto_install=auto_install)
    if suite == "silverbox_public_v1":
        train_val, test = nlb.Silverbox(atleast_2d=True)
    elif suite == "wiener_hammer_public_v1":
        train_val, test = nlb.WienerHammerBenchMark(atleast_2d=True)
    elif suite == "cascaded_tanks_public_v1":
        train_val, test = nlb.Cascaded_Tanks(atleast_2d=True)
    elif suite == "emps_public_v1":
        train_val, test = nlb.EMPS(atleast_2d=True)
    elif suite == "ced_public_v1":
        train_val, test = nlb.CED(atleast_2d=True)
    else:
        raise ValueError(f"Unsupported public suite: {suite}")

    train_dataset = train_val[0] if isinstance(train_val, (tuple, list)) else train_val
    test_dataset = test[0] if isinstance(test, (tuple, list)) else test
    u_train_full, y_train_full = _to_1d_pair(train_dataset)
    u_test_full, y_test_full = _to_1d_pair(test_dataset)
    return train_dataset, test_dataset, u_train_full, y_train_full, u_test_full, y_test_full


@dataclass(frozen=True)
class PublicBenchmarkProtocol:
    suite: str
    suite_dir: Path
    meta: Dict[str, object]
    train_u: np.ndarray
    train_y: np.ndarray
    val_u: np.ndarray
    val_y: np.ndarray
    x_val_windows: np.ndarray
    y_val_windows: np.ndarray
    x_test_windows: np.ndarray
    y_test_windows: np.ndarray
    metric_ignore_prefix_steps: int


def load_public_benchmark_protocol(
    *,
    repo_root: Path,
    suite: str,
    data_root: Optional[Path] = None,
    auto_install: bool = False,
) -> PublicBenchmarkProtocol:
    repo_root = Path(repo_root).resolve()
    suite_dir = _resolve_suite_dir(repo_root, suite, data_root)
    meta = json.loads((suite_dir / "meta.json").read_text(encoding="utf-8"))

    raw = np.load(suite_dir / "data.npz")
    x_val_windows = raw["x_val"].astype(np.float64)
    y_val_windows = raw["y_val"].astype(np.float64)
    x_test_windows = raw["x_test"].astype(np.float64)
    y_test_windows = raw["y_test"].astype(np.float64)

    if "train_stream_u" in raw and "train_stream_y" in raw and "val_stream_u" in raw and "val_stream_y" in raw:
        train_u = raw["train_stream_u"].astype(np.float64)
        train_y = raw["train_stream_y"].astype(np.float64)
        val_u = raw["val_stream_u"].astype(np.float64)
        val_y = raw["val_stream_y"].astype(np.float64)
    else:
        _, _, u_train_full, y_train_full, _, _ = _load_suite_streams(suite, auto_install=auto_install)
        sequence_length = int(meta["sequence_length"])
        split_idx = int(0.8 * len(u_train_full))
        split_idx = max(split_idx, sequence_length + 1)

        train_u = u_train_full[:split_idx]
        train_y = y_train_full[:split_idx]
        val_u = u_train_full[split_idx:]
        val_y = y_train_full[split_idx:]

        norm = meta["normalization"]
        u_mean = float(norm["u_mean"])
        u_std = float(norm["u_std"])
        y_mean = float(norm["y_mean"])
        y_std = float(norm["y_std"])

        train_u = ((train_u - u_mean) / u_std).astype(np.float64)
        train_y = ((train_y - y_mean) / y_std).astype(np.float64)
        val_u = ((val_u - u_mean) / u_std).astype(np.float64)
        val_y = ((val_y - y_mean) / y_std).astype(np.float64)

    return PublicBenchmarkProtocol(
        suite=suite,
        suite_dir=suite_dir,
        meta=meta,
        train_u=train_u,
        train_y=train_y,
        val_u=val_u,
        val_y=val_y,
        x_val_windows=x_val_windows,
        y_val_windows=y_val_windows,
        x_test_windows=x_test_windows,
        y_test_windows=y_test_windows,
        metric_ignore_prefix_steps=int(meta.get("metric_ignore_prefix_steps", 0)),
    )
