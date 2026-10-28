from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np


@dataclass(frozen=True)
class TrainValData:
    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    ignore_prefix_steps: int


@dataclass(frozen=True)
class TestData:
    x_test: np.ndarray
    y_test: np.ndarray
    ignore_prefix_steps: int


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_summary(value: np.ndarray) -> dict[str, object]:
    contiguous = np.ascontiguousarray(value)
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "mean": float(value.mean()),
        "std": float(value.std()),
        "sha256": hashlib.sha256(contiguous.view(np.uint8)).hexdigest(),
    }


def export_case(source_dir: Path, output_dir: Path) -> dict[str, object]:
    source_file = source_dir / "data.npz"
    meta_file = source_dir / "meta.json"
    if not source_file.is_file():
        raise FileNotFoundError(source_file)

    meta = json.loads(meta_file.read_text(encoding="utf-8")) if meta_file.exists() else {}
    ignore_prefix_steps = int(meta.get("metric_ignore_prefix_steps", 0))

    with np.load(source_file) as source:
        arrays = {
            key: np.asarray(source[key], dtype=np.float32)
            for key in (
                "x_train",
                "y_train",
                "x_val",
                "y_val",
                "x_test",
                "y_test",
                "grid",
            )
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    train_val_file = output_dir / "train_val.npz"
    test_file = output_dir / "test.npz"
    np.savez_compressed(
        train_val_file,
        x_train=arrays["x_train"],
        y_train=arrays["y_train"],
        x_val=arrays["x_val"],
        y_val=arrays["y_val"],
        grid=arrays["grid"],
        metric_ignore_prefix_steps=np.asarray(ignore_prefix_steps, dtype=np.int64),
    )
    np.savez_compressed(
        test_file,
        x_test=arrays["x_test"],
        y_test=arrays["y_test"],
        grid=arrays["grid"],
        metric_ignore_prefix_steps=np.asarray(ignore_prefix_steps, dtype=np.int64),
    )

    manifest = {
        "source": str(source_file.resolve()),
        "source_sha256": sha256_file(source_file),
        "source_meta": meta,
        "metric_ignore_prefix_steps": ignore_prefix_steps,
        "arrays": {key: array_summary(value) for key, value in arrays.items()},
        "train_val_file": str(train_val_file.resolve()),
        "train_val_sha256": sha256_file(train_val_file),
        "test_file": str(test_file.resolve()),
        "test_sha256": sha256_file(test_file),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return manifest


def load_train_val(path: Path) -> TrainValData:
    with np.load(path) as data:
        expected = {"x_train", "y_train", "x_val", "y_val"}
        missing = expected.difference(data.files)
        if missing:
            raise KeyError(f"Missing train/validation arrays in {path}: {sorted(missing)}")
        if {"x_test", "y_test"}.intersection(data.files):
            raise ValueError(f"Training view must not contain test arrays: {path}")
        return TrainValData(
            x_train=np.asarray(data["x_train"], dtype=np.float32),
            y_train=np.asarray(data["y_train"], dtype=np.float32),
            x_val=np.asarray(data["x_val"], dtype=np.float32),
            y_val=np.asarray(data["y_val"], dtype=np.float32),
            ignore_prefix_steps=int(data["metric_ignore_prefix_steps"])
            if "metric_ignore_prefix_steps" in data.files
            else 0,
        )


def load_test(path: Path) -> TestData:
    with np.load(path) as data:
        return TestData(
            x_test=np.asarray(data["x_test"], dtype=np.float32),
            y_test=np.asarray(data["y_test"], dtype=np.float32),
            ignore_prefix_steps=int(data["metric_ignore_prefix_steps"])
            if "metric_ignore_prefix_steps" in data.files
            else 0,
        )


def iter_batches(
    inputs: np.ndarray,
    targets: np.ndarray,
    batch_size: int,
    *,
    indices: np.ndarray | None = None,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    order = np.arange(inputs.shape[0]) if indices is None else indices
    for start in range(0, order.size, batch_size):
        batch_indices = order[start : start + batch_size]
        yield inputs[batch_indices], targets[batch_indices]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export test-isolated S5 data views.")
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = export_case(args.source_dir.resolve(), args.output_dir.resolve())
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

