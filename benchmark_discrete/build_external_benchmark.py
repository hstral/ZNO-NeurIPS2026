import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

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
        u = np.asarray(dataset.u, dtype=np.float32)
        y = np.asarray(dataset.y, dtype=np.float32)
    else:
        u, y = dataset
        u = np.asarray(u, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
    return u.reshape(-1), y.reshape(-1)


def _as_dataset_list(dataset_group) -> List[object]:
    if isinstance(dataset_group, (tuple, list)):
        return list(dataset_group)
    return [dataset_group]


def _split_sequence_for_train_val(
    u: np.ndarray,
    y: np.ndarray,
    *,
    min_train_length: int,
    min_val_length: int,
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    split_idx = int(0.8 * len(u))
    split_idx = max(split_idx, min_train_length)
    split_idx = min(split_idx, len(u) - min_val_length)
    if split_idx < min_train_length or len(u) - split_idx < min_val_length:
        raise ValueError(
            f"Sequence too short for split: len={len(u)}, "
            f"need train>={min_train_length}, val>={min_val_length}"
        )
    return (u[:split_idx], y[:split_idx]), (u[split_idx:], y[split_idx:])


def _make_windows(
    u: np.ndarray,
    y: np.ndarray,
    *,
    length: int,
    stride: int,
    max_windows: int,
) -> Tuple[np.ndarray, np.ndarray]:
    xs = []
    ys = []
    if len(u) < length:
        raise ValueError(f"Sequence too short for length={length}: got {len(u)}")
    for start in range(0, len(u) - length + 1, stride):
        end = start + length
        xs.append(u[start:end, None])
        ys.append(y[start:end, None])
        if len(xs) >= max_windows:
            break
    if not xs:
        raise ValueError("No windows generated")
    return np.stack(xs, axis=0).astype(np.float32), np.stack(ys, axis=0).astype(np.float32)


def _make_windows_multi(
    sequences: Sequence[Tuple[np.ndarray, np.ndarray]],
    *,
    length: int,
    stride: int,
    max_windows: int,
) -> Tuple[np.ndarray, np.ndarray]:
    xs = []
    ys = []
    for u, y in sequences:
        if len(u) < length:
            continue
        for start in range(0, len(u) - length + 1, stride):
            end = start + length
            xs.append(u[start:end, None])
            ys.append(y[start:end, None])
            if len(xs) >= max_windows:
                break
        if len(xs) >= max_windows:
            break
    if not xs:
        raise ValueError("No windows generated across all sequences")
    return np.stack(xs, axis=0).astype(np.float32), np.stack(ys, axis=0).astype(np.float32)


def _concat_streams(sequences: Sequence[Tuple[np.ndarray, np.ndarray]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not sequences:
        raise ValueError("No sequences provided")
    u_concat = np.concatenate([u.reshape(-1) for u, _ in sequences], axis=0).astype(np.float32)
    y_concat = np.concatenate([y.reshape(-1) for _, y in sequences], axis=0).astype(np.float32)
    offsets = []
    cursor = 0
    for u, _ in sequences:
        offsets.append(cursor)
        cursor += int(len(u))
    return u_concat[:, None], y_concat[:, None], np.asarray(offsets, dtype=np.int64)


def parse_args():
    parser = argparse.ArgumentParser(description="Build external public system-ID benchmarks.")
    parser.add_argument(
        "--suite",
        choices=[
            "silverbox_public_v1",
            "cascaded_tanks_public_v1",
            "wiener_hammer_public_v1",
            "emps_public_v1",
            "ced_public_v1",
        ],
        default="silverbox_public_v1",
    )
    parser.add_argument("--output-dir", default="data/public")
    parser.add_argument("--sequence-length", type=int, default=2048)
    parser.add_argument("--train-size", type=int, default=512)
    parser.add_argument("--val-size", type=int, default=128)
    parser.add_argument("--test-size", type=int, default=256)
    parser.add_argument("--train-stride", type=int, default=128)
    parser.add_argument("--val-stride", type=int, default=128)
    parser.add_argument("--test-stride", type=int, default=128)
    parser.add_argument(
        "--protocol",
        choices=["legacy", "public", "stage20"],
        default="public",
        help="public uses all released sequences and masks the benchmark warm-up prefix; legacy keeps the old single-sequence/no-warmup convention.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--auto-install", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.output_dir).resolve() / args.suite
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_home = out_dir / ".nonlinear_benchmarks_home"
    cache_home.mkdir(parents=True, exist_ok=True)
    os.environ["HOME"] = str(cache_home)
    ready_file = out_dir / "_READY"
    if ready_file.exists():
        ready_file.unlink()
    if not args.overwrite and (out_dir / "data.npz").exists():
        ready_file.write_text("ready\n", encoding="utf-8")
        print(f"External benchmark already exists: {out_dir}")
        return

    nlb = _import_nonlinear_benchmarks(auto_install=args.auto_install)

    if args.suite == "silverbox_public_v1":
        train_val, test = nlb.Silverbox(atleast_2d=True)
        source_url = "https://www.nonlinearbenchmark.org/benchmarks/silverbox"
    elif args.suite == "cascaded_tanks_public_v1":
        train_val, test = nlb.Cascaded_Tanks(atleast_2d=True)
        source_url = "https://www.nonlinearbenchmark.org/benchmarks/cascaded-tanks"
    elif args.suite == "wiener_hammer_public_v1":
        train_val, test = nlb.WienerHammerBenchMark(atleast_2d=True)
        source_url = "https://www.nonlinearbenchmark.org/benchmarks/wiener-hammerstein"
    elif args.suite == "emps_public_v1":
        train_val, test = nlb.EMPS(atleast_2d=True)
        source_url = "https://www.nonlinearbenchmark.org/benchmarks/emps"
    elif args.suite == "ced_public_v1":
        train_val, test = nlb.CED(atleast_2d=True)
        source_url = "https://www.nonlinearbenchmark.org/benchmarks/coupled-electric-drives"
    else:  # pragma: no cover
        raise ValueError(f"Unsupported suite: {args.suite}")

    train_datasets = _as_dataset_list(train_val)
    test_datasets = _as_dataset_list(test)
    sampling_time = float(getattr(train_datasets[0], "sampling_time", 1.0))
    init_window = max(int(getattr(dataset, "state_initialization_window_length", 0)) for dataset in test_datasets)

    if args.protocol == "legacy":
        train_dataset = train_datasets[0]
        test_dataset = test_datasets[0]
        u_train_full, y_train_full = _to_1d_pair(train_dataset)
        u_test_full, y_test_full = _to_1d_pair(test_dataset)
        (u_train_stream, y_train_stream), (u_val_stream, y_val_stream) = _split_sequence_for_train_val(
            u_train_full,
            y_train_full,
            min_train_length=args.sequence_length + 1,
            min_val_length=args.sequence_length,
        )
        scored_length = int(args.sequence_length)
        total_length = scored_length
        train_streams = [(u_train_stream, y_train_stream)]
        val_streams = [(u_val_stream, y_val_stream)]
        test_streams = [(u_test_full, y_test_full)]
        metric_ignore_prefix_steps = 0
        sequence_policy = "first_sequence_only"
    else:
        scored_length = int(args.sequence_length)
        metric_ignore_prefix_steps = int(init_window)
        total_length = scored_length + metric_ignore_prefix_steps
        train_streams = []
        val_streams = []
        for dataset in train_datasets:
            u_full, y_full = _to_1d_pair(dataset)
            train_split, val_split = _split_sequence_for_train_val(
                u_full,
                y_full,
                min_train_length=total_length + 1,
                min_val_length=total_length,
            )
            train_streams.append(train_split)
            val_streams.append(val_split)
        test_streams = [_to_1d_pair(dataset) for dataset in test_datasets]
        sequence_policy = "all_sequences"

    u_train_concat = np.concatenate([u.reshape(-1) for u, _ in train_streams], axis=0)
    y_train_concat = np.concatenate([y.reshape(-1) for _, y in train_streams], axis=0)
    u_mean = float(u_train_concat.mean())
    u_std = float(u_train_concat.std() + 1e-6)
    y_mean = float(y_train_concat.mean())
    y_std = float(y_train_concat.std() + 1e-6)

    def normalize(u_arr: np.ndarray, y_arr: np.ndarray):
        return ((u_arr - u_mean) / u_std).astype(np.float32), ((y_arr - y_mean) / y_std).astype(np.float32)

    train_norm_streams = [normalize(u, y) for u, y in train_streams]
    val_norm_streams = [normalize(u, y) for u, y in val_streams]
    test_norm_streams = [normalize(u, y) for u, y in test_streams]

    x_train, y_train = _make_windows_multi(
        train_norm_streams,
        length=total_length,
        stride=args.train_stride,
        max_windows=args.train_size,
    )
    x_val, y_val = _make_windows_multi(
        val_norm_streams,
        length=total_length,
        stride=args.val_stride,
        max_windows=args.val_size,
    )
    x_test, y_test = _make_windows_multi(
        test_norm_streams,
        length=total_length,
        stride=args.test_stride,
        max_windows=args.test_size,
    )

    train_stream_u, train_stream_y, train_stream_offsets = _concat_streams(train_norm_streams)
    val_stream_u, val_stream_y, val_stream_offsets = _concat_streams(val_norm_streams)

    grid = np.arange(total_length, dtype=np.float32)
    tmp_data_path = out_dir / "data.tmp.npz"
    np.savez_compressed(
        tmp_data_path,
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        x_test=x_test,
        y_test=y_test,
        train_stream_u=train_stream_u,
        train_stream_y=train_stream_y,
        train_stream_offsets=train_stream_offsets,
        val_stream_u=val_stream_u,
        val_stream_y=val_stream_y,
        val_stream_offsets=val_stream_offsets,
        grid=grid,
    )
    tmp_data_path.replace(out_dir / "data.npz")

    meta = {
        "suite": args.suite,
        "protocol": args.protocol,
        "source": source_url,
        "loader_package": "nonlinear-benchmarks",
        "sequence_length": total_length,
        "scored_sequence_length": scored_length,
        "train_size": int(x_train.shape[0]),
        "val_size": int(x_val.shape[0]),
        "test_size": int(x_test.shape[0]),
        "sampling_time": sampling_time,
        "test_state_initialization_window_length": init_window,
        "metric_ignore_prefix_steps": metric_ignore_prefix_steps,
        "sequence_policy": sequence_policy,
        "train_sequence_count": len(train_streams),
        "test_sequence_count": len(test_streams),
        "normalization": {
            "u_mean": u_mean,
            "u_std": u_std,
            "y_mean": y_mean,
            "y_std": y_std,
        },
        "split_policy": "official_train_for_train_val; official_test_multisine_for_test; 80/20 sequential split on train stream",
        "output_dir": str(out_dir),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    ready_file.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Saved external benchmark to: {out_dir}")


if __name__ == "__main__":
    main()
