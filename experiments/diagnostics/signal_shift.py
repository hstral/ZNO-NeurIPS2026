"""Train/validation/test signal statistics of the scored public windows (EMPS, Tanks).

Usage::

    python -m experiments.diagnostics.signal_shift --data-root data/public \\
        --output-dir runs/diagnostics/shift
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


CASES = ("emps_public_v1", "cascaded_tanks_public_v1")


# "high_frequency_fraction": share of power at normalized frequency >= 0.25
# cycles/sample, i.e. the upper half of the band below the Nyquist frequency.
def statistics(value: np.ndarray) -> dict[str, float]:
    flattened = value.reshape(-1)
    difference = np.diff(value, axis=1)
    spectrum = np.fft.rfft(value, axis=1)
    power = np.square(np.abs(spectrum)).mean(axis=(0, 2))
    frequency = np.fft.rfftfreq(value.shape[1])
    total = max(float(power.sum()), 1e-12)
    return {
        "mean": float(flattened.mean()),
        "std": float(flattened.std()),
        "rms": float(np.sqrt(np.mean(np.square(flattened)))),
        "difference_rms": float(np.sqrt(np.mean(np.square(difference)))),
        "spectral_centroid": float((frequency * power).sum() / total),
        "high_frequency_fraction": float(
            power[frequency >= 0.25].sum() / total
        ),
        "minimum": float(flattened.min()),
        "maximum": float(flattened.max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    ratios = {}
    for case in CASES:
        case_dir = args.data_root / case
        meta = json.loads((case_dir / "meta.json").read_text(encoding="utf-8"))
        prefix = int(meta["metric_ignore_prefix_steps"])
        with np.load(case_dir / "data.npz") as data:
            for signal in ("x", "y"):
                split_stats = {}
                for split in ("train", "val", "test"):
                    value = np.asarray(
                        data[f"{signal}_{split}"][:, prefix:, :],
                        dtype=np.float64,
                    )
                    item = {
                        "case": case,
                        "signal": signal,
                        "split": split,
                        "windows": value.shape[0],
                        "scored_length": value.shape[1],
                        **statistics(value),
                    }
                    rows.append(item)
                    split_stats[split] = item
                ratios[f"{case}:{signal}"] = {
                    f"test_to_train_{metric}": (
                        split_stats["test"][metric]
                        / max(abs(split_stats["train"][metric]), 1e-12)
                    )
                    for metric in (
                        "std",
                        "rms",
                        "difference_rms",
                        "spectral_centroid",
                        "high_frequency_fraction",
                    )
                }
    with (args.output_dir / "split_signal_statistics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / "split_shift_ratios.json").write_text(
        json.dumps(ratios, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(ratios, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
