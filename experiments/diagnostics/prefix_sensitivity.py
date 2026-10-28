"""Prefix sensitivity and per-window error features from exported predictions.

The model always receives the full window; ``prefix`` only sets how many
initial steps are excluded from the relative L2 error. Usage::

    python -m experiments.diagnostics.prefix_sensitivity \\
        --prediction-root runs/diagnostics/predictions --output-dir runs/diagnostics/prefix
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr


CASE_PREFIXES = {
    "emps_public_v1": (0, 20, 50, 100, 200, 500, 1000),
    "cascaded_tanks_public_v1": (0, 50, 60, 80, 100, 120, 150),
}
OFFICIAL_PREFIX = {
    "emps_public_v1": 20,
    "cascaded_tanks_public_v1": 50,
}


def relative_l2_per_sequence(
    prediction: np.ndarray, target: np.ndarray, prefix: int
) -> np.ndarray:
    error = prediction[:, prefix:, :] - target[:, prefix:, :]
    numerator = np.linalg.norm(error.reshape(error.shape[0], -1), axis=1)
    denominator = np.linalg.norm(
        target[:, prefix:, :].reshape(target.shape[0], -1), axis=1
    )
    return numerator / np.maximum(denominator, 1e-8)


def signal_features(x: np.ndarray, y: np.ndarray, prefix: int) -> dict[str, np.ndarray]:
    x = x[:, prefix:, :]
    y = y[:, prefix:, :]
    spectrum = np.fft.rfft(x, axis=1)
    power = np.square(np.abs(spectrum)).mean(axis=2)
    frequency = np.fft.rfftfreq(x.shape[1])
    total = np.maximum(power.sum(axis=1), 1e-12)
    return {
        "input_rms": np.sqrt(np.mean(np.square(x), axis=(1, 2))),
        "input_difference_rms": np.sqrt(
            np.mean(np.square(np.diff(x, axis=1)), axis=(1, 2))
        ),
        "input_spectral_centroid": (power * frequency[None, :]).sum(axis=1) / total,
        "input_high_frequency_fraction": power[:, frequency >= 0.25].sum(axis=1)
        / total,
        "output_rms": np.sqrt(np.mean(np.square(y), axis=(1, 2))),
        "output_mean": np.mean(y, axis=(1, 2)),
        "output_range": np.ptp(y, axis=1).mean(axis=1),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(
        (args.prediction_root / "prediction_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for record in manifest:
        grouped[(record["case"], record["model"])].append(record)

    prefix_rows = []
    sequence_rows = []
    correlation_rows = []
    quartile_rows = []
    report = {}
    for (case, model), records in sorted(grouped.items()):
        arrays = [np.load(record["predictions"]) for record in records]
        official_prefix = OFFICIAL_PREFIX[case]
        for split in ("val", "test"):
            target = np.asarray(arrays[0][f"y_{split}"], dtype=np.float64)
            x = np.asarray(arrays[0][f"x_{split}"], dtype=np.float64)
            seed_errors = []
            for record, data in zip(records, arrays):
                prediction = np.asarray(data[f"y_{split}_pred"], dtype=np.float64)
                for prefix in CASE_PREFIXES[case]:
                    if prefix >= target.shape[1] - 1:
                        continue
                    errors = relative_l2_per_sequence(prediction, target, prefix)
                    prefix_rows.append(
                        {
                            "case": case,
                            "model": model,
                            "seed": record["seed"],
                            "split": split,
                            "prefix": prefix,
                            "mean_per_sequence_rel_l2": float(errors.mean()),
                            "median_per_sequence_rel_l2": float(np.median(errors)),
                        }
                    )
                seed_errors.append(
                    relative_l2_per_sequence(prediction, target, official_prefix)
                )
            mean_error = np.mean(seed_errors, axis=0)
            features = signal_features(x, target, official_prefix)
            for index, error in enumerate(mean_error):
                sequence_rows.append(
                    {
                        "case": case,
                        "model": model,
                        "split": split,
                        "sequence": index,
                        "mean_seed_rel_l2": float(error),
                        **{name: float(values[index]) for name, values in features.items()},
                    }
                )
            for feature_name, values in features.items():
                correlation, pvalue = spearmanr(values, mean_error)
                correlation_rows.append(
                    {
                        "case": case,
                        "model": model,
                        "split": split,
                        "feature": feature_name,
                        "spearman_r": float(correlation),
                        "pvalue": float(pvalue),
                    }
                )
                quantiles = np.quantile(values, [0.25, 0.5, 0.75])
                bins = np.digitize(values, quantiles)
                for bin_index in range(4):
                    mask = bins == bin_index
                    quartile_rows.append(
                        {
                            "case": case,
                            "model": model,
                            "split": split,
                            "feature": feature_name,
                            "quartile": bin_index + 1,
                            "count": int(mask.sum()),
                            "mean_rel_l2": float(mean_error[mask].mean())
                            if mask.any()
                            else None,
                        }
                    )

        val_official = [
            row["mean_per_sequence_rel_l2"]
            for row in prefix_rows
            if row["case"] == case
            and row["model"] == model
            and row["split"] == "val"
            and row["prefix"] == official_prefix
        ]
        test_official = [
            row["mean_per_sequence_rel_l2"]
            for row in prefix_rows
            if row["case"] == case
            and row["model"] == model
            and row["split"] == "test"
            and row["prefix"] == official_prefix
        ]
        test_longest = [
            row["mean_per_sequence_rel_l2"]
            for row in prefix_rows
            if row["case"] == case
            and row["model"] == model
            and row["split"] == "test"
            and row["prefix"] == max(
                prefix
                for prefix in CASE_PREFIXES[case]
                if prefix < arrays[0]["y_test"].shape[1] - 1
            )
        ]
        report[f"{case}:{model}"] = {
            "validation_mean_over_seeds": float(np.mean(val_official)),
            "test_mean_over_seeds": float(np.mean(test_official)),
            "validation_test_ratio": float(
                np.mean(test_official) / max(np.mean(val_official), 1e-12)
            ),
            "test_longest_prefix_mean_over_seeds": float(np.mean(test_longest)),
        }
        for data in arrays:
            data.close()

    write_csv(args.output_dir / "prefix_sensitivity.csv", prefix_rows)
    write_csv(args.output_dir / "sequence_features_and_errors.csv", sequence_rows)
    write_csv(args.output_dir / "feature_error_correlations.csv", correlation_rows)
    write_csv(args.output_dir / "feature_error_quartiles.csv", quartile_rows)
    (args.output_dir / "diagnostic_summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
