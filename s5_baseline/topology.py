"""Choose the S5 base state size at the original operating point by parameter count."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from .model import S5RegressionConfig, count_real_parameters, initialize_model


# Nominal targets of the original operating point (historical element counts of
# the reference topologies). The S5 width/depth is fixed at 20/4 and only the
# base state size is varied; selection uses parameter counts only.
TASKS = {
    "resonant_arma": {"input_channels": 7, "output_channels": 1, "target": 8589},
    "iir_cascade": {"input_channels": 16, "output_channels": 1, "target": 8757},
    "nonlinear_narx": {"input_channels": 1, "output_channels": 1, "target": 8537},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select S5 topology by parameter count only.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--state-sizes",
        default=",".join(str(value) for value in range(8, 81, 4)),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    state_sizes = [int(value) for value in args.state_sizes.split(",")]
    rows: list[dict[str, object]] = []
    for candidate_order, state_size in enumerate(state_sizes):
        config = S5RegressionConfig(ssm_size_base=state_size)
        reference_input_channels = 1
        _, params = initialize_model(
            config,
            input_channels=reference_input_channels,
            output_channels=1,
            sequence_length=8,
            batch_size=1,
            seed=0,
        )
        reference_count = count_real_parameters(params)
        counts: dict[str, int] = {}
        deviations: dict[str, float] = {}
        for task, metadata in TASKS.items():
            if int(metadata["output_channels"]) != 1:
                raise ValueError("The exact input-lift adjustment assumes one output")
            input_channels = int(metadata["input_channels"])
            count = reference_count + (
                input_channels - reference_input_channels
            ) * config.d_model
            target = int(metadata["target"])
            counts[task] = count
            deviations[task] = abs(count - target) / target
        rows.append(
            {
                "candidate_order": candidate_order,
                "ssm_size_base": state_size,
                **{f"{task}_parameters": value for task, value in counts.items()},
                **{f"{task}_relative_deviation": value for task, value in deviations.items()},
                "max_relative_deviation": max(deviations.values()),
                "mean_relative_deviation": sum(deviations.values()) / len(deviations),
                "mean_parameter_count": sum(counts.values()) / len(counts),
            }
        )

    eligible = [row for row in rows if float(row["max_relative_deviation"]) <= 0.05]
    if not eligible:
        raise SystemExit("No shared S5 topology satisfies the predeclared 5% bound")
    selected = min(
        eligible,
        key=lambda row: (
            float(row["max_relative_deviation"]),
            float(row["mean_relative_deviation"]),
            float(row["mean_parameter_count"]),
            int(row["candidate_order"]),
        ),
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_csv = args.output_dir / "matched_topology_candidates.csv"
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / "matched_topology_selected.json").write_text(
        json.dumps(selected, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(selected, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
