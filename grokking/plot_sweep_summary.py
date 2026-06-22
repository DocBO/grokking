from __future__ import annotations

from argparse import ArgumentParser
import csv
from pathlib import Path

import matplotlib.pyplot as plt


COLORS = {
    "adamw": "#4063D8",
    "ensemble-simple": "#E46C2A",
}
LABELS = {
    "adamw": "AdamW",
    "ensemble-simple": "Ensemble-simple",
}


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument(
        "summary",
        type=Path,
        nargs="?",
        default=Path("boundary_sweep_runs/sweep_summary.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("boundary_sweep_runs/sweep_summary.png"),
    )
    parser.add_argument("--threshold", type=float, default=0.95)
    args = parser.parse_args()

    rows = read_rows(args.summary)
    optimizers = [
        name
        for name in ("adamw", "ensemble-simple")
        if any(row["optimizer"] == name for row in rows)
    ]

    plt.style.use("seaborn-v0_8-whitegrid")
    figure, (accuracy_axis, steps_axis) = plt.subplots(
        2,
        1,
        figsize=(9, 8),
        sharex=True,
        constrained_layout=True,
    )

    for optimizer in optimizers:
        optimizer_rows = sorted(
            (row for row in rows if row["optimizer"] == optimizer),
            key=lambda row: float(row["training_fraction"]),
        )
        fractions = [float(row["training_fraction"]) for row in optimizer_rows]
        accuracies = [float(row["best_val_acc"]) for row in optimizer_rows]
        color = COLORS[optimizer]
        label = LABELS[optimizer]

        accuracy_axis.plot(
            fractions,
            accuracies,
            marker="o",
            linewidth=2.2,
            markersize=6,
            color=color,
            label=label,
        )

        successful = [
            row for row in optimizer_rows if float(row["success_rate"]) > 0
        ]
        failed = [
            row for row in optimizer_rows if float(row["success_rate"]) == 0
        ]
        steps_axis.plot(
            [float(row["training_fraction"]) for row in successful],
            [float(row["mean_steps_to_transition"]) for row in successful],
            marker="o",
            linewidth=2.2,
            markersize=6,
            color=color,
            label=label,
        )
        if failed:
            steps_axis.scatter(
                [float(row["training_fraction"]) for row in failed],
                [float(row.get("max_steps") or 100_000) for row in failed],
                marker="x",
                s=65,
                linewidths=2,
                color=color,
                zorder=4,
            )

    accuracy_axis.axhline(
        args.threshold,
        color="#333333",
        linestyle="--",
        linewidth=1.4,
        label=f"Success threshold ({args.threshold:.2f})",
    )
    accuracy_axis.set_title("Optimization boundary sweep")
    accuracy_axis.set_ylabel("Best validation accuracy")
    accuracy_axis.set_ylim(0, 1.02)
    accuracy_axis.legend(frameon=True)

    steps_axis.text(
        0.995,
        0.96,
        "× = no transition by optimizer step limit",
        transform=steps_axis.transAxes,
        horizontalalignment="right",
        verticalalignment="top",
        color="#555555",
    )
    steps_axis.set_xlabel("Training fraction")
    steps_axis.set_ylabel("Steps to validation accuracy > 0.95")
    maximum_step_limit = max(
        float(row.get("max_steps") or 100_000) for row in rows
    )
    steps_axis.set_ylim(0, maximum_step_limit * 1.05)
    steps_axis.legend(frameon=True)

    fractions = sorted({float(row["training_fraction"]) for row in rows})
    steps_axis.set_xticks(fractions)
    steps_axis.set_xticklabels([f"{fraction:.2f}" for fraction in fractions])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=220)
    figure.savefig(args.output.with_suffix(".svg"))
    print(f"Wrote {args.output}")
    print(f"Wrote {args.output.with_suffix('.svg')}")


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as input_file:
        return list(csv.DictReader(input_file))


if __name__ == "__main__":
    main()
