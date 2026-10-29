# File: src/moocradar/ncdm_plot_cli.py
# Purpose: Render NCDM plots in a process isolated from PyTorch training backends.

from __future__ import annotations

import csv
import sys
from pathlib import Path

from .ncdm_plots import plot_prediction_diagnostics, plot_training_history


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python -m moocradar.ncdm_plot_cli RUN_DIR")
    run_dir = Path(sys.argv[1])
    history: list[dict] = []
    with (run_dir / "training_history.csv").open(encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            history.append({
                "epoch": int(row["epoch"]),
                "train_loss": float(row["train_loss"]),
                "validation_roc_auc": float(row["validation_roc_auc"]),
                "validation_brier": float(row["validation_brier"]),
            })
    plot_training_history(history, run_dir / "plots")
    plot_prediction_diagnostics(run_dir / "predictions.csv", run_dir / "plots")


if __name__ == "__main__":
    main()
