# File: src/moocradar/evaluation.py
# Purpose: Compute honest validation and test metrics from saved predictions.

import csv
import math
from collections import defaultdict
from pathlib import Path

from sklearn.metrics import accuracy_score, brier_score_loss, f1_score, roc_auc_score


def evaluate(prediction_path: Path, metrics_path: Path) -> list[dict]:
    """Calculate metrics per model and split; AUC is blank for one-class data."""
    groups = defaultdict(lambda: ([], []))
    with prediction_path.open(encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            if row["split"] not in ("validation", "test"):
                continue
            labels, probabilities = groups[(row["model"], row["split"])]
            labels.append(int(row["is_correct"]))
            probabilities.append(float(row["probability_correct"]))
    metrics = []
    for (model, split), (labels, probabilities) in sorted(groups.items()):
        predicted = [int(p >= 0.5) for p in probabilities]
        metrics.append({
            "model": model, "split": split, "n": len(labels),
            "roc_auc": round(roc_auc_score(labels, probabilities), 6) if len(set(labels)) == 2 else "",
            "accuracy": round(accuracy_score(labels, predicted), 6),
            "f1": round(f1_score(labels, predicted, zero_division=0), 6),
            "brier": round(brier_score_loss(labels, probabilities), 6),
            "tp": sum(y == 1 and p == 1 for y, p in zip(labels, predicted)),
            "tn": sum(y == 0 and p == 0 for y, p in zip(labels, predicted)),
            "fp": sum(y == 0 and p == 1 for y, p in zip(labels, predicted)),
            "fn": sum(y == 1 and p == 0 for y, p in zip(labels, predicted)),
        })
    with metrics_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, ["model", "split", "n", "roc_auc", "accuracy", "f1", "brier", "tp", "tn", "fp", "fn"])
        writer.writeheader()
        writer.writerows(metrics)
    return metrics
