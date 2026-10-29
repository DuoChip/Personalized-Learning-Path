# File: src/moocradar/evaluation.py
# Purpose: Compute honest validation and test metrics from saved predictions.

import csv
from collections import defaultdict
from pathlib import Path

from sklearn.metrics import (accuracy_score, average_precision_score,
                             brier_score_loss,
                             confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)


METRIC_FIELDS = [
    "model", "split", "n", "threshold", "roc_auc", "pr_auc", "accuracy",
    "balanced_accuracy", "f1", "precision", "recall", "specificity",
    "brier", "prevalence", "positive_prediction_rate", "tp", "tn", "fp", "fn",
]


def metric_row(model: str, split: str, labels: list[int], probabilities: list[float],
               threshold: float = 0.5) -> dict:
    """Compute imbalance-aware binary metrics at a declared threshold."""
    predicted = [int(probability >= threshold) for probability in probabilities]
    tn, fp, fn, tp = confusion_matrix(labels, predicted, labels=[0, 1]).ravel()
    two_classes = len(set(labels)) == 2
    return {
        "model": model,
        "split": split,
        "n": len(labels),
        "threshold": threshold,
        "roc_auc": round(roc_auc_score(labels, probabilities), 6) if two_classes else "",
        "pr_auc": round(average_precision_score(labels, probabilities), 6) if two_classes else "",
        "accuracy": round(accuracy_score(labels, predicted), 6),
        "balanced_accuracy": round(
            ((tp / (tp + fn)) + (tn / (tn + fp))) / 2, 6
        ) if two_classes else "",
        "f1": round(f1_score(labels, predicted, zero_division=0), 6),
        "precision": round(precision_score(labels, predicted, zero_division=0), 6),
        "recall": round(recall_score(labels, predicted, zero_division=0), 6),
        "specificity": round(float(tn / (tn + fp)), 6) if tn + fp else "",
        "brier": round(brier_score_loss(labels, probabilities), 6),
        "prevalence": round(sum(labels) / len(labels), 6),
        "positive_prediction_rate": round(sum(predicted) / len(predicted), 6),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
    }


def evaluate(prediction_path: Path, metrics_path: Path,
             thresholds: dict[str, float] | None = None) -> list[dict]:
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
        threshold = (thresholds or {}).get(model, 0.5)
        metrics.append(metric_row(model, split, labels, probabilities, threshold))
    with metrics_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, METRIC_FIELDS)
        writer.writeheader()
        writer.writerows(metrics)
    return metrics
