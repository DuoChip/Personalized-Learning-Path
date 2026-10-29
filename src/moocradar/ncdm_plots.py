# File: src/moocradar/ncdm_plots.py
# Purpose: Plot NCDM training and fair multi-model evaluation diagnostics.

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def _roc_points(labels: list[int], probabilities: list[float]) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(-np.asarray(probabilities))
    truth = np.asarray(labels)[order]
    positives = max(1, int(truth.sum()))
    negatives = max(1, len(truth) - int(truth.sum()))
    return np.r_[0.0, np.cumsum(1 - truth) / negatives], np.r_[0.0, np.cumsum(truth) / positives]


def _pr_points(labels: list[int], probabilities: list[float]) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(-np.asarray(probabilities))
    truth = np.asarray(labels)[order]
    true_positive = np.cumsum(truth)
    precision = true_positive / np.arange(1, len(truth) + 1)
    recall = true_positive / max(1, int(truth.sum()))
    return np.r_[0.0, recall], np.r_[1.0, precision]


def plot_training_history(history: list[dict], output_dir: Path) -> None:
    """Save loss and validation-AUC curves from measured epochs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    epochs = [row["epoch"] for row in history]
    losses = [row["train_loss"] for row in history]
    aucs = [row["validation_roc_auc"] for row in history]
    for values, ylabel, filename in (
        (losses, "Binary cross-entropy", "training_loss.png"),
        (aucs, "Validation ROC-AUC", "validation_auc.png"),
    ):
        figure, axis = plt.subplots(figsize=(7, 4))
        axis.plot(epochs, values, marker="o")
        axis.set(xlabel="Epoch", ylabel=ylabel)
        axis.grid(alpha=0.25)
        figure.tight_layout()
        figure.savefig(output_dir / filename, dpi=180)
        plt.close(figure)


def _test_predictions(path: Path) -> dict[str, tuple[list[int], list[float]]]:
    groups = defaultdict(lambda: ([], []))
    with path.open(encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            if row["split"] != "test":
                continue
            labels, probabilities = groups[row["model"]]
            labels.append(int(row["is_correct"]))
            probabilities.append(float(row["probability_correct"]))
    return dict(groups)


def plot_prediction_diagnostics(prediction_path: Path, output_dir: Path,
                                threshold: float = 0.5) -> None:
    """Save ROC, PR, confusion, and probability plots for the test split."""
    output_dir.mkdir(parents=True, exist_ok=True)
    groups = _test_predictions(prediction_path)

    figure, axis = plt.subplots(figsize=(6, 5))
    for model, (labels, probabilities) in sorted(groups.items()):
        if len(set(labels)) < 2:
            continue
        false_positive, true_positive = _roc_points(labels, probabilities)
        axis.plot(false_positive, true_positive, label=model)
    axis.plot([0, 1], [0, 1], linestyle="--", color="grey")
    axis.set(xlabel="False positive rate", ylabel="True positive rate")
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output_dir / "roc_curve.png", dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(6, 5))
    for model, (labels, probabilities) in sorted(groups.items()):
        recall, precision = _pr_points(labels, probabilities)
        axis.plot(recall, precision, label=model)
    axis.set(xlabel="Recall", ylabel="Precision")
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output_dir / "precision_recall_curve.png", dpi=180)
    plt.close(figure)

    columns = 2
    rows = max(1, int(np.ceil(len(groups) / columns)))
    figure, axes = plt.subplots(rows, columns, figsize=(8, 3.5 * rows), squeeze=False)
    for axis, (model, (labels, probabilities)) in zip(axes.flat, sorted(groups.items())):
        predicted = [int(value >= threshold) for value in probabilities]
        matrix = np.asarray([
            [sum(y == 0 and p == 0 for y, p in zip(labels, predicted)),
             sum(y == 0 and p == 1 for y, p in zip(labels, predicted))],
            [sum(y == 1 and p == 0 for y, p in zip(labels, predicted)),
             sum(y == 1 and p == 1 for y, p in zip(labels, predicted))],
        ])
        axis.imshow(matrix, cmap="Blues")
        for row in range(2):
            for column in range(2):
                axis.text(column, row, str(matrix[row, column]), ha="center", va="center")
        axis.set_xticks([0, 1], labels=["incorrect", "correct"])
        axis.set_yticks([0, 1], labels=["incorrect", "correct"])
        axis.set(xlabel="Predicted", ylabel="Actual")
        axis.set_title(model)
    for axis in axes.flat[len(groups):]:
        axis.axis("off")
    figure.tight_layout()
    figure.savefig(output_dir / "confusion_matrix.png", dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8, 5))
    bins = np.linspace(0, 1, 31)
    for model, (_, probabilities) in sorted(groups.items()):
        axis.hist(probabilities, bins=bins, density=True, histtype="step", label=model)
    axis.set(xlabel="Predicted probability of correct response", ylabel="Density")
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output_dir / "probability_distribution.png", dpi=180)
    plt.close(figure)
