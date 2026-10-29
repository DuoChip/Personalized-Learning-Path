# File: src/moocradar/ncdm_training.py
# Purpose: Train, validate, checkpoint, and run reproducible NCDM inference.

from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from sklearn.metrics import brier_score_loss, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .ncdm import NCDM, NCDMConfig, load_ncdm_checkpoint, save_ncdm_checkpoint
from .ncdm_data import NCDMDataBundle, NCDMInteractionDataset, make_collate_fn


@dataclass(frozen=True)
class NCDMTrainingConfig:
    """Training controls that are recorded in every run."""

    batch_size: int = 512
    learning_rate: float = 0.002
    epochs: int = 20
    patience: int = 4
    hidden_dims: tuple[int, ...] = (256, 128)
    dropout: float = 0.5
    num_workers: int = 0
    early_stopping_metric: str = "roc_auc"
    seed: int = 42


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch without requiring a GPU."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except AttributeError:
        pass


def resolve_device(requested: str = "auto") -> torch.device:
    """Select CUDA when available, otherwise use a supported local fallback."""
    if requested != "auto":
        device = torch.device(requested)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is unavailable")
        return device
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _loader(bundle: NCDMDataBundle, split: str, config: NCDMTrainingConfig,
            *, shuffle: bool) -> DataLoader:
    dataset = NCDMInteractionDataset(bundle.events_for(split))
    generator = torch.Generator().manual_seed(config.seed)
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        collate_fn=make_collate_fn(bundle.q_rows, len(bundle.concept_ids)),
        generator=generator,
        pin_memory=torch.cuda.is_available(),
    )


def predict_loader(model: NCDM, loader: DataLoader,
                   device: torch.device) -> tuple[list[int], list[float]]:
    """Predict in dataset order and return labels with probabilities."""
    model.eval()
    labels: list[int] = []
    probabilities: list[float] = []
    with torch.no_grad():
        for batch in loader:
            output = model(
                batch["student"].to(device),
                batch["problem"].to(device),
                batch["q"].to(device),
            )
            labels.extend(int(value) for value in batch["label"].tolist())
            probabilities.extend(float(value) for value in output.detach().cpu().tolist())
    return labels, probabilities


def _validation_scores(labels: list[int], probabilities: list[float]) -> dict[str, float]:
    return {
        "roc_auc": float(roc_auc_score(labels, probabilities)) if len(set(labels)) == 2 else float("nan"),
        "brier": float(brier_score_loss(labels, probabilities)),
    }


def train_ncdm_model(bundle: NCDMDataBundle, checkpoint_path: Path,
                     config: NCDMTrainingConfig, *, device_name: str = "auto",
                     resume_path: Path | None = None,
                     progress: bool = True,
                     log: Callable[[str], None] | None = None) -> tuple[NCDM, list[dict], torch.device]:
    """Fit NCDM with validation-based early stopping and save the best epoch."""
    if config.early_stopping_metric not in ("roc_auc", "brier"):
        raise ValueError("early_stopping_metric must be roc_auc or brier")
    if config.epochs < 1 or config.patience < 1 or config.batch_size < 1:
        raise ValueError("epochs, patience, and batch_size must be positive")
    seed_everything(config.seed)
    device = resolve_device(device_name)
    model_config = NCDMConfig(
        student_count=len(bundle.student_ids),
        problem_count=len(bundle.problem_ids),
        concept_count=len(bundle.concept_ids),
        hidden_dims=config.hidden_dims,
        dropout=config.dropout,
    )
    model = NCDM(model_config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    start_epoch = 1
    if resume_path is not None:
        resumed_model, payload = load_ncdm_checkpoint(resume_path, device)
        if resumed_model.config != model.config:
            raise ValueError("Resume checkpoint dimensions do not match current mappings")
        model.load_state_dict(resumed_model.state_dict())
        if "optimizer_state" in payload:
            optimizer.load_state_dict(payload["optimizer_state"])
        start_epoch = int(payload["epoch"]) + 1

    train_loader = _loader(bundle, "train", config, shuffle=True)
    validation_loader = _loader(bundle, "validation", config, shuffle=False)
    if not len(train_loader.dataset) or not len(validation_loader.dataset):
        raise ValueError("NCDM requires non-empty train and validation splits")
    criterion = nn.BCELoss()
    history: list[dict] = []
    best_value = float("-inf")
    epochs_without_improvement = 0

    for epoch in range(start_epoch, config.epochs + 1):
        model.train()
        total_loss = 0.0
        total_examples = 0
        batches = tqdm(train_loader, desc=f"NCDM epoch {epoch}/{config.epochs}",
                       leave=False, disable=not progress)
        for batch in batches:
            optimizer.zero_grad(set_to_none=True)
            labels = batch["label"].to(device)
            probabilities = model(
                batch["student"].to(device),
                batch["problem"].to(device),
                batch["q"].to(device),
            )
            loss = criterion(probabilities, labels)
            loss.backward()
            optimizer.step()
            model.clip_nonnegative()
            count = labels.numel()
            total_loss += float(loss.item()) * count
            total_examples += count
            batches.set_postfix(loss=f"{loss.item():.4f}")

        labels, probabilities = predict_loader(model, validation_loader, device)
        scores = _validation_scores(labels, probabilities)
        row = {
            "epoch": epoch,
            "train_loss": total_loss / total_examples,
            "validation_roc_auc": scores["roc_auc"],
            "validation_brier": scores["brier"],
        }
        history.append(row)
        message = (
            f"epoch={epoch} train_loss={row['train_loss']:.6f} "
            f"validation_auc={row['validation_roc_auc']:.6f} "
            f"validation_brier={row['validation_brier']:.6f}"
        )
        (log or print)(message)
        value = scores[config.early_stopping_metric]
        if config.early_stopping_metric == "roc_auc" and np.isnan(value):
            comparison_value = -scores["brier"]
        elif config.early_stopping_metric == "brier":
            comparison_value = -value
        else:
            comparison_value = value
        improved = comparison_value > best_value
        if improved:
            best_value = comparison_value
            epochs_without_improvement = 0
            save_ncdm_checkpoint(
                checkpoint_path,
                model,
                epoch=epoch,
                validation_metric=value,
                optimizer=optimizer,
                training_config=asdict(config),
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.patience:
                break

    best_model, _ = load_ncdm_checkpoint(checkpoint_path, device)
    return best_model, history, device


def predict_split(model: NCDM, bundle: NCDMDataBundle, split: str,
                  config: NCDMTrainingConfig, device: torch.device) -> tuple[list, list[float]]:
    """Return split events and aligned probabilities from a frozen checkpoint."""
    events = bundle.events_for(split)
    loader = _loader(bundle, split, config, shuffle=False)
    _, probabilities = predict_loader(model, loader, device)
    if len(events) != len(probabilities):
        raise RuntimeError("Prediction and event counts are misaligned")
    return events, probabilities
