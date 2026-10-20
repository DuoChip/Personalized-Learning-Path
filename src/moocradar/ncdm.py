# File: src/moocradar/ncdm.py
# Purpose: Monotonic Neural Cognitive Diagnosis Model implementation.

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class NCDMConfig:
    """Serializable dimensions and architecture settings."""

    student_count: int
    problem_count: int
    concept_count: int
    hidden_dims: tuple[int, ...] = (256, 128)
    dropout: float = 0.5
    discrimination_scale: float = 10.0


class NonNegativeLinear(nn.Module):
    """Linear layer constrained to non-negative weights for monotonicity."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        nn.init.xavier_uniform_(self.weight)
        with torch.no_grad():
            self.weight.abs_()

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return F.linear(values, self.weight.clamp(min=0), self.bias)

    def clip_nonnegative(self) -> None:
        with torch.no_grad():
            self.weight.clamp_(min=0)


class NCDM(nn.Module):
    """NeuralCDM with Q-masked concept factors and monotonic prediction layers."""

    def __init__(self, config: NCDMConfig):
        super().__init__()
        if not config.hidden_dims:
            raise ValueError("hidden_dims must not be empty")
        self.config = config
        self.student_proficiency = nn.Embedding(config.student_count, config.concept_count)
        self.exercise_difficulty = nn.Embedding(config.problem_count, config.concept_count)
        self.exercise_discrimination = nn.Embedding(config.problem_count, 1)
        nn.init.xavier_normal_(self.student_proficiency.weight)
        nn.init.xavier_normal_(self.exercise_difficulty.weight)
        nn.init.xavier_normal_(self.exercise_discrimination.weight)

        layers: list[nn.Module] = []
        input_size = config.concept_count
        for hidden_size in config.hidden_dims:
            layers.extend([
                NonNegativeLinear(input_size, hidden_size),
                nn.Sigmoid(),
                nn.Dropout(config.dropout),
            ])
            input_size = hidden_size
        layers.append(NonNegativeLinear(input_size, 1))
        self.prediction = nn.Sequential(*layers)

    def forward(self, student_ids: torch.Tensor, problem_ids: torch.Tensor,
                q_matrix_rows: torch.Tensor) -> torch.Tensor:
        proficiency = torch.sigmoid(self.student_proficiency(student_ids))
        difficulty = torch.sigmoid(self.exercise_difficulty(problem_ids))
        discrimination = (
            torch.sigmoid(self.exercise_discrimination(problem_ids))
            * self.config.discrimination_scale
        )
        interaction = discrimination * (proficiency - difficulty) * q_matrix_rows
        return torch.sigmoid(self.prediction(interaction)).squeeze(-1)

    def mastery(self, student_ids: torch.Tensor) -> torch.Tensor:
        """Return concept-level student proficiency, not response probability."""
        return torch.sigmoid(self.student_proficiency(student_ids))

    def clip_nonnegative(self) -> None:
        """Project prediction-layer weights onto the monotonic feasible set."""
        for module in self.prediction.modules():
            if isinstance(module, NonNegativeLinear):
                module.clip_nonnegative()

    def export_config(self) -> dict:
        config = asdict(self.config)
        config["hidden_dims"] = list(self.config.hidden_dims)
        return config


def save_ncdm_checkpoint(path: Path, model: NCDM, *, epoch: int,
                         validation_metric: float, optimizer: torch.optim.Optimizer | None = None,
                         training_config: dict | None = None) -> None:
    """Save enough state for inference and optional optimizer resume."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state": model.state_dict(),
        "model_config": model.export_config(),
        "epoch": epoch,
        "validation_metric": validation_metric,
        "training_config": training_config or {},
    }
    if optimizer is not None:
        payload["optimizer_state"] = optimizer.state_dict()
    torch.save(payload, path)


def load_ncdm_checkpoint(path: Path, device: str | torch.device = "cpu") -> tuple[NCDM, dict]:
    """Restore an NCDM checkpoint without retraining."""
    target = torch.device(device)
    try:
        payload = torch.load(path, map_location=target, weights_only=False)
    except TypeError:  # PyTorch before weights_only was introduced.
        payload = torch.load(path, map_location=target)
    raw_config = dict(payload["model_config"])
    raw_config["hidden_dims"] = tuple(raw_config["hidden_dims"])
    model = NCDM(NCDMConfig(**raw_config))
    model.load_state_dict(payload["model_state"])
    model.to(target)
    model.eval()
    return model, payload
