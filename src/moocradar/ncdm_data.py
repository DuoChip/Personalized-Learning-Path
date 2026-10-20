# File: src/moocradar/ncdm_data.py
# Purpose: Build deterministic NCDM mappings, sparse Q-matrix rows, and data loaders.

from __future__ import annotations

import csv
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from .modeling import iter_students, load_links, split_name

SUPPORTED_GRANULARITIES = ("coarse", "middle", "fine")


@dataclass(frozen=True, slots=True)
class NCDMEvent:
    """One exercise-level response with stable integer IDs."""

    log_id: str
    user_id: str
    problem_id: str
    student_index: int
    problem_index: int
    label: int
    split: str
    sequence_position: int


@dataclass(slots=True)
class NCDMDataBundle:
    """All deterministic metadata required by NCDM training and inference."""

    events: list[NCDMEvent]
    student_ids: list[str]
    problem_ids: list[str]
    concept_ids: list[str]
    concept_names: dict[str, str]
    q_rows: list[tuple[int, ...]]
    granularity: str
    dropped_without_q: int = 0

    @property
    def student_to_index(self) -> dict[str, int]:
        return {value: index for index, value in enumerate(self.student_ids)}

    @property
    def problem_to_index(self) -> dict[str, int]:
        return {value: index for index, value in enumerate(self.problem_ids)}

    @property
    def concept_to_index(self) -> dict[str, int]:
        return {value: index for index, value in enumerate(self.concept_ids)}

    def events_for(self, split: str) -> list[NCDMEvent]:
        return [event for event in self.events if event.split == split]


class NCDMInteractionDataset(Dataset):
    """Minimal dataset; the Q-vector is materialized only for each mini-batch."""

    def __init__(self, events: Sequence[NCDMEvent]):
        self.events = list(events)

    def __len__(self) -> int:
        return len(self.events)

    def __getitem__(self, index: int) -> tuple[int, int, float, int]:
        event = self.events[index]
        return event.student_index, event.problem_index, float(event.label), index


def _links_for_granularity(processed_dir: Path, granularity: str) -> tuple[dict, dict[str, list[str]], dict[str, str]]:
    if granularity not in SUPPORTED_GRANULARITIES:
        raise ValueError(f"granularity must be one of {SUPPORTED_GRANULARITIES}")
    return load_links(processed_dir, granularity)


def load_ncdm_data(processed_dir: Path, granularity: str = "middle") -> NCDMDataBundle:
    """Read processed interactions and create mappings shared by every split."""
    _, links, names = _links_for_granularity(processed_dir, granularity)
    raw_events: list[tuple[str, str, str, int, str, int]] = []
    active_students: set[str] = set()
    active_problems: set[str] = set()
    dropped_without_q = 0
    for user_id, rows in iter_students(processed_dir / "interactions.csv"):
        for position, row in enumerate(rows):
            if row["is_correct"] not in ("0", "1"):
                continue
            problem_id = row["problem_id"]
            if not links.get(problem_id):
                dropped_without_q += 1
                continue
            active_students.add(user_id)
            active_problems.add(problem_id)
            raw_events.append((
                row["log_id"], user_id, problem_id, int(row["is_correct"]),
                split_name(position, len(rows)), position,
            ))
    if not raw_events:
        raise ValueError("No valid interactions with a Q-matrix row")

    student_ids = sorted(active_students)
    problem_ids = sorted(active_problems)
    active_concepts = {
        concept_id
        for problem_id in active_problems
        for concept_id in links.get(problem_id, [])
    }
    concept_ids = sorted(active_concepts)
    student_index = {value: index for index, value in enumerate(student_ids)}
    problem_index = {value: index for index, value in enumerate(problem_ids)}
    concept_index = {value: index for index, value in enumerate(concept_ids)}
    q_rows = [
        tuple(sorted({concept_index[cid] for cid in links[problem_id] if cid in concept_index}))
        for problem_id in problem_ids
    ]
    events = [
        NCDMEvent(
            log_id=log_id,
            user_id=user_id,
            problem_id=problem_id,
            student_index=student_index[user_id],
            problem_index=problem_index[problem_id],
            label=label,
            split=split,
            sequence_position=position,
        )
        for log_id, user_id, problem_id, label, split, position in raw_events
    ]
    return NCDMDataBundle(
        events=events,
        student_ids=student_ids,
        problem_ids=problem_ids,
        concept_ids=concept_ids,
        concept_names={cid: names.get(cid, cid) for cid in concept_ids},
        q_rows=q_rows,
        granularity=granularity,
        dropped_without_q=dropped_without_q,
    )


def make_collate_fn(q_rows: Sequence[Sequence[int]], concept_count: int) -> Callable:
    """Return a collator that creates only the current batch's dense Q rows."""
    def collate(batch: list[tuple[int, int, float, int]]) -> dict[str, torch.Tensor]:
        students, problems, labels, event_indices = zip(*batch)
        q = torch.zeros((len(batch), concept_count), dtype=torch.float32)
        for row_index, problem_index in enumerate(problems):
            concept_indices = q_rows[problem_index]
            if concept_indices:
                q[row_index, list(concept_indices)] = 1.0
        return {
            "student": torch.tensor(students, dtype=torch.long),
            "problem": torch.tensor(problems, dtype=torch.long),
            "q": q,
            "label": torch.tensor(labels, dtype=torch.float32),
            "event_index": torch.tensor(event_indices, dtype=torch.long),
        }

    return collate


def save_ncdm_metadata(bundle: NCDMDataBundle, model_dir: Path) -> None:
    """Persist stable ID mappings and the Q-matrix in sparse COO form."""
    model_dir.mkdir(parents=True, exist_ok=True)
    mappings = {
        "granularity": bundle.granularity,
        "student_ids": bundle.student_ids,
        "problem_ids": bundle.problem_ids,
        "concept_ids": bundle.concept_ids,
        "concept_names": bundle.concept_names,
    }
    (model_dir / "id_mappings.json").write_text(
        json.dumps(mappings, ensure_ascii=False), encoding="utf-8"
    )
    row_indices: list[int] = []
    column_indices: list[int] = []
    for row_index, concepts in enumerate(bundle.q_rows):
        row_indices.extend([row_index] * len(concepts))
        column_indices.extend(concepts)
    np.savez_compressed(
        model_dir / "q_matrix.npz",
        row=np.asarray(row_indices, dtype=np.int32),
        col=np.asarray(column_indices, dtype=np.int32),
        shape=np.asarray([len(bundle.problem_ids), len(bundle.concept_ids)], dtype=np.int32),
    )


def load_ncdm_metadata(model_dir: Path) -> tuple[dict, list[tuple[int, ...]]]:
    """Load persisted mappings and reconstruct sparse Q rows."""
    mappings = json.loads((model_dir / "id_mappings.json").read_text(encoding="utf-8"))
    matrix = np.load(model_dir / "q_matrix.npz")
    shape = tuple(int(value) for value in matrix["shape"])
    q_rows: list[list[int]] = [[] for _ in range(shape[0])]
    for row, column in zip(matrix["row"], matrix["col"]):
        q_rows[int(row)].append(int(column))
    return mappings, [tuple(values) for values in q_rows]


def training_observation_counts(bundle: NCDMDataBundle) -> Counter[tuple[int, int]]:
    """Count train evidence per student-concept without using future labels."""
    counts: Counter[tuple[int, int]] = Counter()
    for event in bundle.events:
        if event.split != "train":
            continue
        for concept_index in bundle.q_rows[event.problem_index]:
            counts[(event.student_index, concept_index)] += 1
    return counts


def read_split_indices(path: Path) -> dict[str, str]:
    """Read a persisted split contract for alignment checks."""
    with path.open(encoding="utf-8", newline="") as file:
        return {row["log_id"]: row["split"] for row in csv.DictReader(file)}
