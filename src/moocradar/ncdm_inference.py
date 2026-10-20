# File: src/moocradar/ncdm_inference.py
# Purpose: Load a saved NCDM checkpoint for learner profiles and prediction.

from __future__ import annotations

import csv
from pathlib import Path

import torch
import yaml

from .modeling import load_links
from .ncdm import load_ncdm_checkpoint
from .ncdm_data import load_ncdm_metadata
from .recommendation import build_course_index, gaps_for_course, recommend


def _observations(run_dir: Path, user_id: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    with (run_dir / "student_mastery.csv").open(encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            if row["user_id"] == user_id:
                counts[row["concept_id"]] = int(row.get("observation_count") or row["observations"])
    return counts


def _history(processed_dir: Path, user_id: str) -> list[dict]:
    with (processed_dir / "interactions.csv").open(encoding="utf-8", newline="") as file:
        return [row for row in csv.DictReader(file) if row["user_id"] == user_id]


def load_ncdm_for_inference(run_dir: Path, device: str = "cpu") -> tuple:
    """Load model, mappings, and sparse Q rows from one completed run."""
    model, payload = load_ncdm_checkpoint(run_dir / "models" / "best_ncdm.pt", device)
    mappings, q_rows = load_ncdm_metadata(run_dir / "models")
    return model, mappings, q_rows, payload


def predict_ncdm_response(run_dir: Path, user_id: str, problem_id: str,
                          device: str = "cpu") -> float:
    """Predict one known student–problem response from a frozen checkpoint."""
    model, mappings, q_rows, _ = load_ncdm_for_inference(run_dir, device)
    student_index = {value: index for index, value in enumerate(mappings["student_ids"])}
    problem_index = {value: index for index, value in enumerate(mappings["problem_ids"])}
    if user_id not in student_index:
        raise ValueError("NCDM cannot infer an unseen student without retraining")
    if problem_id not in problem_index:
        raise ValueError("NCDM cannot infer an unseen problem without retraining")
    p_index = problem_index[problem_id]
    q = torch.zeros((1, len(mappings["concept_ids"])), dtype=torch.float32, device=device)
    q[0, list(q_rows[p_index])] = 1.0
    with torch.no_grad():
        probability = model(
            torch.tensor([student_index[user_id]], dtype=torch.long, device=device),
            torch.tensor([p_index], dtype=torch.long, device=device),
            q,
        )
    return float(probability.item())


def infer_ncdm(user_id: str, processed_dir: Path, run_dir: Path,
               target_course_id: str | None = None, top_n: int = 5,
               mastery_target: float | None = None, device: str = "cpu") -> dict:
    """Build a learner profile and path from a frozen NCDM checkpoint."""
    model, mappings, _, _ = load_ncdm_for_inference(run_dir, device)
    student_to_index = {value: index for index, value in enumerate(mappings["student_ids"])}
    if user_id not in student_to_index:
        raise ValueError("NCDM cannot infer an unseen student without periodic retraining")
    config = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    if mastery_target is None:
        mastery_target = float(config["mastery_target"])
    granularity = mappings["granularity"]
    problems, links, names = load_links(processed_dir, granularity)
    concept_ids = mappings["concept_ids"]
    allowed = set(concept_ids)
    links = {
        problem_id: [cid for cid in values if cid in allowed]
        for problem_id, values in links.items()
    }
    history = _history(processed_dir, user_id)
    if not history:
        raise ValueError("User not present in processed interactions")
    if target_course_id is None:
        for row in reversed(history):
            if row["problem_id"] in problems:
                target_course_id = problems[row["problem_id"]]["course_id"]
                break
    if target_course_id is None:
        raise ValueError("No target course can be resolved for this student")
    course_index = build_course_index(problems, links)
    if target_course_id not in course_index:
        raise ValueError("Target course has no concepts in the saved NCDM vocabulary")
    index_by_concept = {value: index for index, value in enumerate(concept_ids)}
    student_tensor = torch.tensor([student_to_index[user_id]], dtype=torch.long, device=device)
    with torch.no_grad():
        mastery_vector = model.mastery(student_tensor).squeeze(0).detach().cpu().tolist()
    observation_counts = _observations(run_dir, user_id)
    relevant = sorted(
        set(observation_counts) | set(course_index[target_course_id]),
        key=lambda value: index_by_concept.get(value, len(concept_ids)),
    )
    mastery = {cid: float(mastery_vector[index_by_concept[cid]]) for cid in relevant}
    gaps = gaps_for_course(
        user_id, target_course_id, mastery, observation_counts,
        course_index, mastery_target,
    )[:20]
    course_problems = {
        problem_id: problem
        for problem_id, problem in problems.items()
        if problem["course_id"] == target_course_id
    }
    seen = {row["problem_id"] for row in history}
    paths = recommend(user_id, target_course_id, gaps, course_problems, links, seen, top_n)
    return {
        "user_id": user_id,
        "course_id": target_course_id,
        "model_name": "ncdm",
        "adaptive_update": False,
        "mastery": [
            {
                "concept_id": cid,
                "name_zh": names.get(cid, cid),
                "mastery": round(mastery[cid], 6),
                "mastery_probability": round(mastery[cid], 6),
                "observations": observation_counts.get(cid, 0),
                "insufficient_evidence": observation_counts.get(cid, 0) < 3,
                "evidence_status": "observed_model_estimate" if observation_counts.get(cid, 0)
                else "unobserved_model_estimate",
            }
            for cid in relevant
        ],
        "knowledge_gaps": gaps,
        "learning_path": paths,
    }
