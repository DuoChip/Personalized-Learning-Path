# File: src/moocradar/inference.py
# Purpose: Load saved parameters and update one learner profile & path without retraining (Section 4.4 & 4.6).

import csv
from collections import Counter, defaultdict
from pathlib import Path

import yaml

from .modeling import load_links, load_models, student_concept_mastery
from .recommendation import build_course_index, gaps_for_course, recommend


def infer(user_id: str, processed_dir: Path, run_dir: Path,
          new_problem_id: str | None = None, new_is_correct: int | None = None,
          target_course_id: str | None = None, top_n: int = 5,
          mastery_target: float | None = None) -> dict:
    """Replay a learner's history and optional new response using the selected smoothed_concept model."""
    if (new_problem_id is None) != (new_is_correct is None):
        raise ValueError("Provide both new_problem_id and new_is_correct")
    if new_is_correct is not None and new_is_correct not in (0, 1):
        raise ValueError("new_is_correct must be 0 or 1")
    baseline, _ = load_models(run_dir / "models")
    if mastery_target is None:
        config = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
        mastery_target = float(config["mastery_target"])
    problems, links, names = load_links(processed_dir)
    if new_problem_id and new_problem_id not in problems:
        raise ValueError("Unknown new_problem_id")
    if target_course_id and target_course_id not in {p["course_id"] for p in problems.values()}:
        raise ValueError("Unknown target_course_id")
    student_counts = defaultdict(lambda: [0, 0])
    student_overall = [0, 0]
    observations = Counter()
    bloom = defaultdict(lambda: [0, 0])
    seen = set()
    course = target_course_id
    with (processed_dir / "interactions.csv").open(encoding="utf-8", newline="") as file:
        history = [row for row in csv.DictReader(file) if row["user_id"] == user_id]
    if not history and not new_problem_id:
        raise ValueError("User not present in processed interactions")
    if new_problem_id:
        history.append({"problem_id": new_problem_id, "is_correct": str(new_is_correct)})
    for row in history:
        problem_id = row["problem_id"]
        problem = problems.get(problem_id)
        if not problem or row["is_correct"] not in ("0", "1"):
            continue
        if not target_course_id:
            course = problem["course_id"]
        seen.add(problem_id)
        correct = int(row["is_correct"])
        student_overall[0] += correct
        student_overall[1] += 1
        for cid in links.get(problem_id, []):
            student_counts[cid][0] += correct
            student_counts[cid][1] += 1
            observations[cid] += 1
        level = problem["cognitive_dimension"]
        if level:
            bloom[level][0] += 1
            bloom[level][1] += correct
    states = {
        cid: student_concept_mastery(cid, baseline, student_counts, student_overall)
        for cid in observations
    }
    course_index = build_course_index(problems, links)
    gaps = gaps_for_course(user_id, course, states, observations,
                           course_index, mastery_target)[:20]
    course_problems = {pid: p for pid, p in problems.items() if p["course_id"] == course}
    paths = recommend(user_id, course, gaps, course_problems, links, seen, top_n)
    return {
        "user_id": user_id, "course_id": course,
        "selected_model": "smoothed_concept",
        "mastery": [
            {"concept_id": cid, "name_zh": names.get(cid, ""),
             "mastery": round(value, 6),
             "probability_correct": round(value, 6),
             "observations": observations[cid],
             "confidence": round(observations[cid] / (observations[cid] + 5), 6),
             "insufficient_evidence": observations[cid] < 3}
            for cid, value in sorted(states.items())
        ],
        "bloom_profile": [
            {"cognitive_dimension": level, "observations": values[0],
             "accuracy": round(values[1] / values[0], 6)}
            for level, values in sorted(bloom.items())
        ],
        "knowledge_gaps": gaps, "learning_path": paths,
    }
