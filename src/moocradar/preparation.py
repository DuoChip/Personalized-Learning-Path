# File: src/moocradar/preparation.py
# Purpose: Validate and normalize problem metadata and student interactions.

import csv
import hashlib
import json
import logging
import statistics
from collections import Counter
from datetime import datetime
from pathlib import Path

from .reading import iter_json_array, iter_jsonl

LOG = logging.getLogger(__name__)
PROBLEM_FIELDS = ["problem_id", "exercise_id", "course_id", "cognitive_dimension", "knowledge_type", "title", "question_type", "max_score"]
INTERACTION_FIELDS = ["log_id", "user_id", "problem_id", "is_correct", "attempts", "score", "submit_time", "sequence_position"]


def concept_id(name: str) -> str:
    """Give identical Chinese concept names one stable ID."""
    return "K_" + hashlib.sha256(name.encode("utf-8")).hexdigest()[:20]


def load_problem_index(path: Path) -> dict[str, dict]:
    """Read the small problem metadata file and validate its join key."""
    problems = {}
    for item in iter_jsonl(path):
        problem_id = item.get("problem_id")
        if not problem_id or problem_id in problems:
            raise ValueError("Missing or duplicate problem_id in problem.json")
        problems[problem_id] = item
    return problems


def _detail(item: dict) -> dict:
    """Parse Python-literal metadata safely; malformed detail stays empty."""
    import ast

    try:
        detail = ast.literal_eval(item.get("detail") or "{}")
        return detail if isinstance(detail, dict) else {}
    except (SyntaxError, ValueError, TypeError, MemoryError):
        return {}


def _timestamp(value: str | None) -> bool:
    if not value:
        return False
    try:
        datetime.fromisoformat(value)
        return True
    except ValueError:
        return False


def prepare(data_root: Path, output_dir: Path, max_students: int | None = None) -> dict:
    """Stream normalized CSVs and a quality report; leave raw files untouched."""
    output_dir.mkdir(parents=True, exist_ok=True)
    problems = load_problem_index(data_root / "problem.json")
    quality = Counter()
    concepts = {}
    courses = set()
    exercises = set()
    concept_links = 0
    with (output_dir / "problems.csv").open("w", newline="", encoding="utf-8") as pf, \
         (output_dir / "problem_concepts.csv").open("w", newline="", encoding="utf-8") as cf:
        pw = csv.DictWriter(pf, PROBLEM_FIELDS)
        cw = csv.DictWriter(cf, ["problem_id", "concept_id"])
        pw.writeheader()
        cw.writeheader()
        for problem_id, item in problems.items():
            courses.add(item.get("course_id"))
            exercises.add(item.get("exercise_id"))
            if not item.get("course_id"):
                quality["missing_problem_course_id"] += 1
            if not item.get("exercise_id"):
                quality["missing_problem_exercise_id"] += 1
            detail = _detail(item)
            if not detail:
                quality["malformed_detail"] += 1
            raw_concepts = item.get("concepts") or []
            if not isinstance(raw_concepts, list):
                raw_concepts = []
                quality["invalid_concept_list"] += 1
            if not raw_concepts:
                quality["problems_without_concepts"] += 1
            if item.get("cognitive_dimension") not in (1, 2, 3, 4, 5, 6):
                quality["invalid_cognitive_dimension"] += 1
            for name in dict.fromkeys(raw_concepts):
                if isinstance(name, str) and name.strip():
                    cid = concept_id(name)
                    concepts[cid] = name
                    cw.writerow({"problem_id": problem_id, "concept_id": cid})
                    concept_links += 1
            pw.writerow({
                "problem_id": problem_id,
                "exercise_id": item.get("exercise_id"),
                "course_id": item.get("course_id"),
                "cognitive_dimension": item.get("cognitive_dimension"),
                "knowledge_type": item.get("knowledge_type"),
                "title": detail.get("title"),
                "question_type": detail.get("typetext"),
                "max_score": detail.get("score"),
            })
            quality["problems"] += 1
            quality[f"bloom_{item.get('cognitive_dimension')}"] += 1
    with (output_dir / "concepts.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, ["concept_id", "name_zh"])
        writer.writeheader()
        for cid, name in sorted(concepts.items()):
            writer.writerow({"concept_id": cid, "name_zh": name})
    quality["concepts"] = len(concepts)
    quality["problem_concept_links"] = concept_links
    quality["courses"] = len(courses)
    quality["exercises"] = len(exercises)

    log_ids = set()
    student_ids = set()
    used_problem_ids = set()
    student_lengths = []
    with (output_dir / "interactions.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, INTERACTION_FIELDS)
        writer.writeheader()
        for student_index, group in enumerate(iter_json_array(data_root / "student-problem-fine.json")):
            if max_students is not None and student_index >= max_students:
                break
            sequence = group.get("seq")
            if not isinstance(sequence, list):
                quality["invalid_student_sequences"] += 1
                continue
            quality["student_sequences"] += 1
            student_lengths.append(len(sequence))
            # Sequence order is preserved for identical or absent timestamps.
            ordered = sorted(
                enumerate(sequence),
                key=lambda pair: (pair[1].get("submit_time") or "", pair[0]),
            )
            for position, (_, row) in enumerate(ordered):
                quality["interactions"] += 1
                student_ids.add(row.get("user_id"))
                used_problem_ids.add(row.get("problem_id"))
                for field in ("log_id", "user_id", "problem_id", "is_correct", "submit_time"):
                    if row.get(field) is None:
                        quality[f"missing_{field}"] += 1
                for field in ("score", "attempts"):
                    if row.get(field) is None:
                        quality[f"missing_{field}"] += 1
                if row.get("problem_id") not in problems:
                    quality["unknown_problem_id"] += 1
                if row.get("is_correct") not in (0, 1):
                    quality["invalid_is_correct"] += 1
                else:
                    quality[f"label_{row['is_correct']}"] += 1
                attempts = row.get("attempts")
                if attempts is not None and (not isinstance(attempts, (int, float)) or attempts < 0):
                    quality["invalid_attempts"] += 1
                score = row.get("score")
                if score is not None and (not isinstance(score, (int, float)) or score < 0):
                    quality["invalid_score"] += 1
                if not _timestamp(row.get("submit_time")):
                    quality["invalid_submit_time"] += 1
                log_id = row.get("log_id")
                if log_id in log_ids:
                    quality["duplicate_log_id"] += 1
                log_ids.add(log_id)
                writer.writerow({field: row.get(field, "") if field != "sequence_position" else position for field in INTERACTION_FIELDS})
            if quality["student_sequences"] % 1000 == 0:
                LOG.info("Prepared %s student sequences", quality["student_sequences"])
    quality["distinct_log_ids"] = len(log_ids)
    quality["distinct_students"] = len(student_ids)
    quality["distinct_interacted_problems"] = len(used_problem_ids)
    quality["student_sequence_length"] = {
        "min": min(student_lengths, default=0),
        "median": statistics.median(student_lengths) if student_lengths else 0,
        "max": max(student_lengths, default=0),
    }
    valid_labels = quality["label_0"] + quality["label_1"]
    quality["correct_rate"] = round(quality["label_1"] / valid_labels, 6) if valid_labels else None
    quality["mode"] = "sample" if max_students is not None else "full"
    quality["max_students"] = max_students
    for field in ("log_id", "user_id", "problem_id", "is_correct", "submit_time",
                  "score", "attempts"):
        quality.setdefault(f"missing_{field}", 0)
    for field in ("unknown_problem_id", "invalid_is_correct", "invalid_submit_time",
                  "duplicate_log_id", "invalid_attempts", "invalid_score",
                  "missing_problem_course_id", "missing_problem_exercise_id",
                  "label_0", "label_1"):
        quality.setdefault(field, 0)
    quality["source_files"] = {
        name: {"bytes": (data_root / name).stat().st_size,
               "sha256": _sha256(data_root / name)}
        for name in ("problem.json", "student-problem-fine.json")
    }
    report = dict(quality)
    (output_dir / "data_quality.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()
