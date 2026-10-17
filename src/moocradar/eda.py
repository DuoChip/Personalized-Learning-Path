# File: src/moocradar/eda.py
# Purpose: Audit raw MOOC-Radar data and export reproducible EDA reports.

import csv
import hashlib
import json
import logging
import math
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from .preparation import _detail, concept_id
from .reading import iter_json_array, iter_jsonl

LOG = logging.getLogger(__name__)
EVENT_FIELDS = (
    "log_id", "user_id", "problem_id", "is_correct",
    "attempts", "score", "submit_time",
)
PROBLEM_FIELDS = (
    "problem_id", "exercise_id", "course_id", "detail",
    "knowledge_type", "cognitive_dimension", "concepts",
)


def _missing(value: object) -> bool:
    return value is None or value == ""


def _schema_add(schema: dict[str, set[str]], row: dict) -> None:
    for field, value in row.items():
        schema[field].add(type(value).__name__)


def _schema_rows(schema: dict[str, set[str]], dataset: str) -> list[dict]:
    return [
        {"dataset": dataset, "field": field, "types": ", ".join(sorted(types))}
        for field, types in sorted(schema.items())
    ]


def inspect_files(data_root: Path) -> list[dict]:
    """Inspect four expected files without reading their contents into memory."""
    result = []
    for name in (
        "problem.json", "student-problem-fine.json",
        "student-problem-middle.json", "student-problem-coarse.json",
    ):
        path = data_root / name
        if not path.is_file():
            result.append({"file": name, "exists": False, "bytes": None,
                           "mib": None, "format": None})
            continue
        result.append({
            "file": name, "exists": True, "bytes": path.stat().st_size,
            "mib": round(path.stat().st_size / 2**20, 2),
            "format": "JSON Lines" if name == "problem.json"
                      else "JSON array of student sequences",
        })
    return result


def inspect_schema(data_root: Path) -> list[dict]:
    """Read one object from each large file and return field/type names only."""
    rows = []
    for name in ("student-problem-fine.json", "student-problem-middle.json",
                 "student-problem-coarse.json"):
        path = data_root / name
        if not path.exists():
            continue
        first = next(iter_json_array(path), {})
        event = (first.get("seq") or [{}])[0] if first else {}
        rows.extend(_schema_rows(
            {field: {type(value).__name__} for field, value in event.items()},
            name,
        ))
    return rows


def _problem_metadata(path: Path) -> tuple[dict, dict]:
    """Read small problem metadata and keep only fields needed for joins."""
    problems = {}
    concept_names = {}
    problem_schema = defaultdict(set)
    missing = Counter()
    counts = Counter()
    bloom_problem = Counter()
    root_video_fields = set()
    detail_video_fields = set()
    courses = set()
    exercises = set()
    for item in iter_jsonl(path):
        counts["problem_records"] += 1
        _schema_add(problem_schema, item)
        for field in PROBLEM_FIELDS:
            missing[field] += _missing(item.get(field))
        pid = item.get("problem_id")
        if not pid or pid in problems:
            counts["duplicate_or_missing_problem_id"] += 1
            continue
        if item.get("course_id"):
            courses.add(item["course_id"])
        if item.get("exercise_id"):
            exercises.add(item["exercise_id"])
        names = item.get("concepts")
        if not isinstance(names, list):
            names = []
            counts["invalid_concept_list"] += 1
        valid_names = [
            name for name in names if isinstance(name, str) and name.strip()
        ]
        unique_names = list(dict.fromkeys(valid_names))
        counts["invalid_concept_entries"] += len(names) - len(valid_names)
        counts["duplicate_problem_concept_links"] += (
            len(valid_names) - len(unique_names)
        )
        cids = [concept_id(name) for name in unique_names]
        for cid, name in zip(cids, unique_names):
            concept_names[cid] = name
        counts["problem_concept_links"] += len(cids)
        counts["multi_concept_problems"] += len(cids) > 1
        counts["problems_without_concepts"] += len(cids) == 0
        level = item.get("cognitive_dimension")
        bloom_problem[str(level)] += 1
        counts["problems_with_valid_bloom"] += level in (1, 2, 3, 4, 5, 6)
        detail = _detail(item)
        if not detail:
            counts["empty_or_unparsed_detail"] += 1
        root_video_fields.update(key for key in item if "video" in key.lower())
        detail_video_fields.update(key for key in detail if "video" in key.lower())
        video_values = [
            value for key, value in {**item, **detail}.items()
            if "video" in key.lower()
        ]
        has_video = any(value not in (None, "", [], {}) for value in video_values)
        counts["problems_with_video_link"] += has_video
        problems[pid] = {
            "course_id": item.get("course_id"),
            "concept_ids": cids,
            "bloom": level,
            "video": has_video,
            "max_score": detail.get("score"),
        }
    return problems, {
        "concept_names": concept_names,
        "schema": _schema_rows(problem_schema, "problem.json"),
        "missing": dict(missing),
        "counts": dict(counts),
        "bloom_problem": dict(bloom_problem),
        "courses": len(courses),
        "exercises": len(exercises),
        "video_fields": sorted(root_video_fields | detail_video_fields),
    }


def _valid_timestamp(value: object) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _audit_events(path: Path, problems: dict, max_students: int | None) -> dict:
    """Stream original sequence order and aggregate quality/EDA statistics."""
    counts = Counter()
    missing = Counter()
    event_schema = defaultdict(set)
    seen_log_ids = set()
    seen_fingerprints = set()
    users = set()
    interacted_problems = set()
    student_events = Counter()
    lengths = []
    labels = Counter()
    bloom_events = Counter()
    bloom_correct = Counter()
    concept_events = Counter()
    concept_correct = Counter()
    course_events = Counter()
    monthly_events = Counter()
    for index, group in enumerate(iter_json_array(path)):
        if max_students is not None and index >= max_students:
            break
        sequence = group.get("seq")
        if not isinstance(sequence, list):
            counts["invalid_sequences"] += 1
            continue
        counts["student_sequences"] += 1
        lengths.append(len(sequence))
        previous_time = None
        sequence_out_of_order = False
        for row in sequence:
            counts["interactions"] += 1
            _schema_add(event_schema, row)
            for field in EVENT_FIELDS:
                missing[field] += _missing(row.get(field))
            uid = row.get("user_id")
            pid = row.get("problem_id")
            if uid:
                users.add(uid)
                student_events[uid] += 1
            if pid:
                interacted_problems.add(pid)
            log_id = row.get("log_id")
            if log_id is not None:
                counts["duplicate_log_id"] += log_id in seen_log_ids
                seen_log_ids.add(log_id)
            fingerprint = hashlib.blake2b(
                json.dumps(row, sort_keys=True, ensure_ascii=False).encode("utf-8"),
                digest_size=12,
            ).digest()
            counts["duplicate_exact_record"] += fingerprint in seen_fingerprints
            seen_fingerprints.add(fingerprint)
            label = row.get("is_correct")
            if label in (0, 1):
                labels[str(label)] += 1
            else:
                counts["invalid_is_correct"] += 1
            attempts = row.get("attempts")
            if attempts is not None and (
                not isinstance(attempts, int) or attempts < 0
            ):
                counts["invalid_attempts"] += 1
            score = row.get("score")
            if score is not None and (
                not isinstance(score, (int, float)) or not math.isfinite(score)
                or score < 0
            ):
                counts["invalid_score"] += 1
            timestamp = _valid_timestamp(row.get("submit_time"))
            if timestamp is None:
                counts["invalid_submit_time"] += 1
            else:
                monthly_events[timestamp.strftime("%Y-%m")] += 1
                if previous_time is not None:
                    try:
                        if timestamp < previous_time:
                            counts["out_of_order_pairs"] += 1
                            sequence_out_of_order = True
                    except TypeError:
                        counts["incomparable_timestamp_pairs"] += 1
                previous_time = timestamp
            problem = problems.get(pid)
            if problem is None:
                counts["unmatched_problem_events"] += 1
                continue
            if not problem["concept_ids"]:
                counts["events_without_concepts"] += 1
            if problem["video"]:
                counts["events_with_video_link"] += 1
            if problem["course_id"]:
                course_events[problem["course_id"]] += 1
            else:
                counts["events_without_course"] += 1
            level = problem["bloom"]
            if level in (1, 2, 3, 4, 5, 6):
                bloom_events[str(level)] += 1
                if label in (0, 1):
                    bloom_correct[str(level)] += label
            else:
                counts["events_without_valid_bloom"] += 1
            max_score = problem["max_score"]
            if isinstance(score, (int, float)) and isinstance(max_score, (int, float)):
                counts["score_above_problem_max"] += score > max_score
            for cid in problem["concept_ids"]:
                concept_events[cid] += 1
                if label in (0, 1):
                    concept_correct[cid] += label
        counts["out_of_order_sequences"] += sequence_out_of_order
        if counts["student_sequences"] % 1000 == 0:
            LOG.info("Đã khảo sát %s chuỗi sinh viên", counts["student_sequences"])
    return {
        "counts": dict(counts),
        "missing": dict(missing),
        "schema": _schema_rows(event_schema, "student-problem-fine.json"),
        "users": len(users),
        "interacted_problems": len(interacted_problems),
        "lengths": lengths,
        "student_interactions": list(student_events.values()),
        "labels": dict(labels),
        "bloom_events": dict(bloom_events),
        "bloom_correct": dict(bloom_correct),
        "concept_events": dict(concept_events),
        "concept_correct": dict(concept_correct),
        "course_events": dict(course_events),
        "monthly_events": dict(monthly_events),
    }


def _write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fields)
        writer.writeheader()
        writer.writerows(rows)


def _summarize(data_root: Path, max_students: int | None) -> tuple[dict, dict]:
    files = inspect_files(data_root)
    required = ("problem.json", "student-problem-fine.json")
    for name in required:
        if not (data_root / name).is_file():
            raise FileNotFoundError(data_root / name)
    problems, metadata = _problem_metadata(data_root / "problem.json")
    events = _audit_events(
        data_root / "student-problem-fine.json", problems, max_students,
    )
    total_events = events["counts"].get("interactions", 0)
    total_problems = metadata["counts"].get("problem_records", 0)
    counts = {
        **metadata["counts"],
        **events["counts"],
        "users": events["users"],
        "unique_problems": len(problems),
        "interacted_problems": events["interacted_problems"],
        "courses": metadata["courses"],
        "exercises": metadata["exercises"],
        "concepts": len(metadata["concept_names"]),
        "files_found": sum(item["exists"] for item in files),
    }
    for item in files:
        if item["file"] == "problem.json":
            item["records_scanned"] = counts["problem_records"]
        elif item["file"] == "student-problem-fine.json":
            item["records_scanned"] = counts.get("student_sequences", 0)
        else:
            item["records_scanned"] = None
    coverage = {
        "bloom_problem_pct": round(
            100 * counts.get("problems_with_valid_bloom", 0) / total_problems, 4
        ) if total_problems else None,
        "bloom_event_pct": round(
            100 * sum(events["bloom_events"].values()) / total_events, 4
        ) if total_events else None,
        "video_problem_pct": round(
            100 * counts.get("problems_with_video_link", 0) / total_problems, 4
        ) if total_problems else None,
        "video_event_pct": round(
            100 * counts.get("events_with_video_link", 0) / total_events, 4
        ) if total_events else None,
    }
    report = {
        "mode": "sample" if max_students is not None else "full",
        "max_students": max_students,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "files": files,
        "schema": metadata["schema"] + events["schema"] + [
            row for row in inspect_schema(data_root)
            if row["dataset"] != "student-problem-fine.json"
        ],
        "counts": counts,
        "coverage": coverage,
        "video_fields": metadata["video_fields"],
        "labels": events["labels"],
        "bloom_problem": metadata["bloom_problem"],
        "bloom_events": events["bloom_events"],
        "bloom_correct": events["bloom_correct"],
        "sequence_length": {
            "min": min(events["lengths"], default=0),
            "median": statistics.median(events["lengths"])
            if events["lengths"] else 0,
            "max": max(events["lengths"], default=0),
        },
        "notes": [
            "Mẫu chỉ áp dụng cho chuỗi sinh viên; problem.json luôn được đọc đầy đủ.",
            "Mỗi lần trả lời được tính một lần ở cấp câu hỏi; thống kê theo concept "
            "có thể tính cùng câu hỏi ở nhiều concept.",
            "Không có ánh xạ Bloom 1–6 sang tên mức độ trong file cục bộ.",
            "Không suy luận quan hệ video từ context_id khi thiếu liên kết tường minh.",
        ],
    }
    missing_rows = []
    for dataset, fields, values, total in (
        ("problems", PROBLEM_FIELDS, metadata["missing"], total_problems),
        ("interactions", EVENT_FIELDS, events["missing"], total_events),
    ):
        for field in fields:
            missing_count = values.get(field, 0)
            missing_rows.append({
                "dataset": dataset, "field": field,
                "missing_count": missing_count, "total_count": total,
                "missing_pct": round(100 * missing_count / total, 4)
                if total else 0,
            })
    return report, {
        "missing_rows": missing_rows,
        "concept_names": metadata["concept_names"],
        "events": events,
    }


def run_eda(data_root: Path, artifact_root: Path,
            max_students: int | None = 25) -> Path:
    """Run raw-data-only EDA and write a new run without touching modeling."""
    if max_students is not None and max_students < 1:
        raise ValueError("max_students must be positive or None")
    report, detail = _summarize(data_root, max_students)
    run_id = (
        "eda_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        + "_" + uuid4().hex[:6]
    )
    output = artifact_root / run_id
    output.mkdir(parents=True, exist_ok=False)
    from .eda_plots import make_eda_plots, write_eda_summary

    events = detail["events"]
    concept_rows = []
    for cid, count in events["concept_events"].items():
        correct = events["concept_correct"].get(cid, 0)
        concept_rows.append({
            "concept_id": cid,
            "name_zh": detail["concept_names"].get(cid, ""),
            "interactions": count,
            "correct": correct,
            "error_rate": round(1 - correct / count, 6),
        })
    concept_rows.sort(key=lambda row: (-row["interactions"], row["concept_id"]))
    course_rows = [
        {"course_id": cid, "interactions": count}
        for cid, count in sorted(
            events["course_events"].items(), key=lambda pair: -pair[1]
        )
    ]
    monthly_rows = [
        {"month": month, "interactions": count}
        for month, count in sorted(events["monthly_events"].items())
    ]
    _write_csv(
        output / "missing_values.csv",
        ["dataset", "field", "missing_count", "total_count", "missing_pct"],
        detail["missing_rows"],
    )
    statistics_rows = []
    for section in ("counts", "coverage", "sequence_length"):
        for metric, value in report[section].items():
            statistics_rows.append({
                "section": section, "metric": metric, "value": value,
                "scope": report["mode"],
            })
    for label, count in report["labels"].items():
        statistics_rows.append({
            "section": "labels", "metric": label,
            "value": count, "scope": report["mode"],
        })
    _write_csv(
        output / "dataset_statistics.csv",
        ["section", "metric", "value", "scope"], statistics_rows,
    )
    _write_csv(
        output / "concept_statistics.csv",
        ["concept_id", "name_zh", "interactions", "correct", "error_rate"],
        concept_rows,
    )
    _write_csv(
        output / "course_statistics.csv",
        ["course_id", "interactions"], course_rows,
    )
    _write_csv(
        output / "monthly_statistics.csv",
        ["month", "interactions"], monthly_rows,
    )
    notes = make_eda_plots(
        report, events, concept_rows, course_rows, monthly_rows,
        output / "plots" / "eda",
    )
    report["plot_notes"] = notes
    (output / "data_quality_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    write_eda_summary(report, detail["missing_rows"], notes,
                      output / "eda_summary.md")
    manifest = {
        "run_id": run_id, "type": "data_quality_eda", "status": "complete",
        "mode": report["mode"], "max_students": max_students,
        "created_at_utc": report["created_at_utc"],
        "data_root": str(data_root),
        "files": {item["file"]: item["bytes"] for item in report["files"]},
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    return output
