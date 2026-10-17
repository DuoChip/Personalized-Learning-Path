# File: tests/test_eda.py
# Purpose: Verify raw-data EDA quality checks, outputs, and source preservation.

import csv
import hashlib
import json

from moocradar.eda import run_eda


def test_eda_detects_raw_order_and_preserves_source(tmp_path, monkeypatch):
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "mpl"))
    raw = tmp_path / "raw"
    raw.mkdir()
    problems = [
        {
            "problem_id": "p1", "exercise_id": "e1", "course_id": "c1",
            "detail": "{}", "concepts": ["甲", "乙"],
            "cognitive_dimension": 1, "knowledge_type": 1,
        },
        {
            "problem_id": "p2", "exercise_id": "e1", "course_id": "c1",
            "detail": "{}", "concepts": ["乙"],
            "cognitive_dimension": 2, "knowledge_type": 1,
        },
    ]
    (raw / "problem.json").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in problems),
        encoding="utf-8",
    )
    groups = [{
        "seq": [
            {
                "log_id": f"l{i}", "user_id": "u1",
                "problem_id": "p1" if i != 2 else "unknown",
                "is_correct": i % 2, "attempts": 1,
                "score": -1.0 if i == 0 else None,
                "submit_time": f"2020-01-01 00:00:{second:02d}",
            }
            for i, second in enumerate((3, 2, 4))
        ]
    }]
    (raw / "student-problem-fine.json").write_text(
        json.dumps(groups, ensure_ascii=False), encoding="utf-8",
    )
    before = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in raw.iterdir()
    }

    output = run_eda(raw, tmp_path / "artifacts", max_students=None)
    report = json.loads(
        (output / "data_quality_report.json").read_text(encoding="utf-8")
    )
    assert report["counts"]["interactions"] == 3
    assert report["counts"]["out_of_order_sequences"] == 1
    assert report["counts"]["out_of_order_pairs"] == 1
    assert report["counts"]["invalid_score"] == 1
    assert report["counts"]["unmatched_problem_events"] == 1
    assert report["counts"]["multi_concept_problems"] == 1
    assert report["counts"]["duplicate_exact_record"] == 0
    assert report["coverage"]["video_event_pct"] == 0
    assert report["files"][2]["exists"] is False
    with (output / "missing_values.csv").open(encoding="utf-8") as file:
        missing = list(csv.DictReader(file))
    assert next(
        row for row in missing
        if row["dataset"] == "interactions" and row["field"] == "score"
    )["missing_count"] == "2"
    assert (output / "eda_summary.md").is_file()
    assert list((output / "plots/eda").glob("*.png"))
    assert before == {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in raw.iterdir()
    }
