# File: tests/test_pipeline.py
# Purpose: Check streaming, temporal splits, multi-concept mapping, models, and output persistence.

import csv
import json
from pathlib import Path

import pytest

from moocradar.modeling import (baseline_predict, bkt_predict, bkt_update,
                                fit_models, load_models, save_models, split_name)
from moocradar.inference import infer
from moocradar.pipeline import train
from moocradar.preparation import concept_id, prepare
from moocradar.reading import iter_json_array
from moocradar.recommendation import gaps_for_course, recommend
from moocradar.translation import apply, extract


def fixture_data(tmp_path: Path) -> tuple[Path, Path]:
    raw = tmp_path / "raw"
    raw.mkdir()
    problems = [
        {"problem_id": "p1", "exercise_id": "e1", "course_id": "c1",
         "detail": "{'title': '你好', 'content': '世界', 'option': {'A': '选项'}, 'score': 1}",
         "concepts": ["概念甲", "概念乙"], "cognitive_dimension": 1, "knowledge_type": 2},
        {"problem_id": "p2", "exercise_id": "e1", "course_id": "c1",
         "detail": "{}", "concepts": ["概念乙"], "cognitive_dimension": 2, "knowledge_type": 2},
    ]
    (raw / "problem.json").write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in problems), encoding="utf-8")
    groups = []
    for user in ("u1", "u2", "u3"):
        sequence = [
            {"log_id": f"{user}_{i}", "user_id": user,
             "problem_id": "p1" if i % 2 == 0 else "p2",
             "is_correct": i % 2, "attempts": 1, "score": None,
             "submit_time": f"2020-01-01 00:00:{i:02d}"}
            for i in range(8)
        ]
        groups.append({"seq": sequence})
    (raw / "student-problem-fine.json").write_text(json.dumps(groups, ensure_ascii=False), encoding="utf-8")
    return raw, tmp_path / "processed"


def test_streaming_and_normalization(tmp_path):
    raw, processed = fixture_data(tmp_path)
    assert len(list(iter_json_array(raw / "student-problem-fine.json", chunk_size=13))) == 3
    quality = prepare(raw, processed)
    assert quality["interactions"] == 24
    assert quality["distinct_students"] == 3
    assert quality["unknown_problem_id"] == 0
    with (processed / "problem_concepts.csv").open(encoding="utf-8") as file:
        links = list(csv.DictReader(file))
    assert len(links) == 3
    assert sum(row["problem_id"] == "p1" for row in links) == 2
    with (processed / "interactions.csv").open(encoding="utf-8") as file:
        assert len(list(csv.DictReader(file))) == 24
    for invalid in ("[{},]", "[{}{}]", "[{}] extra"):
        path = tmp_path / "invalid.json"
        path.write_text(invalid, encoding="utf-8")
        with pytest.raises(ValueError):
            list(iter_json_array(path, chunk_size=2))


def test_temporal_split_and_train_only_counts(tmp_path):
    raw, processed = fixture_data(tmp_path)
    prepare(raw, processed)
    baseline, bkt, details = fit_models(processed)
    assert [split_name(i, 8) for i in range(8)] == [
        "train", "train", "train", "train", "train", "validation", "test", "test"
    ]
    assert details["train_interactions"] == 15
    assert sum(total for _, total in baseline["concept_counts"].values()) == 24
    assert 0 <= baseline_predict([concept_id("概念甲")], baseline) <= 1
    assert 0 <= bkt_predict(bkt["initial"], bkt) <= 1
    assert 0 <= bkt_update(bkt["initial"], 1, bkt) <= 1
    model_dir = tmp_path / "models"
    save_models(model_dir, baseline, bkt)
    assert load_models(model_dir) == (baseline, bkt)

    # Changing a later test response must not affect either fitted model.
    groups = json.loads((raw / "student-problem-fine.json").read_text(encoding="utf-8"))
    for group in groups:
        group["seq"][-1]["is_correct"] = 1 - group["seq"][-1]["is_correct"]
    (raw / "student-problem-fine.json").write_text(json.dumps(groups), encoding="utf-8")
    changed_processed = tmp_path / "changed"
    prepare(raw, changed_processed)
    changed_baseline, changed_bkt, _ = fit_models(changed_processed)
    assert changed_baseline == baseline
    assert changed_bkt == bkt


def test_gap_order_and_translation_id_preservation(tmp_path):
    raw, processed = fixture_data(tmp_path)
    cache = tmp_path / "translation.csv"
    assert extract(raw / "problem.json", cache) >= 5
    rows = list(csv.DictReader(cache.open(encoding="utf-8")))
    for row in rows:
        if row["source_zh"] == "你好":
            row["english"] = "Hello"
    with cache.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, ["source_zh", "english"])
        writer.writeheader()
        writer.writerows(rows)
    output = tmp_path / "translated.csv"
    report = apply(raw / "problem.json", cache, output)
    translated = list(csv.DictReader(output.open(encoding="utf-8")))
    assert report["problem_rows"] == 2
    assert translated[0]["problem_id"] == "p1"
    assert translated[0]["title_en"] == "Hello"
    assert translated[1]["problem_id"] == "p2"
    cid_a, cid_b = concept_id("概念甲"), concept_id("概念乙")
    index = {"c1": {cid_a: ["p1"], cid_b: ["p1", "p2"]}}
    gaps = gaps_for_course("u1", "c1", {cid_a: 0.2}, {cid_a: 2}, index)
    assert gaps[0]["concept_id"] == cid_a
    assert gaps[1]["evidence_status"] == "unobserved"
    paths = recommend("u1", "c1", gaps,
                      {"p1": {"course_id": "c1"}, "p2": {"course_id": "c1"}},
                      {"p1": [cid_a, cid_b], "p2": [cid_b]}, set(), 2)
    assert paths[0]["problem_id"] == "p1"
    assert paths[0]["rank"] == 1


def test_artifact_and_model_inference(tmp_path, monkeypatch):
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "mpl"))
    raw, processed = fixture_data(tmp_path)
    prepare(raw, processed)
    run_dir, metrics = train(processed, tmp_path / "artifacts")
    assert len(metrics) == 4
    assert json.loads((run_dir / "manifest.json").read_text())["status"] == "complete"
    with (run_dir / "predictions.csv").open(encoding="utf-8") as file:
        predictions = list(csv.DictReader(file))
    assert len(predictions) == 18
    assert all(0 <= float(row["probability_correct"]) <= 1 for row in predictions)
    before = infer("u1", processed, run_dir)
    after = infer("u1", processed, run_dir, new_problem_id="p1", new_is_correct=0)
    assert before["learning_path"]
    assert before["mastery"] != after["mastery"]
