# File: tests/test_ncdm.py
# Purpose: Validate NCDM data, model, checkpoint, artifacts, and recommendation integration.

import csv
import json
from collections import defaultdict
from pathlib import Path

import torch

from moocradar.ncdm import (NCDM, NCDMConfig, load_ncdm_checkpoint,
                            save_ncdm_checkpoint)
from moocradar.ncdm_data import load_ncdm_data, make_collate_fn
from moocradar.ncdm_inference import infer_ncdm, predict_ncdm_response
from moocradar.ncdm_pipeline import train_ncdm
from moocradar.preparation import prepare


def ncdm_fixture(tmp_path: Path) -> Path:
    raw = tmp_path / "raw"
    raw.mkdir()
    problems = [
        {
            "problem_id": "p1", "exercise_id": "e1", "course_id": "c1",
            "detail": "{'title': 'one', 'score': 1}",
            "concepts": ["concept-a", "concept-b"],
            "cognitive_dimension": 1, "knowledge_type": 2,
        },
        {
            "problem_id": "p2", "exercise_id": "e2", "course_id": "c1",
            "detail": "{'title': 'two', 'score': 1}",
            "concepts": ["concept-b"],
            "cognitive_dimension": 2, "knowledge_type": 2,
        },
    ]
    (raw / "problem.json").write_text(
        "\n".join(json.dumps(row) for row in problems), encoding="utf-8"
    )
    groups = []
    for student_index in range(4):
        user_id = f"u{student_index}"
        sequence = []
        for position in range(10):
            sequence.append({
                "log_id": f"{user_id}_{position}",
                "user_id": user_id,
                "problem_id": "p1" if position % 2 == 0 else "p2",
                "is_correct": (position + student_index) % 2,
                "attempts": 1,
                "score": None,
                "submit_time": f"2020-01-01 00:00:{position:02d}",
            })
        groups.append({"seq": sequence})
    (raw / "student-problem-fine.json").write_text(json.dumps(groups), encoding="utf-8")
    processed = tmp_path / "processed"
    prepare(raw, processed)
    return processed


def test_ncdm_data_q_matrix_and_mapping_consistency(tmp_path):
    processed = ncdm_fixture(tmp_path)
    fine = load_ncdm_data(processed, "fine")
    middle = load_ncdm_data(processed, "middle")
    assert len(fine.q_rows[fine.problem_to_index["p1"]]) == 2
    assert len(fine.q_rows[fine.problem_to_index["p2"]]) == 1
    assert len(middle.concept_ids) == 2
    assert all(event.student_index == middle.student_to_index[event.user_id] for event in middle.events)
    assert all(event.problem_index == middle.problem_to_index[event.problem_id] for event in middle.events)
    for user_id in middle.student_ids:
        positions = [event.sequence_position for event in middle.events if event.user_id == user_id]
        assert positions == sorted(positions)
    assert {event.split for event in middle.events} == {"train", "validation", "test"}


def test_ncdm_forward_gradient_checkpoint_and_reproducibility(tmp_path):
    config = NCDMConfig(4, 2, 3, hidden_dims=(8, 4), dropout=0.0)
    torch.manual_seed(7)
    model = NCDM(config)
    students = torch.tensor([0, 1, 2])
    problems = torch.tensor([0, 1, 0])
    q = torch.tensor([[1, 1, 0], [0, 1, 0], [1, 0, 1]], dtype=torch.float32)
    probabilities = model(students, problems, q)
    assert probabilities.shape == (3,)
    assert torch.all((probabilities >= 0) & (probabilities <= 1))
    torch.nn.functional.binary_cross_entropy(probabilities, torch.tensor([1.0, 0.0, 1.0])).backward()
    assert model.student_proficiency.weight.grad is not None
    assert torch.count_nonzero(model.student_proficiency.weight.grad) > 0
    model.eval()
    one_concept_q = torch.tensor([[1.0, 0.0, 0.0]])
    with torch.no_grad():
        model.student_proficiency.weight[0, 0] = -2.0
        lower = model(torch.tensor([0]), torch.tensor([0]), one_concept_q)
        model.student_proficiency.weight[0, 0] = 2.0
        higher = model(torch.tensor([0]), torch.tensor([0]), one_concept_q)
    assert higher.item() >= lower.item()

    checkpoint = tmp_path / "best_ncdm.pt"
    save_ncdm_checkpoint(checkpoint, model, epoch=1, validation_metric=0.5)
    restored, payload = load_ncdm_checkpoint(checkpoint)
    assert payload["epoch"] == 1
    model.eval()
    with torch.no_grad():
        assert torch.equal(model(students, problems, q), restored(students, problems, q))


def test_collate_materializes_only_batch_q(tmp_path):
    processed = ncdm_fixture(tmp_path)
    bundle = load_ncdm_data(processed, "fine")
    collate = make_collate_fn(bundle.q_rows, len(bundle.concept_ids))
    first = bundle.events[0]
    batch = collate([(first.student_index, first.problem_index, float(first.label), 0)])
    assert batch["q"].shape == (1, len(bundle.concept_ids))
    assert int(batch["q"].sum()) == 2


def test_ncdm_smoke_artifacts_mastery_and_recommendation(tmp_path, monkeypatch):
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "mpl"))
    processed = ncdm_fixture(tmp_path)
    run_dir, metrics = train_ncdm(
        processed,
        tmp_path / "artifacts",
        granularity="middle",
        batch_size=8,
        epochs=2,
        patience=2,
        hidden_dims=(8, 4),
        dropout=0.0,
        device="cpu",
        progress=False,
    )
    required = [
        "config.yaml", "manifest.json", "metrics.csv", "predictions.csv",
        "student_mastery.csv", "knowledge_gaps.csv", "learning_paths.csv",
        "models/best_ncdm.pt", "models/id_mappings.json", "models/q_matrix.npz",
        "plots/training_loss.png", "plots/roc_curve.png", "plots/confusion_matrix.png",
    ]
    assert all((run_dir / name).exists() for name in required)
    assert len(metrics) == 8
    assert json.loads((run_dir / "manifest.json").read_text())["status"] == "complete"
    with (run_dir / "student_mastery.csv").open(encoding="utf-8") as file:
        mastery = list(csv.DictReader(file))
    assert mastery
    assert all(0 <= float(row["mastery_probability"]) <= 1 for row in mastery)
    assert {row["model_name"] for row in mastery} == {"ncdm"}
    with (run_dir / "predictions.csv").open(encoding="utf-8") as file:
        predictions = list(csv.DictReader(file))
    aligned = defaultdict(set)
    for row in predictions:
        aligned[(row["model"], row["split"])].add((row["log_id"], row["is_correct"]))
    for split in ("validation", "test"):
        expected = aligned[("ncdm", split)]
        assert expected == aligned[("smoothed_concept", split)]
        assert expected == aligned[("bkt_frozen", split)]
        assert expected == aligned[("bkt_online", split)]
    with (run_dir / "split_indices.csv").open(encoding="utf-8") as file:
        split_rows = list(csv.DictReader(file))
    split_order = {"train": 0, "validation": 1, "test": 2}
    by_student = defaultdict(list)
    for row in split_rows:
        by_student[row["user_id"]].append(split_order[row["split"]])
    assert all(values == sorted(values) for values in by_student.values())
    with (run_dir / "learning_paths.csv").open(encoding="utf-8") as file:
        paths = list(csv.DictReader(file))
    assert len({row["user_id"] for row in paths}) >= 3

    first_probability = predict_ncdm_response(run_dir, "u0", "p1")
    second_probability = predict_ncdm_response(run_dir, "u0", "p1")
    assert first_probability == second_probability
    profile = infer_ncdm("u0", processed, run_dir)
    assert profile["adaptive_update"] is False
    assert profile["mastery"]
