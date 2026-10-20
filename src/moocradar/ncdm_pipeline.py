# File: src/moocradar/ncdm_pipeline.py
# Purpose: Integrate NCDM training, fair evaluation, mastery, and recommendations.

from __future__ import annotations

import csv
import json
import logging
import math
import os
import shutil
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from contextlib import ExitStack
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import torch
import yaml

from .evaluation import evaluate
from .modeling import (baseline_predict, bkt_predict, bkt_update, fit_models,
                       load_links, save_models)
from .ncdm_data import (NCDMDataBundle, load_ncdm_data, save_ncdm_metadata,
                        training_observation_counts)
from .ncdm_training import (NCDMTrainingConfig, predict_split,
                            train_ncdm_model)
from .recommendation import build_course_index, gaps_for_course, recommend

LOG = logging.getLogger(__name__)
PREDICTION_FIELDS = [
    "log_id", "user_id", "problem_id", "split", "model", "is_correct",
    "probability_correct", "evaluation_protocol", "cold_start_problem",
]
MASTERY_FIELDS = [
    "user_id", "concept_id", "concept_name_zh", "mastery", "mastery_probability",
    "mastery_level", "probability_correct", "observations", "observation_count",
    "confidence", "insufficient_evidence", "evidence_status", "model_name", "run_id",
]
GAP_FIELDS = [
    "user_id", "course_id", "concept_id", "mastery", "observations",
    "evidence_status", "priority",
]
PATH_FIELDS = [
    "user_id", "course_id", "rank", "activity_type", "problem_id",
    "concept_ids", "score", "reason",
]
BLOOM_FIELDS = ["user_id", "cognitive_dimension", "observations", "correct", "accuracy"]


def _writer(stack: ExitStack, path: Path, fields: list[str]) -> csv.DictWriter:
    file = stack.enter_context(path.open("w", encoding="utf-8", newline=""))
    writer = csv.DictWriter(file, fields)
    writer.writeheader()
    return writer


def _write_ncdm_predictions(writer: csv.DictWriter, model, bundle: NCDMDataBundle,
                            config: NCDMTrainingConfig, device: torch.device,
                            trained_problems: set[int]) -> None:
    for split in ("validation", "test"):
        events, probabilities = predict_split(model, bundle, split, config, device)
        for event, probability in zip(events, probabilities):
            writer.writerow({
                "log_id": event.log_id,
                "user_id": event.user_id,
                "problem_id": event.problem_id,
                "split": split,
                "model": "ncdm",
                "is_correct": event.label,
                "probability_correct": round(probability, 8),
                "evaluation_protocol": "frozen_train_checkpoint",
                "cold_start_problem": int(event.problem_index not in trained_problems),
            })


def _write_classic_predictions(writer: csv.DictWriter, bundle: NCDMDataBundle,
                               links: dict[str, list[str]], baseline: dict, bkt: dict,
                               trained_problems: set[int]) -> None:
    current_user = None
    frozen_states: dict[str, float] = {}
    online_states: dict[str, float] = {}
    for event in bundle.events:
        if event.user_id != current_user:
            current_user = event.user_id
            frozen_states = {}
            online_states = {}
        concepts = links.get(event.problem_id, [])
        if event.split == "train":
            for concept in concepts:
                frozen_states[concept] = bkt_update(
                    frozen_states.get(concept, bkt["initial"]), event.label, bkt
                )
                online_states[concept] = bkt_update(
                    online_states.get(concept, bkt["initial"]), event.label, bkt
                )
            continue
        baseline_probability = baseline_predict(concepts, baseline)
        frozen_probability = (
            sum(bkt_predict(frozen_states.get(cid, bkt["initial"]), bkt) for cid in concepts)
            / len(concepts)
        )
        online_probability = (
            sum(bkt_predict(online_states.get(cid, bkt["initial"]), bkt) for cid in concepts)
            / len(concepts)
        )
        common = {
            "log_id": event.log_id,
            "user_id": event.user_id,
            "problem_id": event.problem_id,
            "split": event.split,
            "is_correct": event.label,
            "cold_start_problem": int(event.problem_index not in trained_problems),
        }
        for name, probability, protocol in (
            ("smoothed_concept", baseline_probability, "frozen_train_statistics"),
            ("bkt_frozen", frozen_probability, "frozen_after_train"),
            ("bkt_online", online_probability, "prequential_updates_after_each_label"),
        ):
            writer.writerow({
                **common,
                "model": name,
                "probability_correct": round(probability, 8),
                "evaluation_protocol": protocol,
            })
        for concept in concepts:
            online_states[concept] = bkt_update(
                online_states.get(concept, bkt["initial"]), event.label, bkt
            )


def _write_history(history: list[dict], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, [
            "epoch", "train_loss", "validation_roc_auc", "validation_brier"
        ])
        writer.writeheader()
        writer.writerows(history)


def _make_plots(run_dir: Path) -> None:
    """Render plots in an isolated process to avoid BLAS/OpenMP backend conflicts."""
    environment = os.environ.copy()
    with tempfile.TemporaryDirectory(prefix="moocradar-mpl-") as mpl_config:
        environment.setdefault("MPLCONFIGDIR", mpl_config)
        subprocess.run(
            [sys.executable, "-m", "moocradar.ncdm_plot_cli", str(run_dir)],
            check=True,
            env=environment,
        )


def _mastery_level(value: float, target: float) -> str:
    if value >= target:
        return "mastered"
    if value >= 0.5:
        return "developing"
    return "needs_support"


def _export_profiles(model, bundle: NCDMDataBundle, processed_dir: Path,
                     run_dir: Path, run_id: str, mastery_target: float,
                     top_n: int, device: torch.device) -> dict:
    problems, links, _ = load_links(processed_dir, bundle.granularity)
    allowed_concepts = set(bundle.concept_ids)
    links = {
        problem_id: [cid for cid in concepts if cid in allowed_concepts]
        for problem_id, concepts in links.items()
    }
    links = {problem_id: concepts for problem_id, concepts in links.items() if concepts}
    course_index = build_course_index(problems, links)
    course_problems = defaultdict(dict)
    for problem_id, problem in problems.items():
        course_problems[problem["course_id"]][problem_id] = problem

    observation_pairs = training_observation_counts(bundle)
    observations_by_student: dict[int, Counter[int]] = defaultdict(Counter)
    latest_course: dict[int, str] = {}
    seen_problems: dict[int, set[str]] = defaultdict(set)
    bloom: dict[int, dict[str, list[int]]] = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    for event in bundle.events:
        if event.split != "train":
            continue
        problem = problems[event.problem_id]
        latest_course[event.student_index] = problem["course_id"]
        seen_problems[event.student_index].add(event.problem_id)
        level = problem.get("cognitive_dimension")
        if level:
            bloom[event.student_index][level][0] += 1
            bloom[event.student_index][level][1] += event.label
    for (student_index, concept_index), count in observation_pairs.items():
        observations_by_student[student_index][concept_index] = count

    counts = Counter()
    concept_to_index = bundle.concept_to_index
    with ExitStack() as stack:
        mastery_writer = _writer(stack, run_dir / "student_mastery.csv", MASTERY_FIELDS)
        gap_writer = _writer(stack, run_dir / "knowledge_gaps.csv", GAP_FIELDS)
        path_writer = _writer(stack, run_dir / "learning_paths.csv", PATH_FIELDS)
        bloom_writer = _writer(stack, run_dir / "cognitive_profile.csv", BLOOM_FIELDS)
        batch_size = 256
        model.eval()
        with torch.no_grad():
            for start in range(0, len(bundle.student_ids), batch_size):
                stop = min(start + batch_size, len(bundle.student_ids))
                student_indices = torch.arange(start, stop, device=device)
                mastery_batch = model.mastery(student_indices).detach().cpu()
                for offset, student_index in enumerate(range(start, stop)):
                    user_id = bundle.student_ids[student_index]
                    course_id = latest_course.get(student_index)
                    observed_indices = set(observations_by_student[student_index])
                    course_indices = {
                        concept_to_index[cid]
                        for cid in course_index.get(course_id, {})
                        if cid in concept_to_index
                    }
                    export_indices = sorted(observed_indices | course_indices)
                    values: dict[str, float] = {}
                    observation_map: dict[str, int] = {}
                    for concept_index in export_indices:
                        concept_id = bundle.concept_ids[concept_index]
                        value = float(mastery_batch[offset, concept_index])
                        observed = observations_by_student[student_index][concept_index]
                        values[concept_id] = value
                        observation_map[concept_id] = observed
                        mastery_writer.writerow({
                            "user_id": user_id,
                            "concept_id": concept_id,
                            "concept_name_zh": bundle.concept_names.get(concept_id, concept_id),
                            "mastery": round(value, 6),
                            "mastery_probability": round(value, 6),
                            "mastery_level": _mastery_level(value, mastery_target),
                            "probability_correct": "",
                            "observations": observed,
                            "observation_count": observed,
                            "confidence": round(observed / (observed + 5), 6),
                            "insufficient_evidence": int(observed < 3),
                            "evidence_status": "observed_model_estimate" if observed else "unobserved_model_estimate",
                            "model_name": "ncdm",
                            "run_id": run_id,
                        })
                        counts["mastery_rows"] += 1
                    for level, (total, correct) in bloom[student_index].items():
                        bloom_writer.writerow({
                            "user_id": user_id,
                            "cognitive_dimension": level,
                            "observations": total,
                            "correct": correct,
                            "accuracy": round(correct / total, 6),
                        })
                    if not course_id:
                        continue
                    gaps = gaps_for_course(
                        user_id, course_id, values, observation_map,
                        course_index, mastery_target,
                    )[:20]
                    gap_writer.writerows(gaps)
                    paths = recommend(
                        user_id, course_id, gaps, course_problems[course_id], links,
                        seen_problems[student_index], top_n,
                    )
                    path_writer.writerows(paths)
                    counts["gap_rows"] += len(gaps)
                    counts["path_rows"] += len(paths)
                    if paths:
                        counts["students_with_paths"] += 1
    return dict(counts)


def train_ncdm(processed_dir: Path, artifact_root: Path, *, seed: int = 42,
               granularity: str = "middle", batch_size: int = 512,
               learning_rate: float = 0.002, epochs: int = 20,
               patience: int = 4, hidden_dims: tuple[int, ...] = (256, 128),
               dropout: float = 0.5, num_workers: int = 0,
               device: str = "auto", mastery_target: float = 0.7,
               top_n: int = 5, resume_path: Path | None = None,
               progress: bool = True) -> tuple[Path, list[dict]]:
    """Create an immutable NCDM run while preserving baseline and BKT outputs."""
    if not (processed_dir / "data_quality.json").exists():
        raise FileNotFoundError("Run prepare before NCDM training")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "_ncdm_" + uuid4().hex[:6]
    run_dir = artifact_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "models").mkdir()
    (run_dir / "logs").mkdir()
    (run_dir / "plots").mkdir()
    quality = json.loads((processed_dir / "data_quality.json").read_text(encoding="utf-8"))
    training_config = NCDMTrainingConfig(
        batch_size=batch_size,
        learning_rate=learning_rate,
        epochs=epochs,
        patience=patience,
        hidden_dims=hidden_dims,
        dropout=dropout,
        num_workers=num_workers,
        seed=seed,
    )
    config = {
        "processed_dir": str(processed_dir),
        "seed": seed,
        "mastery_target": mastery_target,
        "top_n": top_n,
        "granularity": granularity,
        "split": "per_student_chronological_70_15_15_min5",
        "threshold_strategy": "fixed_0.5",
        "evaluation_protocol_primary": "frozen_train_state",
        "ncdm": asdict(training_config),
        "device_requested": device,
    }
    (run_dir / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    shutil.copy2(processed_dir / "data_quality.json", run_dir / "data_quality.json")
    manifest = {
        "run_id": run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "running",
        "models": ["smoothed_concept", "bkt_frozen", "bkt_online", "ncdm"],
        "granularity": granularity,
        "seed": seed,
        "dataset_version": quality.get("source_files", {}),
        "evaluation_protocol": {
            "primary": "Frozen train state: baseline, bkt_frozen, and NCDM use train only",
            "supplemental": "bkt_online predicts then consumes each validation/test label",
            "threshold": 0.5,
        },
    }
    manifest_path = run_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    handler = logging.FileHandler(run_dir / "logs" / "training.log", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    LOG.addHandler(handler)
    try:
        bundle = load_ncdm_data(processed_dir, granularity)
        save_ncdm_metadata(bundle, run_dir / "models")
        train_events = bundle.events_for("train")
        trained_problems = {event.problem_index for event in train_events}
        split_counts = Counter(event.split for event in bundle.events)
        LOG.info(
            "NCDM data: %s students, %s problems, %s concepts, splits=%s",
            len(bundle.student_ids), len(bundle.problem_ids), len(bundle.concept_ids), dict(split_counts),
        )
        model, history, resolved_device = train_ncdm_model(
            bundle,
            run_dir / "models" / "best_ncdm.pt",
            training_config,
            device_name=device,
            resume_path=resume_path,
            progress=progress,
            log=LOG.info,
        )
        _write_history(history, run_dir / "training_history.csv")

        baseline, bkt, classic_fit = fit_models(processed_dir, seed, granularity)
        save_models(run_dir / "models", baseline, bkt)
        _, links, _ = load_links(processed_dir, granularity)
        with (run_dir / "predictions.csv").open("w", encoding="utf-8", newline="") as file:
            prediction_writer = csv.DictWriter(file, PREDICTION_FIELDS)
            prediction_writer.writeheader()
            _write_ncdm_predictions(
                prediction_writer, model, bundle, training_config, resolved_device, trained_problems
            )
            _write_classic_predictions(
                prediction_writer, bundle, links, baseline, bkt, trained_problems
            )
        with (run_dir / "split_indices.csv").open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, ["log_id", "user_id", "sequence_position", "split"])
            writer.writeheader()
            for event in bundle.events:
                writer.writerow({
                    "log_id": event.log_id,
                    "user_id": event.user_id,
                    "sequence_position": event.sequence_position,
                    "split": event.split,
                })
        metrics = evaluate(run_dir / "predictions.csv", run_dir / "metrics.csv")
        _make_plots(run_dir)
        profile_counts = _export_profiles(
            model, bundle, processed_dir, run_dir, run_id,
            mastery_target, top_n, resolved_device,
        )
        manifest.update({
            "status": "complete",
            "device": str(resolved_device),
            "checkpoint": "models/best_ncdm.pt",
            "counts": {
                "students": len(bundle.student_ids),
                "problems": len(bundle.problem_ids),
                "concepts": len(bundle.concept_ids),
                "dropped_without_q": bundle.dropped_without_q,
                "cold_start_problems": len(set(range(len(bundle.problem_ids))) - trained_problems),
                **dict(split_counts),
                **profile_counts,
            },
            "ncdm_training": {
                "epochs_completed": len(history),
                "best_epoch": max(
                    history,
                    key=lambda row: row["validation_roc_auc"]
                    if math.isfinite(row["validation_roc_auc"])
                    else -row["validation_brier"],
                )["epoch"],
            },
            "classic_training": classic_fit,
            "metric_rows": len(metrics),
        })
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return run_dir, metrics
    except Exception as error:
        LOG.exception("NCDM run %s failed", run_id)
        manifest.update({"status": "failed", "error": str(error)})
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        raise
    finally:
        LOG.removeHandler(handler)
        handler.close()
