# File: src/moocradar/pipeline.py
# Purpose: Run training, temporal evaluation, profiling, and recommendations.

import csv
import json
import logging
import shutil
from collections import Counter, defaultdict
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import yaml

from .evaluation import evaluate
from .modeling import (baseline_predict, bkt_predict, bkt_update, fit_models,
                       iter_students, load_links, save_models, split_name)
from .recommendation import build_course_index, gaps_for_course, recommend

LOG = logging.getLogger(__name__)
PREDICTION_FIELDS = ["log_id", "user_id", "problem_id", "split", "model", "is_correct", "probability_correct"]
MASTERY_FIELDS = ["user_id", "concept_id", "concept_name_zh", "mastery", "probability_correct", "observations", "confidence", "insufficient_evidence"]
GAP_FIELDS = ["user_id", "course_id", "concept_id", "mastery", "observations", "evidence_status", "priority"]
PATH_FIELDS = ["user_id", "course_id", "rank", "activity_type", "problem_id", "concept_ids", "score", "reason"]
BLOOM_FIELDS = ["user_id", "cognitive_dimension", "observations", "correct", "accuracy"]


def _writer(stack: ExitStack, path: Path, fields: list[str]) -> csv.DictWriter:
    file = stack.enter_context(path.open("w", encoding="utf-8", newline=""))
    writer = csv.DictWriter(file, fields)
    writer.writeheader()
    return writer


def train(processed_dir: Path, artifact_root: Path, *, seed: int = 42,
          mastery_target: float = 0.7, top_n: int = 5) -> tuple[Path, list[dict]]:
    """Create an immutable run directory with models, predictions, and paths."""
    if not 0 < mastery_target < 1 or top_n < 1:
        raise ValueError("mastery_target must be in (0,1) and top_n positive")
    if not (processed_dir / "data_quality.json").exists():
        raise FileNotFoundError("Run prepare before train")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "_" + uuid4().hex[:6]
    run_dir = artifact_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    config = {"processed_dir": str(processed_dir), "seed": seed,
              "mastery_target": mastery_target, "top_n": top_n,
              "split": "per_student_chronological_70_15_15_min5"}
    (run_dir / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    shutil.copy2(processed_dir / "data_quality.json", run_dir / "data_quality.json")
    quality_report = json.loads((processed_dir / "data_quality.json").read_text())
    manifest = {"run_id": run_id, "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "status": "running", "seed": seed, "models": ["smoothed_concept", "bkt"],
                "dataset_version": quality_report.get("source_files", {}),
                "data_quality_mode": quality_report["mode"]}
    manifest_path = run_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    log_dir = run_dir / "logs"
    log_dir.mkdir()
    handler = logging.FileHandler(log_dir / "training.log", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    LOG.addHandler(handler)
    try:
        LOG.info("Starting run %s from %s", run_id, processed_dir)
        baseline, bkt, fit_info = fit_models(processed_dir, seed)
        LOG.info("Fitted models using %s train responses", fit_info["train_interactions"])
        save_models(run_dir / "models", baseline, bkt)
        problems, links, names = load_links(processed_dir)
        course_index = build_course_index(problems, links)
        course_problems = defaultdict(dict)
        for problem_id, problem in problems.items():
            course_problems[problem["course_id"]][problem_id] = problem
        plot_stats = {"student_lengths": [], "concept_attempts": Counter(),
                      "concept_outcomes": defaultdict(lambda: [0, 0]),
                      "bloom": defaultdict(lambda: [0, 0]), "daily": Counter(), "correct": Counter()}
        counts = Counter()
        with ExitStack() as stack:
            predictions = _writer(stack, run_dir / "predictions.csv", PREDICTION_FIELDS)
            splits = _writer(stack, run_dir / "split_indices.csv", ["log_id", "user_id", "sequence_position", "split"])
            mastery_writer = _writer(stack, run_dir / "student_mastery.csv", MASTERY_FIELDS)
            bloom_writer = _writer(stack, run_dir / "cognitive_profile.csv", BLOOM_FIELDS)
            gap_writer = _writer(stack, run_dir / "knowledge_gaps.csv", GAP_FIELDS)
            path_writer = _writer(stack, run_dir / "learning_paths.csv", PATH_FIELDS)
            for user_id, rows in iter_students(processed_dir / "interactions.csv"):
                if not user_id:
                    continue
                counts["students"] += 1
                plot_stats["student_lengths"].append(len(rows))
                states = {}
                observations = Counter()
                bloom = defaultdict(lambda: [0, 0])
                seen_problems = set()
                target_course = None
                for i, row in enumerate(rows):
                    split = split_name(i, len(rows))
                    counts[split] += 1
                    splits.writerow({"log_id": row["log_id"], "user_id": user_id,
                                     "sequence_position": i, "split": split})
                    problem_id = row["problem_id"]
                    problem = problems.get(problem_id)
                    cids = links.get(problem_id, [])
                    if problem:
                        target_course = problem["course_id"]
                    seen_problems.add(problem_id)
                    if row["is_correct"] not in ("0", "1"):
                        continue
                    correct = int(row["is_correct"])
                    plot_stats["correct"][correct] += 1
                    if row["submit_time"]:
                        plot_stats["daily"][row["submit_time"][:10]] += 1
                    for cid in cids:
                        plot_stats["concept_attempts"][cid] += 1
                        plot_stats["concept_outcomes"][cid][0] += 1
                        plot_stats["concept_outcomes"][cid][1] += correct
                    if problem and problem["cognitive_dimension"]:
                        level = problem["cognitive_dimension"]
                        bloom[level][0] += 1
                        bloom[level][1] += correct
                        plot_stats["bloom"][level][0] += 1
                        plot_stats["bloom"][level][1] += correct
                    if split in ("validation", "test"):
                        p_base = baseline_predict(cids, baseline)
                        p_bkt = (sum(bkt_predict(states.get(cid, bkt["initial"]), bkt) for cid in cids) / len(cids)) if cids else bkt_predict(bkt["initial"], bkt)
                        for model, probability in (("smoothed_concept", p_base), ("bkt", p_bkt)):
                            predictions.writerow({"log_id": row["log_id"], "user_id": user_id,
                                                 "problem_id": problem_id, "split": split,
                                                 "model": model, "is_correct": correct,
                                                 "probability_correct": round(probability, 8)})
                    for cid in cids:
                        states[cid] = bkt_update(states.get(cid, bkt["initial"]), correct, bkt)
                        observations[cid] += 1
                for cid, value in states.items():
                    n = observations[cid]
                    mastery_writer.writerow({
                        "user_id": user_id, "concept_id": cid,
                        "concept_name_zh": names.get(cid, ""), "mastery": round(value, 6),
                        "probability_correct": round(bkt_predict(value, bkt), 6),
                        "observations": n, "confidence": round(n / (n + 5), 6),
                        "insufficient_evidence": int(n < 3),
                    })
                for level, (n, correct) in bloom.items():
                    bloom_writer.writerow({"user_id": user_id, "cognitive_dimension": level,
                                           "observations": n, "correct": correct,
                                           "accuracy": round(correct / n, 6)})
                if target_course:
                    gaps = gaps_for_course(user_id, target_course, states, observations,
                                           course_index, mastery_target)[:20]
                    gap_writer.writerows(gaps)
                    paths = recommend(user_id, target_course, gaps,
                                      course_problems[target_course], links,
                                      seen_problems, top_n)
                    path_writer.writerows(paths)
                    if paths:
                        counts["students_with_paths"] += 1
                if counts["students"] % 1000 == 0:
                    LOG.info("Evaluated %s students", counts["students"])
        metrics = evaluate(run_dir / "predictions.csv", run_dir / "metrics.csv")
        LOG.info("Saved %s metric rows", len(metrics))
        from .plots import make_plots
        make_plots(plot_stats, run_dir / "plots")
        manifest.update({"status": "complete", "counts": dict(counts),
                         "training": fit_info, "metric_rows": len(metrics)})
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        LOG.info("Completed run %s", run_id)
        return run_dir, metrics
    except Exception as error:
        LOG.exception("Run %s failed", run_id)
        manifest.update({"status": "failed", "error": str(error)})
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        raise
    finally:
        LOG.removeHandler(handler)
        handler.close()
