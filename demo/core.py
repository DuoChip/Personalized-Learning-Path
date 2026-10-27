# File: demo/core.py
# Purpose: Load pipeline outputs for the Streamlit demo and call the real train/infer code.

from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

import pandas as pd

DEMO_DIR = Path(__file__).resolve().parent
REPO_ROOT = DEMO_DIR.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from moocradar.inference import infer  # noqa: E402
from moocradar.pipeline import train  # noqa: E402

SAMPLE_PROCESSED = DEMO_DIR / "sample_data" / "processed"
CACHE_ARTIFACTS = DEMO_DIR / ".cache" / "artifacts"

BLOOM_VI = {1: "Nhớ", 2: "Hiểu", 3: "Vận dụng", 4: "Phân tích", 5: "Đánh giá", 6: "Sáng tạo"}
REASON_VI = {
    "Practice concepts with observed gaps": "Luyện concept đã có bằng chứng còn yếu",
    "Explore concepts without prior evidence": "Khám phá concept chưa có bằng chứng",
}

# Full-dataset results, quoted from docs/RESULTS.md (run 20261017T150803777229Z_009410).
FULL_DATA = {
    "run_id": "20261017T150803777229Z_009410",
    "students": 14224, "interactions": 898933, "problems": 9383,
    "courses": 73, "concepts": 9985, "multi_concept_problems": 7203,
    "train": 624837, "validation": 128789, "test": 145307,
    "missing_score": 543847, "negative_score": 114,
    "out_of_order_sequences": 8168, "out_of_order_pairs": 95384,
    "positive_rate_test": 0.803,
    "metrics": [
        {"model": "smoothed_concept", "split": "validation", "roc_auc": 0.735351,
         "accuracy": 0.827276, "f1": 0.903734, "brier": 0.129782},
        {"model": "smoothed_concept", "split": "test", "roc_auc": 0.738546,
         "accuracy": 0.811255, "f1": 0.894076, "brier": 0.138044},
        {"model": "bkt", "split": "validation", "roc_auc": 0.663296,
         "accuracy": 0.820435, "f1": 0.901361, "brier": 0.138928},
        {"model": "bkt", "split": "test", "roc_auc": 0.653821,
         "accuracy": 0.803320, "f1": 0.890934, "brier": 0.148772},
    ],
    "bkt_test_confusion": {"tp": 116728, "fp": 28579, "tn": 0, "fn": 0},
    "outputs": {"student_mastery": 1074367, "cognitive_profile": 51156,
                "knowledge_gaps": 266497, "learning_paths": 69874},
}


def processed_dir() -> Path:
    return Path(os.environ.get("MOOC_PROCESSED_DIR", SAMPLE_PROCESSED))


def _complete_runs(root: Path, processed: Path) -> list[Path]:
    runs = []
    for manifest in root.glob("*/manifest.json"):
        info = json.loads(manifest.read_text(encoding="utf-8"))
        config = manifest.parent / "config.yaml"
        if info.get("status") == "complete" and config.exists() \
                and str(processed) in config.read_text(encoding="utf-8"):
            runs.append(manifest.parent)
    return sorted(runs)


def ensure_run(processed: Path) -> Path:
    """Use MOOC_RUN_DIR when given; otherwise reuse or create a run with the real pipeline."""
    if os.environ.get("MOOC_RUN_DIR"):
        return Path(os.environ["MOOC_RUN_DIR"])
    runs = _complete_runs(CACHE_ARTIFACTS, processed)
    if runs:
        return runs[-1]
    run_dir, _ = train(processed, CACHE_ARTIFACTS, seed=42)
    return run_dir


def _read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"user_id": str, "problem_id": str, "concept_id": str,
                                    "course_id": str, "log_id": str})


def load_glossary() -> dict[str, str]:
    with (DEMO_DIR / "glossary_vi.csv").open(encoding="utf-8") as file:
        return {row["name_zh"]: row["name_vi"] for row in csv.DictReader(file)}


def load_bundle(processed: Path, run_dir: Path) -> dict:
    glossary = load_glossary()
    concepts = _read(processed / "concepts.csv")
    concepts["name_vi"] = concepts["name_zh"].map(glossary).fillna(concepts["name_zh"])
    problems = _read(processed / "problems.csv")
    links = _read(processed / "problem_concepts.csv")
    interactions = _read(processed / "interactions.csv")
    interactions["submit_time"] = pd.to_datetime(interactions["submit_time"])
    splits = run_dir / "split_indices.csv"
    if splits.exists():
        interactions = interactions.merge(_read(splits)[["log_id", "split"]], on="log_id", how="left")
    return {
        "processed": processed, "run_dir": run_dir,
        "manifest": json.loads((run_dir / "manifest.json").read_text(encoding="utf-8")),
        "quality": json.loads((processed / "data_quality.json").read_text(encoding="utf-8")),
        "concepts": concepts, "problems": problems, "links": links,
        "interactions": interactions,
        "mastery": _read(run_dir / "student_mastery.csv"),
        "bloom": _read(run_dir / "cognitive_profile.csv"),
        "gaps": _read(run_dir / "knowledge_gaps.csv"),
        "paths": _read(run_dir / "learning_paths.csv"),
        "metrics": _read(run_dir / "metrics.csv"),
        "predictions": _read(run_dir / "predictions.csv"),
    }


def concept_names(bundle: dict) -> dict[str, str]:
    return dict(zip(bundle["concepts"]["concept_id"], bundle["concepts"]["name_vi"]))


def problem_label(bundle: dict, problem_id: str) -> str:
    """Readable title: the sample titles embed the Chinese concept name, shown here in Vietnamese."""
    row = bundle["problems"].loc[bundle["problems"]["problem_id"] == problem_id]
    if row.empty:
        return problem_id
    title = str(row.iloc[0]["title"])
    for zh, vi in zip(bundle["concepts"]["name_zh"], bundle["concepts"]["name_vi"]):
        title = title.replace(f"({zh})", f"· {vi}")
    return title


def run_infer(bundle: dict, user_id: str, course_id: str | None = None,
              new_problem_id: str | None = None, new_is_correct: int | None = None,
              top_n: int = 5) -> dict:
    return infer(user_id, bundle["processed"], bundle["run_dir"],
                 new_problem_id=new_problem_id, new_is_correct=new_is_correct,
                 target_course_id=course_id, top_n=top_n)


def explain_score(bundle: dict, path_row: dict, gaps: list[dict], seen: set[str]) -> dict:
    """Split a learning-path score into the terms used by recommendation.recommend()."""
    priority = {gap["concept_id"]: float(gap["priority"]) for gap in gaps}
    gap_mastery = {gap["concept_id"]: float(gap["mastery"]) for gap in gaps
                   if gap["mastery"] not in ("", None) and not pd.isna(gap["mastery"])}
    matched = [cid for cid in str(path_row["concept_ids"]).split("|") if cid in priority]
    observed = [priority[cid] for cid in matched if priority[cid] >= 0]
    gap_term = sum(observed) / max(1, len(observed))
    coverage = 0.05 * len(matched)
    known = [gap_mastery[cid] for cid in matched if cid in gap_mastery]
    avg_mastery = sum(known) / len(known) if known else 0.45
    problem = bundle["problems"].set_index("problem_id").loc[path_row["problem_id"]]
    bloom = int(problem["cognitive_dimension"])
    target_bloom = 1.0 + 4.0 * avg_mastery
    bloom_bonus = 0.05 * max(0.0, 1.0 - abs(bloom - target_bloom) / 5.0) if 1 <= bloom <= 6 else 0.0
    novelty = 0.1 if path_row["problem_id"] not in seen else 0.0
    total = gap_term + coverage + bloom_bonus + novelty
    return {"Mức cấp thiết của gap": gap_term, "Độ phủ nhiều concept": coverage,
            "Khớp bậc Bloom (ZPD)": bloom_bonus, "Bài chưa làm": novelty,
            "total": total, "target_bloom": target_bloom, "bloom": bloom,
            "consistent": abs(total - float(path_row["score"])) < 1e-4}
