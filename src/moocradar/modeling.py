# File: src/moocradar/modeling.py
# Purpose: Fit a smoothed baseline and global BKT without future-label leakage.

import csv
import json
import math
import random
from collections import Counter, defaultdict
from itertools import groupby
from pathlib import Path

from scipy.optimize import minimize


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def iter_students(path: Path):
    """Yield one student's chronological events at a time."""
    with path.open(encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        for user_id, rows in groupby(reader, key=lambda row: row["user_id"]):
            yield user_id, list(rows)


def split_name(position: int, length: int) -> str:
    """Use per-student chronological 70/15/15 warm-start partitions."""
    if length < 5:
        return "train"
    train_end = max(1, int(0.7 * length))
    validation_end = min(length - 1, train_end + max(1, int(0.15 * length)))
    if position < train_end:
        return "train"
    if position < validation_end:
        return "validation"
    return "test"


def bkt_predict(mastery: float, params: dict) -> float:
    return mastery * (1 - params["slip"]) + (1 - mastery) * params["guess"]


def bkt_update(mastery: float, correct: int, params: dict) -> float:
    """Observe a response, then apply the learning transition."""
    predicted = bkt_predict(mastery, params)
    if correct:
        posterior = mastery * (1 - params["slip"]) / max(predicted, 1e-10)
    else:
        posterior = mastery * params["slip"] / max(1 - predicted, 1e-10)
    return min(1.0, max(0.0, posterior + (1 - posterior) * params["learn"]))


def fit_bkt(sequences: list[list[int]]) -> dict:
    """Estimate pooled BKT parameters by train-only sequence likelihood."""
    defaults = {"initial": 0.35, "learn": 0.12, "guess": 0.2, "slip": 0.1}
    if len(sequences) < 10:
        return {**defaults, "fit_status": "insufficient_sequences_defaults"}

    def objective(values):
        params = dict(zip(("initial", "learn", "guess", "slip"), values))
        loss = 0.0
        for sequence in sequences:
            mastery = params["initial"]
            for correct in sequence:
                probability = bkt_predict(mastery, params)
                loss -= math.log(max(probability if correct else 1 - probability, 1e-10))
                mastery = bkt_update(mastery, correct, params)
        return loss

    result = minimize(
        objective,
        list(defaults.values()),
        method="L-BFGS-B",
        bounds=[(0.01, 0.95), (0.001, 0.8), (0.001, 0.49), (0.001, 0.49)],
        options={"maxiter": 35},
    )
    return {**dict(zip(defaults, map(float, result.x))), "fit_status": "optimized" if result.success else "optimizer_limit"}


def load_links(processed_dir: Path) -> tuple[dict, dict, dict]:
    problems = {row["problem_id"]: row for row in read_csv(processed_dir / "problems.csv")}
    links = defaultdict(list)
    for row in read_csv(processed_dir / "problem_concepts.csv"):
        links[row["problem_id"]].append(row["concept_id"])
    names = {row["concept_id"]: row["name_zh"] for row in read_csv(processed_dir / "concepts.csv")}
    return problems, dict(links), names


def fit_models(processed_dir: Path, seed: int = 42) -> tuple[dict, dict, dict]:
    """Fit train-only concept frequencies and a bounded sample of BKT sequences."""
    _, links, _ = load_links(processed_dir)
    counts = defaultdict(lambda: [0, 0])
    global_correct = 0
    global_total = 0
    sampled = []
    randomizer = random.Random(seed)
    seen_sequences = 0
    for _, rows in iter_students(processed_dir / "interactions.csv"):
        by_concept = defaultdict(list)
        for i, row in enumerate(rows):
            if split_name(i, len(rows)) != "train" or row["is_correct"] not in ("0", "1"):
                continue
            label = int(row["is_correct"])
            global_correct += label
            global_total += 1
            for cid in links.get(row["problem_id"], []):
                counts[cid][0] += label
                counts[cid][1] += 1
                by_concept[cid].append(label)
        for sequence in by_concept.values():
            if len(sequence) < 2:
                continue
            seen_sequences += 1
            sequence = sequence[:30]
            if len(sampled) < 800:
                sampled.append(sequence)
            else:
                index = randomizer.randrange(seen_sequences)
                if index < 800:
                    sampled[index] = sequence
    if global_total == 0:
        raise ValueError("No valid training interactions")
    prior = (global_correct + 1) / (global_total + 2)
    baseline = {"global_probability": prior, "smoothing": 10, "concept_counts": dict(counts)}
    bkt = fit_bkt(sampled)
    training = {"train_interactions": global_total, "sampled_bkt_sequences": len(sampled), "seed": seed}
    return baseline, bkt, training


def baseline_predict(concept_ids: list[str], model: dict) -> float:
    prior = model["global_probability"]
    alpha = model["smoothing"]
    if not concept_ids:
        return prior
    values = []
    for cid in concept_ids:
        correct, total = model["concept_counts"].get(cid, [0, 0])
        values.append((correct + alpha * prior) / (total + alpha))
    return sum(values) / len(values)


def save_models(model_dir: Path, baseline: dict, bkt: dict) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "baseline.json").write_text(json.dumps(baseline), encoding="utf-8")
    (model_dir / "bkt.json").write_text(json.dumps(bkt), encoding="utf-8")


def load_models(model_dir: Path) -> tuple[dict, dict]:
    return (
        json.loads((model_dir / "baseline.json").read_text(encoding="utf-8")),
        json.loads((model_dir / "bkt.json").read_text(encoding="utf-8")),
    )
