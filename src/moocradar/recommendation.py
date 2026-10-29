# File: src/moocradar/recommendation.py
# Purpose: Detect evidence-aware knowledge gaps and rank practice problems with ZPD & Bloom progression.

import math
from collections import defaultdict


def build_course_index(problems: dict, links: dict) -> dict:
    """Group directly linked problems by course and concept."""
    index = defaultdict(lambda: defaultdict(list))
    for problem_id, problem in problems.items():
        course_id = problem["course_id"]
        for cid in links.get(problem_id, []):
            index[course_id][cid].append(problem_id)
    return index


def gaps_for_course(user_id: str, course_id: str, mastery: dict, observations: dict,
                    course_index: dict, target: float = 0.7) -> list[dict]:
    """Keep unseen concepts unknown; put observed weak concepts first, weighted by evidence confidence."""
    gaps = []
    for cid in course_index.get(course_id, {}):
        observed = observations.get(cid, 0)
        value = mastery.get(cid) if observed else None
        if value is not None and value >= target:
            continue
        if value is not None:
            # Weight gap severity by evidence reliability so multi-error concepts rank highest
            evidence_weight = 1.0 + 0.12 * min(observed, 5)
            priority = (target - value) * evidence_weight
        else:
            priority = -1.0
        gaps.append({
            "user_id": user_id, "course_id": course_id, "concept_id": cid,
            "mastery": "" if value is None else round(value, 6),
            "observations": observed,
            "evidence_status": "observed_gap" if observed else "unobserved",
            "priority": round(priority, 6),
        })
    gaps.sort(key=lambda item: (-item["priority"], item["concept_id"]))
    return gaps


def recommend(user_id: str, course_id: str, gaps: list[dict], problems: dict,
              links: dict, seen_problems: set[str], top_n: int = 5,
              problem_difficulty: dict | None = None) -> list[dict]:
    """
    Rank course exercises by:
      1. Directly linked knowledge gap priority
      2. Zone of Proximal Development (ZPD): match problem difficulty to student's current mastery
      3. Bloom cognitive scaffolding: prefer foundational Bloom levels (1-3) when mastery is low
      4. Novelty bonus for unseen exercises
    """
    priority = {gap["concept_id"]: gap["priority"] for gap in gaps}
    gap_mastery = {gap["concept_id"]: float(gap["mastery"]) for gap in gaps if gap["mastery"] != ""}
    candidates = []
    for problem_id, problem in problems.items():
        if problem["course_id"] != course_id:
            continue
        matched = [cid for cid in links.get(problem_id, []) if cid in priority]
        if not matched:
            continue
        observed = [priority[cid] for cid in matched if priority[cid] >= 0]
        base_score = sum(observed) / max(1, len(observed)) + 0.05 * len(matched)

        # Zone of Proximal Development (ZPD) & Bloom Cognitive Alignment bonus
        known_masteries = [gap_mastery[cid] for cid in matched if cid in gap_mastery]
        avg_m = sum(known_masteries) / len(known_masteries) if known_masteries else 0.45

        zpd_bonus = 0.0
        if problem_difficulty and problem_id in problem_difficulty:
            d_prob = 1.0 / (1.0 + math.exp(-max(min(problem_difficulty[problem_id], 10.0), -10.0)))
            # Optimal challenge is slightly above or near current mastery
            zpd_bonus = 0.08 * max(0.0, 1.0 - abs(d_prob - (avg_m + 0.05)))

        bloom_bonus = 0.0
        try:
            bloom_lvl = int(problem.get("cognitive_dimension") or 0)
            if 1 <= bloom_lvl <= 6:
                target_bloom = 1.0 + 4.0 * avg_m  # Low mastery -> Bloom 1-2; higher -> Bloom 3-5
                bloom_bonus = 0.05 * max(0.0, 1.0 - abs(bloom_lvl - target_bloom) / 5.0)
        except (ValueError, TypeError):
            pass

        score = base_score + zpd_bonus + bloom_bonus
        if problem_id not in seen_problems:
            score += 0.1
        candidates.append((score, problem_id, matched))
    candidates.sort(key=lambda item: (-item[0], item[1]))
    result = []
    for rank, (score, problem_id, matched) in enumerate(candidates[:top_n], 1):
        result.append({
            "user_id": user_id, "course_id": course_id, "rank": rank,
            "activity_type": "practice_problem", "problem_id": problem_id,
            "concept_ids": "|".join(sorted(matched)), "score": round(score, 6),
            "reason": "Practice concepts with observed gaps" if any(priority[cid] >= 0 for cid in matched)
                      else "Explore concepts without prior evidence",
        })
    return result
