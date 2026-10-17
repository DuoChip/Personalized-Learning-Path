# File: src/moocradar/recommendation.py
# Purpose: Detect evidence-aware knowledge gaps and rank practice problems.

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
    """Keep unseen concepts unknown; put observed weak concepts first."""
    gaps = []
    for cid in course_index.get(course_id, {}):
        observed = observations.get(cid, 0)
        value = mastery.get(cid) if observed else None
        if value is not None and value >= target:
            continue
        priority = (target - value) if value is not None else -1.0
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
              links: dict, seen_problems: set[str], top_n: int = 5) -> list[dict]:
    """Rank course exercises by directly linked gaps and prior exposure."""
    priority = {gap["concept_id"]: gap["priority"] for gap in gaps}
    candidates = []
    for problem_id, problem in problems.items():
        if problem["course_id"] != course_id:
            continue
        matched = [cid for cid in links.get(problem_id, []) if cid in priority]
        if not matched:
            continue
        observed = [priority[cid] for cid in matched if priority[cid] >= 0]
        score = sum(observed) / max(1, len(observed)) + 0.05 * len(matched)
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
