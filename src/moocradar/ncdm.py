# File: src/moocradar/ncdm.py
# Purpose: Neural Cognitive Diagnosis Model (NCDM) with monotonic diagnostic interaction and online warm-start updates.

import math
import numpy as np


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20.0, 20.0)))


def fit_ncdm(train_events: list[dict], links: dict[str, list[str]], problems: dict[str, dict],
             seed: int = 42, epochs: int = 18, lr: float = 0.08, l2: float = 0.002) -> dict:
    """
    Fit a fast, monotonic Neural Cognitive Diagnosis Model (NCDM) on train interactions.
    Architecture:
      - Student mastery logits: alpha_{u, k} -> mastery m_{u, k} = sigmoid(alpha_{u, k} + 0.5 * theta_u)
      - Problem concept difficulty: d_{p, k} = sigmoid(beta_p + 0.5 * beta_c[k])
      - Problem discrimination (monotonic positive): disc_p = softplus(raw_disc_p)
      - Cognitive (Bloom) level difficulty offset: bloom_offset[b]
      - Diagnostic interaction layer:
          z = global_logit + 4.0 * disc_p * mean_k(m_{u, k} - d_{p, k}) - bloom_offset[b]
          y_hat = sigmoid(z)
    """
    rng = np.random.default_rng(seed)
    users = sorted({e["user_id"] for e in train_events})
    probs = sorted(problems.keys())
    concepts = sorted({cid for cids in links.values() for cid in cids})

    u_map = {u: i for i, u in enumerate(users)}
    p_map = {p: i for i, p in enumerate(probs)}
    c_map = {c: i for i, c in enumerate(concepts)}

    n_u = len(users)
    n_p = len(probs)
    n_c = len(concepts)

    pos_rate = float(np.mean([e["is_correct"] for e in train_events])) if train_events else 0.75
    pos_rate = min(max(pos_rate, 0.05), 0.95)
    global_logit = math.log(pos_rate / (1.0 - pos_rate))

    alpha_uk = np.zeros((n_u, n_c), dtype=np.float64)
    theta_u = np.zeros(n_u, dtype=np.float64)
    beta_p = np.zeros(n_p, dtype=np.float64)
    beta_c = np.zeros(n_c, dtype=np.float64)
    raw_disc_p = np.full(n_p, 1.2, dtype=np.float64)
    bloom_offset = np.zeros(7, dtype=np.float64)

    indexed = []
    for e in train_events:
        u_idx = u_map.get(e["user_id"])
        p_idx = p_map.get(e["problem_id"])
        if u_idx is None or p_idx is None:
            continue
        c_idxs = [c_map[c] for c in links.get(e["problem_id"], []) if c in c_map]
        b_lvl = int(problems.get(e["problem_id"], {}).get("cognitive_dimension") or 0)
        b_lvl = b_lvl if 1 <= b_lvl <= 6 else 0
        indexed.append((u_idx, p_idx, c_idxs, b_lvl, float(e["is_correct"])))

    order = np.arange(len(indexed))
    for epoch in range(epochs):
        rng.shuffle(order)
        step_lr = lr / (1.0 + 0.08 * epoch)
        for idx in order:
            u_idx, p_idx, c_idxs, b_lvl, y = indexed[idx]
            disc = math.log1p(math.exp(raw_disc_p[p_idx]))
            d_sig_disc = 1.0 / (1.0 + math.exp(-raw_disc_p[p_idx]))

            if c_idxs:
                m_uk = _sigmoid(alpha_uk[u_idx, c_idxs] + 0.5 * theta_u[u_idx])
                d_pk = _sigmoid(beta_p[p_idx] + 0.5 * beta_c[c_idxs])
                diff_gap = float(np.mean(m_uk - d_pk))
            else:
                m_uk = np.array([_sigmoid(theta_u[u_idx])])
                d_pk = np.array([_sigmoid(beta_p[p_idx])])
                diff_gap = float(m_uk[0] - d_pk[0])

            z = global_logit + 4.0 * disc * diff_gap - bloom_offset[b_lvl]
            pred = 1.0 / (1.0 + math.exp(-max(min(z, 20.0), -20.0)))
            err = pred - y

            grad_disc = err * 4.0 * diff_gap * d_sig_disc + l2 * (raw_disc_p[p_idx] - 1.2)
            raw_disc_p[p_idx] = float(np.clip(raw_disc_p[p_idx] - step_lr * grad_disc, -1.5, 3.5))

            if b_lvl > 0:
                bloom_offset[b_lvl] += step_lr * (err - l2 * bloom_offset[b_lvl]) * 0.25

            if c_idxs:
                k_len = len(c_idxs)
                grad_m = (err * 4.0 * disc / k_len) * (m_uk * (1.0 - m_uk))
                grad_d = -(err * 4.0 * disc / k_len) * (d_pk * (1.0 - d_pk))

                alpha_uk[u_idx, c_idxs] -= step_lr * (grad_m + l2 * alpha_uk[u_idx, c_idxs])
                theta_u[u_idx] -= step_lr * (0.5 * float(np.sum(grad_m)) + l2 * theta_u[u_idx])
                beta_p[p_idx] -= step_lr * (float(np.sum(grad_d)) + l2 * beta_p[p_idx])
                beta_c[c_idxs] -= step_lr * (0.5 * grad_d + l2 * beta_c[c_idxs])
            else:
                dm = float(m_uk[0] * (1.0 - m_uk[0]))
                dd = float(d_pk[0] * (1.0 - d_pk[0]))
                theta_u[u_idx] -= step_lr * (err * 4.0 * disc * dm + l2 * theta_u[u_idx])
                beta_p[p_idx] -= step_lr * (-err * 4.0 * disc * dd + l2 * beta_p[p_idx])

    return {
        "global_logit": round(global_logit, 6),
        "online_lr": 0.35,
        "bloom_offset": {str(i): round(float(bloom_offset[i]), 6) for i in range(1, 7)},
        "problem_difficulty": {p: round(float(beta_p[p_map[p]]), 6) for p in probs},
        "problem_discrimination": {p: round(float(math.log1p(math.exp(raw_disc_p[p_map[p]]))), 6) for p in probs},
        "concept_difficulty": {c: round(float(beta_c[c_map[c]]), 6) for c in concepts},
    }


def ncdm_init_student() -> dict:
    """Initialize a student's cognitive diagnosis state for online warm-start tracking."""
    return {"theta": 0.0, "alpha": {}}


def ncdm_concept_mastery(state: dict, cid: str) -> float:
    """Return latent mastery probability in (0, 1) for a concept under NCDM."""
    logit = state["alpha"].get(cid, 0.0) + 0.5 * state["theta"]
    return 1.0 / (1.0 + math.exp(-max(min(logit, 20.0), -20.0)))


def ncdm_predict(state: dict, problem_id: str, cids: list[str], bloom_level: int | str | None, model: dict) -> float:
    """Predict probability of a correct response using NCDM item difficulty & discrimination."""
    global_logit = model.get("global_logit", 0.0)
    disc = model.get("problem_discrimination", {}).get(problem_id, 1.3)
    b_p = model.get("problem_difficulty", {}).get(problem_id, 0.0)
    c_diffs = model.get("concept_difficulty", {})
    b_off = model.get("bloom_offset", {}).get(str(bloom_level or ""), 0.0)

    if cids:
        gaps = []
        for cid in cids:
            m_uk = ncdm_concept_mastery(state, cid)
            d_pk = 1.0 / (1.0 + math.exp(-max(min(b_p + 0.5 * c_diffs.get(cid, 0.0), 20.0), -20.0)))
            gaps.append(m_uk - d_pk)
        diff_gap = sum(gaps) / len(gaps)
    else:
        m_u = 1.0 / (1.0 + math.exp(-max(min(state["theta"], 20.0), -20.0)))
        d_p = 1.0 / (1.0 + math.exp(-max(min(b_p, 20.0), -20.0)))
        diff_gap = m_u - d_p

    z = global_logit + 4.0 * disc * diff_gap - b_off
    return 1.0 / (1.0 + math.exp(-max(min(z, 20.0), -20.0)))


def ncdm_update(state: dict, problem_id: str, cids: list[str], bloom_level: int | str | None,
                correct: int, model: dict) -> None:
    """Online gradient / Bayesian mastery state update after observing a student's response."""
    pred = ncdm_predict(state, problem_id, cids, bloom_level, model)
    err = pred - float(correct)
    disc = model.get("problem_discrimination", {}).get(problem_id, 1.3)
    lr = model.get("online_lr", 0.35)

    if cids:
        k_len = len(cids)
        total_grad_m = 0.0
        for cid in cids:
            m_uk = ncdm_concept_mastery(state, cid)
            grad_m = (err * 4.0 * disc / k_len) * (m_uk * (1.0 - m_uk))
            state["alpha"][cid] = max(min(state["alpha"].get(cid, 0.0) - lr * grad_m, 6.0), -6.0)
            total_grad_m += grad_m
        state["theta"] = max(min(state["theta"] - lr * 0.4 * total_grad_m, 5.0), -5.0)
    else:
        m_u = 1.0 / (1.0 + math.exp(-max(min(state["theta"], 20.0), -20.0)))
        grad_theta = err * 4.0 * disc * (m_u * (1.0 - m_u))
        state["theta"] = max(min(state["theta"] - lr * 0.4 * grad_theta, 5.0), -5.0)
