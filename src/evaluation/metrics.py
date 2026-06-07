from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def classification_metrics(
    y_true: list[int],
    y_pred: list[int],
) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def correlation_metrics(
    human_scores: list[float],
    predicted_scores: list[float],
) -> dict[str, float]:
    rho, p_value = spearmanr(human_scores, predicted_scores)
    return {
        "spearman_rho": float(0.0 if np.isnan(rho) else rho),
        "spearman_p_value": float(1.0 if np.isnan(p_value) else p_value),
    }


def ranking_metrics(
    rows: pd.DataFrame,
    group_col: str = "jd_id",
    score_col: str = "final_score",
    relevance_col: str = "human_label",
    k: int = 10,
) -> dict[str, float]:
    grouped = rows.groupby(group_col)
    precision_values = []
    recall_values = []
    ndcg_values = []
    mrr_values = []

    for _, group in grouped:
        ranked = group.sort_values(score_col, ascending=False).head(k)
        relevant_total = int((group[relevance_col] > 0).sum())
        relevant_at_k = int((ranked[relevance_col] > 0).sum())

        precision_values.append(relevant_at_k / max(1, len(ranked)))
        recall_values.append(relevant_at_k / max(1, relevant_total))
        ndcg_values.append(ndcg_at_k(ranked[relevance_col].tolist(), k))
        mrr_values.append(reciprocal_rank(ranked[relevance_col].tolist()))

    return {
        f"precision_at_{k}": float(np.mean(precision_values)) if precision_values else 0.0,
        f"recall_at_{k}": float(np.mean(recall_values)) if recall_values else 0.0,
        f"ndcg_at_{k}": float(np.mean(ndcg_values)) if ndcg_values else 0.0,
        "mrr": float(np.mean(mrr_values)) if mrr_values else 0.0,
    }


def ndcg_at_k(relevance: list[float], k: int) -> float:
    relevance = np.asarray(relevance[:k], dtype=float)
    if relevance.size == 0:
        return 0.0
    discounts = np.log2(np.arange(2, relevance.size + 2))
    dcg = np.sum((2**relevance - 1) / discounts)
    ideal = np.sort(relevance)[::-1]
    idcg = np.sum((2**ideal - 1) / discounts)
    return float(dcg / idcg) if idcg > 0 else 0.0


def reciprocal_rank(relevance: list[float]) -> float:
    for index, value in enumerate(relevance, start=1):
        if value > 0:
            return 1.0 / index
    return 0.0
