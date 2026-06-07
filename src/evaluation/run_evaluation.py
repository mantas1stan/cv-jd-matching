from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.evaluation.metrics import classification_metrics, correlation_metrics, ranking_metrics


def evaluate_predictions(
    predictions_path: str | Path,
    labels_path: str | Path,
    output_path: str | Path | None = None,
) -> dict[str, float]:
    predictions = pd.read_csv(predictions_path)
    labels = pd.read_csv(labels_path)
    rows = predictions.merge(labels, on=["cv_id", "jd_id"], how="inner")

    if rows.empty:
        raise ValueError("No prediction rows matched the labelled CV-JD pairs.")

    rows["predicted_label"] = rows["final_score"].apply(score_to_label)
    metrics = {}
    metrics.update(classification_metrics(rows["human_label"].astype(int), rows["predicted_label"]))

    if "human_score" in rows.columns:
        metrics.update(correlation_metrics(rows["human_score"], rows["final_score"]))

    metrics.update(ranking_metrics(rows, k=10))

    if output_path:
        pd.DataFrame([metrics]).to_csv(output_path, index=False)
    return metrics


def score_to_label(score: float) -> int:
    if score >= 0.80:
        return 3
    if score >= 0.60:
        return 2
    if score >= 0.35:
        return 1
    return 0
