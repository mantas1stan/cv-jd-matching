from __future__ import annotations

from pathlib import Path

import pandas as pd


def largest_disagreements(
    predictions_path: str | Path,
    labels_path: str | Path,
    n: int = 25,
) -> pd.DataFrame:
    predictions = pd.read_csv(predictions_path)
    labels = pd.read_csv(labels_path)
    rows = predictions.merge(labels, on=["cv_id", "jd_id"], how="inner")

    if "human_score" in rows.columns:
        rows["absolute_error"] = (rows["final_score"] - rows["human_score"]).abs()
    else:
        normalized_label = rows["human_label"].astype(float) / max(1.0, rows["human_label"].max())
        rows["absolute_error"] = (rows["final_score"] - normalized_label).abs()

    return rows.sort_values("absolute_error", ascending=False).head(n)
