from __future__ import annotations

import numpy as np


class IdentityCalibrator:
    """Default calibrator used before labelled CV-JD data is available."""

    def fit(self, scores: list[float], labels: list[float]) -> "IdentityCalibrator":
        return self

    def predict(self, scores: list[float]) -> list[float]:
        return [float(min(1.0, max(0.0, score))) for score in scores]


class QuantileCalibrator:
    """Small non-parametric calibrator for mapping raw scores to empirical quantiles."""

    def __init__(self) -> None:
        self._sorted_scores: np.ndarray | None = None

    def fit(self, scores: list[float], labels: list[float] | None = None) -> "QuantileCalibrator":
        if not scores:
            raise ValueError("Cannot calibrate without scores.")
        self._sorted_scores = np.sort(np.asarray(scores, dtype=float))
        return self

    def predict(self, scores: list[float]) -> list[float]:
        if self._sorted_scores is None:
            raise RuntimeError("Calibrator must be fitted before prediction.")
        ranks = np.searchsorted(self._sorted_scores, np.asarray(scores, dtype=float), side="right")
        calibrated = ranks / len(self._sorted_scores)
        return [float(min(1.0, max(0.0, score))) for score in calibrated]
