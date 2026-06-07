from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

try:
    import yaml
except ImportError:  # pragma: no cover - optional dependency guard
    yaml = None

from src.scoring.score_schema import ScoreBreakdown


DEFAULT_WEIGHTS = {
    "must_have_requirement_coverage": 0.45,
    "experience_or_seniority_match": 0.20,
    "responsibility_alignment": 0.15,
    "nice_to_have_coverage": 0.10,
    "domain_context_alignment": 0.10,
}


@dataclass(frozen=True)
class ScoreWeights:
    weights: dict[str, float]

    @classmethod
    def default(cls) -> "ScoreWeights":
        return cls(dict(DEFAULT_WEIGHTS))

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ScoreWeights":
        if yaml is None:
            raise RuntimeError("PyYAML is required to load weighting configuration.")

        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        raw_weights = data.get("weights", data)
        return cls(normalize_weights(raw_weights))

    def score(self, breakdown: ScoreBreakdown) -> float:
        total = 0.0
        for component, weight in self.weights.items():
            value = getattr(breakdown, component)
            if value is not None:
                total += weight * value
        return round(max(0.0, min(1.0, total)), 6)


def normalize_weights(weights: Mapping[str, float]) -> dict[str, float]:
    clean = {key: float(value) for key, value in weights.items() if float(value) >= 0.0}
    total = sum(clean.values())
    if total <= 0.0:
        raise ValueError("At least one scoring weight must be positive.")
    return {key: value / total for key, value in clean.items()}
