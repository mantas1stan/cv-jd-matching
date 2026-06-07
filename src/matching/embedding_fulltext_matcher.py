from __future__ import annotations

import numpy as np

from src.matching.base import BaseMatcher, CVDocument, JobDescription
from src.scoring.score_schema import MatchResult, ScoreBreakdown


DEFAULT_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"


class EmbeddingFullTextMatcher(BaseMatcher):
    """Full-text semantic baseline using sentence-transformer embeddings."""

    method_name = "embedding_fulltext"

    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL, local_files_only: bool = False) -> None:
        self.model_name = model_name
        self.local_files_only = local_files_only
        self._model = None

    def _load_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name, local_files_only=self.local_files_only)
        return self._model

    def match(self, cv: CVDocument, job: JobDescription) -> MatchResult:
        if not cv.text.strip() or not job.text.strip():
            similarity = 0.0
        else:
            model = self._load_model()
            vectors = model.encode([cv.text, job.text], normalize_embeddings=True)
            similarity = float(np.dot(vectors[0], vectors[1]))
            similarity = (similarity + 1.0) / 2.0

        similarity = max(0.0, min(1.0, similarity))
        return MatchResult(
            cv_id=cv.cv_id,
            jd_id=job.jd_id,
            method=self.method_name,
            final_score=round(similarity, 6),
            score_breakdown=ScoreBreakdown(
                domain_context_alignment=round(similarity, 6),
                full_text_similarity=round(similarity, 6),
            ),
            explanation="Full-text sentence embedding cosine baseline.",
            metadata={"embedding_model": self.model_name},
        )
