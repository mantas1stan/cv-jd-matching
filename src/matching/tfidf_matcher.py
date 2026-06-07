from __future__ import annotations

from collections import Counter
import re

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity

from src.matching.base import BaseMatcher, CVDocument, JobDescription
from src.scoring.score_schema import MatchResult, ScoreBreakdown


class TfidfMatcher(BaseMatcher):
    """Simple lexical baseline using TF-IDF cosine similarity."""

    method_name = "tfidf_cosine"

    def __init__(self, max_features: int = 20000, ngram_range: tuple[int, int] = (1, 2)) -> None:
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            stop_words="english",
            ngram_range=ngram_range,
            max_features=max_features,
            min_df=1,
        )

    def match(self, cv: CVDocument, job: JobDescription) -> MatchResult:
        if not cv.text.strip() or not job.text.strip():
            score = 0.0
        else:
            matrix = self.vectorizer.fit_transform([cv.text, job.text])
            score = float(cosine_similarity(matrix[0], matrix[1])[0, 0])

        score = max(0.0, min(1.0, score))
        common_terms = ", ".join(top_shared_terms(cv.text, job.text, limit=12))
        explanation = "TF-IDF cosine baseline."
        if common_terms:
            explanation = f"{explanation} Top shared terms: {common_terms}."

        return MatchResult(
            cv_id=cv.cv_id,
            jd_id=job.jd_id,
            method=self.method_name,
            final_score=round(score, 6),
            score_breakdown=ScoreBreakdown(
                domain_context_alignment=round(score, 6),
                full_text_similarity=round(score, 6),
            ),
            explanation=explanation,
        )


def top_shared_terms(cv_text: str, jd_text: str, limit: int = 10) -> list[str]:
    token_pattern = re.compile(r"(?u)\b[\w+#.-]{2,}\b")
    stopwords = set(ENGLISH_STOP_WORDS) | {"also", "etc", "job", "work", "team", "candidate"}
    cv_counts = Counter(
        token.lower()
        for token in token_pattern.findall(cv_text)
        if token.lower() not in stopwords
    )
    jd_counts = Counter(
        token.lower()
        for token in token_pattern.findall(jd_text)
        if token.lower() not in stopwords
    )
    shared = cv_counts.keys() & jd_counts.keys()
    ranked = sorted(shared, key=lambda token: np.sqrt(cv_counts[token] * jd_counts[token]), reverse=True)
    return ranked[:limit]
