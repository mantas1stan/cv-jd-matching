"""Matching methods that implement the shared matcher interface."""

from src.matching.base import BaseMatcher, CVDocument, JobDescription
from src.matching.embedding_model_matcher import EmbeddingModelMatcher
from src.matching.llm_matcher import LlmMatcher
from src.matching.tfidf_matcher import TfidfMatcher

__all__ = [
    "BaseMatcher",
    "CVDocument",
    "EmbeddingModelMatcher",
    "JobDescription",
    "LlmMatcher",
    "TfidfMatcher",
]
