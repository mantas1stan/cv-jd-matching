from __future__ import annotations

import json
import re
import urllib.error
import urllib.request

from src.matching.base import BaseMatcher, CVDocument, JobDescription
from src.scoring.score_schema import MatchResult, ScoreBreakdown


class LlmMatcher(BaseMatcher):
    """Ollama-backed LLM evaluator with structured JSON scoring."""

    method_name = "ollama_llm_evaluator"

    def __init__(
        self,
        model_name: str = "llama3.1:8b",
        base_url: str = "http://localhost:11434",
        timeout_seconds: int = 120,
        max_chars: int = 6000,
    ) -> None:
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.max_chars = max_chars

    def match(self, cv: CVDocument, job: JobDescription) -> MatchResult:
        payload = {
            "model": self.model_name,
            "prompt": self._build_prompt(cv.text, job.text),
            "stream": False,
            "format": "json",
            "options": {
                "temperature": 0,
                "num_predict": 700,
            },
        }
        response = self._post_json("/api/generate", payload)
        content = response.get("response", "")
        parsed = parse_llm_json(content)

        breakdown = ScoreBreakdown(
            must_have_requirement_coverage=percent_to_unit(parsed.get("must_have_requirement_coverage", 0)),
            experience_or_seniority_match=percent_to_unit(parsed.get("experience_or_seniority_match", 0)),
            responsibility_alignment=percent_to_unit(parsed.get("responsibility_alignment", 0)),
            nice_to_have_coverage=percent_to_unit(parsed.get("nice_to_have_coverage", 0)),
            domain_context_alignment=percent_to_unit(parsed.get("domain_context_alignment", 0)),
        )
        final_score = percent_to_unit(parsed.get("final_score", 0))
        explanation = str(parsed.get("explanation", "Ollama LLM evaluator."))

        return MatchResult(
            cv_id=cv.cv_id,
            jd_id=job.jd_id,
            method=self.method_name,
            final_score=final_score,
            score_breakdown=breakdown,
            explanation=explanation,
            metadata={"ollama_model": self.model_name},
        )

    def _post_json(self, path: str, payload: dict) -> dict:
        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}{path}",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"Could not connect to Ollama at {self.base_url}. "
                "Start Ollama and make sure the requested model is available."
            ) from exc

    def _build_prompt(self, cv_text: str, jd_text: str) -> str:
        cv_text = cv_text[: self.max_chars]
        jd_text = jd_text[: self.max_chars]
        return f"""
You are evaluating whether a CV satisfies a job description.

Return only valid JSON with these numeric fields from 0 to 100:
- final_score
- must_have_requirement_coverage
- experience_or_seniority_match
- responsibility_alignment
- nice_to_have_coverage
- domain_context_alignment

Also include:
- explanation: one short sentence

Score requirement satisfaction based on evidence in the CV, not generic text similarity.

JOB DESCRIPTION:
{jd_text}

CV:
{cv_text}
""".strip()


def parse_llm_json(content: str) -> dict:
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", content, flags=re.DOTALL)
        if not match:
            raise ValueError(f"LLM did not return JSON: {content[:500]}")
        return json.loads(match.group(0))


def percent_to_unit(value: object) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = 0.0
    if numeric > 1.0:
        numeric /= 100.0
    return round(max(0.0, min(1.0, numeric)), 6)
