from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.project_paths import DATASETS_GENERATED_DIR, DATASETS_ORIGINAL_DIR, PROJECT_ROOT


LABELLED_DIR = PROJECT_ROOT / "data" / "labelled"

NORMALIZED_CVS = DATASETS_GENERATED_DIR / "normalized_resumes.csv"
NORMALIZED_JDS = DATASETS_GENERATED_DIR / "normalized_jds.csv"
PAIR_SCORES = DATASETS_GENERATED_DIR / "custom_cv_jd_pairs_scored.csv"

CANDIDATE_OUTPUT = LABELLED_DIR / "candidate_pairs_1000.csv"
HUMAN_REVIEW_OUTPUT = LABELLED_DIR / "human_review_sample_100.csv"
LLM_LABEL_OUTPUT = LABELLED_DIR / "llm_labels_sample_100.csv"


FIT_BANDS = [
    (0, 20, "poor fit"),
    (21, 40, "weak fit"),
    (41, 60, "possible fit"),
    (61, 80, "good fit"),
    (81, 100, "strong fit"),
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Create candidate CV-JD pairs for LLM/human labelling.")
    parser.add_argument("--jd-count", type=int, default=50)
    parser.add_argument("--pairs-per-jd", type=int, default=20)
    parser.add_argument("--human-review-count", type=int, default=100)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--run-llm", action="store_true")
    parser.add_argument("--llm-limit", type=int, default=0, help="0 means label all selected candidate pairs.")
    parser.add_argument(
        "--llm-source",
        choices=["human_review", "candidates"],
        default="human_review",
        help="Rows to send to the LLM when --run-llm is enabled.",
    )
    parser.add_argument("--llm-output-path", default=str(LLM_LABEL_OUTPUT))
    parser.add_argument("--ollama-model", default="llama3.2:1b")
    parser.add_argument("--ollama-url", default="http://localhost:11434")
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    args = parser.parse_args()

    LABELLED_DIR.mkdir(parents=True, exist_ok=True)
    cvs = pd.read_csv(NORMALIZED_CVS)
    jds = pd.read_csv(NORMALIZED_JDS)
    pairs = pd.read_csv(PAIR_SCORES)

    candidate_pairs = build_candidate_pairs(
        pairs=pairs,
        jds=jds,
        jd_count=args.jd_count,
        pairs_per_jd=args.pairs_per_jd,
    )
    candidate_pairs = add_text_and_metadata(candidate_pairs, cvs, jds)
    candidate_pairs = add_empty_human_columns(candidate_pairs)
    candidate_pairs.to_csv(CANDIDATE_OUTPUT, index=False, encoding="utf-8")

    human_review = sample_human_review(candidate_pairs, args.human_review_count, args.random_seed)
    human_review.to_csv(HUMAN_REVIEW_OUTPUT, index=False, encoding="utf-8")

    print(f"Wrote {len(candidate_pairs):,} candidate pairs to {CANDIDATE_OUTPUT}")
    print(f"Wrote {len(human_review):,} human review rows to {HUMAN_REVIEW_OUTPUT}")

    if args.run_llm:
        llm_source = human_review if args.llm_source == "human_review" else candidate_pairs
        label_count = len(llm_source) if args.llm_limit <= 0 else min(args.llm_limit, len(llm_source))
        llm_labels = label_with_ollama(
            llm_source.head(label_count),
            model=args.ollama_model,
            base_url=args.ollama_url,
            sleep_seconds=args.sleep_seconds,
        )
        llm_output_path = Path(args.llm_output_path)
        llm_output_path.parent.mkdir(parents=True, exist_ok=True)
        llm_labels.to_csv(llm_output_path, index=False, encoding="utf-8")
        print(f"Wrote {len(llm_labels):,} LLM labels to {llm_output_path}")


def build_candidate_pairs(
    pairs: pd.DataFrame,
    jds: pd.DataFrame,
    jd_count: int,
    pairs_per_jd: int,
) -> pd.DataFrame:
    jd_ids_with_pairs = set(pairs["jd_id"])
    eligible_jds = jds[jds["jd_id"].isin(jd_ids_with_pairs)].copy()
    selected_jds = select_diverse_jds(eligible_jds, jd_count)
    selected = pairs[pairs["jd_id"].isin(selected_jds["jd_id"])].copy()

    selected = (
        selected.sort_values(["jd_id", "pair_type", "selection_rank"])
        .groupby("jd_id", group_keys=False)
        .head(pairs_per_jd)
        .reset_index(drop=True)
    )
    selected = selected.rename(columns={"pair_type": "candidate_bucket"})
    selected.insert(0, "candidate_pair_id", [f"cand_{idx:05d}" for idx in range(1, len(selected) + 1)])
    return selected


def select_diverse_jds(jds: pd.DataFrame, jd_count: int) -> pd.DataFrame:
    groups = {
        group: group_df.sort_values(["source_dataset", "quality_score"], ascending=[True, False])
        for group, group_df in jds.groupby("occupation_group")
    }
    selected_indexes: list[int] = []
    while len(selected_indexes) < jd_count:
        made_progress = False
        for group in sorted(groups):
            group_df = groups[group]
            remaining = group_df[~group_df.index.isin(selected_indexes)]
            if remaining.empty:
                continue
            selected_indexes.append(int(remaining.index[0]))
            made_progress = True
            if len(selected_indexes) >= jd_count:
                break
        if not made_progress:
            break
    return jds.loc[selected_indexes].reset_index(drop=True)


def add_text_and_metadata(pairs: pd.DataFrame, cvs: pd.DataFrame, jds: pd.DataFrame) -> pd.DataFrame:
    cv_cols = ["cv_id", "cv_text", "resume_group", "resume_category"]
    jd_cols = ["jd_id", "jd_text", "job_title", "occupation_group", "jd_category"]
    output = pairs.merge(cvs[cv_cols], on="cv_id", how="left")
    output = output.merge(jds[jd_cols], on="jd_id", how="left")
    output = output.rename(
        columns={
            "resume_group": "cv_group_full",
            "resume_category": "cv_category_full",
            "occupation_group": "jd_group_full",
            "jd_category": "jd_category_full",
        }
    )
    return output


def add_empty_human_columns(rows: pd.DataFrame) -> pd.DataFrame:
    rows = rows.copy()
    rows["llm_fit_score"] = np.nan
    rows["llm_fit_category"] = ""
    rows["llm_missing_requirements"] = ""
    rows["llm_explanation"] = ""
    rows["human_reviewed"] = False
    rows["human_score"] = np.nan
    rows["human_category"] = ""
    rows["human_notes"] = ""
    rows["final_label_score"] = np.nan
    rows["final_label_category"] = ""
    return rows


def sample_human_review(rows: pd.DataFrame, n: int, random_seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(random_seed)
    buckets = sorted(rows["candidate_bucket"].dropna().unique())
    per_bucket = max(1, n // max(1, len(buckets)))
    selected_parts = []
    selected_indexes: set[int] = set()

    for bucket in buckets:
        bucket_rows = rows[rows["candidate_bucket"] == bucket]
        sample_n = min(per_bucket, len(bucket_rows))
        sampled = bucket_rows.sample(n=sample_n, random_state=random_seed)
        selected_parts.append(sampled)
        selected_indexes.update(sampled.index.tolist())

    remaining_n = n - sum(len(part) for part in selected_parts)
    if remaining_n > 0:
        remaining = rows[~rows.index.isin(selected_indexes)]
        if len(remaining) > 0:
            sampled_indexes = rng.choice(remaining.index.to_numpy(), size=min(remaining_n, len(remaining)), replace=False)
            selected_parts.append(remaining.loc[sampled_indexes])

    human_review = pd.concat(selected_parts, ignore_index=True)
    human_review["human_reviewed"] = ""
    return human_review.head(n)


def label_with_ollama(
    rows: pd.DataFrame,
    model: str,
    base_url: str,
    sleep_seconds: float,
) -> pd.DataFrame:
    labels = []
    for idx, row in enumerate(rows.itertuples(index=False), start=1):
        print(f"LLM labelling {idx:,}/{len(rows):,}: {row.candidate_pair_id}")
        parsed = call_ollama_label(
            cv_text=row.cv_text,
            jd_text=row.jd_text,
            model=model,
            base_url=base_url,
        )
        score = clamp_score(parsed.get("score", parsed.get("fit_score", 0)))
        labels.append(
            {
                "candidate_pair_id": row.candidate_pair_id,
                "cv_id": row.cv_id,
                "jd_id": row.jd_id,
                "candidate_bucket": row.candidate_bucket,
                "llm_fit_score": score,
                "llm_fit_category": parsed.get("fit_category") or score_to_category(score),
                "llm_missing_requirements": normalize_list_text(parsed.get("missing_requirements", "")),
                "llm_explanation": str(parsed.get("short_explanation", parsed.get("explanation", "")))[:1000],
                "llm_model": model,
            }
        )
        if sleep_seconds:
            time.sleep(sleep_seconds)
    return pd.DataFrame(labels)


def call_ollama_label(cv_text: str, jd_text: str, model: str, base_url: str) -> dict:
    payload = {
        "model": model,
        "stream": False,
        "format": "json",
        "prompt": build_label_prompt(cv_text[:5500], jd_text[:5500]),
        "options": {"temperature": 0, "num_predict": 800},
    }
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/api/generate",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            raw = json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Could not connect to Ollama at {base_url}.") from exc
    return parse_json_object(raw.get("response", ""))


def build_label_prompt(cv_text: str, jd_text: str) -> str:
    return f"""
You are creating labels for a CV-job description matching dataset.

Judge the candidate's fit for the job from 0 to 100 using this rubric:
- 0-20: poor fit. CV is unrelated or misses most must-have requirements.
- 21-40: weak fit. Some domain overlap, but major must-have gaps.
- 41-60: possible fit. Partial match, several important gaps remain.
- 61-80: good fit. Most must-have requirements are supported by CV evidence.
- 81-100: strong fit. Clear evidence for nearly all major requirements.

Consider required skills, years of experience, seniority, domain experience,
education/certifications, and missing must-have requirements.

Return only valid JSON:
{{
  "score": number from 0 to 100,
  "fit_category": "poor fit|weak fit|possible fit|good fit|strong fit",
  "missing_requirements": ["short missing item 1", "short missing item 2"],
  "short_explanation": "one or two sentences explaining the score"
}}

Do not give 80 by default. Use the full 0-100 scale.

JOB DESCRIPTION:
{jd_text}

CV:
{cv_text}
""".strip()


def parse_json_object(content: str) -> dict:
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", content, flags=re.DOTALL)
        if not match:
            return {"score": 0, "fit_category": "poor fit", "missing_requirements": [], "short_explanation": content[:500]}
        return json.loads(match.group(0))


def clamp_score(value: object) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        score = 0.0
    return round(max(0.0, min(100.0, score)), 2)


def score_to_category(score: float) -> str:
    for low, high, label in FIT_BANDS:
        if low <= score <= high:
            return label
    return "strong fit"


def normalize_list_text(value: object) -> str:
    if isinstance(value, list):
        return "; ".join(str(item) for item in value)
    return str(value)


if __name__ == "__main__":
    main()
