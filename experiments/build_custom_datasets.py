from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
import re
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.project_paths import DATASETS_GENERATED_DIR, DATASETS_ORIGINAL_DIR, PROJECT_ROOT

JD_OUTPUT = DATASETS_GENERATED_DIR / "custom_jd_dataset.csv"
RESUME_OUTPUT = DATASETS_GENERATED_DIR / "custom_resume_dataset.csv"
REPORT_OUTPUT = DATASETS_GENERATED_DIR / "custom_dataset_selection_report.md"


@dataclass(frozen=True)
class TextQualityConfig:
    min_words: int
    target_words: int
    max_words: int
    marker_terms: tuple[str, ...]


JD_QUALITY = TextQualityConfig(
    min_words=80,
    target_words=450,
    max_words=1200,
    marker_terms=(
        "requirements",
        "responsibilities",
        "qualifications",
        "experience",
        "skills",
        "preferred",
        "benefits",
        "duties",
        "must",
        "required",
        "education",
    ),
)

TITLE_JD_QUALITY = TextQualityConfig(
    min_words=60,
    target_words=420,
    max_words=1200,
    marker_terms=JD_QUALITY.marker_terms,
)

RESUME_QUALITY = TextQualityConfig(
    min_words=180,
    target_words=850,
    max_words=3000,
    marker_terms=(
        "experience",
        "skills",
        "education",
        "summary",
        "projects",
        "certifications",
        "responsibilities",
        "accomplishments",
        "work experience",
    ),
)


def main() -> None:
    jd_frames = [
        prepare_monster_jobs(DATASETS_ORIGINAL_DIR / "monster_com-job_sample.csv"),
        prepare_title_jobs(DATASETS_ORIGINAL_DIR / "job_title_des.csv"),
    ]
    resume_frames = [
        prepare_resume_csv(DATASETS_ORIGINAL_DIR / "Resume.csv"),
        prepare_resume_data_csv(DATASETS_ORIGINAL_DIR / "Resume data.csv"),
    ]

    selected_monster = select_diverse(jd_frames[0], 2000, "occupation_group")
    selected_title_jobs = select_diverse(jd_frames[1], 2000, "job_title")
    custom_jds = pd.concat([selected_monster, selected_title_jobs], ignore_index=True)
    custom_jds = custom_jds.sort_values(["source_dataset", "occupation_group", "quality_score"], ascending=[True, True, False])
    custom_jds.insert(0, "custom_jd_id", [f"jd_custom_{idx:05d}" for idx in range(1, len(custom_jds) + 1)])

    selected_resume_csv = select_diverse(resume_frames[0], 3000, "resume_group")
    remaining_resume_target = 6000 - len(selected_resume_csv)
    selected_resume_data = select_diverse(resume_frames[1], remaining_resume_target, "resume_group")
    custom_resumes = pd.concat([selected_resume_csv, selected_resume_data], ignore_index=True)
    custom_resumes = custom_resumes.sort_values(["source_dataset", "resume_group", "quality_score"], ascending=[True, True, False])
    custom_resumes.insert(
        0,
        "custom_resume_id",
        [f"resume_custom_{idx:05d}" for idx in range(1, len(custom_resumes) + 1)],
    )

    custom_jds.to_csv(JD_OUTPUT, index=False, encoding="utf-8")
    custom_resumes.to_csv(RESUME_OUTPUT, index=False, encoding="utf-8")
    write_report(custom_jds, custom_resumes, resume_frames[0])

    print(f"Wrote {len(custom_jds):,} JDs to {JD_OUTPUT}")
    print(f"Wrote {len(custom_resumes):,} resumes to {RESUME_OUTPUT}")
    print(f"Wrote selection report to {REPORT_OUTPUT}")


def prepare_monster_jobs(path: Path) -> pd.DataFrame:
    columns = [
        "uniq_id",
        "job_title",
        "job_description",
        "sector",
        "job_type",
        "location",
        "organization",
        "page_url",
    ]
    df = pd.read_csv(path, usecols=columns)
    df = df.rename(
        columns={
            "uniq_id": "source_id",
            "job_title": "job_title",
            "job_description": "description",
        }
    )
    df["source_dataset"] = "monster_com-job_sample.csv"
    df["source_category"] = df["sector"].fillna("Unknown").astype(str)
    df["occupation_group"] = df.apply(
        lambda row: infer_job_group(
            " ".join(
                [
                    str(row.get("job_title", "")),
                    str(row.get("sector", "")),
                    str(row.get("description", ""))[:1500],
                ]
            )
        ),
        axis=1,
    )
    df = score_and_filter_text(df, "description", JD_QUALITY)
    return df[
        [
            "source_dataset",
            "source_id",
            "job_title",
            "description",
            "occupation_group",
            "source_category",
            "quality_score",
            "word_count",
            "job_type",
            "location",
            "organization",
            "page_url",
        ]
    ]


def prepare_title_jobs(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.rename(
        columns={
            "Unnamed: 0": "source_id",
            "Job Title": "job_title",
            "Job Description": "description",
        }
    )
    df["source_dataset"] = "job_title_des.csv"
    df["source_category"] = df["job_title"]
    df["occupation_group"] = "Technology"
    df["job_type"] = ""
    df["location"] = ""
    df["organization"] = ""
    df["page_url"] = ""
    df = score_and_filter_text(df, "description", TITLE_JD_QUALITY)
    return df[
        [
            "source_dataset",
            "source_id",
            "job_title",
            "description",
            "occupation_group",
            "source_category",
            "quality_score",
            "word_count",
            "job_type",
            "location",
            "organization",
            "page_url",
        ]
    ]


def prepare_resume_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, usecols=["ID", "Resume_str", "Category"])
    df = df.rename(columns={"ID": "source_id", "Resume_str": "resume_text", "Category": "source_category"})
    df["source_dataset"] = "Resume.csv"
    df["resume_group"] = df["source_category"].fillna("Unknown").astype(str).str.replace("-", " ", regex=False).str.title()
    df = score_and_filter_text(df, "resume_text", RESUME_QUALITY)
    return df[
        [
            "source_dataset",
            "source_id",
            "resume_text",
            "resume_group",
            "source_category",
            "quality_score",
            "word_count",
        ]
    ]


def prepare_resume_data_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, usecols=["Resume", "Label"])
    df = df.reset_index(names="source_id")
    df = df.rename(columns={"Resume": "resume_text", "Label": "source_category"})
    df["source_dataset"] = "Resume data.csv"
    df["resume_group"] = df["source_category"].fillna("Unknown").apply(infer_resume_group)
    df = score_and_filter_text(df, "resume_text", RESUME_QUALITY)
    return df[
        [
            "source_dataset",
            "source_id",
            "resume_text",
            "resume_group",
            "source_category",
            "quality_score",
            "word_count",
        ]
    ]


def score_and_filter_text(
    df: pd.DataFrame,
    text_col: str,
    config: TextQualityConfig,
) -> pd.DataFrame:
    df = df.copy()
    df[text_col] = df[text_col].fillna("").astype(str).map(clean_text)
    df["normalized_text"] = df[text_col].map(normalize_text)
    df = df[df["normalized_text"].str.len() > 0]
    df = df.drop_duplicates("normalized_text")

    df["word_count"] = df[text_col].map(lambda value: len(value.split()))
    df = df[df["word_count"].between(config.min_words, config.max_words)]
    df["quality_score"] = df[text_col].map(lambda value: quality_score(value, config))
    df = df.drop(columns=["normalized_text"])
    return df.sort_values("quality_score", ascending=False).reset_index(drop=True)


def select_diverse(df: pd.DataFrame, target_n: int, group_col: str) -> pd.DataFrame:
    if len(df) <= target_n:
        selected = df.copy()
        selected["selection_note"] = "all_available_unique_rows_below_target"
        return selected

    groups = {
        group: group_df.sort_values("quality_score", ascending=False).copy()
        for group, group_df in df.groupby(group_col, dropna=False)
    }
    group_names = sorted(groups, key=lambda group: (-min(len(groups[group]), target_n), str(group)))
    per_group_cap = max(1, math.ceil(target_n / max(1, len(groups))) * 2)

    selected_parts = []
    selected_indexes: set[int] = set()
    made_progress = True
    while len(selected_indexes) < target_n and made_progress:
        made_progress = False
        for group in group_names:
            if len(selected_indexes) >= target_n:
                break
            already_from_group = sum(1 for part in selected_parts if part[group_col] == group)
            if already_from_group >= per_group_cap:
                continue

            group_df = groups[group]
            candidates = group_df[~group_df.index.isin(selected_indexes)]
            if candidates.empty:
                continue
            row = candidates.iloc[0]
            selected_parts.append(row)
            selected_indexes.add(int(row.name))
            made_progress = True

    if len(selected_indexes) < target_n:
        remaining = df[~df.index.isin(selected_indexes)].sort_values("quality_score", ascending=False)
        for _, row in remaining.head(target_n - len(selected_indexes)).iterrows():
            selected_parts.append(row)
            selected_indexes.add(int(row.name))

    selected = pd.DataFrame(selected_parts).reset_index(drop=True)
    selected["selection_note"] = "quality_ranked_balanced_by_" + group_col
    return selected


def quality_score(text: str, config: TextQualityConfig) -> float:
    words = re.findall(r"[A-Za-z0-9+#.]+", text)
    word_count = len(words)
    if word_count == 0:
        return 0.0

    lower = text.lower()
    unique_ratio = len(set(word.lower() for word in words)) / word_count
    marker_score = min(1.0, sum(1 for term in config.marker_terms if term in lower) / 5)
    structure_score = min(1.0, (text.count("\n") + text.count("•") + text.count("- ")) / 12)

    if word_count <= config.target_words:
        length_score = word_count / config.target_words
    else:
        length_score = max(0.0, 1.0 - ((word_count - config.target_words) / config.max_words))

    score = (
        0.45 * length_score
        + 0.25 * marker_score
        + 0.20 * min(1.0, unique_ratio * 2)
        + 0.10 * structure_score
    )
    return round(float(score), 6)


def infer_job_group(text: str) -> str:
    normalized = normalize_text(text)
    rules = [
        ("Technology", ("software", "developer", "programmer", "database", "network", "it ", "systems", "web", "data", "cloud", "security analyst")),
        ("Healthcare", ("medical", "health", "nurse", "clinical", "patient", "dental", "pharmacy", "therapist")),
        ("Finance", ("finance", "accounting", "accountant", "bank", "mortgage", "insurance", "payroll", "tax")),
        ("Sales", ("sales", "retail", "account executive", "business development", "store")),
        ("Marketing", ("marketing", "brand", "product marketing", "public relations", "social media")),
        ("Operations", ("operations", "program management", "project management", "business strategic", "manager")),
        ("Manufacturing", ("manufacturing", "production", "quality", "warehouse", "assembly")),
        ("Logistics", ("logistics", "transportation", "driver", "shipping", "supply chain")),
        ("Customer Support", ("customer support", "client care", "call center", "customer service")),
        ("Hospitality", ("food", "hospitality", "restaurant", "chef", "hotel")),
        ("Engineering", ("engineering", "mechanical", "electrical", "civil", "structural")),
        ("Construction", ("construction", "skilled trades", "building")),
        ("Education", ("education", "training", "teacher", "instruction")),
        ("Legal", ("legal", "paralegal", "attorney", "law")),
        ("Human Resources", ("human resources", "recruiting", "recruiter", "hr ")),
        ("Administrative", ("administrative", "clerical", "data entry", "office assistant")),
        ("Science", ("science", "biotech", "laboratory", "r&d", "research")),
        ("Security", ("protective services", "security guard", "public safety")),
        ("Creative Media", ("creative", "design", "editorial", "writing", "media")),
    ]
    for group, keywords in rules:
        if any(keyword in normalized for keyword in keywords):
            return group
    return "Other"


def infer_resume_group(label: str) -> str:
    normalized = normalize_text(str(label).replace("_", " "))
    rules = [
        ("Data and Database", ("database", "data analyst", "data scientist", "business intelligence")),
        ("Software Development", ("software", "developer", "java", "python", "web", "front end", "backend", "full stack", "ios")),
        ("Infrastructure and Systems", ("systems", "network", "administrator", "devops", "cloud")),
        ("Cybersecurity", ("security", "cyber")),
        ("Project Management", ("project manager", "program manager", "scrum", "product manager")),
    ]
    for group, keywords in rules:
        if any(keyword in normalized for keyword in keywords):
            return group
    return "Other Technology"


def clean_text(text: str) -> str:
    text = str(text).replace("\u00a0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def normalize_text(text: str) -> str:
    text = clean_text(text).lower()
    text = re.sub(r"[^a-z0-9+#.]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def write_report(jds: pd.DataFrame, resumes: pd.DataFrame, prepared_resume_csv: pd.DataFrame) -> None:
    resume_csv_available = len(prepared_resume_csv)
    lines = [
        "# Custom Dataset Selection Report",
        "",
        "Generated by `experiments/build_custom_datasets.py`.",
        "",
        "## Output Files",
        "",
        f"- `{JD_OUTPUT.name}`: {len(jds):,} job descriptions.",
        f"- `{RESUME_OUTPUT.name}`: {len(resumes):,} resumes.",
        "",
        "## Selection Rules",
        "",
        "- Removed empty text and exact duplicate normalized text.",
        "- Kept informative rows using word-count thresholds and a text quality score.",
        "- Quality score rewards sufficient length, requirement/experience/skills markers, vocabulary variety, and visible structure.",
        "- Selected rows by quality-ranked round-robin across occupation/category groups.",
        "",
        "## Important Constraint",
        "",
        (
            f"`Resume.csv` only has {resume_csv_available:,} usable unique resumes after cleaning, "
            "so a unique 3,000-row sample from that source is not possible. "
            "The custom resume dataset keeps the 6,000 total target by using all usable rows from "
            "`Resume.csv` and filling the remainder from `Resume data.csv`."
        ),
        "",
        "## JD Source Counts",
        "",
        dataframe_to_markdown(jds["source_dataset"].value_counts().rename_axis("source_dataset").reset_index(name="rows")),
        "",
        "## JD Occupation Groups",
        "",
        dataframe_to_markdown(jds["occupation_group"].value_counts().rename_axis("occupation_group").reset_index(name="rows")),
        "",
        "## Resume Source Counts",
        "",
        dataframe_to_markdown(resumes["source_dataset"].value_counts().rename_axis("source_dataset").reset_index(name="rows")),
        "",
        "## Resume Groups",
        "",
        dataframe_to_markdown(resumes["resume_group"].value_counts().rename_axis("resume_group").reset_index(name="rows")),
        "",
    ]
    REPORT_OUTPUT.write_text("\n".join(lines), encoding="utf-8")


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    try:
        return df.to_markdown(index=False)
    except ImportError:
        return df.to_string(index=False)


if __name__ == "__main__":
    main()
