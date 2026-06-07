from __future__ import annotations

from datetime import datetime
from io import BytesIO
from pathlib import Path
import sys

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loading.cv_reader import make_cv_document
from src.data_loading.jd_reader import make_jd_document
from src.matching.method_runner import (
    DEFAULT_BGE_MODEL,
    DEFAULT_E5_MODEL,
    DEFAULT_FAST_EMBEDDING_MODEL,
    DEFAULT_GTE_MODEL,
    DEFAULT_OLLAMA_MODEL,
    build_current_methods,
    score_pair_all_methods,
)
from src.project_paths import (
    DATASETS_RESULTS_DIR,
    NORMALIZED_CVS,
    NORMALIZED_JDS,
    PAIR_SCORES,
)


REPORTS_DIR = DATASETS_RESULTS_DIR

METHOD_OPTIONS = {
    "tfidf": "TF-IDF cosine",
    "sbert": "SBERT full-text embedding",
    "bge": "BGE full-text embedding",
    "e5": "E5 full-text embedding",
    "gte": "GTE full-text embedding",
    "section": "Section-aware embedding",
    "requirement": "Requirement-level rules",
    "llm": "Ollama LLM evaluator",
}

METHOD_HELP = {
    "tfidf": "Fast lexical baseline. Compares exact and near-exact word/phrase overlap between the CV and JD.",
    "sbert": "Embedding baseline. Converts the full CV and full JD into semantic vectors and compares cosine similarity.",
    "bge": "Alternative embedding model. Same full-text idea as SBERT, but using the BGE model family.",
    "e5": "Alternative embedding model. Uses E5 query/passage formatting for JD-to-CV retrieval-style scoring.",
    "gte": "Alternative embedding model. Same full-text embedding comparison using the GTE model family.",
    "section": "Compares CV and JD sections separately, then combines section similarities instead of using one full text block.",
    "requirement": "Rule-based explainable score. Extracts JD requirements and checks whether the CV text contains evidence for them.",
    "llm": "Ollama-based evaluator. Sends the CV and JD to a local LLM and asks it for a fit score and explanation.",
}

METHOD_GUIDE_ROWS = [
    {
        "method": "TF-IDF",
        "what it measures": "Shared important words and short phrases.",
        "how it works": "Builds TF-IDF vectors from the CV and JD text, then calculates cosine similarity.",
        "best use": "Simple baseline; fast; good for checking lexical overlap.",
    },
    {
        "method": "SBERT",
        "what it measures": "Overall semantic similarity.",
        "how it works": "Encodes the full CV and full JD with sentence-transformers/all-MiniLM-L6-v2 and compares vectors.",
        "best use": "Main fast semantic baseline.",
    },
    {
        "method": "BGE",
        "what it measures": "Overall semantic similarity using BGE embeddings.",
        "how it works": "Encodes JD as a matching query and CV as a resume document, then compares vectors.",
        "best use": "Compare against SBERT to see whether another embedding model ranks pairs better.",
    },
    {
        "method": "E5",
        "what it measures": "Retrieval-style semantic similarity.",
        "how it works": "Adds E5 prefixes, `query:` for JD and `passage:` for CV, then compares embeddings.",
        "best use": "Useful because CV-JD matching is similar to retrieving relevant CVs for a job query.",
    },
    {
        "method": "GTE",
        "what it measures": "Overall semantic similarity using GTE embeddings.",
        "how it works": "Encodes full CV and full JD with the GTE model and calculates cosine similarity.",
        "best use": "Another embedding-family comparison point.",
    },
    {
        "method": "Section-aware",
        "what it measures": "Similarity between meaningful parts of the CV and JD.",
        "how it works": "Parses sections such as skills, experience, education, responsibilities, and requirements, then combines section scores.",
        "best use": "More explainable than full-text embedding because different CV/JD parts can matter differently.",
    },
    {
        "method": "Requirement rules",
        "what it measures": "Estimated requirement coverage.",
        "how it works": "Extracts requirements from the JD and searches the CV for evidence, including skill and experience clues.",
        "best use": "Closest to the thesis idea: score means how much of the JD the CV appears to satisfy.",
    },
    {
        "method": "Ollama LLM",
        "what it measures": "LLM judgement of candidate fit.",
        "how it works": "Sends the CV and JD to a local Ollama model and asks for a structured score and explanation.",
        "best use": "Optional qualitative evaluator; slower and less stable, especially with small local models.",
    },
]

SCORE_COLUMNS = [
    "tfidf_score",
    "bge_score",
    "e5_score",
    "gte_score",
    "sbert_score",
    "section_score",
    "requirement_score",
    "llm_score",
    "final_score",
]


st.set_page_config(page_title="CV-JD Matching Scorer", layout="wide")
st.title("CV-JD Matching Scorer")


@st.cache_data
def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cvs = pd.read_csv(NORMALIZED_CVS)
    jds = pd.read_csv(NORMALIZED_JDS)
    pairs = pd.read_csv(PAIR_SCORES)
    return cvs, jds, pairs


@st.cache_resource(show_spinner=False)
def load_selected_methods(
    selected_methods: tuple[str, ...],
    embedding_model: str,
    bge_model: str,
    e5_model: str,
    gte_model: str,
    ollama_model: str,
    local_files_only: bool,
):
    all_methods = build_current_methods(
        embedding_model=embedding_model,
        bge_model=bge_model,
        e5_model=e5_model,
        gte_model=gte_model,
        ollama_model=ollama_model,
        local_files_only=local_files_only,
        include_llm="llm" in selected_methods,
    )
    return {key: all_methods[key] for key in selected_methods if key in all_methods}


def score_to_category(score: float | None) -> str:
    if score is None or pd.isna(score):
        return ""
    score = float(score)
    if score <= 20:
        return "poor fit"
    if score <= 40:
        return "weak fit"
    if score <= 60:
        return "possible fit"
    if score <= 80:
        return "good fit"
    return "strong fit"


def choose_pairs(
    pairs: pd.DataFrame,
    jds: pd.DataFrame,
    pair_count: int,
    pair_type: str,
    jd_group: str,
    sample_mode: str,
    random_seed: int,
) -> pd.DataFrame:
    available = pairs.copy()

    if pair_type != "All":
        available = available[available["pair_type"] == pair_type]

    if jd_group != "All":
        jd_ids = set(jds.loc[jds["occupation_group"] == jd_group, "jd_id"])
        available = available[available["jd_id"].isin(jd_ids)]

    if available.empty:
        return available

    pair_count = min(pair_count, len(available))
    if sample_mode == "Random":
        return available.sample(n=pair_count, random_state=random_seed).reset_index(drop=True)
    if sample_mode == "Highest previous score":
        return available.sort_values("final_score", ascending=False).head(pair_count).reset_index(drop=True)
    if sample_mode == "Mixed by pair type":
        parts = []
        selected_indexes: set[int] = set()
        per_type = max(1, pair_count // max(1, available["pair_type"].nunique()))
        for _, group_df in available.groupby("pair_type"):
            sampled = group_df.sample(n=min(per_type, len(group_df)), random_state=random_seed)
            parts.append(sampled)
            selected_indexes.update(sampled.index.tolist())
        mixed = pd.concat(parts, ignore_index=True)
        if len(mixed) < pair_count:
            rest = available[~available.index.isin(selected_indexes)]
            if not rest.empty:
                extra = rest.sample(n=min(pair_count - len(mixed), len(rest)), random_state=random_seed)
                mixed = pd.concat([mixed, extra], ignore_index=True)
        return mixed.head(pair_count).reset_index(drop=True)
    return available.head(pair_count).reset_index(drop=True)


def score_batch(
    selected_pairs: pd.DataFrame,
    cvs_by_id: pd.DataFrame,
    jds_by_id: pd.DataFrame,
    methods: dict[str, object],
) -> pd.DataFrame:
    progress = st.progress(0)
    status = st.empty()
    rows: list[dict] = []

    total = len(selected_pairs)
    for index, pair in enumerate(selected_pairs.itertuples(index=False), start=1):
        status.write(f"Scoring pair {index} of {total}: {pair.cv_id} -> {pair.jd_id}")

        cv_row = cvs_by_id.loc[pair.cv_id]
        jd_row = jds_by_id.loc[pair.jd_id]
        cv = make_cv_document(
            str(pair.cv_id),
            str(cv_row["cv_text"]),
            {
                "resume_group": cv_row.get("resume_group", ""),
                "resume_category": cv_row.get("resume_category", ""),
            },
        )
        jd = make_jd_document(
            str(pair.jd_id),
            str(jd_row["jd_text"]),
            {
                "job_title": jd_row.get("job_title", ""),
                "occupation_group": jd_row.get("occupation_group", ""),
                "jd_category": jd_row.get("jd_category", ""),
            },
        )

        score_row, _ = score_pair_all_methods(cv, jd, methods)
        score_row.update(
            {
                "pair_type": getattr(pair, "pair_type", ""),
                "job_title": jd_row.get("job_title", ""),
                "cv_group": cv_row.get("resume_group", ""),
                "jd_group": jd_row.get("occupation_group", ""),
                "previous_tfidf_score": getattr(pair, "tfidf_score", None),
                "previous_sbert_score": getattr(pair, "sbert_score", None),
                "previous_final_score": getattr(pair, "final_score", None),
            }
        )
        score_row["fit_category"] = score_to_category(score_row["final_score"])
        rows.append(score_row)
        progress.progress(index / total)

    status.empty()
    progress.empty()
    return pd.DataFrame(rows)


def save_outputs(results: pd.DataFrame) -> tuple[Path, Path, bytes, bytes]:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = REPORTS_DIR / f"streamlit_batch_scores_{timestamp}.csv"
    excel_path = REPORTS_DIR / f"streamlit_batch_scores_{timestamp}.xlsx"

    results.to_csv(csv_path, index=False, encoding="utf-8")
    csv_bytes = results.to_csv(index=False).encode("utf-8")

    excel_buffer = BytesIO()
    with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
        results.to_excel(writer, index=False, sheet_name="scores")
    excel_bytes = excel_buffer.getvalue()
    excel_path.write_bytes(excel_bytes)

    return csv_path, excel_path, csv_bytes, excel_bytes


cvs, jds, pairs = load_data()
cvs_by_id = cvs.set_index("cv_id", drop=False)
jds_by_id = jds.set_index("jd_id", drop=False)

st.caption(
    f"Loaded {len(cvs):,} CVs, {len(jds):,} job descriptions, and {len(pairs):,} prebuilt candidate pairs. "
    "The app scores selected existing pairs instead of comparing every CV to every JD."
)

batch_tab, single_tab = st.tabs(["Batch scoring", "Single pair check"])

with batch_tab:
    st.subheader("1. Choose methods")
    fast_left, fast_middle, fast_right, slow_left = st.columns(4)
    method_checks = {}
    with fast_left:
        method_checks["tfidf"] = st.checkbox("TF-IDF", value=True, help=METHOD_HELP["tfidf"])
        method_checks["section"] = st.checkbox("Section-aware", value=False, help=METHOD_HELP["section"])
    with fast_middle:
        method_checks["sbert"] = st.checkbox("SBERT", value=True, help=METHOD_HELP["sbert"])
        method_checks["requirement"] = st.checkbox("Requirement rules", value=True, help=METHOD_HELP["requirement"])
    with fast_right:
        method_checks["bge"] = st.checkbox("BGE", value=False, help=METHOD_HELP["bge"])
        method_checks["e5"] = st.checkbox("E5", value=False, help=METHOD_HELP["e5"])
    with slow_left:
        method_checks["gte"] = st.checkbox("GTE", value=False, help=METHOD_HELP["gte"])
        method_checks["llm"] = st.checkbox("Ollama LLM", value=False, help=METHOD_HELP["llm"])

    selected_methods = tuple(key for key, enabled in method_checks.items() if enabled)
    st.caption(
        "Recommended first run: TF-IDF + SBERT + Requirement rules. "
        "BGE/E5/GTE need local Hugging Face models. Ollama LLM is much slower."
    )

    with st.expander("What do these methods mean?", expanded=True):
        st.write(
            "All scores are calculated only from the CV text and job-description text. "
            "They are not human labels. Higher scores mean the selected method found stronger evidence that the CV fits the JD."
        )
        st.dataframe(pd.DataFrame(METHOD_GUIDE_ROWS), width="stretch", hide_index=True)

    with st.expander("Model settings", expanded=False):
        left, right = st.columns(2)
        with left:
            embedding_model = st.text_input("SBERT / section model", value=DEFAULT_FAST_EMBEDDING_MODEL)
            bge_model = st.text_input("BGE model", value=DEFAULT_BGE_MODEL)
            e5_model = st.text_input("E5 model", value=DEFAULT_E5_MODEL)
            gte_model = st.text_input("GTE model", value=DEFAULT_GTE_MODEL)
        with right:
            ollama_model = st.text_input("Ollama model", value=DEFAULT_OLLAMA_MODEL)
            local_files_only = st.checkbox(
                "Use only locally cached embedding models",
                value=True,
                help="Recommended. If unchecked, sentence-transformers may try to download missing models.",
            )
            st.info("Ollama must already be running if you select the LLM evaluator.")

    st.subheader("2. Choose pairs")
    left, middle, right, fourth = st.columns([1, 1, 1, 1])
    with left:
        pair_count = st.number_input(
            "How many pairs to score",
            min_value=1,
            max_value=min(5000, len(pairs)),
            value=100,
            step=25,
        )
    with middle:
        sample_mode = st.selectbox(
            "Pair selection",
            ["Mixed by pair type", "Random", "First rows", "Highest previous score"],
        )
    with right:
        pair_type = st.selectbox("Pair type filter", ["All"] + sorted(pairs["pair_type"].dropna().unique()))
    with fourth:
        jd_group = st.selectbox("JD group filter", ["All"] + sorted(jds["occupation_group"].dropna().unique()))

    random_seed = st.number_input("Random seed", min_value=0, max_value=999999, value=42, step=1)
    selected_pairs = choose_pairs(
        pairs=pairs,
        jds=jds,
        pair_count=int(pair_count),
        pair_type=pair_type,
        jd_group=jd_group,
        sample_mode=sample_mode,
        random_seed=int(random_seed),
    )

    st.write(f"Selected {len(selected_pairs):,} pairs for this run.")
    st.dataframe(
        selected_pairs[
            [
                "cv_id",
                "jd_id",
                "pair_type",
                "job_title",
                "cv_group",
                "jd_group",
                "tfidf_score",
                "sbert_score",
                "final_score",
            ]
        ].head(20),
        width="stretch",
    )

    run_disabled = not selected_methods or selected_pairs.empty
    if st.button("Rate selected pairs and create result files", type="primary", disabled=run_disabled):
        with st.spinner("Loading selected models and scoring pairs..."):
            methods = load_selected_methods(
                selected_methods=selected_methods,
                embedding_model=embedding_model,
                bge_model=bge_model,
                e5_model=e5_model,
                gte_model=gte_model,
                ollama_model=ollama_model,
                local_files_only=local_files_only,
            )
            results = score_batch(selected_pairs, cvs_by_id, jds_by_id, methods)
            csv_path, excel_path, csv_bytes, excel_bytes = save_outputs(results)

        st.session_state["batch_results"] = results
        st.session_state["batch_csv_path"] = csv_path
        st.session_state["batch_excel_path"] = excel_path
        st.session_state["batch_csv_bytes"] = csv_bytes
        st.session_state["batch_excel_bytes"] = excel_bytes

    if "batch_results" in st.session_state:
        results = st.session_state["batch_results"]
        st.subheader("Results")
        st.success(
            f"Saved CSV to {st.session_state['batch_csv_path']} and Excel to {st.session_state['batch_excel_path']}."
        )

        metric_cols = st.columns(4)
        metric_cols[0].metric("Rows scored", f"{len(results):,}")
        metric_cols[1].metric("Average final score", f"{results['final_score'].mean():.1f}")
        metric_cols[2].metric("Highest final score", f"{results['final_score'].max():.1f}")
        metric_cols[3].metric("Rows with method errors", int(results["method_errors"].fillna("").ne("").sum()))

        visible_cols = [
            "cv_id",
            "jd_id",
            "pair_type",
            "job_title",
            "cv_group",
            "jd_group",
            *SCORE_COLUMNS,
            "fit_category",
            "method_errors",
        ]
        existing_cols = [col for col in visible_cols if col in results.columns]
        st.dataframe(results[existing_cols], width="stretch")

        left, right = st.columns(2)
        with left:
            st.download_button(
                "Download CSV",
                data=st.session_state["batch_csv_bytes"],
                file_name=Path(st.session_state["batch_csv_path"]).name,
                mime="text/csv",
            )
        with right:
            st.download_button(
                "Download Excel",
                data=st.session_state["batch_excel_bytes"],
                file_name=Path(st.session_state["batch_excel_path"]).name,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

with single_tab:
    st.subheader("Single CV-JD pair")
    st.write("Use this only for checking one example in detail.")

    pair_preview = pairs.head(500).copy()
    pair_labels = [
        f"{row.jd_id} | {row.cv_id} | {row.pair_type} | previous final {row.final_score:.1f}"
        for row in pair_preview.itertuples(index=False)
    ]
    selected_label = st.selectbox("Example pair", pair_labels)
    selected_pair = pair_preview.iloc[pair_labels.index(selected_label)]
    cv_row = cvs_by_id.loc[selected_pair["cv_id"]]
    jd_row = jds_by_id.loc[selected_pair["jd_id"]]

    preview_left, preview_right = st.columns(2)
    with preview_left:
        st.write(f"CV: `{selected_pair['cv_id']}`")
        st.write(f"Group: `{cv_row.get('resume_group', '')}`")
        st.text_area("CV text", str(cv_row["cv_text"])[:4000], height=280, disabled=True)
    with preview_right:
        st.write(f"JD: `{selected_pair['jd_id']}`")
        st.write(f"Title: `{jd_row.get('job_title', '')}`")
        st.write(f"Group: `{jd_row.get('occupation_group', '')}`")
        st.text_area("JD text", str(jd_row["jd_text"])[:4000], height=280, disabled=True)
