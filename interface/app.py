# app.py
# Streamlit interface for phenotype extraction pipeline
#
# Run with:
# streamlit run app.py

import html
import sys
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st
import pronto
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PIPELINE_DIR = PROJECT_ROOT / "Pipeline"

if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))


# ============================================================
# IMPORT YOUR PIPELINE HERE
# ============================================================
# Replace "your_pipeline_file" with the real Python file/module name.
#
# Example:
# from phenotype_pipeline import run_pipeline
#
# Expected:
# results = run_pipeline(text, k)
#
# results should be a list of dictionaries like:
# {
#     "text": "capsule endoscopy",
#     "label": "DIAGNOSTIC_PROCEDURE",
#     "is_present": True,
#     "hpo_id": None,
#     "definition": "...",      # optional
#     "synonyms": [...],        # optional
#     "comments": "...",        # optional
#     "top_k_candidates": [...]
# }

try:
    # Try dynamic import so static analyzers don't flag a missing module here.
    import importlib

    _mod = importlib.import_module("pipeline")
    run_pipeline = _mod.run_pipeline
    print("Pipeline imported successfully.")
except Exception as e:
    print(f"Error importing pipeline: {e}")
    # Demo fallback so the interface can be tested immediately.



# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Phenotype Extraction Interface",
    page_icon="🧬",
    layout="wide",
)


# ============================================================
# SESSION STATE
# ============================================================

if "results" not in st.session_state:
    st.session_state.results = []

    
if "selected_phenotype_idx" not in st.session_state:
    st.session_state.selected_phenotype_idx = None

if "has_run_pipeline" not in st.session_state:
    st.session_state.has_run_pipeline = False

HPO_OBO_PATH = Path("resources/HPO/hp.obo")
# ============================================================
# HELPERS
# ============================================================

def escape(value: Any) -> str:
    """Safely escape values before inserting them into HTML."""
    if value is None:
        return ""
    return html.escape(str(value), quote=True)


def get_candidates(item: Dict[str, Any]) -> List[Dict[str, Any]]:
    candidates = item.get("top_k_candidates") or []
    return candidates if isinstance(candidates, list) else []


def get_best_candidate(item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    candidates = get_candidates(item)
    return candidates[0] if candidates else None


def get_display_hpo_id(item: Dict[str, Any]) -> str:
    """
    Prefer the final assigned hpo_id if present.
    If the pipeline has not assigned one, show the best candidate as a suggestion.
    """
    if item.get("hpo_id"):
        return str(item["hpo_id"])

    best = get_best_candidate(item)
    if best and best.get("hpo_id"):
        return f"{best.get('hpo_id')} best candidate"

    return "N/A"


def get_display_hpo_term(item: Dict[str, Any]) -> str:
    best = get_best_candidate(item)
    if best and best.get("hpo_term"):
        return str(best["hpo_term"])
    return "N/A"


def format_list(value: Any) -> str:
    if value is None:
        return "N/A"

    if isinstance(value, list):
        return ", ".join(str(x) for x in value) if value else "N/A"

    return str(value) if str(value).strip() else "N/A"


def get_optional_field(item: Dict[str, Any], field_name: str) -> str:
    """
    Read definition/synonyms/comments from the extracted item.
    If missing, try the best candidate.
    """
    value = item.get(field_name)

    if value:
        return format_list(value)

    best = get_best_candidate(item)
    if best:
        value = best.get(field_name)
        if value:
            return format_list(value)

    return "N/A"


def get_hover_text(item: Dict[str, Any], threshold: float = 0.75) -> str:
    """
    Shows alternative candidates after the top candidate.
    Stops as soon as the score drops below threshold.
    """
    candidates = get_candidates(item)

    alternative_terms = []

    for candidate in candidates[1:]:
        score = candidate.get("score", 0)

        try:
            score = float(score)
        except TypeError:
            score = 0

        if score < threshold:
            break

        term = candidate.get("hpo_term")
        if term:
            alternative_terms.append(str(term))

    if alternative_terms:
        return "Could also be: " + ", ".join(alternative_terms)

    return "No strong alternative candidate above threshold."


def candidates_to_dataframe(item: Dict[str, Any]) -> pd.DataFrame:
    rows = []

    for rank, candidate in enumerate(get_candidates(item), start=1):
        rows.append(
            {
                "Rank": rank,
                "HPO ID": candidate.get("hpo_id", "N/A"),
                "Term": candidate.get("hpo_term", "N/A"),
                "Score": round(float(candidate.get("score", 0) or 0), 3),
                "Cross score": round(float(candidate.get("cross_score", 0) or 0), 3),
                "Parents": ", ".join(candidate.get("parent_ids", []) or []),
                "Children": ", ".join(candidate.get("children_ids", []) or []),
            }
        )

    return pd.DataFrame(rows)


def render_css() -> None:
    st.markdown(
        """
<style>
.main-title {
    font-size: 2.25rem;
    font-weight: 750;
    margin-bottom: 0.25rem;
    text-align: center;
}

.subtitle {
    color: #6b7280;
    margin-bottom: 1.5rem;
    text-align: center;
}

.results-summary {
    color: #4b5563;
    margin: 0.5rem 0 1rem 0;
    text-align: center;
}

.phenotype-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(290px, 360px));
    gap: 1rem;
    justify-content: center;
    align-items: stretch;

    width: 100%;
    max-width: 1150px;
    margin: 1.25rem auto 2rem auto;
    padding: 0 1rem;
}

.phenotype-chip {
    box-sizing: border-box;
    position: relative;

    width: 100%;
    min-height: 92px;

    display: flex;
    align-items: center;
    gap: 0.75rem;

    padding: 0.85rem 1rem;
    border-radius: 18px;

    border: 1px solid #e5e7eb;
    background: #f9fafb;

    text-decoration: none !important;
    color: #111827 !important;

    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.06);
    transition: transform 0.12s ease, box-shadow 0.12s ease, border-color 0.12s ease;
}

.phenotype-chip:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 18px rgba(0, 0, 0, 0.10);
    border-color: #d1d5db;
}

.present-chip {
    border-left: 7px solid #34c759;
}

.absent-chip {
    border-left: 7px solid #ff6b6b;
    opacity: 0.78;
}

.status-dot {
    width: 0.65rem;
    height: 0.65rem;
    border-radius: 999px;
    flex: 0 0 auto;
}

.present-dot {
    background: #34c759;
}

.absent-dot {
    background: #ff6b6b;
}

.chip-main {
    display: flex;
    flex-direction: column;
    gap: 0.18rem;
    min-width: 0;
    flex: 1;
}

.chip-title {
    font-weight: 700;
    font-size: 0.98rem;
    line-height: 1.2rem;
    color: #111827;

    white-space: normal;
    overflow-wrap: anywhere;
}

.chip-subtitle {
    color: #6b7280;
    font-size: 0.78rem;
    line-height: 1rem;

    white-space: normal;
    overflow-wrap: anywhere;
}

.chip-status {
    flex: 0 0 auto;

    padding: 0.22rem 0.55rem;
    border-radius: 999px;

    font-size: 0.72rem;
    font-weight: 650;
    white-space: nowrap;
}

.present-status {
    background: #e8f8ee;
    color: #157347;
}

.absent-status {
    background: #fff0ef;
    color: #c1121f;
}

.candidate-tooltip {
    visibility: hidden;
    opacity: 0;

    position: absolute;
    left: 50%;
    top: calc(100% + 8px);
    transform: translateX(-50%);

    width: max-content;
    max-width: 310px;

    background: #111827;
    color: white;

    padding: 0.55rem 0.75rem;
    border-radius: 10px;

    font-size: 0.78rem;
    line-height: 1.15rem;
    text-align: center;

    z-index: 9999;
    pointer-events: none;

    transition: opacity 0.15s ease;
}

.phenotype-chip:hover .candidate-tooltip {
    visibility: visible;
    opacity: 1;
}

.info-box {
    padding: 0.85rem 1rem;
    border-radius: 14px;
    border: 1px solid #e5e7eb;
    background: #f9fafb;
    color: #111827 !important;
    margin-bottom: 1rem;
}

.info-box strong {
    color: #111827 !important;
}
/* ============================
   Dialog / popup readability
   ============================ */

div[data-testid="stDialog"] {
    color: #111827 !important;
}

div[data-testid="stDialog"] div[role="dialog"] {
    background-color: #ffffff !important;
    color: #111827 !important;
}

/* Normal text inside the popup */
div[data-testid="stDialog"] h1,
div[data-testid="stDialog"] h2,
div[data-testid="stDialog"] h3,
div[data-testid="stDialog"] h4,
div[data-testid="stDialog"] p,
div[data-testid="stDialog"] span,
div[data-testid="stDialog"] li,
div[data-testid="stDialog"] label,
div[data-testid="stDialog"] strong {
    color: #111827 !important;
}

/* Info box inside popup */
div[data-testid="stDialog"] .info-box {
    background: #f9fafb !important;
    color: #111827 !important;
    border: 1px solid #e5e7eb !important;
}

div[data-testid="stDialog"] .info-box strong {
    color: #111827 !important;
}

/* Ontology hierarchy expander */
div[data-testid="stDialog"] div[data-testid="stExpander"] {
    border: 1px solid #e5e7eb !important;
    border-radius: 10px !important;
    overflow: hidden !important;
}

div[data-testid="stDialog"] div[data-testid="stExpander"] details {
    background-color: #ffffff !important;
    color: #111827 !important;
}

div[data-testid="stDialog"] div[data-testid="stExpander"] summary {
    background-color: #f3f4f6 !important;
    color: #111827 !important;
    border-radius: 10px 10px 0 0 !important;
}

div[data-testid="stDialog"] div[data-testid="stExpander"] summary * {
    color: #111827 !important;
}

div[data-testid="stDialog"] div[data-testid="stExpander"] details div {
    background-color: #ffffff !important;
    color: #111827 !important;
}

/* Bottom Close button */
div[data-testid="stDialog"] div[class*="stButton"] button {
    background-color: #f3f4f6 !important;
    color: #111827 !important;
    border: 1px solid #d1d5db !important;
    border-radius: 10px !important;
}

div[data-testid="stDialog"] div[class*="stButton"] button:hover {
    background-color: #e5e7eb !important;
    color: #111827 !important;
    border: 1px solid #9ca3af !important;
}

/* Top-right X close icon */
div[data-testid="stDialog"] button[aria-label="Close"] {
    background-color: transparent !important;
    color: #111827 !important;
    border: none !important;
}

div[data-testid="stDialog"] button[aria-label="Close"] svg {
    color: #111827 !important;
    fill: #111827 !important;
    stroke: #111827 !important;
}
</style>
""",
        unsafe_allow_html=True,
    )


def render_phenotype_chips(results: List[Dict[str, Any]]) -> None:
    """
    Render clickable phenotype boxes using native Streamlit buttons.
    This avoids browser navigation and allows st.dialog() to open as an inner popup.
    """

    # Style only phenotype buttons using their Streamlit keys.
    css_rules = [
    """
<style>
div[class*="st-key-phenotype_chip_"] button {
    min-height: 95px;
    width: 100%;
    border-radius: 18px;
    padding: 0.85rem 1rem;
    text-align: left;
    justify-content: flex-start;
    align-items: center;
    white-space: normal;

    background: #f9fafb !important;
    color: #111827 !important;

    border: 1px solid #e5e7eb;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.06);
    transition: transform 0.12s ease, box-shadow 0.12s ease, border-color 0.12s ease;
}

div[class*="st-key-phenotype_chip_"] button:hover {
    background: #f3f4f6 !important;
    color: #111827 !important;
    transform: translateY(-2px);
    box-shadow: 0 8px 18px rgba(0, 0, 0, 0.10);
    border-color: #d1d5db;
}

div[class*="st-key-phenotype_chip_"] button:focus,
div[class*="st-key-phenotype_chip_"] button:active {
    background: #f3f4f6 !important;
    color: #111827 !important;
}

div[class*="st-key-phenotype_chip_"] button * {
    color: #111827 !important;
}

div[class*="st-key-phenotype_chip_"] button p {
    font-size: 0.9rem;
    line-height: 1.25rem;
    color: #111827 !important;
}
</style>
"""
]

    for idx, item in enumerate(results):
        is_present = bool(item.get("is_present", True))
        border_color = "#34c759" if is_present else "#ff6b6b"
        opacity = "1" if is_present else "0.78"

        css_rules.append(
            f"""
<style>
div[class*="st-key-phenotype_chip_{idx}"] button {{
    border-left: 7px solid {border_color} !important;
    opacity: {opacity};
}}
</style>
"""
        )

    st.markdown("\n".join(css_rules), unsafe_allow_html=True)

    n_cols = min(3, len(results))
    cols = st.columns(n_cols, gap="medium")

    for idx, item in enumerate(results):
        with cols[idx % n_cols]:
            is_present = bool(item.get("is_present", True))

            status_text = "Present" if is_present else "Absent"
            status_icon = "🟢" if is_present else "🔴"

            mention = item.get("text", "Unknown phenotype")
            label = item.get("label", "UNKNOWN")
            best_term = get_display_hpo_term(item)
            hover_text = get_hover_text(item)

            subtitle = f"{label} · mentioned in text: {mention}"

            button_label = (
                f"{status_icon} **{best_term}**\n\n"
                f"{subtitle}\n\n"
                f"*{status_text}*"
            )

            if st.button(
                button_label,
                key=f"phenotype_chip_{idx}",
                help=hover_text,
                use_container_width=True,
            ):
                st.session_state.selected_phenotype_idx = idx

@st.cache_resource
def load_hpo_ontology(obo_path: str):
    """
    Load the HPO ontology once and cache it.
    """
    path = Path(obo_path)

    if not path.exists():
        return None

    return pronto.Ontology(str(path))


def get_main_hpo_id(item: Dict[str, Any]) -> Optional[str]:
    """
    Prefer the pipeline-assigned HPO ID.
    If missing, use the best candidate HPO ID.
    """
    if item.get("hpo_id"):
        return str(item["hpo_id"])

    best = get_best_candidate(item)
    if best and best.get("hpo_id"):
        return str(best["hpo_id"])

    return None


def get_hpo_term_from_ontology(hpo_id: Optional[str]):
    """
    Retrieve a pronto Term object from the ontology.
    """
    if not hpo_id:
        return None

    ontology = load_hpo_ontology(str(HPO_OBO_PATH))

    if ontology is None:
        return None

    try:
        return ontology.get(hpo_id)
    except Exception:
        return None


def get_hpo_definition(term) -> str:
    if term is None:
        return "N/A"

    definition = getattr(term, "definition", None)

    if definition:
        return str(definition)

    return "N/A"


def get_hpo_synonyms(term) -> str:
    if term is None:
        return "N/A"

    synonyms = getattr(term, "synonyms", [])

    values = []

    for synonym in synonyms:
        try:
            values.append(str(synonym.description))
        except Exception:
            values.append(str(synonym))

    return ", ".join(values) if values else "N/A"


def get_hpo_parents(term) -> str:
    if term is None:
        return "N/A"

    try:
        parents = list(term.superclasses(distance=1, with_self=False))
    except Exception:
        return "N/A"

    values = [f"{parent.id} — {parent.name}" for parent in parents]

    return ", ".join(values) if values else "N/A"


def get_hpo_children(term) -> str:
    if term is None:
        return "N/A"

    try:
        children = list(term.subclasses(distance=1, with_self=False))
    except Exception:
        return "N/A"

    values = [f"{child.id} — {child.name}" for child in children]

    return ", ".join(values) if values else "N/A"

def render_phenotype_card(item: Dict[str, Any]) -> None:
    mention = item.get("text", "Unknown phenotype")
    label = item.get("label", "UNKNOWN")
    is_present = bool(item.get("is_present", True))
    status_text = "Present" if is_present else "Absent"

    hpo_id = get_main_hpo_id(item)
    hpo_term = get_hpo_term_from_ontology(hpo_id)

    if hpo_term is not None:
        hpo_name = hpo_term.name or "N/A"
        hpo_definition = get_hpo_definition(hpo_term)
        hpo_synonyms = get_hpo_synonyms(hpo_term)
        hpo_parents = get_hpo_parents(hpo_term)
        hpo_children = get_hpo_children(hpo_term)
        hpo_comments = format_list(getattr(hpo_term, "comment", None))
    else:
        hpo_name = get_display_hpo_term(item)
        hpo_definition = get_optional_field(item, "definition")
        hpo_synonyms = get_optional_field(item, "synonyms")
        hpo_comments = get_optional_field(item, "comments")
        hpo_parents = "N/A"
        hpo_children = "N/A"

    st.markdown(f"### {hpo_name}")

    st.markdown(
        f"""
<div class="info-box">
    <strong>Mention in text:</strong> {escape(mention)}<br>
    <strong>Status:</strong> {escape(status_text)}<br>
    <strong>Extraction label:</strong> {escape(label)}<br>
    <strong>HPO ID:</strong> {escape(hpo_id or "N/A")}
</div>
""",
        unsafe_allow_html=True,
    )

    if not HPO_OBO_PATH.exists():
        st.warning(
            f"Could not find the HPO ontology file at `{HPO_OBO_PATH}`. "
            "Place `hp.obo` next to `app.py`, or update `HPO_OBO_PATH`."
        )

    if hpo_id and hpo_term is None and HPO_OBO_PATH.exists():
        st.warning(
            f"The HPO ID `{hpo_id}` was not found in the ontology file."
        )


    st.markdown("#### HPO information")
    st.write("**Definition:**", hpo_definition)
    st.write("**Synonyms:**", hpo_synonyms)
    st.write("**Comments:**", hpo_comments)

    with st.expander("Ontology hierarchy"):
        st.write("**Parents:**", hpo_parents)
        st.write("**Children:**", hpo_children)



def clear_selected_phenotype() -> None:
    st.session_state.selected_phenotype_idx = None

    rerun = getattr(st, "rerun", None) or getattr(st, "experimental_rerun", None)
    if rerun:
        rerun()


_DIALOG_DECORATOR = getattr(st, "dialog", None) or getattr(st, "experimental_dialog", None)

if _DIALOG_DECORATOR is not None:
    @_DIALOG_DECORATOR("Phenotype card")
    def phenotype_dialog(item: Dict[str, Any]) -> None:
        render_phenotype_card(item)

        st.divider()

        if st.button("Close"):
            clear_selected_phenotype()
else:
    def phenotype_dialog(item: Dict[str, Any]) -> None:
        st.warning(
            "Your Streamlit version does not support modal dialogs. "
            "Upgrade Streamlit or use the inline card below."
        )
        render_phenotype_card(item)



# ============================================================
# UI
# ============================================================

render_css()

st.markdown(
    '<div class="main-title">Phenotype Extraction Interface</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="subtitle">
Extract phenotype concepts from clinical text using your NLP + HPO matching pipeline.
</div>
""",
    unsafe_allow_html=True,
)


with st.form("phenotype_input_form"):
    text_input = st.text_area(
        "Input clinical text",
        height=230,
        placeholder="Paste clinical text here...",
    )

    k = st.slider(
        "Top-k candidates",
        min_value=1,
        max_value=15,
        value=10,
        help="Number of HPO candidates requested from the pipeline for each extracted mention.",
    )

    submitted = st.form_submit_button("Run pipeline", type="primary")


if submitted:
    text_input = text_input.strip()
    text_input += "\n"  # Ensure non-empty input is not just whitespace
    st.session_state.last_text = text_input

    if not text_input.strip():
        st.warning("Please enter some clinical text first.")
        st.session_state.results = []
        st.session_state.selected_phenotype_idx = None
        st.session_state.has_run_pipeline = False
    else:
        st.session_state.has_run_pipeline = True
        with st.spinner("Running phenotype extraction pipeline..."):
            try:
                try:
                    results = run_pipeline(text_input, k=k)
                except TypeError:
                    results = run_pipeline(text_input, k)

                if results is None:
                    results = []

                if not isinstance(results, list):
                    st.error(
                        "The pipeline should return a list of dictionaries. "
                        f"Received: {type(results)}"
                    )
                    results = []

                st.session_state.results = results
                st.session_state.selected_phenotype_idx = None

            except Exception as exc:
                st.session_state.results = []
                st.error("The pipeline failed.")
                st.exception(exc)


# ============================================================
# OUTPUT
# ============================================================

results = st.session_state.results

if results:
    present_count = sum(1 for item in results if item.get("is_present", True))
    absent_count = len(results) - present_count

    st.markdown(
        f"""
<div class="results-summary">
Found <strong>{len(results)}</strong> extracted concept(s):
<strong>{present_count}</strong> present,
<strong>{absent_count}</strong> absent.
</div>
""",
        unsafe_allow_html=True,
    )

    st.caption(
        "Hover over a phenotype to see strong alternative candidates. "
        "Click a phenotype to open its detailed card."
    )

    render_phenotype_chips(results)

    selected_idx = st.session_state.get("selected_phenotype_idx")


    if selected_idx is not None:
        if 0 <= selected_idx < len(results):
            phenotype_dialog(results[selected_idx])
        else:
            st.session_state.selected_phenotype_idx = None

else:
    if st.session_state.has_run_pipeline:
        st.info("No phenotypes found.")
    else:
        st.info("Run the pipeline to display extracted phenotypes.")

