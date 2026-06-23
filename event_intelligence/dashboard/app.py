from __future__ import annotations

import copy
import html
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    import streamlit as st
except Exception as exc:  # pragma: no cover - dashboard dependency
    raise RuntimeError("Install streamlit with `pip install -r requirements.txt` to run the dashboard.") from exc

import plotly.express as px
import plotly.graph_objects as go
import streamlit.components.v1 as components

from event_intelligence.src.common.runtime import build_event, load_or_train
from event_intelligence.src.llm_assistant import LLMCopilot, LLMService
from event_intelligence.src.prediction import ImpactPredictionEngine


BENGALURU_CENTER = (12.9716, 77.5946)
FINAL_PAGES = [
    "Live Event Assessment",
    "Control Room",
    "Diversion Management",
    "Resource Management",
    "Similar Incidents",
    "Scenario Simulator",
    "Executive Analytics",
    "AI Copilot",
]
STATUSES = ["Reported", "Assigned", "En Route", "On Scene", "Diversion Active", "Resolved", "Closed"]
ACTIVE_STATUSES = {"Reported", "Assigned", "En Route", "On Scene", "Diversion Active"}
RISK_COLORS = {"Low": "#16a34a", "Moderate": "#eab308", "High": "#f97316", "Critical": "#dc2626"}
ROUTE_COLORS = {"route_1": "#2563eb", "route_2": "#f97316", "route_3": "#16a34a"}


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        /* ═══════════════════════════════════════════════════════════════
           FORCE DARK MODE — browser-agnostic, overrides light-mode bleed
           ═══════════════════════════════════════════════════════════════ */

        /* 1. Tell the browser this entire document is dark — kills
              system-colour inheritance that causes white boxes in light mode */
        :root {
            color-scheme: dark only !important;
        }
        html {
            color-scheme: dark only !important;
        }

        /* 2. Root backgrounds */
        html, body,
        .stApp,
        [data-testid="stAppViewContainer"],
        [data-testid="stMain"],
        [data-testid="stMainBlockContainer"],
        .main,
        .main > div {
            background-color: #0b1120 !important;
            color: #f8fafc !important;
        }

        .block-container {
            background-color: #0b1120 !important;
            padding-top: 1.3rem;
        }

        /* 3. Every generic wrapper div — transparent so parent dark bg shows */
        div[data-testid="stVerticalBlock"],
        div[data-testid="stVerticalBlock"] > div,
        div[data-testid="stHorizontalBlock"] > div[data-testid="column"],
        div[data-testid="element-container"],
        div[data-testid="stMarkdownContainer"] {
            background-color: transparent !important;
        }

        /* 4. Border-wrapper containers (the main culprit for white boxes) */
        div[data-testid="stVerticalBlockBorderWrapper"],
        div[data-testid="stVerticalBlockBorderWrapper"] > div,
        section[data-testid="stSidebar"] ~ div div[data-testid="stVerticalBlockBorderWrapper"] {
            background-color: #111827 !important;
            border-color: #334155 !important;
            color: #f8fafc !important;
        }

        /* 5. Metrics */
        div[data-testid="stMetric"] {
            background: #111827 !important;
            border: 1px solid #334155 !important;
            border-radius: 8px;
            padding: 0.75rem 0.9rem;
            color: #f8fafc !important;
        }
        div[data-testid="stMetric"] label,
        div[data-testid="stMetric"] [data-testid="stMetricValue"],
        div[data-testid="stMetric"] [data-testid="stMetricDelta"] {
            color: #f8fafc !important;
        }

        /* 6. Expanders */
        div[data-testid="stExpander"] {
            background: #111827 !important;
            border: 1px solid #334155 !important;
            border-radius: 8px;
            color: #f8fafc !important;
        }
        div[data-testid="stExpander"] summary,
        div[data-testid="stExpander"] details,
        div[data-testid="stExpander"] p,
        div[data-testid="stExpander"] span,
        div[data-testid="stExpander"] label,
        div[data-testid="stExpander"] div {
            background-color: transparent !important;
            color: #f8fafc !important;
        }

        /* 7. Alert / info / warning / success / error boxes */
        div[data-testid="stAlert"],
        div[data-baseweb="notification"],
        [data-testid="stAlert"] > div,
        .stAlert {
            background-color: #1e293b !important;
            color: #f8fafc !important;
            border-color: #334155 !important;
        }
        div[data-testid="stAlert"] p,
        div[data-testid="stAlert"] span,
        div[data-baseweb="notification"] p,
        div[data-baseweb="notification"] span {
            color: #f8fafc !important;
        }

        /* 8. DataFrames / Tables */
        div[data-testid="stDataFrame"],
        div[data-testid="stDataFrame"] > div,
        div[data-testid="stDataFrame"] iframe,
        .stDataFrame,
        div[data-testid="stTable"],
        div[data-testid="stTable"] table {
            background-color: #111827 !important;
            border-color: #334155 !important;
            color: #f8fafc !important;
        }

        /* 9. Forms */
        div[data-testid="stForm"] {
            background: #111827 !important;
            border-color: #334155 !important;
            color: #f8fafc !important;
        }

        /* 10. Chat messages */
        div[data-testid="stChatMessage"],
        div[data-testid="stChatMessageContent"] {
            background-color: #1e293b !important;
            color: #f8fafc !important;
        }

        /* 11. Tabs */
        div[data-baseweb="tab-list"],
        button[data-baseweb="tab"] {
            background-color: #0b1120 !important;
            color: #f8fafc !important;
            border-color: #334155 !important;
        }
        button[data-baseweb="tab"][aria-selected="true"] {
            background-color: #1e293b !important;
            color: #f8fafc !important;
            border-bottom-color: #3b82f6 !important;
        }
        div[data-baseweb="tab-panel"] {
            background-color: #111827 !important;
        }

        /* ── Custom classes ── */
        .command-subtitle {
            color: #cbd5e1;
            margin-top: -0.65rem;
            margin-bottom: 0.85rem;
        }
        .nav-caption {
            color: #cbd5e1;
            font-size: 0.82rem;
            margin-bottom: 0.25rem;
            text-transform: uppercase;
            letter-spacing: 0.04em;
        }

        /* ── Sidebar ── */
        [data-testid="stSidebar"],
        [data-testid="stSidebar"] > div:first-child,
        [data-testid="stSidebar"] section {
            background-color: #0E1117 !important;
        }
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] span,
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3 {
            color: #FAFAFA !important;
        }

        /* ── Inputs ── */
        [data-testid="stSidebar"] input,
        [data-testid="stSidebar"] select,
        [data-testid="stSidebar"] textarea,
        select, textarea, input,
        .stTextInput input,
        .stSelectbox select,
        .stMultiselect select,
        .stDateInput input,
        .stNumberInput input,
        .stTextArea textarea,
        .stTextInput textarea {
            background-color: #262730 !important;
            color: #FAFAFA !important;
            border-color: #334155 !important;
        }
        [data-testid="stSidebar"] select option,
        select option {
            background-color: #111827 !important;
            color: #f8fafc !important;
        }

        /* ── Dropdowns / listboxes ── */
        div[role="listbox"],
        div[role="presentation"] {
            background-color: #111827 !important;
            color: #f8fafc !important;
        }
        div[role="option"],
        div[role="menuitem"] {
            background-color: #111827 !important;
            color: #f8fafc !important;
        }
        div[role="option"][aria-selected="true"],
        div[role="option"]:hover,
        div[role="menuitem"]:hover,
        div[role="menuitem"][aria-selected="true"] {
            background-color: #1f2937 !important;
            color: #f8fafc !important;
        }

        /* ── BaseWeb Selectbox ── */
        div[data-baseweb="select"] > div:first-child {
            background-color: #262730 !important;
            border-color: #334155 !important;
        }
        div[data-baseweb="select"] > div:first-child:hover,
        div[data-baseweb="select"] > div:first-child:focus-within {
            background-color: #262730 !important;
            border-color: #475569 !important;
        }
        div[data-baseweb="select"] input,
        div[data-baseweb="select"] span {
            background-color: transparent !important;
            color: #f8fafc !important;
        }
        div[data-baseweb="select"] svg {
            fill: #f8fafc !important;
        }
        ul[data-baseweb="menu"],
        div[data-baseweb="popover"],
        div[data-baseweb="menu"] {
            background-color: #111827 !important;
            border-color: #334155 !important;
        }
        li[role="option"],
        div[data-baseweb="menu"] li {
            background-color: #111827 !important;
            color: #f8fafc !important;
        }
        li[role="option"]:hover,
        li[role="option"][aria-selected="true"],
        div[data-baseweb="menu"] li:hover {
            background-color: #1f2937 !important;
            color: #f8fafc !important;
        }
        span[data-baseweb="tag"],
        div[data-baseweb="tag"] {
            background-color: #1e293b !important;
            color: #f8fafc !important;
        }

        /* ── Header / Radio nav ── */
        div[data-testid="stRadio"] > div[role="radiogroup"],
        header[data-testid="stHeader"] {
            background-color: #0E1117 !important;
        }
        div[data-testid="stRadio"] > div[role="radiogroup"] label p,
        header[data-testid="stHeader"] * {
            color: #FAFAFA !important;
        }

        /* ── All regular (non-nav) buttons ── */
        button[kind="secondary"],
        div[data-testid="stButton"] > button,
        div[data-testid="stFormSubmitButton"] > button {
            background-color: #1e293b !important;
            color: #f8fafc !important;
            border: 1px solid #334155 !important;
        }
        button[kind="secondary"]:hover,
        div[data-testid="stButton"] > button:hover,
        div[data-testid="stFormSubmitButton"] > button:hover {
            background-color: #334155 !important;
            color: #f8fafc !important;
            border: 1px solid #475569 !important;
        }
        div[data-testid="stButton"] > button p,
        div[data-testid="stButton"] > button span,
        div[data-testid="stFormSubmitButton"] > button p,
        div[data-testid="stFormSubmitButton"] > button span {
            color: #f8fafc !important;
        }

        /* ── Multiselect: tag chips (Map Layers, Incident dropdown tags) ── */
        div[data-testid="stMultiSelect"] span[data-baseweb="tag"],
        div[data-testid="stMultiSelect"] div[data-baseweb="tag"],
        span[data-baseweb="tag"],
        div[data-baseweb="tag"] {
            background-color: #1e293b !important;
            color: #f8fafc !important;
            border-color: #334155 !important;
        }
        /* Text inside tag chips */
        span[data-baseweb="tag"] span,
        div[data-baseweb="tag"] span,
        div[data-testid="stMultiSelect"] span[data-baseweb="tag"] span,
        div[data-testid="stMultiSelect"] [data-baseweb="tag"] * {
            color: #f8fafc !important;
            background-color: transparent !important;
        }
        /* The × close button inside each tag */
        span[data-baseweb="tag"] [role="presentation"],
        div[data-baseweb="tag"] [role="presentation"],
        span[data-baseweb="tag"] svg,
        div[data-baseweb="tag"] svg {
            fill: #f8fafc !important;
            color: #f8fafc !important;
        }
        /* Multiselect control box itself */
        div[data-testid="stMultiSelect"] div[data-baseweb="select"] > div:first-child {
            background-color: #1e293b !important;
            border-color: #334155 !important;
        }
        /* Placeholder / input text inside multiselect */
        div[data-testid="stMultiSelect"] input {
            color: #f8fafc !important;
        }
        /* ── Multiselect label */
        div[data-testid="stMultiSelect"] label {
            color: #f8fafc !important;
        }

        /* ── Selectbox: label + selected value text (main content, not just sidebar) ── */
        div[data-testid="stSelectbox"] label,
        div[data-testid="stSelectbox"] p {
            color: #f8fafc !important;
        }
        div[data-testid="stSelectbox"] div[data-baseweb="select"] > div:first-child {
            background-color: #1e293b !important;
            border-color: #334155 !important;
        }
        div[data-testid="stSelectbox"] div[data-baseweb="select"] span,
        div[data-testid="stSelectbox"] div[data-baseweb="select"] div,
        div[data-testid="stSelectbox"] div[data-baseweb="select"] input {
            color: #f8fafc !important;
            background-color: transparent !important;
        }
        div[data-testid="stSelectbox"] div[data-baseweb="select"] svg {
            fill: #f8fafc !important;
        }

        /* ── Nav buttons ── */
        div[data-testid="stHorizontalBlock"] button {
            background-color: #1e293b !important;
            color: #f8fafc !important;
            border: 1px solid #334155 !important;
        }
        div[data-testid="stHorizontalBlock"] button:hover {
            background-color: #334155 !important;
            color: #f8fafc !important;
            border: 1px solid #475569 !important;
        }
        div[data-testid="stHorizontalBlock"] button[kind="primary"] {
            background-color: #dc2626 !important;
            color: #ffffff !important;
            border: 1px solid #dc2626 !important;
        }
        div[data-testid="stHorizontalBlock"] button[kind="primary"]:hover {
            background-color: #b91c1c !important;
            border: 1px solid #b91c1c !important;
        }
        div[data-testid="stHorizontalBlock"] button[data-nav-active="true"] {
            background-color: #dc2626 !important;
            color: #ffffff !important;
            border: 1px solid #dc2626 !important;
        }
        div[data-testid="stHorizontalBlock"] button[data-nav-active="true"]:hover {
            background-color: #b91c1c !important;
            border: 1px solid #b91c1c !important;
        }
        div[data-testid="stHorizontalBlock"] button p,
        div[data-testid="stHorizontalBlock"] button span,
        div[data-testid="stHorizontalBlock"] button div {
            transition: all 0.2s ease !important;
            color: inherit !important;
        }

        /* ── Tooltips / popovers ── */
        div[data-baseweb="tooltip"],
        div[data-baseweb="popover"] > div {
            background-color: #1e293b !important;
            color: #f8fafc !important;
            border-color: #334155 !important;
        }

        /* ── Slider ── */
        /* Track rail (unfilled portion) */
        div[data-testid="stSlider"] [data-baseweb="slider"] > div:first-child {
            background-color: #334155 !important;
        }
        /* Filled / active portion of the track */
        div[data-testid="stSlider"] [data-baseweb="slider"] [role="progressbar"],
        div[data-testid="stSlider"] [data-baseweb="slider"] [data-testid="stSliderTrackFill"],
        div[data-testid="stSlider"] [data-baseweb="slider"] > div > div[style*="background"] {
            background-color: #dc2626 !important;
        }
        /* Thumb (the draggable circle) */
        div[data-testid="stSlider"] [data-baseweb="slider"] [role="slider"],
        div[data-testid="stSlider"] [data-baseweb="slider"] div[data-testid="stThumbValue"] {
            background-color: #dc2626 !important;
            border-color: #dc2626 !important;
        }
        /* Value/tick labels below the slider */
        div[data-testid="stSlider"] [data-testid="stTickBarMin"],
        div[data-testid="stSlider"] [data-testid="stTickBarMax"],
        div[data-testid="stSlider"] p,
        div[data-testid="stSlider"] span {
            color: #f8fafc !important;
        }
        /* Slider wrapper background */
        div[data-testid="stSlider"] {
            background-color: transparent !important;
        }

        /* ── Sidebar: ALL text elements ── */
        [data-testid="stSidebar"] *,
        [data-testid="stSidebar"] div,
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] span,
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] small,
        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3,
        [data-testid="stSidebar"] h4 {
            color: #FAFAFA !important;
        }
        /* Sidebar selectbox value text */
        [data-testid="stSidebar"] div[data-baseweb="select"] span,
        [data-testid="stSidebar"] div[data-baseweb="select"] div,
        [data-testid="stSidebar"] div[data-baseweb="select"] input {
            color: #FAFAFA !important;
            background-color: transparent !important;
        }
        [data-testid="stSidebar"] div[data-baseweb="select"] > div:first-child {
            background-color: #1e293b !important;
            border-color: #334155 !important;
        }
        /* Sidebar number input +/- buttons */
        [data-testid="stSidebar"] button[data-testid="stNumberInputStepUp"],
        [data-testid="stSidebar"] button[data-testid="stNumberInputStepDown"],
        [data-testid="stNumberInputStepUp"],
        [data-testid="stNumberInputStepDown"] {
            background-color: #1e293b !important;
            color: #f8fafc !important;
            border-color: #334155 !important;
        }
        /* Sidebar submit / other buttons */
        [data-testid="stSidebar"] button,
        [data-testid="stSidebar"] button span,
        [data-testid="stSidebar"] button p {
            background-color: #1e293b !important;
            color: #f8fafc !important;
            border-color: #334155 !important;
        }
        [data-testid="stSidebar"] button[kind="primary"],
        [data-testid="stSidebar"] button[kind="primary"] span,
        [data-testid="stSidebar"] button[kind="primary"] p {
            background-color: #dc2626 !important;
            color: #ffffff !important;
            border-color: #dc2626 !important;
        }

        /* ── Checkbox & radio labels ── */
        div[data-testid="stCheckbox"] label,
        div[data-testid="stCheckbox"] span,
        div[data-testid="stRadio"] label,
        div[data-testid="stRadio"] span {
            color: #f8fafc !important;
        }

        /* ── Markdown text ── */
        .stMarkdown p,
        .stMarkdown li,
        .stMarkdown span {
            color: #f8fafc !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Inject a <meta> tag to reinforce dark color-scheme at the HTML level,
    # which prevents Chrome/Edge from applying light-mode system colours.
    st.markdown(
        '<meta name="color-scheme" content="dark">',
        unsafe_allow_html=True,
    )


# def render_top_nav() -> str:
#     st.session_state.setdefault("current_page", FINAL_PAGES[0])
#     if st.session_state["current_page"] not in FINAL_PAGES:
#         st.session_state["current_page"] = FINAL_PAGES[0]

#     current = st.session_state["current_page"]
#     st.markdown('<div class="nav-caption">Traffic Police Command Center</div>', unsafe_allow_html=True)
#     columns = st.columns(len(FINAL_PAGES))
#     for column, page in zip(columns, FINAL_PAGES):
#         if column.button(page, key=f"nav-{page}", use_container_width=True):
#             st.session_state["current_page"] = page
#             st.rerun()

#     # Use a MutationObserver so the active highlight survives Streamlit re-renders
#     active_js = f"""
#     <script>
#     (function() {{
#         const ACTIVE_PAGE = {repr(current)};

#         function applyActive() {{
#             const blocks = document.querySelectorAll('div[data-testid="stHorizontalBlock"]');
#             if (!blocks.length) return;
#             const navBlock = blocks[0];
#             navBlock.querySelectorAll('button').forEach(function(btn) {{
#                 if (btn.innerText.trim() === ACTIVE_PAGE) {{
#                     btn.style.setProperty('background-color', '#dc2626', 'important');
#                     btn.style.setProperty('border-color', '#dc2626', 'important');
#                     btn.style.setProperty('color', '#ffffff', 'important');
#                 }} else {{
#                     btn.style.removeProperty('background-color');
#                     btn.style.removeProperty('border-color');
#                     btn.style.removeProperty('color');
#                 }}
#             }});
#         }}

#         // Run immediately and keep re-applying via MutationObserver
#         applyActive();

#         const observer = new MutationObserver(applyActive);
#         observer.observe(document.body, {{ childList: true, subtree: true }});

#         // Also re-apply on a short interval to catch any late renders
#         setInterval(applyActive, 300);
#     }})();
#     </script>
#     """
#     st.markdown(active_js, unsafe_allow_html=True)
#     return current

# def render_top_nav() -> str:
#     st.session_state.setdefault("current_page", FINAL_PAGES[0])
#     if st.session_state["current_page"] not in FINAL_PAGES:
#         st.session_state["current_page"] = FINAL_PAGES[0]

#     current = st.session_state["current_page"]
#     st.markdown('<div class="nav-caption">Traffic Police Command Center</div>', unsafe_allow_html=True)
#     columns = st.columns(len(FINAL_PAGES))
#     for column, page in zip(columns, FINAL_PAGES):
#         if column.button(page, key=f"nav-{page}", use_container_width=True):
#             st.session_state["current_page"] = page
#             st.rerun()

#     # Determine the 1-indexed position of the active page for the CSS nth-child selector
#     active_idx = FINAL_PAGES.index(current) + 1

#     # Pure CSS approach: We use an empty marker span right after the columns block.
#     # The :has selector isolates the navigation container without affecting buttons elsewhere on the page.
#     active_css = f"""
#     <span class="nav-active-marker"></span>
#     <style>
#     div[data-testid="element-container"]:has(+ div[data-testid="element-container"] .nav-active-marker) div[data-testid="column"]:nth-child({active_idx}) button {{
#         background-color: #dc2626 !important;
#         color: #ffffff !important;
#         border: 1px solid #dc2626 !important;
#     }}
#     div[data-testid="element-container"]:has(+ div[data-testid="element-container"] .nav-active-marker) div[data-testid="column"]:nth-child({active_idx}) button:hover {{
#         background-color: #b91c1c !important;
#         border: 1px solid #b91c1c !important;
#     }}
#     div[data-testid="element-container"]:has(+ div[data-testid="element-container"] .nav-active-marker) div[data-testid="column"]:nth-child({active_idx}) button * {{
#         color: #ffffff !important;
#     }}
#     </style>
#     """
#     st.markdown(active_css, unsafe_allow_html=True)
#     return current

def render_top_nav() -> str:
    st.session_state.setdefault("current_page", FINAL_PAGES[0])
    if st.session_state["current_page"] not in FINAL_PAGES:
        st.session_state["current_page"] = FINAL_PAGES[0]

    st.markdown('<div class="nav-caption">Traffic Police Command Center</div>', unsafe_allow_html=True)
    columns = st.columns(len(FINAL_PAGES))
    for column, page in zip(columns, FINAL_PAGES):
        # Check if this button matches the currently active page
        active = page == st.session_state["current_page"]
        
        # Use "primary" type for active buttons (red) and "secondary" for inactive buttons (default)
        button_type = "primary" if active else "secondary"
        if column.button(page, key=f"nav-{page}", type=button_type, use_container_width=True):
            st.session_state["current_page"] = page
            st.rerun()
            
    return st.session_state["current_page"]

@st.cache_resource(show_spinner="Loading command-center modules...")
def modules():
    loaded = load_or_train()
    normalize_diversion_graph(loaded["diversion"])
    return loaded


@st.cache_data(show_spinner="Loading event categories...")
def cleaned_rows():
    loaded = modules()
    return loaded["preprocessor"].load_and_transform(loaded["dataset"])


def coordinate_distance_km(left_lat: float, left_lon: float, right_lat: float, right_lon: float) -> float:
    lat_delta = left_lat - right_lat
    lon_delta = left_lon - right_lon
    return max(0.05, ((lat_delta * lat_delta + lon_delta * lon_delta) ** 0.5) * 111)


def graph_node_coordinates(graph: Any, node: str) -> tuple[float, float] | None:
    data = graph.nodes.get(node, {})
    lat = data.get("latitude")
    lon = data.get("longitude")
    if lat is None or lon is None:
        return None
    return float(lat), float(lon)


def normalize_diversion_graph(diversion: Any) -> None:
    graph = getattr(diversion, "graph", None)
    if graph is None:
        return
    for left, right, data in graph.edges(data=True):
        left_point = graph_node_coordinates(graph, left)
        right_point = graph_node_coordinates(graph, right)
        if not left_point or not right_point:
            continue
        distance = coordinate_distance_km(left_point[0], left_point[1], right_point[0], right_point[1])
        if data.get("corridor") == "transfer":
            data["weight"] = round(distance * 1.15, 3)
        elif not data.get("weight") or float(data.get("weight", 0)) <= 0:
            data["weight"] = round(distance, 3)


def options(rows: list[dict], column: str, fallback: list[str]) -> list[str]:
    values = sorted({str(row.get(column)) for row in rows if row.get(column) and row.get(column) != "Unknown"})
    return values or fallback


def infer_location(rows: list[dict], event_values: dict[str, Any]) -> tuple[float | None, float | None]:
    junction = event_values.get("junction")
    corridor = event_values.get("corridor")
    for matcher in (
        lambda row: row.get("junction") == junction and row.get("corridor") == corridor,
        lambda row: row.get("junction") == junction,
        lambda row: row.get("corridor") == corridor,
    ):
        for row in rows:
            if matcher(row) and row.get("latitude") is not None and row.get("longitude") is not None:
                return float(row["latitude"]), float(row["longitude"])
    return None, None


def option_index(values: list[str], current: Any) -> int:
    try:
        return values.index(str(current))
    except ValueError:
        return 0


def default_event_values(rows: list[dict]) -> dict[str, Any]:
    event_type_values = options(rows, "event_type", ["unplanned", "planned"])
    cause_values = options(rows, "event_cause", ["vehicle_breakdown"])
    zone_values = options(rows, "zone", ["Unknown"])
    corridor_values = options(rows, "corridor", ["Non-corridor"])
    junction_values = options(rows, "junction", ["Unknown"])
    latitude, longitude = infer_location(rows, {"corridor": corridor_values[0], "junction": junction_values[0]})
    return {
        "event_type": event_type_values[0],
        "event_cause": cause_values[0],
        "priority": "High",
        "requires_road_closure": False,
        "zone": zone_values[0],
        "corridor": corridor_values[0],
        "junction": junction_values[0],
        "hour": 18,
        "latitude": latitude,
        "longitude": longitude,
        "event_description": "",
    }


def event_input(rows: list[dict]) -> dict[str, Any]:
    st.session_state.setdefault("live_event_values", default_event_values(rows))
    current = st.session_state["live_event_values"]
    event_types = options(rows, "event_type", ["unplanned", "planned"])
    causes = options(rows, "event_cause", ["vehicle_breakdown"])
    zones = options(rows, "zone", ["Unknown"])
    corridors = options(rows, "corridor", ["Non-corridor"])
    junctions = options(rows, "junction", ["Unknown"])
    priorities = options(rows, "priority", ["High", "Low"])

    st.sidebar.header("Live Incident Input")
    with st.sidebar.form("live-incident-input"):
        event_type = st.selectbox("Event Type", event_types, index=option_index(event_types, current.get("event_type")))
        cause = st.selectbox("Cause", causes, index=option_index(causes, current.get("event_cause")))
        priority = st.selectbox("Priority", priorities, index=option_index(priorities, current.get("priority")))
        closure = st.checkbox("Road Closure", value=bool(current.get("requires_road_closure")))
        zone = st.selectbox("Zone", zones, index=option_index(zones, current.get("zone")))
        corridor = st.selectbox("Corridor", corridors, index=option_index(corridors, current.get("corridor")))
        junction = st.selectbox("Junction", junctions, index=option_index(junctions, current.get("junction")))
        hour = st.slider("Hour", 0, 23, int(current.get("hour", 18)))
        description = st.text_area("Description", value=str(current.get("event_description", "")))
        submitted = st.form_submit_button("Submit Event", type="primary")

    if submitted:
        latitude, longitude = infer_location(rows, {"corridor": corridor, "junction": junction})
        st.session_state["live_event_values"] = {
            "event_type": event_type,
            "event_cause": cause,
            "priority": priority,
            "requires_road_closure": closure,
            "zone": zone,
            "corridor": corridor,
            "junction": junction,
            "hour": hour,
            "latitude": latitude,
            "longitude": longitude,
            "event_description": description,
        }
        current = st.session_state["live_event_values"]
        st.sidebar.success("Live assessment updated.")

    st.sidebar.header("Operational Inputs")
    st.session_state["available_barricades"] = st.sidebar.number_input(
        "Available Barricades",
        min_value=0,
        max_value=100,
        value=int(st.session_state.get("available_barricades", 8)),
    )
    return current


def event_payload(values: dict[str, Any]) -> dict[str, Any]:
    payload = dict(values)
    hour = int(payload.pop("hour", 18))
    payload["start_datetime"] = datetime(2026, 1, 1, hour, 0).isoformat()
    return payload


def corridor_risk_lookup(loaded: dict) -> dict[str, float]:
    return {row["corridor"]: float(row["risk_score"]) for row in loaded["risk"].top_corridors(500)}


def nearest_graph_node(graph: Any, latitude: float | None, longitude: float | None) -> str | None:
    if latitude is None or longitude is None:
        return None
    nearest: tuple[float, str] | None = None
    for node, data in graph.nodes(data=True):
        lat = data.get("latitude")
        lon = data.get("longitude")
        if lat is None or lon is None:
            continue
        distance = coordinate_distance_km(float(latitude), float(longitude), float(lat), float(lon))
        if nearest is None or distance < nearest[0]:
            nearest = (distance, node)
    return nearest[1] if nearest else None


def route_destination_node(graph: Any, origin: str, corridor: str | None) -> str:
    origin_point = graph_node_coordinates(graph, origin)
    if not origin_point:
        nodes = [node for node in graph.nodes if node != origin]
        return nodes[0] if nodes else origin

    candidates = []
    for node, data in graph.nodes(data=True):
        if node == origin:
            continue
        point = graph_node_coordinates(graph, node)
        if not point:
            continue
        distance = coordinate_distance_km(origin_point[0], origin_point[1], point[0], point[1])
        same_corridor = data.get("corridor") == corridor
        if same_corridor and distance >= 0.8:
            candidates.append((0, abs(distance - 2.5), distance, node))
        elif distance >= 1.2:
            candidates.append((1, abs(distance - 3.5), distance, node))
    if not candidates:
        for neighbor in graph.neighbors(origin):
            return neighbor
        return origin
    return min(candidates)[3]


def route_endpoints(loaded: dict, event: dict[str, Any], rows: list[dict]) -> tuple[str, str]:
    graph = getattr(loaded["diversion"], "graph", None)
    if graph is None or not graph.nodes:
        return "Unknown", "Unknown"
    event_junction = event.get("junction")
    origin = event_junction if event_junction in graph else nearest_graph_node(graph, event.get("latitude"), event.get("longitude"))
    if origin is None:
        return "Unknown", "Unknown"
    destination = route_destination_node(graph, origin, event.get("corridor"))
    return origin, destination


def build_command_context(
    loaded: dict,
    event: dict[str, Any],
    rows: list[dict],
    available_barricades: int,
    origin: str,
    destination: str,
) -> dict[str, Any]:
    duration = loaded["duration"].predict_one(event)
    impact = loaded["impact"].predict_one(event)
    risk_level = ImpactPredictionEngine.risk_level(impact)
    corridor_risk = loaded["risk"].get_corridor_risk(event.get("corridor"))
    resources = loaded["resources"].recommend(
        predicted_impact=impact,
        predicted_duration=duration,
        road_closure=bool(event.get("requires_road_closure")),
        corridor_risk=corridor_risk,
        event_type=event.get("event_type", ""),
        event_cause=event.get("event_cause", ""),
        veh_type=event.get("veh_type", ""),
    )
    dispatch = loaded["dispatch"].assign(event.get("latitude"), event.get("longitude"), resources["officers"])
    sufficiency = loaded["resource_checker"].check(
        required_officers=resources["officers"],
        required_barricades=resources["barricades"],
        available_officers=dispatch.get("available_officers", 0),
        available_barricades=available_barricades,
    )
    diversion = loaded["diversion"].plan_multiple(
        origin,
        destination,
        affected_corridor=event.get("corridor"),
        road_closure=bool(event.get("requires_road_closure")),
        corridor_risks=corridor_risk_lookup(loaded),
    )
    timeline = loaded["timeline"].simulate(impact, duration)
    similar = loaded["similar"].query(event, top_k=5, planner=loaded["resources"])
    action_plan = loaded["action_plan"].generate(
        impact_score=impact,
        duration_minutes=duration,
        resources=resources,
        dispatch=dispatch,
        diversion_plan=diversion,
        event=event,
    )
    explanations = loaded["explainability"].explain(event, loaded["impact"], loaded["duration"])
    return {
        "event": event,
        "predicted_duration": duration,
        "predicted_impact": impact,
        "risk_level": risk_level,
        "corridor_risk": corridor_risk,
        "resources": resources,
        "dispatch": dispatch,
        "resource_sufficiency": sufficiency,
        "diversion": diversion,
        "timeline": timeline,
        "similar_incidents": similar,
        "action_plan": action_plan,
        "explanations": explanations,
        "officer_allocation": loaded["officer_allocation"].recommend_by_corridor(loaded["risk"], 20),
    }


def confidence_panel(context: dict[str, Any]) -> dict[str, Any]:
    similar_count = len(context.get("similar_incidents", {}).get("matches", []))
    location_bonus = 6 if context.get("event", {}).get("latitude") is not None else -8
    risk_penalty = {"Low": 0, "Moderate": 3, "High": 5, "Critical": 7}.get(context.get("risk_level"), 4)
    confidence = max(62, min(94, 78 + similar_count * 2 + location_bonus - risk_penalty))
    impact_margin = max(4, round(14 - confidence / 12))
    duration_margin = max(5, round(float(context.get("predicted_duration", 30)) * (1 - confidence / 115)))
    return {
        "impact": round(float(context.get("predicted_impact", 0)), 2),
        "impact_margin": impact_margin,
        "duration": round(float(context.get("predicted_duration", 0)), 2),
        "duration_margin": duration_margin,
        "confidence": confidence,
    }


def impact_radius_meters(risk_level: str, impact_score: float) -> int:
    base = {"Low": 500, "Moderate": 1000, "High": 2000, "Critical": 3000}.get(risk_level, 1000)
    return int(base * max(0.8, min(1.2, float(impact_score) / 75)))


def public_advisory(context: dict[str, Any]) -> str:
    event = context["event"]
    location = event.get("junction") or event.get("corridor") or "the affected corridor"
    cause = str(event.get("event_cause", "incident")).replace("_", " ")
    duration = round(float(context.get("predicted_duration", 0)))
    route = recommended_route(context.get("diversion", {}))
    alternate = route.get("route_name") if route else "the signed alternate route"
    return (
        f"Traffic congestion expected near {location} for approximately {duration} minutes due to {cause}. "
        f"Use {alternate} and follow Bengaluru Traffic Police directions."
    )


def enrich_context(context: dict[str, Any], incident_id: str, status: str = "Reported") -> dict[str, Any]:
    enriched = copy.deepcopy(context)
    now = datetime.now().isoformat(timespec="seconds")
    enriched["incident_id"] = incident_id
    enriched.setdefault("created_at", now)
    enriched.setdefault("status", status)
    enriched.setdefault("status_timestamps", {status: now})
    enriched["impact_radius_m"] = impact_radius_meters(enriched["risk_level"], enriched["predicted_impact"])
    enriched["prediction_confidence"] = confidence_panel(enriched)
    enriched["public_advisory"] = public_advisory(enriched)
    return enriched


def initialize_incidents(loaded: dict, rows: list[dict], current_context: dict[str, Any], available_barricades: int) -> None:
    if "incident_contexts" in st.session_state:
        return

    incidents = [enrich_context(current_context, "INC-0001", "Assigned")]
    seen = {current_context["event"].get("junction")}
    for row in rows:
        if len(incidents) >= 5:
            break
        if row.get("latitude") is None or row.get("longitude") is None:
            continue
        if row.get("junction") in seen or row.get("junction") == "Unknown":
            continue
        origin, destination = route_endpoints(loaded, row, rows)
        if origin == "Unknown" or destination == "Unknown":
            continue
        context = build_command_context(loaded, row, rows, available_barricades, origin, destination)
        status = "Diversion Active" if context["risk_level"] in {"High", "Critical"} else "On Scene"
        incidents.append(enrich_context(context, f"INC-{len(incidents) + 1:04d}", status))
        seen.add(row.get("junction"))

    st.session_state["incident_contexts"] = incidents
    st.session_state["selected_incident_id"] = incidents[0]["incident_id"]
    st.session_state["similar_drawer_id"] = ""


def active_incidents() -> list[dict[str, Any]]:
    return [ctx for ctx in st.session_state.get("incident_contexts", []) if ctx.get("status") in ACTIVE_STATUSES]


def all_incidents() -> list[dict[str, Any]]:
    return list(st.session_state.get("incident_contexts", []))


def selected_incident() -> dict[str, Any]:
    incidents = all_incidents()
    selected_id = st.session_state.get("selected_incident_id")
    for incident in incidents:
        if incident["incident_id"] == selected_id:
            return incident
    return incidents[0] if incidents else {}


def update_incident_status(incident_id: str, status: str) -> None:
    now = datetime.now().isoformat(timespec="seconds")
    for incident in st.session_state.get("incident_contexts", []):
        if incident.get("incident_id") == incident_id:
            incident["status"] = status
            incident.setdefault("status_timestamps", {}).setdefault(status, now)
            break


def recommended_route(diversion: dict[str, Any]) -> dict[str, Any] | None:
    for key in ("route_1", "route_2", "route_3"):
        route = diversion.get(key)
        if route and route.get("path"):
            return route
    return None


def command_stats(incidents: list[dict[str, Any]]) -> dict[str, Any]:
    active = [item for item in incidents if item.get("status") in ACTIVE_STATUSES]
    resolved = [item for item in incidents if item.get("status") in {"Resolved", "Closed"}]
    durations = [float(item.get("predicted_duration", 0)) for item in resolved or incidents if item.get("predicted_duration")]
    return {
        "Active Events": len(active),
        "Critical Events": sum(1 for item in active if item.get("risk_level") == "Critical"),
        "Resolved Today": len(resolved),
        "Diversions Active": sum(1 for item in active if recommended_route(item.get("diversion", {}))),
        "Officers Deployed": sum(int(item.get("dispatch", {}).get("assigned_officers", 0)) for item in active),
        "Barricades Deployed": sum(int(item.get("resources", {}).get("barricades", 0)) for item in active),
        "Average Clearance Time": round(mean(durations), 1) if durations else 0,
    }


def render_stats_board(incidents: list[dict[str, Any]]) -> None:
    stats = command_stats(incidents)
    columns = st.columns(7)
    for column, (label, value) in zip(columns, stats.items()):
        suffix = " min" if label == "Average Clearance Time" else ""
        column.metric(label, f"{value}{suffix}")


def render_city_highlights(incidents: list[dict[str, Any]]) -> None:
    active = [item for item in incidents if item.get("status") in ACTIVE_STATUSES]
    if not active:
        st.info("No active city incidents match the current filters.")
        return
    highest = max(active, key=lambda item: float(item.get("corridor_risk") or 0))
    zone_counts: dict[str, int] = {}
    for item in active:
        zone = item.get("event", {}).get("zone") or "Unknown"
        zone_counts[zone] = zone_counts.get(zone, 0) + 1
    most_congested_zone = max(zone_counts.items(), key=lambda item: item[1])
    c1, c2 = st.columns(2)
    c1.metric(
        "Highest Risk Corridor",
        highest.get("event", {}).get("corridor", "Unknown"),
        f"{highest.get('corridor_risk', 0)}/100",
    )
    c2.metric("Most Congested Zone", most_congested_zone[0], f"{most_congested_zone[1]} active")


def incident_table_rows(incidents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for item in incidents:
        event = item["event"]
        dispatch = item["dispatch"]
        rows.append(
            {
                "Incident ID": item["incident_id"],
                "Type": event.get("event_type"),
                "Cause": event.get("event_cause"),
                "Zone": event.get("zone"),
                "Corridor": event.get("corridor"),
                "Junction": event.get("junction"),
                "Risk": item.get("risk_level"),
                "Impact": item.get("predicted_impact"),
                "Duration": item.get("predicted_duration"),
                "Station": dispatch.get("station_name"),
                "Status": item.get("status"),
            }
        )
    return rows


def filtered_incidents(incidents: list[dict[str, Any]], rows: list[dict]) -> list[dict[str, Any]]:
    with st.expander("City-Wide Incident Filter", expanded=True):
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        zone = c1.selectbox("Zone", ["All"] + options(rows, "zone", ["Unknown"]))
        risk = c2.selectbox("Risk Level", ["All", "Low", "Moderate", "High", "Critical"])
        event_type = c3.selectbox("Event Type", ["All"] + options(rows, "event_type", ["unplanned", "planned"]))
        status = c4.selectbox("Status", ["All"] + STATUSES)
        stations = sorted({item.get("dispatch", {}).get("station_name", "Unknown") for item in incidents})
        station = c5.selectbox("Police Station", ["All"] + stations)
        selected_date = c6.date_input("Date", value=datetime.now().date())
        search = st.text_input("Search Incident ID, Corridor, Junction, Police Station, or Event Type")

    result = incidents
    if zone != "All":
        result = [item for item in result if item["event"].get("zone") == zone]
    if risk != "All":
        result = [item for item in result if item.get("risk_level") == risk]
    if event_type != "All":
        result = [item for item in result if item["event"].get("event_type") == event_type]
    if status != "All":
        result = [item for item in result if item.get("status") == status]
    if station != "All":
        result = [item for item in result if item.get("dispatch", {}).get("station_name") == station]
    if selected_date:
        result = [
            item
            for item in result
            if str(item.get("created_at", datetime.now().date().isoformat()))[:10] == selected_date.isoformat()
        ]
    if search:
        needle = search.lower()
        result = [
            item
            for item in result
            if any(
                needle in str(value).lower()
                for value in (
                    item.get("incident_id"),
                    item["event"].get("corridor"),
                    item["event"].get("junction"),
                    item["event"].get("event_type"),
                    item.get("dispatch", {}).get("station_name"),
                )
            )
        ]
        if result:
            st.session_state["selected_incident_id"] = result[0]["incident_id"]
            st.success(f"Jumped to {result[0]['incident_id']}")
    return result


def circle_points(lat: float, lon: float, radius_m: int) -> tuple[list[float], list[float]]:
    lats = []
    lons = []
    lat_radius = radius_m / 111_320
    lon_radius = radius_m / (111_320 * max(0.2, math.cos(math.radians(lat))))
    for degree in range(0, 361, 10):
        radians = math.radians(degree)
        lats.append(lat + lat_radius * math.sin(radians))
        lons.append(lon + lon_radius * math.cos(radians))
    return lats, lons


def densify_coordinates(coordinates: list[tuple[float, float]], max_step_km: float = 0.35) -> list[tuple[float, float]]:
    if len(coordinates) < 2:
        return coordinates
    densified = [coordinates[0]]
    for start, end in zip(coordinates, coordinates[1:]):
        distance = coordinate_distance_km(start[0], start[1], end[0], end[1])
        steps = max(1, math.ceil(distance / max_step_km))
        for step in range(1, steps + 1):
            ratio = step / steps
            densified.append(
                (
                    start[0] + (end[0] - start[0]) * ratio,
                    start[1] + (end[1] - start[1]) * ratio,
                )
            )
    return densified


def route_plot_coordinates(item: dict[str, Any], route: dict[str, Any]) -> list[tuple[float, float]]:
    coordinates = [(float(lat), float(lon)) for lat, lon in route.get("coordinates", []) if lat is not None and lon is not None]
    if len(coordinates) < 2 and route.get("path"):
        coordinates = modules()["diversion"].route_coordinates(route["path"])

    event = item.get("event", {})
    event_lat = event.get("latitude")
    event_lon = event.get("longitude")
    if event_lat is not None and event_lon is not None:
        event_point = (float(event_lat), float(event_lon))
        if not coordinates:
            coordinates = [event_point]
        elif coordinate_distance_km(event_point[0], event_point[1], coordinates[0][0], coordinates[0][1]) > 0.15:
            coordinates = [event_point] + coordinates

    return densify_coordinates(coordinates)


def build_control_room_map(incidents: list[dict[str, Any]], layers: list[str], height: int = 610) -> go.Figure:
    fig = go.Figure()
    station_points: dict[str, dict[str, Any]] = {}
    for item in incidents:
        event = item["event"]
        lat = event.get("latitude")
        lon = event.get("longitude")
        if lat is None or lon is None:
            continue
        hover = (
            f"Incident Type: {event.get('event_type')}<br>"
            f"Location: {event.get('junction')} / {event.get('corridor')}<br>"
            f"Impact Score: {item.get('predicted_impact')}<br>"
            f"Duration: {item.get('predicted_duration')} min<br>"
            f"Assigned Station: {item.get('dispatch', {}).get('station_name')}<br>"
            f"Status: {item.get('status')}"
        )
        if "Impact Radius" in layers:
            circle_lat, circle_lon = circle_points(float(lat), float(lon), int(item.get("impact_radius_m", 1000)))
            fig.add_trace(
                go.Scattermap(
                    lat=circle_lat,
                    lon=circle_lon,
                    mode="lines",
                    fill="toself",
                    fillcolor="rgba(220,38,38,0.12)",
                    line={"width": 1, "color": "rgba(220,38,38,0.35)"},
                    name=f"{item['incident_id']} Radius",
                    hoverinfo="skip",
                )
            )
        if "Active Incidents" in layers:
            fig.add_trace(
                go.Scattermap(
                    lat=[lat],
                    lon=[lon],
                    mode="markers",
                    marker={"size": 18, "color": RISK_COLORS.get(item.get("risk_level"), "#64748b")},
                    text=[item["incident_id"]],
                    hovertext=[hover],
                    hoverinfo="text",
                    name=f"{item['incident_id']} {item.get('risk_level')}",
                )
            )
        if "Active Diversions" in layers:
            for key, color in ROUTE_COLORS.items():
                route = item.get("diversion", {}).get(key) or {}
                coordinates = route_plot_coordinates(item, route)
                if not coordinates or len(coordinates) < 2:
                    continue
                route_hover = (
                    f"{route.get('route_name')}<br>"
                    f"Reason: {event.get('event_cause')} at {event.get('junction')}<br>"
                    f"Distance: {route.get('distance_km')} km<br>"
                    f"Delay: {route.get('estimated_delay_min')} min<br>"
                    f"Status: {route.get('status')}"
                )
                fig.add_trace(
                    go.Scattermap(
                        lat=[point[0] for point in coordinates],
                        lon=[point[1] for point in coordinates],
                        mode="lines+markers",
                        line={"width": 5, "color": color},
                        marker={"size": 6, "color": color},
                        hovertext=[route_hover] * len(coordinates),
                        hoverinfo="text",
                        name=f"{item['incident_id']} {route.get('route_name', key)}",
                    )
                )
        dispatch = item.get("dispatch", {})
        station_name = dispatch.get("station_name")
        if station_name and station_name not in {"Location unavailable", "No station data"}:
            station_points[station_name] = dispatch
        if "Officer Deployment" in layers:
            officer_count = max(1, int(dispatch.get("assigned_officers", 0)))
            for idx in range(officer_count):
                offset = (idx + 1) * 0.00035
                fig.add_trace(
                    go.Scattermap(
                        lat=[float(lat) + offset],
                        lon=[float(lon) - offset],
                        mode="markers",
                        marker={"size": 9, "color": "#0f172a"},
                        hovertext=[f"Officer deployment<br>{station_name}<br>{item['incident_id']}"],
                        hoverinfo="text",
                        name="Officer Location",
                        showlegend=False,
                    )
                )
        if "Barricades" in layers:
            barricades = int(item.get("resources", {}).get("barricades", 0))
            for idx in range(barricades):
                fig.add_trace(
                    go.Scattermap(
                        lat=[float(lat) - 0.0003 * (idx + 1)],
                        lon=[float(lon) + 0.0003 * (idx + 1)],
                        mode="markers",
                        marker={"size": 10, "color": "#7c2d12", "symbol": "square"},
                        hovertext=[f"Barricade<br>{item['incident_id']}"],
                        hoverinfo="text",
                        name="Barricade",
                        showlegend=False,
                    )
                )

    if "Police Stations" in layers:
        dispatch_engine = modules()["dispatch"]
        for station in dispatch_engine.stations:
            fig.add_trace(
                go.Scattermap(
                    lat=[station["latitude"]],
                    lon=[station["longitude"]],
                    mode="markers",
                    marker={"size": 13, "color": "#7c3aed"},
                    hovertext=[f"{station['station_name']}<br>Available officers: {station['available_officers']}"],
                    hoverinfo="text",
                    name="Police Station",
                    showlegend=False,
                )
            )

    center = map_center(incidents)
    fig.update_layout(
        map_style="open-street-map",
        map={"center": {"lat": center[0], "lon": center[1]}, "zoom": 11},
        height=height,
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
        legend={"orientation": "h"},
    )
    return fig


def map_center(incidents: list[dict[str, Any]]) -> tuple[float, float]:
    points = [
        (item["event"].get("latitude"), item["event"].get("longitude"))
        for item in incidents
        if item["event"].get("latitude") is not None and item["event"].get("longitude") is not None
    ]
    if not points:
        return BENGALURU_CENTER
    return (sum(float(lat) for lat, _ in points) / len(points), sum(float(lon) for _, lon in points) / len(points))


def render_prediction_confidence(context: dict[str, Any]) -> None:
    confidence = context.get("prediction_confidence") or confidence_panel(context)
    c1, c2, c3 = st.columns(3)
    c1.metric("Impact Prediction", f"{confidence['impact']} +/- {confidence['impact_margin']}")
    c2.metric("Duration Prediction", f"{confidence['duration']} +/- {confidence['duration_margin']} min")
    c3.metric("Confidence", f"{confidence['confidence']}%")


def render_lifecycle(context: dict[str, Any]) -> None:
    current = context.get("status", "Reported")
    st.caption("Incident Lifecycle")
    cols = st.columns(len(STATUSES))
    for col, step in zip(cols, STATUSES):
        marker = "●" if STATUSES.index(current) >= STATUSES.index(step) else "○"
        timestamp = context.get("status_timestamps", {}).get(step, "")
        col.write(f"{marker} {step}")
        if timestamp:
            col.caption(timestamp.replace("T", " "))


def render_similar_panel(context: dict[str, Any]) -> None:
    similar = context.get("similar_incidents", {})
    matches = similar.get("matches", [])
    resources = similar.get("recommended_resources", {})
    officers = [match.get("resources_used", {}).get("officers", 0) for match in matches]
    barricades = [match.get("resources_used", {}).get("barricades", 0) for match in matches]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Average Duration", f"{similar.get('average_duration', 0)} min")
    c2.metric("Average Impact", similar.get("average_impact", 0))
    c3.metric("Average Officers", round(mean(officers), 1) if officers else resources.get("officers", 0))
    c4.metric("Average Barricades", round(mean(barricades), 1) if barricades else resources.get("barricades", 0))
    table = []
    for match in matches:
        used = match.get("resources_used", {})
        table.append(
            {
                "Incident Type": match.get("event_type"),
                "Date": match.get("date"),
                "Corridor": match.get("corridor"),
                "Duration": match.get("duration_minutes"),
                "Impact": match.get("impact_score"),
                "Resources Used": f"{used.get('officers', 0)} officers, {used.get('barricades', 0)} barricades",
                "Outcome": match.get("outcome"),
            }
        )
    st.dataframe(table, use_container_width=True, hide_index=True)


def copy_advisory_button(text: str, key: str) -> None:
    escaped = html.escape(text)
    js_text = json.dumps(text)
    # Calculate height dynamically: ~18px per char width of 60 chars + button + padding
    approx_lines = max(3, len(text) // 60 + 1)
    box_height = 52 + approx_lines * 22  # button row + text lines
    components.html(
        f"""
        <button id="{key}" style="padding:0.45rem 0.7rem;border:1px solid #475569;border-radius:6px;background:#1e293b;color:#f8fafc;cursor:pointer;margin-bottom:0.5rem;">
          Copy Advisory
        </button>
        <script>
        const button = document.getElementById("{key}");
        button.onclick = async () => {{
          await navigator.clipboard.writeText({js_text});
          button.innerText = "Copied";
        }};
        </script>
        <p style="font-family:sans-serif;font-size:0.82rem;color:#94a3b8;margin:0;line-height:1.5;">{escaped}</p>
        """,
        height=box_height,
    )


def render_advisory(context: dict[str, Any], key: str) -> None:
    advisory = context["public_advisory"]
    copy_advisory_button(advisory, f"copy-{key}")
    st.download_button(
        "Download Advisory",
        advisory,
        file_name=f"{context['incident_id']}_public_advisory.txt",
        mime="text/plain",
        key=f"download-{key}",
    )


def incident_card(context: dict[str, Any], expanded: bool = False, allow_status_update: bool = True) -> None:
    event = context["event"]
    with st.expander(
        f"{context['incident_id']} | {event.get('event_type')} | {event.get('junction')} | {context['risk_level']}",
        expanded=expanded,
    ):
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Impact", context["predicted_impact"], context["risk_level"])
        c2.metric("Duration", f"{context['predicted_duration']} min")
        c3.metric("Station", context["dispatch"].get("station_name", "Unknown"))
        c4.metric("Officers", context["dispatch"].get("assigned_officers", 0))
        c5.metric("Barricades", context["resources"].get("barricades", 0))
        if allow_status_update:
            status = st.selectbox(
                "Status",
                STATUSES,
                index=STATUSES.index(context.get("status", "Reported")),
                key=f"status-{context['incident_id']}",
            )
            if status != context.get("status"):
                update_incident_status(context["incident_id"], status)
                st.rerun()
        else:
            st.caption(f"Status: {context.get('status', 'Reported')}")
        render_lifecycle(context)
        left, right = st.columns([1, 1])
        with left:
            st.caption("Action Plan")
            for index, step in enumerate(context.get("action_plan", []), start=1):
                st.write(f"{index}. {step}")
        with right:
            st.caption("Citizen Advisory")
            render_advisory(context, context["incident_id"])
        if st.button("View Similar Incidents", key=f"similar-{context['incident_id']}"):
            st.session_state["similar_drawer_id"] = context["incident_id"]


def render_new_incident_form(loaded: dict, rows: list[dict], available_barricades: int, key_prefix: str) -> None:
    with st.form(f"{key_prefix}-new-incident"):
        c1, c2, c3 = st.columns(3)
        event_type = c1.selectbox("Incident Type", options(rows, "event_type", ["unplanned", "planned"]), key=f"{key_prefix}-type")
        cause = c2.selectbox("Cause", options(rows, "event_cause", ["vehicle_breakdown"]), key=f"{key_prefix}-cause")
        priority = c3.selectbox("Priority", ["High", "Low"], key=f"{key_prefix}-priority")
        c4, c5, c6 = st.columns(3)
        latitude = c4.number_input("Latitude", value=BENGALURU_CENTER[0], format="%.6f", key=f"{key_prefix}-lat")
        longitude = c5.number_input("Longitude", value=BENGALURU_CENTER[1], format="%.6f", key=f"{key_prefix}-lon")
        road_closure = c6.checkbox("Road Closure", key=f"{key_prefix}-closure")
        c7, c8, c9 = st.columns(3)
        zone = c7.selectbox("Zone", options(rows, "zone", ["Unknown"]), key=f"{key_prefix}-zone")
        corridor = c8.selectbox("Corridor", options(rows, "corridor", ["Non-corridor"]), key=f"{key_prefix}-corridor")
        junction = c9.selectbox("Junction", options(rows, "junction", ["Unknown"]), key=f"{key_prefix}-junction")
        description = st.text_area("Description", key=f"{key_prefix}-description")
        submitted = st.form_submit_button("Submit Incident")
    if submitted:
        event = build_event(
            loaded["preprocessor"],
            event_type=event_type,
            event_cause=cause,
            latitude=latitude,
            longitude=longitude,
            zone=zone,
            corridor=corridor,
            junction=junction,
            priority=priority,
            requires_road_closure=road_closure,
            event_description=description,
            start_datetime=datetime.now().isoformat(),
        )
        origin, destination = route_endpoints(loaded, event, rows)
        context = build_command_context(loaded, event, rows, available_barricades, origin, destination)
        next_id = f"INC-{len(st.session_state.get('incident_contexts', [])) + 1:04d}"
        st.session_state.setdefault("incident_contexts", []).append(enrich_context(context, next_id, "Reported"))
        st.session_state["selected_incident_id"] = next_id
        st.success(f"{next_id} created and processed through prediction, dispatch, diversion, and action planning.")
        st.rerun()


def report_incident_button(loaded: dict, rows: list[dict], available_barricades: int) -> None:
    if hasattr(st, "dialog"):
        @st.dialog("+ Report New Incident")
        def incident_dialog() -> None:
            render_new_incident_form(loaded, rows, available_barricades, "dialog")

        if st.button("+ Report New Incident", type="primary"):
            incident_dialog()
    else:
        with st.expander("+ Report New Incident"):
            render_new_incident_form(loaded, rows, available_barricades, "inline")


def page_live_assessment(loaded: dict, context: dict[str, Any]) -> None:
    st.subheader("Live Event Assessment")
    render_stats_board(all_incidents())

    live_preview = enrich_context(context, "LIVE-PREVIEW", "Reported")
    st.plotly_chart(
        build_control_room_map(
            [live_preview],
            ["Active Incidents", "Impact Radius", "Active Diversions", "Police Stations", "Officer Deployment", "Barricades"],
            height=660,
        ),
        use_container_width=True,
    )

    st.subheader("Live Preview")
    incident_card(live_preview, expanded=True, allow_status_update=False)

    st.subheader("Assessment Results")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Impact Score", f"{context['predicted_impact']}/100", context["risk_level"])
    c2.metric("Duration Prediction", f"{context['predicted_duration']} min")
    c3.metric("Corridor Risk", f"{context['corridor_risk']}/100")
    c4.metric("Assigned Station", context["dispatch"].get("station_name", "Unknown"))
    render_prediction_confidence(context)
    with st.expander("Prediction Explanation", expanded=True):
        st.dataframe(context["explanations"], use_container_width=True, hide_index=True)

    st.subheader("Recommendations")
    left, right = st.columns([1, 1])
    with left:
        st.caption("Resource Recommendation")
        resource_rows = [
            {"Resource": "Officers", "Recommended": str(context["resources"].get("officers", 0))},
            {"Resource": "Barricades", "Recommended": str(context["resources"].get("barricades", 0))},
            {"Resource": "Tow Truck", "Recommended": "Yes" if context["resources"].get("tow_truck") else "No"},
            {"Resource": "Escalation", "Recommended": context["resources"].get("escalation_level", "None")},
        ]
        st.dataframe(resource_rows, use_container_width=True, hide_index=True)
        st.caption("Dispatch Assignment")
        st.write(
            f"{context['dispatch'].get('station_name', 'Unknown')} | "
            f"{context['dispatch'].get('distance_km', 0)} km | "
            f"ETA {context['dispatch'].get('eta_minutes', 0)} min"
        )
    with right:
        st.caption("Action Plan")
        for index, step in enumerate(context.get("action_plan", []), start=1):
            st.write(f"{index}. {step}")
        st.caption("Citizen Advisory")
        render_advisory(live_preview, "live-preview")


def page_control_room(loaded: dict, rows: list[dict], available_barricades: int) -> None:
    st.subheader("Control Room")
    report_incident_button(loaded, rows, available_barricades)
    render_stats_board(all_incidents())
    render_city_highlights(all_incidents())
    filtered = filtered_incidents(active_incidents(), rows)
    layers = st.multiselect(
        "Map Layers",
        ["Active Incidents", "Impact Radius", "Active Diversions", "Police Stations", "Officer Deployment", "Barricades"],
        default=["Active Incidents", "Impact Radius", "Active Diversions", "Police Stations", "Officer Deployment", "Barricades"],
    )
    st.plotly_chart(build_control_room_map(filtered, layers), use_container_width=True)

    left, right = st.columns([1.35, 0.65])
    with left:
        st.caption("Active Incident Cards")
        for item in filtered:
            incident_card(item, expanded=item["incident_id"] == st.session_state.get("selected_incident_id"))
    with right:
        drawer_id = st.session_state.get("similar_drawer_id")
        drawer_context = next((item for item in all_incidents() if item["incident_id"] == drawer_id), None)
        if drawer_context:
            st.subheader("Similar Incidents")
            st.caption(drawer_context["incident_id"])
            render_similar_panel(drawer_context)
        else:
            st.info("Select View Similar Incidents on any incident card to open the drawer.")


def page_diversion_management() -> None:
    st.subheader("Diversion Management")
    context = selected_incident()
    st.selectbox(
        "Incident",
        [item["incident_id"] for item in all_incidents()],
        index=[item["incident_id"] for item in all_incidents()].index(context["incident_id"]),
        key="selected_incident_id",
    )
    context = selected_incident()
    layers = st.multiselect(
        "Map Layers",
        ["Active Incidents", "Impact Radius", "Active Diversions", "Police Stations", "Officer Deployment", "Barricades"],
        default=["Active Incidents", "Impact Radius", "Active Diversions", "Police Stations", "Officer Deployment"],
        key="diversion-layers",
    )
    st.plotly_chart(build_control_room_map([context], layers), use_container_width=True)
    for key in ("route_1", "route_2", "route_3"):
        route = context.get("diversion", {}).get(key) or {}
        if not route:
            continue
        with st.expander(route.get("route_name", key), expanded=key == "route_1"):
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Distance", f"{route.get('distance_km')} km")
            c2.metric("Estimated Delay", f"{route.get('estimated_delay_min')} min")
            c3.metric("Junction Count", route.get("junction_count", 0))
            c4.metric("Corridor Risk", route.get("corridor_risk", route.get("risk_score", 0)))
            c5.metric("Suitability", route.get("suitability", "Traffic"))
            st.caption("Why this route was selected")
            for reason in route.get("why_selected", ["Balanced distance, delay, junction count, and corridor risk"]):
                st.write(f"- {reason}")
            st.caption("Path")
            st.write(" -> ".join(route.get("path", [])))


def resource_board_rows(loaded: dict, incidents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for station in loaded["dispatch"].stations:
        station_name = station["station_name"]
        assigned_officers = sum(
            int(item.get("dispatch", {}).get("assigned_officers", 0))
            for item in incidents
            if item.get("dispatch", {}).get("station_name") == station_name
        )
        assigned_barricades = sum(
            int(item.get("resources", {}).get("barricades", 0))
            for item in incidents
            if item.get("dispatch", {}).get("station_name") == station_name
        )
        total_barricades = max(6, int(station["available_officers"] * 0.75))
        total_vehicles = max(2, int(station["available_officers"] / 5))
        officer_deficit = max(0, assigned_officers - int(station["available_officers"]))
        barricade_deficit = max(0, assigned_barricades - total_barricades)
        rows.append(
            {
                "Police Station": station_name,
                "Available Officers": max(0, int(station["available_officers"]) - assigned_officers),
                "Assigned Officers": assigned_officers,
                "Available Barricades": max(0, total_barricades - assigned_barricades),
                "Available Vehicles": total_vehicles,
                "Officer Deficit": officer_deficit,
                "Barricade Deficit": barricade_deficit,
                "Status": "Overloaded" if officer_deficit or barricade_deficit else "Available",
            }
        )
    return rows


def page_resource_management(loaded: dict) -> None:
    st.subheader("Resource Management")
    rows = resource_board_rows(loaded, active_incidents())
    st.dataframe(rows, use_container_width=True, hide_index=True)
    shortages = [row for row in rows if row["Officer Deficit"] or row["Barricade Deficit"]]
    if shortages:
        st.warning("Resource Sufficiency Checker detected station-level shortages.")
        st.dataframe(shortages, use_container_width=True, hide_index=True)
    else:
        st.success("All assigned stations have sufficient officers and barricades for active incidents.")
    fig = px.bar(rows, x="Police Station", y=["Available Officers", "Assigned Officers"], barmode="group")
    st.plotly_chart(fig, use_container_width=True)


def page_similar_incidents() -> None:
    st.subheader("Similar Incidents")
    ids = [item["incident_id"] for item in all_incidents()]
    selected = st.selectbox("Incident", ids, index=ids.index(st.session_state.get("selected_incident_id", ids[0])))
    st.session_state["selected_incident_id"] = selected
    render_similar_panel(selected_incident())


def page_scenario_simulator() -> None:
    st.subheader("Scenario Simulator")
    ids = [item["incident_id"] for item in all_incidents()]
    selected = st.selectbox("Incident", ids, index=ids.index(st.session_state.get("selected_incident_id", ids[0])))
    st.session_state["selected_incident_id"] = selected
    context = selected_incident()

    c1, c2, c3, c4 = st.columns(4)
    officers = c1.number_input("Officers", min_value=0, max_value=50, value=int(context["resources"].get("officers", 1)))
    barricades = c2.number_input("Barricades", min_value=0, max_value=50, value=int(context["resources"].get("barricades", 0)))
    road_closure = c3.checkbox("Road Closure", value=bool(context["event"].get("requires_road_closure")))
    diversion = c4.checkbox("Diversion Activated", value=True)

    base_impact = float(context["predicted_impact"])
    base_duration = float(context["predicted_duration"])
    intervention_credit = officers * 3.5 + barricades * 2.5 + (18 if diversion else 0) - (8 if road_closure else 0)
    with_impact = max(5, round(base_impact - intervention_credit, 2))
    with_duration = max(5, round(base_duration * max(0.35, 1 - intervention_credit / 140), 2))
    without = {"Impact": base_impact, "Duration": base_duration, "Risk Level": ImpactPredictionEngine.risk_level(base_impact)}
    with_case = {"Impact": with_impact, "Duration": with_duration, "Risk Level": ImpactPredictionEngine.risk_level(with_impact)}

    left, right = st.columns(2)
    with left:
        st.caption("Without Intervention")
        st.metric("Impact", without["Impact"])
        st.metric("Duration", f"{without['Duration']} min")
        st.metric("Predicted Risk Level", without["Risk Level"])
    with right:
        st.caption("With Intervention")
        st.metric("Impact", with_case["Impact"])
        st.metric("Duration", f"{with_case['Duration']} min")
        st.metric("Predicted Risk Level", with_case["Risk Level"])


def page_executive_analytics() -> None:
    st.subheader("Executive Analytics")
    incidents = all_incidents()
    render_stats_board(incidents)
    table = incident_table_rows(incidents)
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(px.histogram(table, x="Risk", title="Events by Risk Level"), use_container_width=True)
        st.plotly_chart(px.histogram(table, x="Type", title="Events by Type"), use_container_width=True)
    with c2:
        st.plotly_chart(px.histogram(table, x="Zone", title="Events by Zone"), use_container_width=True)
        hourly = [{"Hour": int(item["event"].get("hour", 0)), "Incident": item["incident_id"]} for item in incidents]
        st.plotly_chart(px.histogram(hourly, x="Hour", nbins=24, title="Events by Hour"), use_container_width=True)
    st.dataframe(table, use_container_width=True, hide_index=True)


def copilot_context() -> dict[str, Any]:
    incidents = all_incidents()
    return {
        "active_incidents": incident_table_rows(active_incidents()),
        "diversions": [
            {
                "incident_id": item["incident_id"],
                "routes": [item.get("diversion", {}).get(key) for key in ("route_1", "route_2", "route_3")],
            }
            for item in incidents
        ],
        "resource_availability": resource_board_rows(modules(), active_incidents()),
        "dispatch_assignments": [item.get("dispatch", {}) | {"incident_id": item["incident_id"]} for item in incidents],
        "action_plans": [{"incident_id": item["incident_id"], "steps": item.get("action_plan", [])} for item in incidents],
        "similar_incidents": [
            {"incident_id": item["incident_id"], "similar": item.get("similar_incidents", {})} for item in incidents
        ],
        "timeline_simulator": [{"incident_id": item["incident_id"], "timeline": item.get("timeline", [])} for item in incidents],
        "summary_stats": command_stats(incidents),
    }


def page_ai_copilot() -> None:
    st.subheader("AI Copilot")
    context = copilot_context()
    st.caption("The copilot answers only from active incidents, diversions, resources, dispatches, action plans, similar incidents, and timelines.")
    examples = [
        "Which incident requires immediate attention?",
        "Which police station is overloaded?",
        "Show active diversions in South Zone.",
        "Why was this route selected?",
        "How many officers are currently deployed?",
    ]
    prompt = st.selectbox("Example Questions", [""] + examples)
    query = st.chat_input("Ask the command center copilot")
    final_query = query or prompt
    if final_query:
        with st.chat_message("user"):
            st.write(final_query)
        with st.chat_message("assistant"):
            st.write(LLMCopilot(LLMService.from_env()).answer(final_query, context))
    else:
        st.info(LLMCopilot(LLMService()).answer("Summarize the current command center state.", context))


def main() -> None:
    st.set_page_config(
        page_title="Traffic Police Command Center",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_styles()
    st.title("Traffic Police Operations Command Center")
    # st.markdown(
    #     '',
    #     unsafe_allow_html=True,
    # )
    st.markdown(
        """
        <div class="command-subtitle">Single-page operations workspace for incident assessment, dispatch, diversions, resources, analytics, and copilot support.</div>
        """,
        unsafe_allow_html=True
    )
    page = render_top_nav()
    loaded = modules()
    rows = cleaned_rows()

    st.session_state.setdefault("available_barricades", 8)
    if page == "Live Event Assessment":
        event_values = event_input(rows)
    else:
        event_values = st.session_state.get("live_event_values", default_event_values(rows))
    event = build_event(loaded["preprocessor"], **event_payload(event_values))

    available_barricades = int(st.session_state.get("available_barricades", 8))
    origin, destination = route_endpoints(loaded, event, rows)
    context = build_command_context(loaded, event, rows, available_barricades, origin, destination)
    initialize_incidents(loaded, rows, context, available_barricades)

    if page == "Live Event Assessment":
        page_live_assessment(loaded, context)
    elif page == "Control Room":
        page_control_room(loaded, rows, int(available_barricades))
    elif page == "Diversion Management":
        page_diversion_management()
    elif page == "Resource Management":
        page_resource_management(loaded)
    elif page == "Similar Incidents":
        page_similar_incidents()
    elif page == "Scenario Simulator":
        page_scenario_simulator()
    elif page == "Executive Analytics":
        page_executive_analytics()
    elif page == "AI Copilot":
        page_ai_copilot()


if __name__ == "__main__":
    main()
