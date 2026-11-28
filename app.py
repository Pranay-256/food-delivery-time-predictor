import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import pickle
import warnings
import os
from types import SimpleNamespace
warnings.filterwarnings("ignore")

from eda_engine import render_data_quality_report
import ml_pipeline as mp                                   # shared cleaning / feature-selection / training
from custom_dataset_tab import render_custom_dataset_tab   # "Try With Your Own Dataset" tab
from dynamic_eda import stretch                            # full-width helper that works on old & new Streamlit

# ─────────────────────────────────────────────────────────────
# BASE DIR — anchor all file paths to this script's own folder,
# so it works no matter which directory `streamlit run` is
# launched from. Data/model files must sit in this same folder
# (i.e. inside the "random-forest" directory alongside app.py).
# ─────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG  — must be the very first Streamlit call
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Food Delivery Time Predictor",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─────────────────────────────────────────────────────────────
# GLOBAL CSS  — dark gold/teal theme (mirrors T20 analyzer)
# ─────────────────────────────────────────────────────────────
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Rajdhani:wght@500;600;700&family=DM+Sans:ital,wght@0,300;0,400;0,500;0,600;1,300&display=swap" rel="stylesheet">

<style>

/* ── Root design tokens ───────────────────────────────────── */
:root {
    --bg-base:      #0d1117;
    --bg-card:      #161b22;
    --bg-card2:     #1c2330;
    --border:       #2a3444;
    --gold:         #c9a84c;
    --gold-light:   #e0c070;
    --teal:         #2dd4bf;
    --teal-dim:     #1a9e8f;
    --text-primary: #e6edf3;
    --text-muted:   #8b949e;
    --text-accent:  #c9a84c;
    --danger:       #e05252;
    --success:      #3fb950;
}

/* ── Base ─────────────────────────────────────────────────── */
html, body, [class*="css"] {
    font-family: 'Rajdhani', sans-serif !important;
    background-color: var(--bg-base) !important;
    color: var(--text-primary) !important;
}

/* ── Streamlit chrome ─────────────────────────────────────── */
.stApp { background-color: var(--bg-base) !important; }
header[data-testid="stHeader"] {
    background: var(--bg-base) !important;
    border-bottom: 1px solid var(--border);
}
.main .block-container {
    padding-top: 1rem !important;
    padding-bottom: 1rem !important;
}

/* ── Dataframe ────────────────────────────────────────────── */
div[data-testid="stDataFrame"] {
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    background: var(--bg-card) !important;
    padding: 4px;
}
div[data-testid="stDataFrame"] > div { background: var(--bg-card) !important; }
div[data-testid="stDataFrame"] * { color: var(--text-primary) !important; }
div[data-testid="stDataFrame"] thead { background: rgba(201,168,76,0.08) !important; }

/* ── Tabs ─────────────────────────────────────────────────── */
div[data-baseweb="tab-list"] { border-bottom: 1px solid var(--border) !important; gap: 18px; }
div[data-baseweb="tab"] {
    font-family: 'Rajdhani', sans-serif !important;
    font-weight: 600;
    font-size: 16px !important;
    color: var(--text-muted) !important;
    padding: 10px 6px !important;
    background: transparent !important;
    border: none !important;
    transition: all 0.2s ease;
}
div[data-baseweb="tab"]:hover { color: var(--gold-light) !important; }
div[data-baseweb="tab"][aria-selected="true"] {
    color: var(--gold) !important;
    font-weight: 700 !important;
}
div[data-baseweb="tab-highlight"] {
    background: linear-gradient(90deg, #c9a84c, #e0c070) !important;
    height: 3px !important;
    border-radius: 2px;
}
div[data-baseweb="tab-border"] { background: transparent !important; }

/* ── Sidebar ──────────────────────────────────────────────── */
section[data-testid="stSidebar"] {
    background: var(--bg-card) !important;
    border-right: 1px solid var(--border) !important;
}
section[data-testid="stSidebar"] * { color: var(--text-primary) !important; }

/* ── Radio buttons ────────────────────────────────────────── */
div[role="radiogroup"] label {
    font-family: 'Rajdhani', sans-serif !important;
    color: var(--text-primary) !important;
    font-size: 14px;
}

/* ── Primary buttons (Predict / Confirm) ──────────────────── */
div.stButton > button {
    font-family: 'Rajdhani', sans-serif !important;
    font-weight: 700;
    font-size: 15px;
    letter-spacing: 0.5px;
    background: linear-gradient(135deg, #b8922e 0%, #e0c070 100%) !important;
    color: #0d1117 !important;
    border: none !important;
    border-radius: 8px !important;
    height: 46px !important;
    transition: all 0.25s ease !important;
    box-shadow: 0 2px 12px rgba(201,168,76,0.15) !important;
}
div.stButton > button:hover {
    background: linear-gradient(135deg, #e0c070 0%, #c9a84c 100%) !important;
    box-shadow: 0 4px 20px rgba(201,168,76,0.30) !important;
    transform: translateY(-1px) !important;
}

/* ── Download buttons (template / sample CSV) ─────────────── */
div[data-testid="stDownloadButton"] > button,
div.stDownloadButton > button {
    font-family: 'Rajdhani', sans-serif !important;
    font-weight: 700 !important;
    font-size: 15px !important;
    letter-spacing: 0.3px !important;
    white-space: nowrap !important;
    background: linear-gradient(135deg, #b8922e 0%, #e0c070 100%) !important;
    color: #0d1117 !important;
    border: 1.5px solid rgba(255,255,255,0.85) !important;
    border-radius: 10px !important;
    height: 48px !important;
    padding: 0 18px !important;
    transition: background 0.2s ease, border-color 0.2s ease, transform 0.2s ease !important;
    box-shadow: none !important;
    filter: none !important;
    outline: none !important;
}
div[data-testid="stDownloadButton"] > button:hover,
div[data-testid="stDownloadButton"] > button:focus,
div[data-testid="stDownloadButton"] > button:focus-visible,
div[data-testid="stDownloadButton"] > button:active,
div.stDownloadButton > button:hover,
div.stDownloadButton > button:focus,
div.stDownloadButton > button:focus-visible,
div.stDownloadButton > button:active {
    background: linear-gradient(135deg, #e0c070 0%, #c9a84c 100%) !important;
    border-color: #ffffff !important;
    box-shadow: none !important;
    filter: none !important;
    outline: none !important;
    transform: translateY(-1px) !important;
}
div[data-testid="stDownloadButton"] > button p,
div.stDownloadButton > button p {
    color: #0d1117 !important;
    font-weight: 700 !important;
    white-space: nowrap !important;
}

/* ── File uploader ────────────────────────────────────────── */
div[data-testid="stFileUploaderDropzone"] {
    background: var(--bg-card2) !important;
    border: 1px dashed rgba(201,168,76,0.55) !important;
    border-radius: 10px !important;
}
div[data-testid="stFileUploaderDropzone"] button {
    background: #1f1d18 !important;
    color: var(--text-primary) !important;
    border: 1px solid var(--border) !important;
    height: 44px !important;
}
div[data-testid="stFileUploader"] label p { font-size: 15px !important; }

/* ── Selectbox ────────────────────────────────────────────── */
div[data-baseweb="select"] > div {
    background: var(--bg-card2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text-primary) !important;
}
div[data-baseweb="select"] span { color: var(--text-primary) !important; }

/* ── Number inputs ────────────────────────────────────────── */
div[data-testid="stNumberInput"] input {
    background: var(--bg-card2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text-primary) !important;
    font-family: 'Rajdhani', sans-serif !important;
}

/* ── Slider ───────────────────────────────────────────────── */
div[data-testid="stSlider"] * { color: var(--text-primary) !important; }

/* ── Expanders ────────────────────────────────────────────── */
div[data-testid="stExpander"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    margin-bottom: 10px;
}
div[data-testid="stExpander"] summary {
    font-family: 'Rajdhani', sans-serif !important;
    font-weight: 600 !important;
    font-size: 16px !important;
    color: var(--gold-light) !important;
    padding: 12px 16px !important;
}
div[data-testid="stExpander"] summary:hover {
    background: rgba(201,168,76,0.06) !important;
}
div[data-testid="stExpander"] > div > div {
    padding: 4px 20px 16px 20px !important;
    color: var(--text-primary) !important;
}

/* ── Alerts / messages ────────────────────────────────────── */
div[data-testid="stAlert"] {
    border-radius: 8px !important;
    font-family: 'Rajdhani', sans-serif !important;
}

/* ── Dividers ─────────────────────────────────────────────── */
hr { border-color: var(--border) !important; }

/* ── Text / paragraphs ────────────────────────────────────── */
p, li {
    color: var(--text-primary) !important;
    font-family: 'Rajdhani', sans-serif !important;
    font-size: 15px;
    line-height: 1.7;
}
strong { color: var(--text-primary) !important; font-family: 'Rajdhani', sans-serif !important; }
a { color: var(--teal) !important; text-decoration: none; }
a:hover { text-decoration: underline; color: var(--gold-light) !important; }

/* ── Scrollbar ────────────────────────────────────────────── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--bg-base); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: var(--gold); }

/* ── Equal-height columns & cards ─────────────────────────── */
div[data-testid="stHorizontalBlock"] {
    align-items: stretch !important;
}
div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"],
div[data-testid="stHorizontalBlock"] > div[data-testid="column"] {
    display: flex !important;
    flex-direction: column !important;
}
div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"] > div,
div[data-testid="stHorizontalBlock"] > div[data-testid="column"] > div,
div[data-testid="stHorizontalBlock"] div[data-testid="stVerticalBlock"],
div[data-testid="stHorizontalBlock"] div[data-testid="stElementContainer"],
div[data-testid="stHorizontalBlock"] div.element-container,
div[data-testid="stHorizontalBlock"] div[data-testid="stMarkdownContainer"] {
    flex: 1 1 auto !important;
    display: flex !important;
    flex-direction: column !important;
}
div[data-testid="stHorizontalBlock"] div[data-testid="stMarkdownContainer"] > div {
    flex: 1 1 auto !important;
}

</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# HELPER: gold-accented section heading banner
# ─────────────────────────────────────────────────────────────
def section_heading(icon, title):
    """Renders a left-gold-bordered section heading banner."""
    st.markdown(f"""
    <div style="
        background: linear-gradient(90deg, rgba(201,168,76,0.12) 0%, transparent 100%);
        border-left: 3px solid #c9a84c;
        border-radius: 0 8px 8px 0;
        padding: 10px 18px;
        margin: 18px 0 10px 0;
    ">
        <span style="font-family:'Rajdhani',sans-serif; font-weight:700;
                     font-size:20px; color:#e0c070; letter-spacing:0.4px;">
            {title}
        </span>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# HELPER: column info card used in the Dataset tab
# ─────────────────────────────────────────────────────────────
def column_info(title, text):
    """Renders a styled card describing a single dataset column."""
    st.markdown(f"""
    <div style="
        background: #1c2330;
        border: 1px solid #2a3444;
        border-radius: 8px;
        padding: 10px 16px;
        margin-bottom: 8px;
    ">
        <p style="font-family:'Rajdhani',sans-serif; font-weight:700;
                  font-size:15px; color:#c9a84c; margin:0 0 4px 0;">{title}</p>
        <p style="font-family:'Rajdhani',sans-serif; font-size:13px;
                  color:#8b949e; margin:0; line-height:1.55;">{text}</p>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# HELPER: KPI / stat card
# ─────────────────────────────────────────────────────────────
def kpi_card(icon, label, value, sub=""):
    """Renders a metric highlight card with an icon, label, value, and optional subtitle."""
    icon_html = f'<div style="font-size:26px; margin-bottom:6px; line-height:1;">{icon}</div>' if icon else ""
    sub_html = (
        f"<p style='font-size:12px; color:#8b949e; margin:6px 0 0 0; "
        f"font-family:Rajdhani,sans-serif; line-height:1.3;'>{sub}</p>"
        if sub else ""
    )
    card_html = f"""
    <div style="
        background: linear-gradient(145deg, #1c2330, #161b22);
        border: 1px solid #2a3444;
        border-top: 2px solid #c9a84c;
        border-radius: 10px;
        padding: 18px 14px;
        text-align: center;
        height: 100%;
        box-sizing: border-box;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        box-shadow: 0 4px 16px rgba(0,0,0,0.3);
    ">
        {icon_html}
        <p style="font-family:'Rajdhani',sans-serif; font-size:12px;
                  color:#8b949e; margin:0 0 4px 0; text-transform:uppercase;
                  letter-spacing:0.8px;">{label}</p>
        <p style="font-family:'Rajdhani',sans-serif; font-weight:700;
                  font-size:20px; color:#e0c070; margin:0; line-height:1.2;
                  overflow-wrap:anywhere;">{value}</p>
        {sub_html}
    </div>
    """
    # drop empty / indented lines: a blank line would end the HTML block and show the rest as code
    st.markdown("\n".join(l.strip() for l in card_html.splitlines() if l.strip()), unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# HELPER: teal-bordered insight line
# ─────────────────────────────────────────────────────────────
def insight_line(number, text):
    """Renders a numbered insight bullet with a teal left border."""
    st.markdown(f"""
    <div style="
        background: #161b22;
        border: 1px solid #2a3444;
        border-left: 3px solid #2dd4bf;
        border-radius: 0 8px 8px 0;
        padding: 10px 16px;
        margin-bottom: 8px;
        display: flex;
        align-items: flex-start;
        gap: 12px;
    ">
        <span style="font-family:'Rajdhani',sans-serif; font-weight:700;
                     font-size:16px; color:#2dd4bf; min-width:24px;">{number}.</span>
        <span style="font-family:'Rajdhani',sans-serif; font-size:14px;
                     color:#e6edf3; line-height:1.6;">{text}</span>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# HELPER: prediction result banner
# ─────────────────────────────────────────────────────────────
def prediction_banner(minutes, footnote="Predicted by Random Forest &nbsp;·&nbsp; R² = 0.95"):
    """Renders a large, eye-catching prediction result card."""
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, #1c2330 0%, #161b22 100%);
        border: 2px solid #c9a84c;
        border-radius: 14px;
        padding: 28px 24px;
        text-align: center;
        box-shadow: 0 6px 32px rgba(201,168,76,0.18);
        margin: 20px 0 10px 0;
    ">
        <div style="font-size:44px; margin-bottom:8px;">🍕</div>
        <p style="font-family:'Rajdhani',sans-serif; font-size:14px;
                  color:#8b949e; margin:0 0 6px 0; text-transform:uppercase;
                  letter-spacing:1px;">Estimated Delivery Time</p>
        <p style="font-family:'Rajdhani',sans-serif; font-weight:700;
                  font-size:52px; color:#e0c070; margin:0; line-height:1.1;">
            {minutes:.1f}
            <span style="font-size:22px; color:#c9a84c; font-weight:600;">min</span>
        </p>
        <p style="font-family:'Rajdhani',sans-serif; font-size:13px;
                  color:#8b949e; margin:10px 0 0 0;">
            {footnote}
        </p>
    </div>
    """, unsafe_allow_html=True)


# Bundle of UI helpers handed to the separate modules (custom_dataset_tab.py, dynamic_eda.py)
UI = SimpleNamespace(
    section_heading=section_heading,
    insight_line=insight_line,
    kpi_card=kpi_card,
    prediction_banner=prediction_banner,
)


# ─────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────
def render_footer():
    """Renders the three-column footer with a credit line."""
    components.html("""
    <link href="https://fonts.googleapis.com/css2?family=Rajdhani:wght@600;700&display=swap" rel="stylesheet">
    <style>
        body { margin:0; padding:0; background:transparent; }
        .footer-wrapper {
            display: flex;
            justify-content: space-around;
            text-align: center;
            padding: 28px 0 8px 0;
            border-top: 1px solid #2a3444;
        }
        .footer-item { flex: 1; }
        .footer-item .title {
            font-family: 'Rajdhani', sans-serif;
            font-weight: 700;
            font-size: 16px;
            color: #c9a84c;
            margin-bottom: 4px;
        }
        .footer-item .subtitle {
            font-family: 'Rajdhani', sans-serif;
            color: #8b949e;
            font-size: 13px;
            margin-top: 0;
        }
        .footer-credit {
            text-align: center;
            font-family: 'Rajdhani', sans-serif;
            color: #8b949e;
            font-size: 13px;
            padding: 12px 0 4px 0;
        }
    </style>

    <div class="footer-wrapper">
        <div class="footer-item">
            <p class="title">&#127381; Instant Predictions</p>
            <p class="subtitle">Get delivery estimates in seconds</p>
        </div>
        <div class="footer-item">
            <p class="title">&#129504; ML-Powered</p>
            <p class="subtitle">Random Forest Model</p>
        </div>
        <div class="footer-item">
            <p class="title">&#128202; Data-Driven</p>
            <p class="subtitle">Trained on recorded delivery data</p>
        </div>
    </div>

    <p class="footer-credit">Developed by Pranay Jha, Pradeep Singh Negi and Pranav Jha &nbsp;|&nbsp; Powered by Python, Scikit-learn &amp; Streamlit</p>
    """, height=240)


# ─────────────────────────────────────────────────────────────
# LOAD MODEL & ENCODERS (cached so they load only once)
# ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    """
    Loads the trained RandomForestRegressor and the dict of LabelEncoders
    from disk. Cached by st.cache_resource so pickle files are read only
    once across all sessions.
    """
    with open(os.path.join(BASE_DIR, "rf_model.pkl"), "rb") as f:
        model = pickle.load(f)
    with open(os.path.join(BASE_DIR, "label_encoders.pkl"), "rb") as f:
        encoders = pickle.load(f)
    return model, encoders


# ─────────────────────────────────────────────────────────────
# LOAD THE "BEST FEATURES" MODEL (created by the notebook, Step 31).
# If rf_model_best.pkl / best_features.json are not in the folder yet,
# ml_pipeline trains it once from Food_Delivery_Times.csv and saves it.
# ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_best_artifacts():
    return mp.load_or_build_best_model(BASE_DIR, "Food_Delivery_Times.csv")


# ─────────────────────────────────────────────────────────────
# LOAD DEFAULT DATASET (cached so the CSV is read from disk only
# once, not on every slider / dropdown interaction)
# ─────────────────────────────────────────────────────────────
DATASET_FILENAME   = "Food_Delivery_Times.csv"
DATASET_SOURCE_URL = "https://www.kaggle.com/datasets/denkuznetz/food-delivery-time-prediction"


@st.cache_data(show_spinner=False)
def load_default_dataset():
    """
    Reads the training dataset (Food_Delivery_Times.csv) and cleans it exactly
    like the notebook does (Step 6):
      - missing Weather / Traffic_Level / Time_of_Day -> most frequent value
      - missing Courier_Experience_yrs                -> median
      - Order_ID column dropped (identifier only)
    Cached by st.cache_data, so the file is read only once.
    """
    df = pd.read_csv(os.path.join(BASE_DIR, DATASET_FILENAME))
    for col in ["Weather", "Traffic_Level", "Time_of_Day"]:
        df[col] = df[col].fillna(df[col].mode()[0])
    df["Courier_Experience_yrs"] = df["Courier_Experience_yrs"].fillna(df["Courier_Experience_yrs"].median())
    return df.drop(columns="Order_ID")


# ─────────────────────────────────────────────────────────────
# SIDEBAR — page navigation
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding:12px 0 8px 0;'>
        <span style="font-family:'Rajdhani',sans-serif; font-size:28px;
                     font-weight:700; color:#c9a84c; letter-spacing:1px;">
            Tool Menu
        </span>
    </div>
    """, unsafe_allow_html=True)

    # Radio drives the two main pages
    page = st.radio("", ["Home", "About"])

    st.markdown("<hr style='border-color:#2a3444; margin:12px 0;'>", unsafe_allow_html=True)

    st.markdown("""
    <div style="font-family:'Rajdhani',sans-serif; font-size:13px;
                color:#8b949e; line-height:1.9; padding:4px 0;">
        <strong style="color:#c9a84c;">Current Version:</strong> 1.2<br>
        <strong style="color:#c9a84c;">Release Date:</strong> November 2026<br>
        <strong style="color:#c9a84c;">Model:</strong> Random Forest<br>
        <strong style="color:#c9a84c;">R² Score:</strong> 0.95
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# MAIN LAYOUT — centered column (mirrors T20 analyzer)
# ─────────────────────────────────────────────────────────────
left, center, right = st.columns([1, 3, 1])

with center:

    # ══════════════════════════════════════════════════════
    # HOME PAGE
    # ══════════════════════════════════════════════════════
    if page == "Home":

        # ── Hero header ───────────────────────────────────
        st.markdown("""
        <div style="text-align:center; padding:24px 0 6px 0;">
            <h1 style="font-family:'Rajdhani',sans-serif; font-weight:700;
                       font-size:38px; color:#e0c070; margin:0; letter-spacing:1px;">
                🍕 Food Delivery Time Predictor
            </h1>
            <p style="font-family:'Rajdhani',sans-serif; font-size:16px;
                      color:#8b949e; margin:6px 0 0 0;">
                ML-Powered Delivery Time Estimation Using Random Forest
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("")

        # Best-features model + summary (optional: the app still works without it)
        try:
            best_model, best_meta = load_best_artifacts()
        except Exception:
            best_model, best_meta = None, None
        n_best = len(best_meta["best_features"]) if best_meta else None

        # ── Three tabs: Overview | Test the model | Your own dataset ───
        tab1, tab2, tab3 = st.tabs(["Overview", "Test The Model", "Try With Your Own Dataset"])


        # ══════════════════════════════════════════════════
        # TAB 1 — OVERVIEW
        # ══════════════════════════════════════════════════
        with tab1:

            # ── Project Introduction & Value Proposition ───
            section_heading("", "Welcome to Food Delivery Time Predictor")
            st.markdown("""
            <p style="font-family:'Rajdhani',sans-serif; font-size:15px; color:#e6edf3;
                      margin: -4px 0 12px 0; line-height:1.75;">
                Accurate delivery estimates keep customers informed and dispatch operations running smoothly.
                While major platforms like <strong style="color:#e0c070;">Zomato and Swiggy</strong> rely on
                enterprise-grade logistics software, <strong style="color:#2dd4bf;">local cafes, restaurants, cloud
                kitchens, and neighborhood vendors</strong> rarely have access to expensive predictive tools.
                Powered by a <strong style="color:#e0c070;">Random Forest</strong> model, this application
                gives any business data-driven ETA predictions — accounting for <strong style="color:#2dd4bf;">road
                distance, kitchen preparation time, live weather, traffic congestion, time of day, vehicle type, and
                courier experience</strong>. Use <strong style="color:#e0c070;">all features</strong> or only the
                <strong style="color:#e0c070;">best features</strong>, or upload
                <strong style="color:#2dd4bf;">your own delivery data</strong> to get a private model, charts and
                insights for your business.
            </p>
            """, unsafe_allow_html=True)

            # ── Quick Highlights KPI Row ───────────────────
            h1, h2, h3, h4 = st.columns(4)
            with h1:
                kpi_card("", "Instant ETAs", "Real-Time", "Predictions in seconds")
            with h2:
                kpi_card("", "ML Engine", "Random Forest", "Ensemble regressor")
            with h3:
                kpi_card("", "Easy Inputs", f"7 or {n_best} Factors" if n_best else "7 Factors",
                         "All features or best only")
            with h4:
                kpi_card("", "Built-In EDA", "Auto Report", "Also for your own data")

            st.divider()

            # ── Real-World Business Use Case ───────────────
            section_heading("", "Built for Local Cafes, Restaurants & Small Vendors")
            st.markdown("""
            <p style="font-family:'Rajdhani',sans-serif; font-size:14px; color:#8b949e;
                      margin: -6px 0 14px 0; line-height:1.6;">
                Turning everyday delivery details into confident customer quotes.
            </p>
            """, unsafe_allow_html=True)

            st.markdown("""
            <div style="
                background: #1c2330;
                border: 1px solid #2a3444;
                border-left: 3px solid #2dd4bf;
                border-radius: 0 10px 10px 0;
                padding: 18px 20px;
                box-sizing: border-box;
            ">
                <p style="font-family:'Rajdhani',sans-serif; font-weight:700;
                          font-size:17px; color:#2dd4bf; margin:0 0 8px 0;">
                    Real-Life Phone Order Scenario
                </p>
                <p style="font-family:'Rajdhani',sans-serif; font-size:14px;
                          color:#e6edf3; margin:0; line-height:1.65;">
                    Imagine a customer calls to order food and asks: <em>"How long will it take? I have other tasks to run."</em>
                    Even an inexperienced staff member won't draw a blank — simply select live <strong style="color:#2dd4bf;">traffic,
                    time of day, weather, distance, prep time, and rider experience</strong>. If the model predicts
                    <strong style="color:#e0c070;">30 min</strong> (with ~3–5 min average error), staff can confidently quote
                    <strong style="color:#e0c070;">"around 30 minutes"</strong> or <strong style="color:#e0c070;">"30 to 35 minutes"</strong>.
                </p>
            </div>
            """, unsafe_allow_html=True)

            st.divider()

            # ── Workflow Guide ─────────────────────────────
            section_heading("", "How to Use This Application")
            st.markdown("""
            <p style="font-family:'Rajdhani',sans-serif; font-size:14px; color:#8b949e;
                      margin: -6px 0 14px 0; line-height:1.6;">
                Try the pre-trained model right away — or bring your own delivery records.
            </p>
            """, unsafe_allow_html=True)

            g1, g2 = st.columns(2)
            with g1:
                st.markdown("""
                <div style="
                    background: #1c2330;
                    border: 1px solid #2a3444;
                    border-top: 3px solid #c9a84c;
                    border-radius: 10px;
                    padding: 18px 20px;
                    box-sizing: border-box;
                    height: 100%;
                ">
                    <p style="font-family:'Rajdhani',sans-serif; font-weight:700;
                              font-size:18px; color:#e0c070; margin:0 0 8px 0;">
                        Test The Model
                    </p>
                    <p style="font-family:'Rajdhani',sans-serif; font-size:14px;
                              color:#e6edf3; margin:0 0 10px 0; line-height:1.6;">
                        Live predictions from our pre-trained Random Forest — no setup needed.
                    </p>
                    <p style="font-family:'Rajdhani',sans-serif; font-size:13px;
                              color:#8b949e; margin:0; line-height:1.65;">
                        ▸ Choose <strong>Select all features</strong> or <strong>Use best features</strong>
                        and open the feature summary drop-down to see the difference<br>
                        ▸ Adjust the sliders and dropdowns shown for the chosen features<br>
                        ▸ Click <strong>Predict Delivery Time</strong> for an instant ETA and explore the Data Quality Report below
                    </p>
                </div>
                """, unsafe_allow_html=True)
            with g2:
                st.markdown("""
                <div style="
                    background: #1c2330;
                    border: 1px solid #2a3444;
                    border-top: 3px solid #2dd4bf;
                    border-radius: 10px;
                    padding: 18px 20px;
                    box-sizing: border-box;
                    height: 100%;
                ">
                    <p style="font-family:'Rajdhani',sans-serif; font-weight:700;
                              font-size:18px; color:#2dd4bf; margin:0 0 8px 0;">
                        Try With Your Own Dataset
                    </p>
                    <p style="font-family:'Rajdhani',sans-serif; font-size:14px;
                              color:#e6edf3; margin:0 0 10px 0; line-height:1.6;">
                        Upload your own delivery records and get a model built for <em>your</em> business.
                    </p>
                    <p style="font-family:'Rajdhani',sans-serif; font-size:13px;
                              color:#8b949e; margin:0; line-height:1.65;">
                        ▸ Download the <strong>empty template</strong> or the <strong>sample filled CSV</strong><br>
                        ▸ Upload your CSV — it is cleaned automatically and charts with insights are drawn<br>
                        ▸ Choose all features or the best features, set the inputs and click <strong>Predict Delivery Time</strong>
                    </p>
                </div>
                """, unsafe_allow_html=True)

            st.divider()

            # ── Model performance summary ──────────────────
            section_heading("", "Model Performance at a Glance")
            st.markdown("""
            <p style="font-family:'Rajdhani',sans-serif; font-size:14px; color:#8b949e;
                      margin: -6px 0 12px 0; line-height:1.6;">
                The <strong style="color:#e0c070;">Random Forest</strong> (default settings, 100 trees) clearly
                outperforms the single Decision Tree baseline across all evaluation metrics.
            </p>
            """, unsafe_allow_html=True)

            m1, m2, m3, m4 = st.columns(4)
            with m1:
                kpi_card("", "R² Score",   "0.95",     "vs 0.92 baseline")
            with m2:
                kpi_card("", "MAE",        "3.72 min", "vs 4.79 baseline")
            with m3:
                kpi_card("", "RMSE",       "4.97 min", "vs 6.19 baseline")
            with m4:
                kpi_card("", "Estimators", "100 trees","Default setting")

            if best_meta:
                st.markdown("<br>", unsafe_allow_html=True)
                bm, am = best_meta["metrics_best"], best_meta["metrics_all"]
                st.markdown(f"""
                <p style="font-family:'Rajdhani',sans-serif; font-size:14px; color:#8b949e;
                          margin: 0 0 12px 0; line-height:1.6;">
                    The <strong style="color:#2dd4bf;">best-features model</strong> uses only the
                    <strong style="color:#e0c070;">{len(best_meta['best_features'])} most useful factors</strong>
                    ({', '.join(mp.FEATURE_META[f][1] for f in best_meta['best_features'])}) and is a simpler alternative
                    to the all-features model.
                </p>
                """, unsafe_allow_html=True)
                b1, b2, b3, b4 = st.columns(4)
                with b1:
                    kpi_card("", "Best-Features Model", f"{len(best_meta['best_features'])} of 7", "Factors used")
                with b2:
                    kpi_card("", "R² Score", f"{bm['R2']:.2f}", f"vs {am['R2']:.2f} with all features")
                with b3:
                    kpi_card("", "MAE", f"{bm['MAE']:.2f} min", f"vs {am['MAE']:.2f} with all features")
                with b4:
                    kpi_card("", "RMSE", f"{bm['RMSE']:.2f} min", f"vs {am['RMSE']:.2f} with all features")

            st.markdown("<br>", unsafe_allow_html=True)

            # ── Key findings ───────────────────────────────
            section_heading("", "Key Findings From Our Modeling")
            insight_line(1, "Distance is the strongest driver — longer routes consistently produce longer delivery times.")
            insight_line(2, "Adverse weather (Snowy / Rainy) adds measurable delay compared to Clear conditions.")
            insight_line(3, "High traffic levels independently extend delivery duration regardless of trip distance.")
            insight_line(4, "More experienced couriers (7–9 yrs) deliver noticeably faster than new couriers (0–1 yr).")
            insight_line(5, "One-Way ANOVA testing showed that only Weather and Traffic Level have a statistically significant effect on delivery time (p < 0.05). Time of Day (p = 0.93) and Vehicle Type (p = 0.71) did not pass.")
            insight_line(6, "The main model still keeps <strong>all 7 features</strong>: Time of Day, Vehicle Type and Courier Experience "
                            "are weak on their own, but they show their importance in <strong>combined comparisons</strong> "
                            "(for example Vehicle Type vs Weather vs Traffic Level). Prefer a simpler model? Switch to "
                            "<strong>Use best features</strong> in the Test The Model tab.")
            insight_line(7, "Have your own delivery records? The <strong>Try With Your Own Dataset</strong> tab cleans your file, draws the same "
                            "charts and insights, finds the best features for <em>your</em> data and trains a model on it.")

            st.markdown("<div style='height:50px;'></div>", unsafe_allow_html=True)


        # ══════════════════════════════════════════════════
        # TAB 2 — TEST THE MODEL
        # ══════════════════════════════════════════════════
        with tab2:

            # Load model artifacts (cached)
            try:
                model, label_encoders = load_artifacts()
            except Exception as e:
                st.error(f"Could not load model files: {e}")
                st.stop()

            section_heading("", "Predict Delivery Time")
            st.markdown("""
            <p style="font-family:'Rajdhani',sans-serif; font-size:14px; color:#8b949e;
                      margin: -6px 0 14px 0; line-height:1.6;">
                Choose which features the model should use, fill in the delivery details and click
                <strong style="color:#e0c070;">Predict</strong> to get an estimated delivery time from the trained Random Forest model.
            </p>
            """, unsafe_allow_html=True)

            # ── Feature set: all features (current model) or best features ──
            ALL_LABEL, BEST_LABEL = "Select all features", "Use best features"
            mode_options = [ALL_LABEL] + ([BEST_LABEL] if best_meta else [])
            feature_mode = st.radio(
                "Which features should the model use?",
                mode_options,
                horizontal=True,
                key="feature_mode",
                help="All features = the full Random Forest (7 inputs). Best features = a simpler Random Forest "
                     "trained only on the factors that matter most for delivery time."
            )
            use_best = (feature_mode == BEST_LABEL) and best_meta is not None
            active_features = list(best_meta["best_features"]) if use_best else list(mp.FEATURE_ORDER)
            active_model    = best_model if use_best else model
            active_r2       = best_meta["metrics_best"]["R2"] if use_best else 0.95

            if use_best:
                st.markdown(
                    "<p style=\"font-family:'Rajdhani',sans-serif; font-size:14px; color:#2dd4bf; margin:4px 0 0 0;\">"
                    f"Using the <strong>{len(active_features)} best features</strong>: "
                    + ", ".join(mp.FEATURE_META[f][1] for f in active_features) + "</p>",
                    unsafe_allow_html=True)
            else:
                st.markdown(
                    "<p style=\"font-family:'Rajdhani',sans-serif; font-size:14px; color:#2dd4bf; margin:4px 0 0 0;\">"
                    "Using <strong>all 7 features</strong>.</p>", unsafe_allow_html=True)

            # ── Drop-down summary: best features vs all features ──
            if best_meta:
                with st.expander("Feature summary — best features vs all features", expanded=False):
                    sel = pd.DataFrame(best_meta["selection_table"])
                    am, bm = best_meta["metrics_all"], best_meta["metrics_best"]

                    insight_line(1, "<strong>All features (7):</strong> the full model uses every column — Distance, Weather, Traffic Level, "
                                    "Time of Day, Vehicle Type, Preparation Time and Courier Experience. Some of them are weak "
                                    "when tested one at a time, but they are kept because they still show importance in "
                                    "combined comparisons such as Vehicle Type vs Weather vs Traffic Level.")
                    insight_line(2, f"<strong>Best features ({len(best_meta['best_features'])}):</strong> "
                                    + ", ".join(mp.FEATURE_META[f][1] for f in best_meta["best_features"])
                                    + ". These are the factors with a statistically significant link to delivery time (p &lt; 0.05) "
                                      "in the Pearson correlation test (numbers) or the One-Way ANOVA test (categories).")
                    dropped_feats = [f for f in best_meta["all_features"] if f not in best_meta["best_features"]]
                    if dropped_feats:
                        insight_line(3, "<strong>Left out of the best-features model:</strong> "
                                        + ", ".join(mp.FEATURE_META[f][1] for f in dropped_feats)
                                        + " — on their own they change delivery time very little.")

                    st.markdown("<p style='font-weight:700;font-size:16px;color:#c9a84c;margin:12px 0 6px 0;'>"
                                "How each feature was tested</p>", unsafe_allow_html=True)
                    show = sel[["Feature", "Test", "Statistic", "p-value", "Selected", "Why"]].copy()
                    show["Feature"] = show["Feature"].map(lambda c: f"{mp.FEATURE_ICONS[c]} {mp.FEATURE_META[c][1]}")
                    show["Statistic"] = show["Statistic"].round(3)
                    show["p-value"] = show["p-value"].map(lambda v: "< 0.0001" if v < 1e-4 else f"{v:.4f}")
                    show["Selected"] = show["Selected"].map({True: "⭐ Best", False: "—"})
                    show.insert(5, "In all-features model", "✅")
                    stretch(st.dataframe, show, hide_index=True)

                    st.markdown("<p style='font-weight:700;font-size:16px;color:#c9a84c;margin:12px 0 6px 0;'>"
                                "Accuracy comparison (same test data)</p>", unsafe_allow_html=True)
                    cmp_df = pd.DataFrame({
                        "Model": ["All features", "Best features"],
                        "Features used": [len(best_meta["all_features"]), len(best_meta["best_features"])],
                        "R²": [round(am["R2"], 3), round(bm["R2"], 3)],
                        "MAE (min)": [round(am["MAE"], 2), round(bm["MAE"], 2)],
                        "RMSE (min)": [round(am["RMSE"], 2), round(bm["RMSE"], 2)],
                    })
                    stretch(st.dataframe, cmp_df, hide_index=True)

                # ── R² cards: all features vs best features ──
                am, bm = best_meta["metrics_all"], best_meta["metrics_best"]
                st.markdown("")
                r1, r2 = st.columns(2)
                with r1:
                    kpi_card("", "R² · All Features", f"{am['R2']:.2f}",
                             f"All {len(best_meta['all_features'])} features · MAE {am['MAE']:.2f} min")
                with r2:
                    kpi_card("", "R² · Best Features", f"{bm['R2']:.2f}",
                             f"Best {len(best_meta['best_features'])} features · MAE {bm['MAE']:.2f} min")

            st.markdown("")

            # ── Input form — two-column layout (only the chosen features are shown) ──
            col_left, col_right = st.columns(2)
            raw_inputs = {}

            # ── Left column: Numerical inputs ─────────────
            with col_left:
                section_heading("", "Numerical Inputs")

                if "Distance_km" in active_features:
                    raw_inputs["Distance_km"] = st.slider(
                        "Distance (km)",
                        min_value=0.5,
                        max_value=20.0,
                        value=10.0,
                        step=0.1,
                        format="%.2f",
                        help="Road distance from restaurant to delivery address."
                    )

                if "Preparation_Time_min" in active_features:
                    raw_inputs["Preparation_Time_min"] = st.slider(
                        "Preparation Time (min)",
                        min_value=5,
                        max_value=30,
                        value=15,
                        step=1,
                        help="Time the restaurant needs to prepare the order."
                    )

                if "Courier_Experience_yrs" in active_features:
                    raw_inputs["Courier_Experience_yrs"] = st.slider(
                        "Courier Experience (years)",
                        min_value=0.0,
                        max_value=9.0,
                        value=3.0,
                        step=0.5,
                        format="%.2f",
                        help="Number of years the courier has been delivering."
                    )

            # ── Right column: Categorical inputs ──────────
            with col_right:
                section_heading("", "Categorical Inputs")

                if "Weather" in active_features:
                    raw_inputs["Weather"] = st.selectbox(
                        "Weather Condition",
                        list(label_encoders["Weather"].classes_),
                        help="Current weather at the time of delivery."
                    )

                if "Traffic_Level" in active_features:
                    raw_inputs["Traffic_Level"] = st.selectbox(
                        "Traffic Level",
                        list(label_encoders["Traffic_Level"].classes_),
                        help="Road traffic level — Low, Medium, or High."
                    )

                if "Time_of_Day" in active_features:
                    raw_inputs["Time_of_Day"] = st.selectbox(
                        "Time of Day",
                        list(label_encoders["Time_of_Day"].classes_),
                        help="Part of the day when the delivery takes place."
                    )

                if "Vehicle_Type" in active_features:
                    raw_inputs["Vehicle_Type"] = st.selectbox(
                        "Vehicle Type",
                        list(label_encoders["Vehicle_Type"].classes_),
                        help="Type of vehicle used by the courier."
                    )

            st.markdown("")

            # ── Predict button (full width) ────────────────
            predict_clicked = st.button("🔮 Predict Delivery Time", use_container_width=True)

            if predict_clicked:
                try:
                    # Encode categorical inputs using the saved LabelEncoders
                    encoded_row = {}
                    for feat in active_features:           # column order must match training
                        if feat in mp.CATEGORICAL_FEATURES:
                            encoded_row[feat] = label_encoders[feat].transform([raw_inputs[feat]])[0]
                        else:
                            encoded_row[feat] = raw_inputs[feat]
                    input_df = pd.DataFrame([encoded_row], columns=active_features)

                    # Run inference
                    prediction = active_model.predict(input_df)[0]

                    # Render the large result banner
                    if use_best:
                        prediction_banner(
                            prediction,
                            f"Predicted by Random Forest &nbsp;·&nbsp; best {len(active_features)} features "
                            f"&nbsp;·&nbsp; R² = {active_r2:.2f}")
                    else:
                        prediction_banner(
                            prediction,
                            "Predicted by Random Forest &nbsp;·&nbsp; all 7 features &nbsp;·&nbsp; R² = 0.95")

                    # ── Context cards below result ─────────
                    st.markdown("<br>", unsafe_allow_html=True)
                    section_heading("", "Inputs Used for Prediction")

                    for start in range(0, len(active_features), 4):
                        chunk = active_features[start:start + 4]
                        card_cols = st.columns(len(chunk))
                        for card_col, feat in zip(card_cols, chunk):
                            icon, label, unit = mp.FEATURE_META[feat]
                            val = raw_inputs[feat]
                            text = f"{val:g} {unit}".strip() if feat in mp.NUMERIC_FEATURES else str(val)
                            with card_col:
                                kpi_card(icon, label, text)
                        st.markdown("<br>", unsafe_allow_html=True)

                except Exception as e:
                    st.error(f"Prediction failed: {e}")

            # ── Tip section shown when no prediction yet ───
            else:
                st.markdown("<br>", unsafe_allow_html=True)
                with st.expander("Tips for Using the Predictor", expanded=False):
                    insight_line(1, "Distance has the biggest impact — even a few extra kilometres significantly increases estimated time.")
                    insight_line(2, "Select <strong>High</strong> traffic to simulate peak-hour conditions like lunch or dinner rush.")
                    insight_line(3, "Couriers with <strong>0–1 year</strong> of experience tend to take longer than those with <strong>7+ years</strong>.")
                    insight_line(4, "Snowy and Rainy weather conditions add the most delay compared to Clear weather.")
                    insight_line(5, "Prediction accuracy: <strong>MAE ≈ 3.72 min</strong> with all features — results are typically within a few minutes of actual time.")
                    insight_line(6, "Want fewer inputs? Switch to <strong>Use best features</strong> — only the factors that matter most are shown.")

            # ══════════════════════════════════════════════
            # BELOW THE PREDICTION SECTION
            # Dataset overview + data quality report
            # ══════════════════════════════════════════════
            st.divider()

            # Cached read — does not hit the disk again on widget changes
            default_df = None
            try:
                default_df = load_default_dataset()
            except Exception as e:
                st.warning(f"Could not load {DATASET_FILENAME}: {e}")

            # ── Dataset Overview & Source Link ─────────────
            section_heading("", "Dataset Overview & Source")
            st.markdown("""
            <p style="font-family:'Rajdhani',sans-serif; font-size:14px; color:#8b949e;
                      margin: -6px 0 12px 0; line-height:1.6;">
                The model behind the predictions above was trained on the dataset summarised below.
            </p>
            """, unsafe_allow_html=True)

            n_records = f"{len(default_df):,}" if default_df is not None else "1,000"

            d1, d2, d3 = st.columns(3)
            with d1:
                kpi_card("", "Records", n_records, "Delivery orders")
            with d2:
                kpi_card("", "Predictive Drivers", "7", "3 numeric · 4 categorical")
            with d3:
                kpi_card("", "Target", "Delivery Time", "In minutes")

            st.markdown("<br>", unsafe_allow_html=True)

            st.markdown(f"""
            <div style="background:#1c2330; border:1px solid #2a3444; border-left:3px solid #2dd4bf;
                        border-radius:0 10px 10px 0; padding:14px 18px;">
                <p style="font-family:'Rajdhani',sans-serif; font-weight:700; font-size:16px;
                          color:#c9a84c; margin:0 0 6px 0;">Dataset Reference</p>
                <p style="font-family:'Rajdhani',sans-serif; font-size:14px; color:#e6edf3;
                          margin:0 0 8px 0; line-height:1.6;">
                    <strong>Drivers:</strong> Distance, Weather, Traffic Level, Time of Day,
                    Vehicle Type, Preparation Time and Courier Experience.
                    <code>Order_ID</code> is an identifier only and is never used for modeling.
                </p>
                <a href="{DATASET_SOURCE_URL}" target="_blank" rel="noopener noreferrer"
                   style="font-family:'Rajdhani',sans-serif; font-size:14px; font-weight:600;">
                    View the original dataset source ↗
                </a>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # ── Data Quality Report (eda_engine) ───────────
            section_heading("", "Data Quality Report")
            st.markdown("""
            <p style="font-family:'Rajdhani',sans-serif; font-size:14px; color:#8b949e;
                      margin: -6px 0 12px 0; line-height:1.6;">
                Charts and plain-English insights generated directly from the
                training data — distributions, categorical rankings, average-time bars,
                box plots, two-factor heatmaps, and correlations.
            </p>
            """, unsafe_allow_html=True)

            if default_df is not None:
                render_data_quality_report(default_df)

            st.markdown("<div style='height:50px;'></div>", unsafe_allow_html=True)


        # ══════════════════════════════════════════════════
        # TAB 3 — TRY WITH YOUR OWN DATASET
        # (logic lives in custom_dataset_tab.py, ml_pipeline.py and dynamic_eda.py)
        # ══════════════════════════════════════════════════
        with tab3:
            render_custom_dataset_tab(UI)


    # ══════════════════════════════════════════════════════
    # ABOUT PAGE
    # ══════════════════════════════════════════════════════
    elif page == "About":

        # ── Hero header ───────────────────────────────────
        st.markdown("""
        <div style="text-align:center; padding:24px 0 6px 0;">
            <h1 style="font-family:'Rajdhani',sans-serif; font-weight:700;
                       font-size:38px; color:#e0c070; margin:0; letter-spacing:1px;">
                Food Delivery Time Predictor
            </h1>
            <p style="font-family:'Rajdhani',sans-serif; font-size:16px;
                      color:#8b949e; margin:6px 0 0 0;">
                ML-Powered Delivery Time Estimation Using Random Forest
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.divider()

#        # ── About the developer ────────────────────────────
#        with st.expander("About the Developer — Pranay Jha", expanded=False):
#            st.markdown("""
#            <div style="font-family:'Rajdhani',sans-serif; font-size:14px;
#                        color:#e6edf3; line-height:1.8;">
#                Hello! My name is <strong style="color:#e0c070;">Pranay Jha</strong>, and I am currently pursuing a
#                <strong style="color:#e0c070;">Bachelor of Technology (B.Tech) in Computer Science and Engineering</strong>.<br><br>
#                I am actively learning and exploring the fields of
#                <strong style="color:#e0c070;">Machine Learning, Data Science, and Predictive Analytics</strong>.<br><br>
#                This project — <strong style="color:#c9a84c;">Food Delivery Time Predictor</strong> — is part of my
#                learning journey in applied machine learning, demonstrating how real-world regression problems can
#                be solved with ensemble models and deployed as interactive web tools.<br><br>
#                The tool uses Python libraries including
#                <strong style="color:#2dd4bf;">NumPy, Pandas, Scikit-learn, and Streamlit</strong>.
#            </div>
#            """, unsafe_allow_html=True)
#
#            st.markdown("<br>", unsafe_allow_html=True)
#
#            st.markdown("""
#            <p style="font-family:'Rajdhani',sans-serif; font-weight:700;
#                       font-size:17px; color:#c9a84c; margin-bottom:8px;">
#                Key Highlights
#            </p>
#            """, unsafe_allow_html=True)
#
#            for item in [
#                "Built a <strong>regression model</strong> on real-world structured delivery data",
#                "Performed <strong>data cleaning, EDA, and feature engineering</strong>",
#                "Applied <strong>LabelEncoding</strong> to categorical features for model training",
#                "Deployed the model as an <strong>interactive Streamlit web application</strong>",
#            ]:
#                st.markdown(f"""
#                <div style="background:#1c2330; border:1px solid #2a3444; border-radius:6px;
#                            padding:8px 14px; margin-bottom:6px; font-family:'Rajdhani',sans-serif;
#                            font-size:14px; color:#e6edf3;">
#                    ▸ &nbsp;{item}
#                </div>
#                """, unsafe_allow_html=True)
#
#            st.markdown("<br>", unsafe_allow_html=True)
#
#            st.markdown("""
#            <p style="font-family:'Rajdhani',sans-serif; font-weight:700;
#                       font-size:17px; color:#c9a84c; margin-bottom:8px;">
#                Connect With Me
#            </p>
#            """, unsafe_allow_html=True)
#
#            st.markdown("""
#            <div style="display:flex; gap:14px; flex-wrap:wrap;">
#                <a href="https://github.com/Pranay-256" target="_blank"
#                   style="display:inline-flex; align-items:center; gap:8px;
#                          background:#1c2330; border:1px solid #2a3444;
#                          border-radius:8px; padding:10px 18px;
#                          font-family:'Rajdhani',sans-serif; font-size:14px;
#                          color:#e6edf3; text-decoration:none;">
#                    <span style="font-size:18px;"></span>
#                    <strong>GitHub</strong> — Pranay-256
#                </a>
#                <a href="https://www.linkedin.com/in/pranay-jha-6582a937b/" target="_blank"
#                   style="display:inline-flex; align-items:center; gap:8px;
#                          background:#1c2330; border:1px solid #2a3444;
#                          border-radius:8px; padding:10px 18px;
#                          font-family:'Rajdhani',sans-serif; font-size:14px;
#                          color:#e6edf3; text-decoration:none;">
#                    <span style="font-size:18px;"></span>
#                    <strong>LinkedIn</strong> — Pranay Jha
#                </a>
#            </div>
#            """, unsafe_allow_html=True)
#            st.markdown("<br>", unsafe_allow_html=True)

        # ── Problem statement ──────────────────────────────
        with st.expander("Problem Statement", expanded=False):
            st.markdown("""
            <div style="font-family:'Rajdhani',sans-serif; font-size:14px;
                        color:#e6edf3; line-height:1.8; margin-bottom:10px;">
                Accurate delivery time estimation is a <strong style="color:#e0c070;">critical challenge</strong>
                for food delivery platforms, restaurants, and logistics operations.
                When estimated times are inaccurate, it leads to customer dissatisfaction,
                poor courier planning, and revenue loss.
            </div>
            """, unsafe_allow_html=True)

            st.markdown("""
            <p style="font-family:'Rajdhani',sans-serif; font-weight:700;
                       font-size:17px; color:#c9a84c; margin:8px 0;">Key Challenges</p>
            """, unsafe_allow_html=True)

            for item in [
                "Delivery time depends on <strong>multiple interacting factors</strong> (distance, weather, traffic, courier skill)",
                "Simple rule-based estimates <strong>fail in dynamic real-world conditions</strong>",
                "Categorical factors like <strong>weather and traffic are hard to quantify</strong> manually",
                "No easy way to <strong>compare courier performance</strong> against expected baselines",
                "Customers expect <strong>precise ETAs</strong>, not just rough estimates",
            ]:
                st.markdown(f"""
                <div style="background:#1c2330; border:1px solid #2a3444; border-left:3px solid #e05252;
                            border-radius:0 6px 6px 0; padding:8px 14px; margin-bottom:6px;
                            font-family:'Rajdhani',sans-serif; font-size:14px; color:#e6edf3;">
                    {item}
                </div>
                """, unsafe_allow_html=True)

        # ── Objective ──────────────────────────────────────
        with st.expander("Objective", expanded=False):
            st.markdown("""
            <div style="font-family:'Rajdhani',sans-serif; font-size:14px;
                        color:#e6edf3; line-height:1.8; margin-bottom:10px;">
                The goal is to build a <strong style="color:#e0c070;">reliable, data-driven regression model</strong>
                that predicts food delivery time in minutes based on readily available order and
                environmental information — replacing guesswork with accurate ML-based estimates.
            </div>
            """, unsafe_allow_html=True)

            st.markdown("""
            <p style="font-family:'Rajdhani',sans-serif; font-weight:700;
                       font-size:17px; color:#c9a84c; margin:8px 0;">Core Focus Areas</p>
            """, unsafe_allow_html=True)

            for item in [
                "Identify the key <strong>features that drive delivery time</strong>",
                "Train a <strong>Random Forest Regressor</strong> and compare it with a Decision Tree baseline",
                "Achieve <strong>high predictive accuracy</strong> (target R² > 0.90)",
                "Provide an <strong>interactive tool</strong> for instant predictions",
                "Make the model <strong>transparent and explainable</strong> via dataset documentation",
            ]:
                st.markdown(f"""
                <div style="background:#1c2330; border:1px solid #2a3444; border-left:3px solid #3fb950;
                            border-radius:0 6px 6px 0; padding:8px 14px; margin-bottom:6px;
                            font-family:'Rajdhani',sans-serif; font-size:14px; color:#e6edf3;">
                    {item}
                </div>
                """, unsafe_allow_html=True)

        # ── Tech stack ─────────────────────────────────────
        with st.expander("Tech Stack", expanded=False):
            st.markdown("""
            <p style="font-family:'Rajdhani',sans-serif; font-size:14px; color:#e6edf3;
                      line-height:1.8; margin-bottom:10px;">
                Built using the <strong style="color:#e0c070;">Python ML ecosystem</strong>
                combined with an interactive deployment framework.
            </p>
            """, unsafe_allow_html=True)

            tools = [
                ("Python",           "Core language for data processing and ML"),
                ("NumPy",            "Numerical operations and array handling"),
                ("Pandas",           "Data cleaning, manipulation, and EDA"),
                ("Seaborn",          "Charts in the Data Quality Report (built on Matplotlib)"),
                ("Matplotlib",       "Chart rendering for the dataset report and your own uploaded data"),
                ("SciPy",            "Pearson correlation and One-Way ANOVA used for best-feature selection"),
                ("Scikit-learn",     "LabelEncoder, DecisionTreeRegressor, RandomForestRegressor"),
                ("Pickle",           "Serialisation and loading of trained model artifacts"),
                ("Streamlit",        "Interactive web dashboard for model inference"),
            ]
            for name, desc in tools:
                st.markdown(f"""
                <div style="background:#1c2330; border:1px solid #2a3444; border-radius:8px;
                            padding:10px 16px; margin-bottom:7px;
                            font-family:'Rajdhani',sans-serif; font-size:14px; color:#e6edf3;
                            display:flex; gap:10px; align-items:flex-start;">
                    <strong style="color:#c9a84c; min-width:120px;">{name}</strong>
                    <span style="color:#8b949e;">{desc}</span>
                </div>
                """, unsafe_allow_html=True)

        # ── ML approach ────────────────────────────────────
        with st.expander("ML Approach & Methodology", expanded=False):

            def ml_block(emoji, title, items):
                """Renders a titled list block inside the ML approach expander."""
                st.markdown(f"""
                <div style="background:#1c2330; border:1px solid #2a3444; border-radius:8px;
                            padding:12px 16px; margin-bottom:10px;">
                    <p style="font-family:'Rajdhani',sans-serif; font-weight:700;
                              font-size:17px; color:#c9a84c; margin:0 0 6px 0;">
                        {emoji} {title}
                    </p>
                    {''.join(f"<p style='font-family:Rajdhani,sans-serif; font-size:14px; color:#e6edf3; margin:3px 0;'>▸ &nbsp;{item}</p>" for item in items)}
                </div>
                """, unsafe_allow_html=True)

            ml_block("", "Data Preprocessing", [
                "Dropped Order_ID (non-predictive identifier)",
                "Filled missing Weather, Traffic_Level, Time_of_Day with mode",
                "Filled missing Courier_Experience_yrs with median (30 values)",
                "Applied LabelEncoder to all 4 categorical features",
                "Uploaded datasets go through the same steps, plus: junk entries ('--', '?', 'unknown') treated as blank, "
                "duplicate rows removed, rows without Delivery_Time_min removed",
            ])

            ml_block("", "Feature Selection & Testing", [
                "Pearson correlation for numerical features vs Delivery_Time_min",
                "One-Way ANOVA for categorical features — Weather and Traffic_Level passed (p < 0.05); "
                "Time_of_Day (p = 0.93) and Vehicle_Type (p = 0.71) did not",
                "The main model still keeps all 7 features: Time of Day, Vehicle Type and Courier Experience are weak on their own, "
                "but they show their importance in combined comparisons (e.g. Vehicle Type vs Weather vs Traffic Level)",
                "A second model uses only the best features — those with a statistically significant link to delivery time "
                "(p < 0.05 in the Pearson correlation or One-Way ANOVA test)",
                "Users can switch between the two in the Test The Model tab and read the feature summary drop-down",
            ])

            ml_block("", "Your Own Dataset", [
                "Upload a CSV that follows the template — cleaning, charts with insights, feature selection and model training "
                "all run automatically on your file",
                "Choose all features or the best features (Pearson correlation and One-Way ANOVA, p < 0.05); a Random Forest is trained on your file",
                "Set the inputs and click Predict Delivery Time. The file is processed in memory and is not saved to disk",
            ])

            ml_block("", "Model Training", [
                "Baseline: Decision Tree Regressor (R² = 0.92, MAE = 4.79, RMSE = 6.19)",
                "Final model: Random Forest Regressor with default settings, 100 trees (R² = 0.95, MAE = 3.72, RMSE = 4.97)",
                "Hyperparameter tuning (RandomizedSearchCV) was tested but only raised R² from 0.95 to 0.96, "
                "so it was dropped to keep the model simple, small and fast",
            ])

        # ── Version history ────────────────────────────────
        with st.expander("Version History", expanded=False):

            def version_block(version, date, items):
                """Renders a version changelog block."""
                st.markdown(f"""
                <div style="background:#1c2330; border:1px solid #2a3444; border-radius:8px;
                            padding:12px 16px; margin-bottom:10px;">
                    <p style="font-family:'Rajdhani',sans-serif; font-weight:700;
                              font-size:16px; color:#c9a84c; margin:0 0 4px 0;">
                        Version {version} &nbsp;·&nbsp;
                        <span style="color:#8b949e; font-size:14px; font-weight:500;">{date}</span>
                    </p>
                    {''.join(f"<p style='font-family:Rajdhani,sans-serif; font-size:13px; color:#e6edf3; margin:3px 0;'>— {i}</p>" for i in items)}
                </div>
                """, unsafe_allow_html=True)

            version_block("1.2", "November 2026", [
                "Added a Use best features option to Test The Model, with a drop-down summary of best features vs all features",
                "Added the Try With Your Own Dataset tab: upload, automatic cleaning, visual report with insights, "
                "feature choice, private model and live predictions",
                "Added an empty template CSV and a sample filled CSV for download",
                "Moved cleaning, feature selection, training and charts into ml_pipeline.py, dynamic_eda.py and custom_dataset_tab.py",
            ])

            version_block("1.1", "November 2026", [
                "Added visualizations for the Data Quality Report",
                "Removed hyperparameter tuning: the model is now a default Random Forest (R² = 0.95)",
            ])

            version_block("1.0", "September 2026", [
                "Initial release of Food Delivery Time Predictor",
                "Optimized Random Forest model with R² = 0.96",
                "Full dark-themed Streamlit UI with gold/teal design system",
                "Dataset overview tab with column documentation",
                "Interactive prediction form with instant result banner",
                "About section with developer info, methodology, and version history",
            ])

        st.markdown("<div style='height:50px;'></div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# FOOTER — rendered on every page
# ─────────────────────────────────────────────────────────────
render_footer()