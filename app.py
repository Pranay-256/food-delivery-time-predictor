import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import pickle
import warnings
import os
import io
import hashlib
import inspect
warnings.filterwarnings("ignore")

from eda_engine import render_data_quality_report
from pipeline_engine import (
    validate_and_clean_data,
    rank_features,
    train_custom_model,
    predict_delivery_time,
    PREDICTORS,
    MIN_ROWS_FOR_TRAINING,
    IMPORTANCE_THRESHOLD_PCT,
    MIN_BEST_FEATURES,
)

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

/* ── Download buttons (match primary buttons) ─────────────── */
div[data-testid="stDownloadButton"] > button {
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
div[data-testid="stDownloadButton"] > button:hover {
    background: linear-gradient(135deg, #e0c070 0%, #c9a84c 100%) !important;
    box-shadow: 0 4px 20px rgba(201,168,76,0.30) !important;
    transform: translateY(-1px) !important;
}
div[data-testid="stDownloadButton"] > button p { color: #0d1117 !important; font-weight: 700; }

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

/* ── File uploader ────────────────────────────────────────── */
section[data-testid="stFileUploaderDropzone"],
div[data-testid="stFileUploaderDropzone"] {
    background: var(--bg-card2) !important;
    border: 1px dashed var(--gold) !important;
    border-radius: 10px !important;
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
            {icon}&nbsp; {title}
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
    sub_html = (
        f"<p style='font-size:12px; color:#8b949e; margin:6px 0 0 0; "
        f"font-family:Rajdhani,sans-serif; line-height:1.3;'>{sub}</p>"
        if sub else ""
    )
    st.markdown(f"""
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
        <div style="font-size:26px; margin-bottom:6px; line-height:1;">{icon}</div>
        <p style="font-family:'Rajdhani',sans-serif; font-size:12px;
                  color:#8b949e; margin:0 0 4px 0; text-transform:uppercase;
                  letter-spacing:0.8px;">{label}</p>
        <p style="font-family:'Rajdhani',sans-serif; font-weight:700;
                  font-size:20px; color:#e0c070; margin:0; line-height:1.2;
                  overflow-wrap:anywhere;">{value}</p>
        {sub_html}
    </div>
    """, unsafe_allow_html=True)


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
def prediction_banner(minutes, footnote="Predicted by Optimized Random Forest &nbsp;·&nbsp; R² = 0.96"):
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
            <p class="subtitle">Optimized Random Forest Model</p>
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
    with open(os.path.join(BASE_DIR, "optimized_rf_model.pkl"), "rb") as f:
        model = pickle.load(f)
    with open(os.path.join(BASE_DIR, "label_encoders.pkl"), "rb") as f:
        encoders = pickle.load(f)
    return model, encoders


# ─────────────────────────────────────────────────────────────
# LOAD DEFAULT DATASET (cached so the CSV is read from disk only
# once, not on every slider / dropdown interaction)
# ─────────────────────────────────────────────────────────────
DATASET_FILENAME   = "Food_Delivery_Times.csv"
DATASET_SOURCE_URL = "https://www.kaggle.com/datasets/denkuznetz/food-delivery-time-prediction"


@st.cache_data(show_spinner=False)
def load_default_dataset():
    """
    Reads the default training dataset (Food_Delivery_Times.csv) from disk.
    Cached by st.cache_data, so the file is parsed only once per session
    server-wide and each rerun receives a fresh copy of the cached frame.
    """
    return pd.read_csv(os.path.join(BASE_DIR, DATASET_FILENAME))


# ─────────────────────────────────────────────────────────────
# CUSTOM DATASET TAB — constants, helpers & renderer
# Everything the "📊 Try With Your Own Dataset" tab needs lives
# in this section. All user data is kept in st.session_state
# (keys prefixed "cd_") so widget interactions never reset it.
# ─────────────────────────────────────────────────────────────
SAMPLE_CSV_FILENAME = "food_delivery_sample_prefilled.csv"
TEMPLATE_COLUMNS    = [
    "Order_ID", "Distance_km", "Weather", "Traffic_Level", "Time_of_Day",
    "Vehicle_Type", "Preparation_Time_min", "Courier_Experience_yrs", "Delivery_Time_min",
]

FEATURE_LABELS = {
    "Distance_km":            "📍 Distance (km)",
    "Preparation_Time_min":   "🍳 Preparation Time (min)",
    "Courier_Experience_yrs": "🧑‍💼 Courier Experience (years)",
    "Weather":                "🌦️ Weather Condition",
    "Traffic_Level":          "🚦 Traffic Level",
    "Time_of_Day":            "🕐 Time of Day",
    "Vehicle_Type":           "🏍️ Vehicle Type",
}

MODE_BEST   = "⭐ Choose Best Features"
MODE_ALL    = "🌐 Select All Features"
MODE_CUSTOM = "🛠️ Customized Feature Selection"

# Plotly: new Streamlit uses width="stretch", old uses use_container_width
try:
    _PLOTLY_HAS_WIDTH = "width" in inspect.signature(st.plotly_chart).parameters
except (TypeError, ValueError):  # pragma: no cover
    _PLOTLY_HAS_WIDTH = False


def _plotly_show(fig, key):
    """Version-tolerant st.plotly_chart."""
    if _PLOTLY_HAS_WIDTH:
        st.plotly_chart(fig, width="stretch", theme=None, key=key)
    else:  # pragma: no cover - older Streamlit
        st.plotly_chart(fig, use_container_width=True, theme=None, key=key)


def _template_csv_bytes():
    """Empty CSV — headers only, in the exact required order."""
    return (",".join(TEMPLATE_COLUMNS) + "\n").encode("utf-8")


def _fallback_sample_df(n=50):
    """Synthetic example used only if the bundled sample CSV is missing from the repo."""
    rng = np.random.default_rng(42)
    weather = rng.choice(["Clear", "Rainy", "Foggy", "Snowy", "Windy"], n)
    traffic = rng.choice(["Low", "Medium", "High"], n)
    tod     = rng.choice(["Morning", "Afternoon", "Evening", "Night"], n)
    vehicle = rng.choice(["Bike", "Car", "Scooter"], n)
    dist    = np.round(rng.uniform(1, 20, n), 2)
    prep    = rng.integers(5, 30, n)
    exp     = np.round(rng.uniform(0, 9, n), 1)
    delay   = (
        pd.Series(traffic).map({"Low": 0, "Medium": 5, "High": 10}).to_numpy()
        + pd.Series(weather).map({"Clear": 0, "Windy": 2, "Foggy": 4, "Rainy": 6, "Snowy": 9}).to_numpy()
    )
    delivery = np.round(10 + 3.0 * dist + 0.8 * prep + delay - 0.8 * exp + rng.normal(0, 3, n)).astype(int)
    return pd.DataFrame({
        "Order_ID": np.arange(1001, 1001 + n), "Distance_km": dist, "Weather": weather,
        "Traffic_Level": traffic, "Time_of_Day": tod, "Vehicle_Type": vehicle,
        "Preparation_Time_min": prep, "Courier_Experience_yrs": exp, "Delivery_Time_min": delivery,
    })


@st.cache_data(show_spinner=False)
def load_sample_csv_bytes():
    """Pre-filled example CSV (read from disk once; synthetic fallback if the file is absent)."""
    path = os.path.join(BASE_DIR, SAMPLE_CSV_FILENAME)
    if os.path.exists(path):
        with open(path, "rb") as f:
            return f.read()
    return _fallback_sample_df().to_csv(index=False).encode("utf-8")


def _read_csv_bytes(raw):
    """Parse uploaded bytes into a DataFrame, raising ValueError with a friendly message."""
    if not raw or not raw.strip():
        raise ValueError("The uploaded file is empty.")
    last_err = None
    for enc in ("utf-8-sig", "latin-1"):
        try:
            return pd.read_csv(io.BytesIO(raw), encoding=enc, skipinitialspace=True)
        except UnicodeDecodeError as exc:
            last_err = exc
        except pd.errors.EmptyDataError:
            raise ValueError("The uploaded file has no columns or data.")
        except pd.errors.ParserError as exc:
            raise ValueError(f"The file could not be parsed as a CSV ({exc}).")
    raise ValueError(f"The file could not be decoded as text ({last_err}).")


def _reset_custom_state():
    """Forget everything derived from a previously uploaded dataset."""
    for key in list(st.session_state.keys()):
        if key.startswith(("cd_in_", "cd_models")) or key in (
            "cd_data", "cd_sig", "cd_mode", "cd_mode_saved",
            "cd_custom_widget", "cd_custom_saved",
        ):
            del st.session_state[key]


def _process_upload(uploaded_file):
    """
    Clean + rank a freshly uploaded file exactly once per distinct file
    (identified by a content hash) and cache the outcome in session_state.
    """
    raw = uploaded_file.getvalue()
    sig = f"{uploaded_file.name}:{hashlib.md5(raw).hexdigest()}"
    if st.session_state.get("cd_sig") == sig:
        return  # same file as before — keep what we already have

    _reset_custom_state()
    st.session_state["cd_sig"] = sig
    data = {"filename": uploaded_file.name, "error": None}

    try:
        raw_df = _read_csv_bytes(raw)
        data["raw_rows"] = int(len(raw_df))
        cleaned_df, logs = validate_and_clean_data(raw_df)
        data.update(cleaned_df=cleaned_df, logs=list(logs), ranked_df=None, best_features=None)

        if len(cleaned_df) >= MIN_ROWS_FOR_TRAINING:
            ranked_df, best_features = rank_features(cleaned_df)
            data.update(ranked_df=ranked_df, best_features=best_features)
            top = ", ".join(FEATURE_LABELS[f].split(" ", 1)[1] for f in best_features)
            data["logs"].append(f"Ranked all {len(PREDICTORS)} features by importance — best drivers: {top}.")
    except ValueError as exc:
        data["error"] = str(exc)
    except Exception as exc:  # noqa: BLE001 - never crash the page on bad user data
        data["error"] = f"Unexpected problem while processing the file: {type(exc).__name__}: {exc}"

    st.session_state["cd_data"] = data


def _clear_uploaded_dataset():
    """Button callback: drop the data and reset the uploader widget (via a new key)."""
    _reset_custom_state()
    st.session_state["cd_uploader_nonce"] = st.session_state.get("cd_uploader_nonce", 0) + 1


def _persisted(key, default):
    """
    Restore a widget's value from session_state if Streamlit dropped it because the
    widget was not rendered for a run (e.g. the user visited another page/mode).
    """
    if key not in st.session_state:
        st.session_state[key] = st.session_state.get(f"{key}_saved", default)


def _render_ranking(ranked_df, best_features):
    """Ranked bar chart (+ table) of all 7 predictors."""
    colors = ["#c9a84c" if f in best_features else "#1a9e8f" for f in ranked_df["feature"]]
    labels = [FEATURE_LABELS[f].split(" ", 1)[1] for f in ranked_df["feature"]]

    fig = go.Figure(go.Bar(
        x=ranked_df["importance_pct"], y=labels, orientation="h",
        marker_color=colors,
        text=[f"{v:.1f}%" for v in ranked_df["importance_pct"]], textposition="outside",
        hovertemplate="%{y}: %{x:.1f}%<extra></extra>",
    ))
    fig.add_vline(
        x=IMPORTANCE_THRESHOLD_PCT,
        line_dash="dash",
        line_color="#8b949e",
        annotation_text=f"{IMPORTANCE_THRESHOLD_PCT:.0f}% cut-off",
        annotation_position="top right",
        annotation_font=dict(color="#8b949e", size=12),
    )
    max_imp = float(ranked_df["importance_pct"].max())
    fig.update_layout(
        title=dict(text="Feature impact on Delivery Time (%)", x=0.0, xanchor="left",
                   font=dict(color="#e0c070", size=18)),
        height=390,
        margin=dict(l=175, r=55, t=55, b=55),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Rajdhani, sans-serif", color="#e6edf3", size=14),
        xaxis=dict(
            title=dict(text="Importance (%)", standoff=14, font=dict(color="#8b949e", size=14)),
            tickfont=dict(color="#e6edf3", size=13),
            gridcolor="#2a3444",
            zeroline=False,
            automargin=True,
            range=[0, max(max_imp * 1.18, IMPORTANCE_THRESHOLD_PCT * 2)],
        ),
        yaxis=dict(
            autorange="reversed",
            tickfont=dict(color="#e6edf3", size=13),
            automargin=True,
        ),
        showlegend=False,
    )
    _plotly_show(fig, "cd_rank_chart")
    st.caption(f"🟨 Gold = recommended best features (≥ {IMPORTANCE_THRESHOLD_PCT:.0f}% impact, minimum top {MIN_BEST_FEATURES}) · 🟩 Teal = lower impact. "
               "`Order_ID` is permanently excluded.")

    with st.expander("📋 View ranking table", expanded=False):
        table = pd.DataFrame({
            "Rank": range(1, len(ranked_df) + 1),
            "Feature": [FEATURE_LABELS[f] for f in ranked_df["feature"]],
            "Impact on Delivery Time (%)": ranked_df["importance_pct"].round(2),
            "Recommended": ["⭐ Yes" if f in best_features else "—" for f in ranked_df["feature"]],
        })
        st.dataframe(table, use_container_width=True, hide_index=True)


def _numeric_input(feature, meta, df):
    """Slider for a numeric feature, built from the user's own min/max."""
    lo, hi, med = meta["min"], meta["max"], meta["median"]
    label = FEATURE_LABELS[feature]
    if not hi > lo:  # constant column — a slider cannot be drawn
        st.markdown(f"**{label}** — every row has the value `{lo:g}`; using it for prediction.")
        return float(lo)
    is_int = bool((df[feature] % 1 == 0).all())
    if is_int:
        return float(st.slider(label, int(lo), int(hi), int(round(min(max(med, lo), hi))),
                               step=1, key=f"cd_in_{feature}"))
    return float(st.slider(label, float(lo), float(hi), float(min(max(med, lo), hi)),
                           step=0.1, format="%.2f", key=f"cd_in_{feature}"))


def _categorical_input(feature, meta, df):
    """Dropdown for a categorical feature, built from the user's own categories."""
    cats = meta["categories"]
    default = df[feature].astype(str).mode().iloc[0]
    idx = cats.index(default) if default in cats else 0
    return st.selectbox(FEATURE_LABELS[feature], cats, index=idx, key=f"cd_in_{feature}")


def _render_model_section(cleaned_df, ranked_df, best_features, sig):
    """Feature selector → KPIs → dynamic prediction form. Returns nothing; never stops the page."""
    section_heading("⭐", "Your Best Features")
    st.markdown("""
    <p style="font-family:'Rajdhani',sans-serif; font-size:14px; color:#8b949e;
              margin:-6px 0 12px 0; line-height:1.6;">
        All 7 valid features are ranked below by how strongly they influence
        <strong style="color:#e0c070;">Delivery Time</strong> in <em>your</em> data.
    </p>
    """, unsafe_allow_html=True)

    _render_ranking(ranked_df, best_features)

    # ── 3-mode selector ───────────────────────────────────
    _persisted("cd_mode", MODE_BEST)
    mode = st.radio("How should the model choose its features?",
                    [MODE_BEST, MODE_ALL, MODE_CUSTOM], horizontal=True, key="cd_mode")
    st.session_state["cd_mode_saved"] = mode

    if mode == MODE_BEST:
        chosen = list(best_features)
        st.caption(f"Auto-selected the {len(chosen)} top driver(s) from the ranking above.")
    elif mode == MODE_ALL:
        chosen = list(PREDICTORS)
        st.caption("Using all 7 valid predictors.")
    else:
        _persisted("cd_custom_widget", list(PREDICTORS))
        chosen = st.multiselect(
            "Add or remove features (at least 1 required)", options=list(PREDICTORS),
            key="cd_custom_widget", format_func=lambda f: FEATURE_LABELS[f],
        )
        st.session_state["cd_custom_saved"] = list(chosen)

    if not chosen:
        st.warning("Please select at least 1 feature to train your custom model.")
        return

    # Canonical order keeps the cache key and the form layout stable
    active_features = [f for f in PREDICTORS if f in chosen]

    # ── Train / retrieve the custom model ─────────────────
    models = st.session_state.setdefault("cd_models", {})
    cache_key = (sig, tuple(active_features))
    if cache_key not in models:
        with st.spinner("🌲 Training your custom Random Forest..."):
            models[cache_key] = train_custom_model(cleaned_df, active_features)
    result, err = models[cache_key]
    if err:
        st.error(err)
        return

    metrics = result["metrics"]
    r2, mae = metrics["r2"], metrics["mae"]

    section_heading("📈", "Your Custom Model Performance")
    k1, k2, k3 = st.columns(3)
    with k1:
        kpi_card("🎯", "Custom Model Accuracy (R²)", f"{r2:.2f}",
                 f"Explains {max(r2, 0) * 100:.0f}% of delivery-time variation")
    with k2:
        kpi_card("⏱️", "Average Error", f"± {mae:.1f} min", "Typical miss on unseen orders")
    with k3:
        kpi_card("📦", "Rows Trained", f"{metrics['row_count']:,}", "80% to train · 20% to validate")
    st.caption("Accuracy and error are measured on a 20% hold-out of your data that the model never saw during training.")
    if r2 < 0.5:
        st.info("💡 Accuracy is modest. More rows, or more features, usually help — try the "
                "'Select All Features' option or upload a larger dataset.")

    # ── Dynamic prediction form ───────────────────────────
    section_heading("🔮", "Predict Delivery Time for Your Business")
    meta = result["feature_metadata"]
    num_feats = [f for f in active_features if meta[f]["type"] == "numeric"]
    cat_feats = [f for f in active_features if meta[f]["type"] == "categorical"]

    inputs = {}
    if num_feats and cat_feats:
        col_l, col_r = st.columns(2)
    else:
        col_l = col_r = st.container()
    if num_feats:
        with col_l:
            section_heading("🔢", "Numerical Inputs")
            for f in num_feats:
                inputs[f] = _numeric_input(f, meta[f], cleaned_df)
    if cat_feats:
        with col_r:
            section_heading("🏷️", "Categorical Inputs")
            for f in cat_feats:
                inputs[f] = _categorical_input(f, meta[f], cleaned_df)

    st.markdown("")
    predict_custom_clicked = st.button(
        "🔮 Predict Delivery Time",
        use_container_width=True,
        key="cd_predict_btn",
    )

    if predict_custom_clicked:
        try:
            minutes = predict_delivery_time(result, inputs)
            prediction_banner(
                minutes,
                footnote=(f"Predicted by your custom Random Forest &nbsp;·&nbsp; R² = {r2:.2f} "
                          f"&nbsp;·&nbsp; {len(active_features)} feature(s) &nbsp;·&nbsp; "
                          f"{metrics['row_count']:,} rows"),
            )
        except Exception as exc:  # noqa: BLE001
            st.error(f"Prediction failed: {exc}")


def render_custom_dataset_tab():
    """Complete workflow for the '📊 Try With Your Own Dataset' tab."""

    # ── 1. Template downloads & tutorial ──────────────────
    section_heading("📊", "Try With Your Own Dataset")
    st.markdown("""
    <p style="font-family:'Rajdhani',sans-serif; font-size:14px; color:#8b949e;
              margin:-6px 0 14px 0; line-height:1.6;">
        Upload your own delivery records and the tool will automatically clean them,
        find the factors that matter most for <em>your</em> business, train a private
        model, and let you predict delivery times.
    </p>
    """, unsafe_allow_html=True)

    d1, d2 = st.columns(2)
    with d1:
        st.download_button("📥 Download Empty Template CSV", data=_template_csv_bytes(),
                           file_name="food_delivery_template.csv", mime="text/csv",
                           use_container_width=True, key="cd_dl_template")
    with d2:
        st.download_button("📄 Download Sample Filled CSV", data=load_sample_csv_bytes(),
                           file_name=SAMPLE_CSV_FILENAME, mime="text/csv",
                           use_container_width=True, key="cd_dl_sample")

    with st.expander("📘 How to Use This Tool (Tutorial & Guide)", expanded=False):
        st.markdown("#### Step-by-step")
        st.markdown(
            "1. **Download** the *Empty Template CSV* (or the *Sample Filled CSV* to see a finished example).\n"
            "2. **Open** it in Excel, Google Sheets or any spreadsheet app.\n"
            "3. **Fill in one row per delivery** — keep the header row exactly as it is.\n"
            "4. **Save / export** the sheet as **CSV (comma-separated)**.\n"
            "5. **Upload** the CSV in the box below. Your data is cleaned automatically.\n"
            "6. **Pick your features** — let the tool choose the best ones, use all, or customise.\n"
            "7. **Play with the sliders** to get a live delivery-time prediction for your business."
        )
        st.markdown("#### Column formatting rules")
        st.markdown(
            "| Column | What to enter | Example |\n"
            "|---|---|---|\n"
            "| `Order_ID` | Any unique order number (ignored by the model, optional) | 1001 |\n"
            "| `Distance_km` | Road distance in km — number | 15.57 |\n"
            "| `Weather` | Text category, e.g. Clear, Rainy, Foggy, Snowy, Windy | Rainy |\n"
            "| `Traffic_Level` | Text category, e.g. Low, Medium, High | High |\n"
            "| `Time_of_Day` | Text category, e.g. Morning, Afternoon, Evening, Night | Evening |\n"
            "| `Vehicle_Type` | Text category, e.g. Bike, Scooter, Car | Scooter |\n"
            "| `Preparation_Time_min` | Kitchen preparation time in minutes — number | 21 |\n"
            "| `Courier_Experience_yrs` | Courier experience in years — number | 3.5 |\n"
            "| `Delivery_Time_min` | **Total** delivery time in minutes — number (this is what we predict) | 71 |"
        )
        st.markdown(
            "- **Do not rename, add or reorder** the header columns.\n"
            "- Use plain numbers only (no units like `km` or `min`).\n"
            "- Keep category spelling consistent (`Rainy`, not `rainy` / `Rain`).\n"
            f"- You need **at least {MIN_ROWS_FOR_TRAINING} valid rows** to train a model; 200+ gives better accuracy.\n"
            "- Blank cells are fine: they are filled automatically (median / most frequent value). "
            "Rows with a blank **Delivery_Time_min** are removed."
        )


    # ── 2. File uploader ──────────────────────────────────
    st.markdown("<div style='height:6px;'></div>", unsafe_allow_html=True)
    nonce = st.session_state.get("cd_uploader_nonce", 0)
    uploaded = st.file_uploader("Upload your completed CSV file", type=["csv"],
                                key=f"cd_uploader_{nonce}")
    if uploaded is not None:
        with st.spinner("🧹 Cleaning your data and ranking features..."):
            _process_upload(uploaded)

    data = st.session_state.get("cd_data")
    if data is None:
        st.info("👆 Upload a CSV to get started. Not sure how? Download the sample file above and open the tutorial.")
        return

    # Persistent-data banner (shown whenever the uploader itself is empty but we still hold the data)
    if uploaded is None:
        b1, b2 = st.columns([3, 1])
        with b1:
            st.info(f"📁 Using your previously uploaded file: **{data['filename']}**")
        with b2:
            st.button("🗑️ Remove dataset", on_click=_clear_uploaded_dataset,
                      use_container_width=True, key="cd_remove_btn")

    if data["error"]:
        st.error(f"❌ {data['error']}")
        return

    cleaned_df = data["cleaned_df"]
    n_clean, n_raw = len(cleaned_df), data["raw_rows"]

    # ── Cleaning summary ──────────────────────────────────
    st.success(f"✅ **{data['filename']}** uploaded and cleaned automatically — "
               f"**{n_clean:,} of {n_raw:,}** row(s) are ready for modeling.")
    with st.expander(f"🧹 Automated preprocessing & cleaning summary ({len(data['logs'])} steps)", expanded=True):
        for line in data["logs"]:
            st.markdown(f"- {line}")

    # ── Too few rows: skip modelling, go straight to the report ──
    if n_clean < MIN_ROWS_FOR_TRAINING:
        if n_clean == 0:
            st.error("No usable rows were found in your file, so there is nothing to analyse. "
                     "Please fill in the template and upload it again.")
            return
        st.warning(f"⚠️ Model prediction requires at least **{MIN_ROWS_FOR_TRAINING} valid rows**, "
                   f"but only **{n_clean}** remain after cleaning. Add more rows to unlock feature "
                   "ranking and predictions. Your Data Quality Report is shown below.")
    else:
        # ── 3 & 4. Ranking, feature selector, KPIs, prediction form ──
        _render_model_section(cleaned_df, data["ranked_df"], data["best_features"], st.session_state["cd_sig"])

    # ── 5. Custom Data Quality Report ─────────────────────
    st.divider()
    section_heading("🧪", "Your Data Quality Report")
    st.markdown("""
    <p style="font-family:'Rajdhani',sans-serif; font-size:14px; color:#8b949e;
              margin:-6px 0 12px 0; line-height:1.6;">
        Interactive charts and plain-English insights generated from <strong style="color:#e0c070;">your</strong>
        cleaned dataset.
    </p>
    """, unsafe_allow_html=True)
    render_data_quality_report(cleaned_df, key_prefix="usr_eda")
    st.markdown("<div style='height:50px;'></div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# SIDEBAR — page navigation
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding:12px 0 8px 0;'>
        <span style="font-family:'Rajdhani',sans-serif; font-size:28px;
                     font-weight:700; color:#c9a84c; letter-spacing:1px;">
            🍕 Tool Menu
        </span>
    </div>
    """, unsafe_allow_html=True)

    # Radio drives the two main pages
    page = st.radio("", ["Home", "About"])

    st.markdown("<hr style='border-color:#2a3444; margin:12px 0;'>", unsafe_allow_html=True)

    st.markdown("""
    <div style="font-family:'Rajdhani',sans-serif; font-size:13px;
                color:#8b949e; line-height:1.9; padding:4px 0;">
        <strong style="color:#c9a84c;">Current Version:</strong> 1.0<br>
        <strong style="color:#c9a84c;">Release Date:</strong> September 2026<br>
        <strong style="color:#c9a84c;">Model:</strong> Random Forest (Optimized)<br>
        <strong style="color:#c9a84c;">R² Score:</strong> 0.96
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

        # ── Three tabs: Project overview | Test the model | Own dataset ───
        tab1, tab2, tab3 = st.tabs(
            ["🏠 Overview", "🔮 Test The Model", "📊 Try With Your Own Dataset"]
        )


        # ══════════════════════════════════════════════════
        # TAB 1 — OVERVIEW
        # ══════════════════════════════════════════════════
        with tab1:

            # ── Project Introduction & Value Proposition ───
            section_heading("🚀", "Welcome to Food Delivery Time Predictor")
            st.markdown("""
            <p style="font-family:'Rajdhani',sans-serif; font-size:15px; color:#e6edf3;
                      margin: -4px 0 12px 0; line-height:1.75;">
                Accurate delivery estimates keep customers informed and dispatch operations running smoothly.
                While major platforms like <strong style="color:#e0c070;">Uber Eats, Zomato and Swiggy</strong> rely on
                enterprise-grade logistics software developed and managed by large teams, <strong style="color:#2dd4bf;">local cafes, restaurants, cloud
                kitchens, and neighborhood vendors</strong> rarely have access to expensive predictive tools.
                Powered by an <strong style="color:#e0c070;">Optimized Random Forest</strong> model, this application
                gives any business data-driven ETA predictions — accounting for <strong style="color:#2dd4bf;">road
                distance, kitchen preparation time, live weather, traffic congestion, time of day, vehicle type, and
                courier experience</strong>.
            </p>
            """, unsafe_allow_html=True)

            # ── Quick Highlights KPI Row ───────────────────
            h1, h2, h3, h4 = st.columns(4)
            with h1:
                kpi_card("⚡", "Instant ETAs", "Real-Time", "Predictions in seconds")
            with h2:
                kpi_card("🧠", "ML Engine", "Random Forest", "Tuned ensemble regressor")
            with h3:
                kpi_card("🎛️", "Two Modes", "Demo + Custom", "Pre-trained or your CSV")
            with h4:
                kpi_card("🧪", "Built-In EDA", "Auto Report", "Visual quality & insights")

            st.divider()

            # ── Real-World Business Use Case ───────────────
            section_heading("🏪", "Built for Local Cafes, Restaurants & Small Vendors")
            st.markdown("""
            <p style="font-family:'Rajdhani',sans-serif; font-size:14px; color:#8b949e;
                      margin: -6px 0 14px 0; line-height:1.6;">
                Why we built <strong style="color:#2dd4bf;">Try With Your Own Dataset</strong> — turning everyday order logs into confident customer quotes.
            </p>
            """, unsafe_allow_html=True)

            u1, u2 = st.columns(2)
            with u1:
                st.markdown("""
                <div style="
                    background: #1c2330;
                    border: 1px solid #2a3444;
                    border-left: 3px solid #c9a84c;
                    border-radius: 0 10px 10px 0;
                    padding: 18px 20px;
                    height: 100%;
                    box-sizing: border-box;
                    display: flex;
                    flex-direction: column;
                    justify-content: flex-start;
                ">
                    <p style="font-family:'Rajdhani',sans-serif; font-weight:700;
                              font-size:17px; color:#e0c070; margin:0 0 8px 0;">
                        📋 Collect 15–30 Days of Orders &amp; Train Free
                    </p>
                    <p style="font-family:'Rajdhani',sans-serif; font-size:14px;
                              color:#e6edf3; margin:0; line-height:1.65;">
                        Small cafes and local shopkeepers don't need paid dispatch software to forecast delivery times.
                        Download the free CSV template, record deliveries step-by-step — an active cafe can gather
                        enough order history in just <strong style="color:#e0c070;">15 to 30 days</strong> — and upload
                        it to train a private model tailored to your own kitchen speed, couriers, and local roads.
                    </p>
                </div>
                """, unsafe_allow_html=True)

            with u2:
                st.markdown("""
                <div style="
                    background: #1c2330;
                    border: 1px solid #2a3444;
                    border-left: 3px solid #2dd4bf;
                    border-radius: 0 10px 10px 0;
                    padding: 18px 20px;
                    height: 100%;
                    box-sizing: border-box;
                    display: flex;
                    flex-direction: column;
                    justify-content: flex-start;
                ">
                    <p style="font-family:'Rajdhani',sans-serif; font-weight:700;
                              font-size:17px; color:#2dd4bf; margin:0 0 8px 0;">
                        📞 Real-Life Phone Order Scenario
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

            # ── Two-Mode Workflow Guide ────────────────────
            section_heading("🧭", "How to Use This Application")
            st.markdown("""
            <p style="font-family:'Rajdhani',sans-serif; font-size:14px; color:#8b949e;
                      margin: -6px 0 14px 0; line-height:1.6;">
                Choose between testing the pre-trained model right away or training a private model on your own business records.
            </p>
            """, unsafe_allow_html=True)

            w1, w2 = st.columns(2)
            with w1:
                st.markdown("""
                <div style="
                    background: #1c2330;
                    border: 1px solid #2a3444;
                    border-top: 3px solid #c9a84c;
                    border-radius: 10px;
                    padding: 18px 20px;
                    height: 100%;
                    box-sizing: border-box;
                    display: flex;
                    flex-direction: column;
                    justify-content: flex-start;
                ">
                    <p style="font-family:'Rajdhani',sans-serif; font-weight:700;
                              font-size:18px; color:#e0c070; margin:0 0 8px 0;">
                        1️⃣ &nbsp;🔮 Test The Model (Instant Demo)
                    </p>
                    <p style="font-family:'Rajdhani',sans-serif; font-size:14px;
                              color:#e6edf3; margin:0 0 10px 0; line-height:1.6;">
                        Try live predictions immediately using our pre-trained Random Forest model — no file upload required.
                    </p>
                    <p style="font-family:'Rajdhani',sans-serif; font-size:13px;
                              color:#8b949e; margin:0; line-height:1.65;">
                        ▸ Adjust distance, prep time, and courier experience sliders<br>
                        ▸ Select weather, traffic level, time of day, and vehicle type<br>
                        ▸ Click <strong>Predict Delivery Time</strong> for an instant ETA and explore the benchmark Data Quality Report below
                    </p>
                </div>
                """, unsafe_allow_html=True)

            with w2:
                st.markdown("""
                <div style="
                    background: #1c2330;
                    border: 1px solid #2a3444;
                    border-top: 3px solid #2dd4bf;
                    border-radius: 10px;
                    padding: 18px 20px;
                    height: 100%;
                    box-sizing: border-box;
                    display: flex;
                    flex-direction: column;
                    justify-content: flex-start;
                ">
                    <p style="font-family:'Rajdhani',sans-serif; font-weight:700;
                              font-size:18px; color:#2dd4bf; margin:0 0 8px 0;">
                        2️⃣ &nbsp;📊 Try With Your Own Dataset
                    </p>
                    <p style="font-family:'Rajdhani',sans-serif; font-size:14px;
                              color:#e6edf3; margin:0 0 10px 0; line-height:1.6;">
                        Upload your own delivery CSV to train a dedicated model tailored to your city, kitchen, or fleet.
                    </p>
                    <p style="font-family:'Rajdhani',sans-serif; font-size:13px;
                              color:#8b949e; margin:0; line-height:1.65;">
                        ▸ Download the empty template or pre-filled sample CSV<br>
                        ▸ Automatic data cleaning, missing-value handling &amp; feature ranking<br>
                        ▸ Pick best, all, or custom features, predict ETAs, and inspect your custom Data Quality Report
                    </p>
                </div>
                """, unsafe_allow_html=True)

            st.divider()

            # ── Model performance summary ──────────────────
            section_heading("📊", "Model Performance at a Glance")
            st.markdown("""
            <p style="font-family:'Rajdhani',sans-serif; font-size:14px; color:#8b949e;
                      margin: -6px 0 12px 0; line-height:1.6;">
                The Optimized Random Forest was tuned using <strong style="color:#e0c070;">RandomizedSearchCV
                (100 iterations, 5-fold CV)</strong> and outperforms the default baseline across all evaluation metrics.
            </p>
            """, unsafe_allow_html=True)

            m1, m2, m3, m4 = st.columns(4)
            with m1:
                kpi_card("📈", "R² Score",   "0.96",     "vs 0.91 baseline")
            with m2:
                kpi_card("📉", "MAE",        "3.60 min", "vs 5.11 baseline")
            with m3:
                kpi_card("📉", "RMSE",       "4.59 min", "vs 6.61 baseline")
            with m4:
                kpi_card("🌲", "Estimators", "900 trees","Best via tuning")

            st.markdown("<br>", unsafe_allow_html=True)

            # ── Key findings ───────────────────────────────
            section_heading("💡", "Key Findings From Our Modeling")
            insight_line(1, "Distance is the strongest driver — longer routes consistently produce longer delivery times.")
            insight_line(2, "Adverse weather (Snowy / Rainy) adds measurable delay compared to Clear conditions.")
            insight_line(3, "High traffic levels independently extend delivery duration regardless of trip distance.")
            insight_line(4, "More experienced couriers (7–9 yrs) deliver noticeably faster than new couriers (0–1 yr).")
            insight_line(5, "All four categorical features passed One-Way ANOVA testing (p < 0.05), confirming statistical significance.")

            st.markdown("<div style='height:50px;'></div>", unsafe_allow_html=True)


        # ══════════════════════════════════════════════════
        # TAB 3 — TRY WITH YOUR OWN DATASET
        # (Tab 3 sits to the right of Tab 2 in the UI. Its code is placed
        #  before Tab 2's block on purpose: Tab 2 calls st.stop() if the
        #  pickled model fails to load, which would otherwise prevent this
        #  independent tab from rendering.)
        # ══════════════════════════════════════════════════
        with tab3:
            render_custom_dataset_tab()


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

            section_heading("🔮", "Predict Delivery Time")
            st.markdown("""
            <p style="font-family:'Rajdhani',sans-serif; font-size:14px; color:#8b949e;
                      margin: -6px 0 14px 0; line-height:1.6;">
                Fill in the delivery details below and click <strong style="color:#e0c070;">Predict</strong>
                to get an estimated delivery time from the trained Random Forest model.
            </p>
            """, unsafe_allow_html=True)

            # ── Input form — two-column layout ────────────
            col_left, col_right = st.columns(2)

            # ── Left column: Numerical inputs ─────────────
            with col_left:
                section_heading("🔢", "Numerical Inputs")

                distance_km = st.slider(
                    "📍 Distance (km)",
                    min_value=0.5,
                    max_value=20.0,
                    value=10.0,
                    step=0.1,
                    format="%.2f",
                    help="Road distance from restaurant to delivery address."
                )

                preparation_time = st.slider(
                    "🍳 Preparation Time (min)",
                    min_value=5,
                    max_value=30,
                    value=15,
                    step=1,
                    help="Time the restaurant needs to prepare the order."
                )

                courier_experience = st.slider(
                    "🧑‍💼 Courier Experience (years)",
                    min_value=0.0,
                    max_value=9.0,
                    value=3.0,
                    step=0.5,
                    format="%.2f",
                    help="Number of years the courier has been delivering."
                )

            # ── Right column: Categorical inputs ──────────
            with col_right:
                section_heading("🏷️", "Categorical Inputs")

                weather_options   = list(label_encoders["Weather"].classes_)
                weather_selected  = st.selectbox(
                    "🌦️ Weather Condition",
                    weather_options,
                    help="Current weather at the time of delivery."
                )

                traffic_options   = list(label_encoders["Traffic_Level"].classes_)
                traffic_selected  = st.selectbox(
                    "🚦 Traffic Level",
                    traffic_options,
                    help="Road traffic level — Low, Medium, or High."
                )

                time_options      = list(label_encoders["Time_of_Day"].classes_)
                time_selected     = st.selectbox(
                    "🕐 Time of Day",
                    time_options,
                    help="Part of the day when the delivery takes place."
                )

                vehicle_options   = list(label_encoders["Vehicle_Type"].classes_)
                vehicle_selected  = st.selectbox(
                    "🏍️ Vehicle Type",
                    vehicle_options,
                    help="Type of vehicle used by the courier."
                )

            st.markdown("")

            # ── Predict button (full width) ────────────────
            predict_clicked = st.button("🔮 Predict Delivery Time", use_container_width=True)

            if predict_clicked:
                try:
                    # Encode categorical inputs using the saved LabelEncoders
                    weather_enc  = label_encoders["Weather"].transform([weather_selected])[0]
                    traffic_enc  = label_encoders["Traffic_Level"].transform([traffic_selected])[0]
                    time_enc     = label_encoders["Time_of_Day"].transform([time_selected])[0]
                    vehicle_enc  = label_encoders["Vehicle_Type"].transform([vehicle_selected])[0]

                    # Build input DataFrame — column order must match training
                    input_df = pd.DataFrame(
                        [[distance_km, weather_enc, traffic_enc, time_enc,
                          vehicle_enc, preparation_time, courier_experience]],
                        columns=[
                            "Distance_km", "Weather", "Traffic_Level", "Time_of_Day",
                            "Vehicle_Type", "Preparation_Time_min", "Courier_Experience_yrs"
                        ]
                    )

                    # Run inference
                    prediction = model.predict(input_df)[0]

                    # Render the large result banner
                    prediction_banner(prediction)

                    # ── Context cards below result ─────────
                    st.markdown("<br>", unsafe_allow_html=True)
                    section_heading("📋", "Inputs Used for Prediction")

                    r1c1, r1c2, r1c3, r1c4 = st.columns(4)
                    with r1c1:
                        kpi_card("📍", "Distance",     f"{distance_km:.1f} km")
                    with r1c2:
                        kpi_card("🍳", "Prep Time",    f"{preparation_time} min")
                    with r1c3:
                        kpi_card("🧑‍💼", "Experience",  f"{courier_experience} yrs")
                    with r1c4:
                        kpi_card("🌦️", "Weather",      weather_selected)

                    st.markdown("<br>", unsafe_allow_html=True)

                    r2c1, r2c2, r2c3 = st.columns(3)
                    with r2c1:
                        kpi_card("🚦", "Traffic",      traffic_selected)
                    with r2c2:
                        kpi_card("🕐", "Time of Day",  time_selected)
                    with r2c3:
                        kpi_card("🏍️", "Vehicle",      vehicle_selected)

                except Exception as e:
                    st.error(f"Prediction failed: {e}")

            # ── Tip section shown when no prediction yet ───
            else:
                st.markdown("<br>", unsafe_allow_html=True)
                with st.expander("💡 Tips for Using the Predictor", expanded=True):
                    insight_line(1, "Distance has the biggest impact — even a few extra kilometres significantly increases estimated time.")
                    insight_line(2, "Select <strong>High</strong> traffic to simulate peak-hour conditions like lunch or dinner rush.")
                    insight_line(3, "Couriers with <strong>0–1 year</strong> of experience tend to take longer than those with <strong>7+ years</strong>.")
                    insight_line(4, "Snowy and Rainy weather conditions add the most delay compared to Clear weather.")
                    insight_line(5, "Prediction accuracy: <strong>MAE ≈ 3.60 min</strong> — results are typically within a few minutes of actual time.")

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
            section_heading("🗃️", "Dataset Overview & Source")
            st.markdown("""
            <p style="font-family:'Rajdhani',sans-serif; font-size:14px; color:#8b949e;
                      margin: -6px 0 12px 0; line-height:1.6;">
                The model behind the predictions above was trained on the dataset summarised below.
            </p>
            """, unsafe_allow_html=True)

            n_records = f"{len(default_df):,}" if default_df is not None else "1,000"

            d1, d2, d3, d4 = st.columns(4)
            with d1:
                kpi_card("📄", "Dataset File", DATASET_FILENAME, "CSV format")
            with d2:
                kpi_card("📦", "Records", n_records, "Delivery orders")
            with d3:
                kpi_card("🔢", "Predictive Drivers", "7", "3 numeric · 4 categorical")
            with d4:
                kpi_card("🎯", "Target", "Delivery Time", "In minutes")

            st.markdown("<br>", unsafe_allow_html=True)

            st.markdown(f"""
            <div style="background:#1c2330; border:1px solid #2a3444; border-left:3px solid #2dd4bf;
                        border-radius:0 10px 10px 0; padding:14px 18px;">
                <p style="font-family:'Rajdhani',sans-serif; font-weight:700; font-size:16px;
                          color:#c9a84c; margin:0 0 6px 0;">🔗 Dataset Reference</p>
                <p style="font-family:'Rajdhani',sans-serif; font-size:14px; color:#e6edf3;
                          margin:0 0 8px 0; line-height:1.6;">
                    <strong>Drivers:</strong> Distance, Weather, Traffic Level, Time of Day,
                    Vehicle Type, Preparation Time and Courier Experience.
                    <code>Order_ID</code> is an identifier only and is never used for modeling.
                </p>
                <a href="{DATASET_SOURCE_URL}" target="_blank" rel="noopener noreferrer"
                   style="font-family:'Rajdhani',sans-serif; font-size:14px; font-weight:600;">
                    🌐 View the original dataset source ↗
                </a>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # ── Data Quality Report (eda_engine) ───────────
            section_heading("🧪", "Data Quality Report")
            st.markdown("""
            <p style="font-family:'Rajdhani',sans-serif; font-size:14px; color:#8b949e;
                      margin: -6px 0 12px 0; line-height:1.6;">
                Interactive charts and plain-English insights generated directly from the
                training data — distributions, categorical rankings, average-time bars,
                box plots, two-factor heatmaps, and correlations.
            </p>
            """, unsafe_allow_html=True)

            if default_df is not None:
                render_data_quality_report(default_df)

            st.markdown("<div style='height:50px;'></div>", unsafe_allow_html=True)


    # ══════════════════════════════════════════════════════
    # ABOUT PAGE
    # ══════════════════════════════════════════════════════
    elif page == "About":

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

        st.divider()

#        # ── About the developer ────────────────────────────
#        with st.expander("👨‍💻 About the Developer — Pranay Jha", expanded=False):
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
#                🚀 Key Highlights
#            </p>
#            """, unsafe_allow_html=True)
#
#            for item in [
#                "Built a <strong>regression model</strong> on real-world structured delivery data",
#                "Performed <strong>data cleaning, EDA, and feature engineering</strong>",
#                "Applied <strong>LabelEncoding</strong> to categorical features for model training",
#                "Tuned hyperparameters using <strong>RandomizedSearchCV (100 iterations, 5-fold CV)</strong>",
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
#                🔗 Connect With Me
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
#                    <span style="font-size:18px;">🐙</span>
#                    <strong>GitHub</strong> — Pranay-256
#                </a>
#                <a href="https://www.linkedin.com/in/pranay-jha-6582a937b/" target="_blank"
#                   style="display:inline-flex; align-items:center; gap:8px;
#                          background:#1c2330; border:1px solid #2a3444;
#                          border-radius:8px; padding:10px 18px;
#                          font-family:'Rajdhani',sans-serif; font-size:14px;
#                          color:#e6edf3; text-decoration:none;">
#                    <span style="font-size:18px;">💼</span>
#                    <strong>LinkedIn</strong> — Pranay Jha
#                </a>
#            </div>
#            """, unsafe_allow_html=True)
#            st.markdown("<br>", unsafe_allow_html=True)

        # ── Problem statement ──────────────────────────────
        with st.expander("❗ Problem Statement", expanded=False):
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
                       font-size:17px; color:#c9a84c; margin:8px 0;">⚠️ Key Challenges</p>
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
        with st.expander("🎯 Objective", expanded=False):
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
                       font-size:17px; color:#c9a84c; margin:8px 0;">📊 Core Focus Areas</p>
            """, unsafe_allow_html=True)

            for item in [
                "Identify the key <strong>features that drive delivery time</strong>",
                "Train a <strong>Random Forest Regressor</strong> with hyperparameter tuning",
                "Achieve <strong>high predictive accuracy</strong> (target R² > 0.90)",
                "Provide an <strong>interactive tool</strong> for instant predictions",
                "Make the model <strong>transparent and explainable</strong> via dataset documentation",
            ]:
                st.markdown(f"""
                <div style="background:#1c2330; border:1px solid #2a3444; border-left:3px solid #3fb950;
                            border-radius:0 6px 6px 0; padding:8px 14px; margin-bottom:6px;
                            font-family:'Rajdhani',sans-serif; font-size:14px; color:#e6edf3;">
                    ✓ &nbsp;{item}
                </div>
                """, unsafe_allow_html=True)

        # ── Tech stack ─────────────────────────────────────
        with st.expander("🛠️ Tech Stack", expanded=False):
            st.markdown("""
            <p style="font-family:'Rajdhani',sans-serif; font-size:14px; color:#e6edf3;
                      line-height:1.8; margin-bottom:10px;">
                Built using the <strong style="color:#e0c070;">Python ML ecosystem</strong>
                combined with an interactive deployment framework.
            </p>
            """, unsafe_allow_html=True)

            tools = [
                ("🐍 Python",           "Core language for data processing and ML"),
                ("🔢 NumPy",            "Numerical operations and array handling"),
                ("🐼 Pandas",           "Data cleaning, manipulation, and EDA"),
                ("🤖 Scikit-learn",     "LabelEncoder, RandomForestRegressor, RandomizedSearchCV"),
                ("💾 Pickle",           "Serialisation and loading of trained model artifacts"),
                ("🚀 Streamlit",        "Interactive web dashboard for model inference"),
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
        with st.expander("🤖 ML Approach & Methodology", expanded=False):

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

            ml_block("🧹", "Data Preprocessing", [
                "Dropped Order_ID (non-predictive identifier)",
                "Filled missing Weather, Traffic_Level, Time_of_Day with mode",
                "Filled missing Courier_Experience_yrs with median (30 values)",
                "Applied LabelEncoder to all 4 categorical features",
            ])

            ml_block("🔬", "Feature Selection & Testing", [
                "Pearson correlation for numerical features vs Delivery_Time_min",
                "One-Way ANOVA for categorical features — all 4 passed (p < 0.05)",
                "All 7 features retained based on statistical significance",
            ])

            ml_block("🌲", "Model Training & Tuning", [
                "Baseline: Decision Tree Regressor (R² = ~0.92)",
                "Baseline: Default Random Forest (R² = 0.92)",
                "Tuned: RandomizedSearchCV — 100 iterations, 5-fold CV",
                "Best params: n_estimators=900, max_depth=13, max_features=4, "
                "min_samples_split=5, min_samples_leaf=1",
                "Optimized Random Forest: R² = 0.96, MAE = 3.60, RMSE = 4.59",
            ])

        # ── Version history ────────────────────────────────
        with st.expander("🔄 Version History", expanded=False):

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

            version_block("1.1", "November 2026", [
                "Added visualizations for the Data Quality Report",
                "Added 'Try With Your Own Dataset' feature",
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
