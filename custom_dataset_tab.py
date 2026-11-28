"""
custom_dataset_tab.py
=====================
The "Try With Your Own Dataset" tab of the Food Delivery Time Predictor.

Layout is deliberately identical to the "Test The Model" tab:
  upload -> feature choice (all / best) -> input form -> Predict button -> result banner
  -> divider -> Dataset Overview -> Data Quality Report (same eda_engine charts as Test The Model)

All heavy work lives in ml_pipeline.py and dynamic_eda.py; results are cached so that
moving a slider never re-cleans the data, re-draws the charts or re-trains the model.
"""
from __future__ import annotations

import hashlib
import io
import os

import numpy as np
import pandas as pd
import streamlit as st

import ml_pipeline as mp
from dynamic_eda import stretch
from eda_engine import render_data_quality_report   # same report/charts as the Test The Model tab

TEMPLATE_FILENAME = "Food_Delivery_Template.csv"
SAMPLE_FILENAME = "testing.csv"
SAMPLE_DOWNLOAD_NAME = "Food_Delivery_Sample_Filled.csv"

ALL_LABEL, BEST_LABEL = "Select all features", "Use best features"

# Same wording / help texts as the Test The Model tab
INPUT_LABELS = {
    "Distance_km": "Distance (km)",
    "Preparation_Time_min": "Preparation Time (min)",
    "Courier_Experience_yrs": "Courier Experience (years)",
    "Weather": "Weather Condition",
    "Traffic_Level": "Traffic Level",
    "Time_of_Day": "Time of Day",
    "Vehicle_Type": "Vehicle Type",
}
INPUT_HELP = {
    "Distance_km": "Road distance from restaurant to delivery address.",
    "Preparation_Time_min": "Time the restaurant needs to prepare the order.",
    "Courier_Experience_yrs": "Number of years the courier has been delivering.",
    "Weather": "Current weather at the time of delivery.",
    "Traffic_Level": "Road traffic level.",
    "Time_of_Day": "Part of the day when the delivery takes place.",
    "Vehicle_Type": "Type of vehicle used by the courier.",
}


# ─────────────────────────────────────────────────────────────
# Cached helpers
# ─────────────────────────────────────────────────────────────
def _parse_csv(raw: bytes) -> pd.DataFrame:
    """Reads the upload as text (so bad entries such as '--' survive until cleaning)."""
    last_error = None
    for enc in ("utf-8-sig", "latin-1"):
        try:
            df = pd.read_csv(io.BytesIO(raw), dtype=str, encoding=enc)
            if df.shape[1] == 1 and ";" in str(df.columns[0]):      # semicolon-separated export
                df = pd.read_csv(io.BytesIO(raw), dtype=str, encoding=enc, sep=";")
            return df
        except Exception as e:                                       # noqa: BLE001
            last_error = e
    raise ValueError(f"The file could not be read as a CSV ({last_error}).")


@st.cache_data(show_spinner="Cleaning your data…", max_entries=3)
def _clean_upload(raw: bytes):
    return mp.clean_dataframe(_parse_csv(raw))


@st.cache_data(show_spinner=False, max_entries=3)
def _selection_table(data_key: str, _df: pd.DataFrame):
    return mp.feature_selection_table(_df)


@st.cache_resource(show_spinner="Training your model on your data…", max_entries=3, ttl=3600)
def _train_private(data_key: str, features: tuple, _df: pd.DataFrame):
    return mp.train_model(_df, list(features))


def _read_local(name: str):
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), name)
    if os.path.exists(path):
        with open(path, "rb") as f:
            return f.read()
    return None


# ─────────────────────────────────────────────────────────────
# Tutorial (mirrors the mock-up)
# ─────────────────────────────────────────────────────────────
_COLUMN_RULES = [
    ("Order_ID", "Any unique order number (ignored by the model, optional)", "1001"),
    ("Distance_km", "Road distance in km — number", "15.57"),
    ("Weather", "Text category, e.g. Clear, Rainy, Foggy, Snowy, Windy", "Rainy"),
    ("Traffic_Level", "Text category, e.g. Low, Medium, High", "High"),
    ("Time_of_Day", "Text category, e.g. Morning, Afternoon, Evening, Night", "Evening"),
    ("Vehicle_Type", "Text category, e.g. Bike, Scooter, Car", "Scooter"),
    ("Preparation_Time_min", "Kitchen preparation time in minutes — number", "21"),
    ("Courier_Experience_yrs", "Courier experience in years — number", "3.5"),
    ("Delivery_Time_min", "<strong>Total</strong> delivery time in minutes — number (this is what we predict)", "71"),
]


def _tutorial():
    st.markdown("""
    <h3 style="font-family:'Rajdhani',sans-serif;color:#e6edf3;margin:4px 0 8px 0;">Step-by-step</h3>
    <ol style="font-family:'Rajdhani',sans-serif;font-size:15px;color:#e6edf3;line-height:1.9;">
      <li><strong>Download</strong> the <em>Empty Template CSV</em> (or the <em>Sample Filled CSV</em> to see a finished example).</li>
      <li><strong>Open</strong> it in Excel, Google Sheets or any spreadsheet app.</li>
      <li><strong>Fill in one row per delivery</strong> — keep the header row exactly as it is.</li>
      <li><strong>Save / export</strong> the sheet as <strong>CSV (comma-separated)</strong>.</li>
      <li><strong>Upload</strong> the CSV in the box below. Your data is cleaned automatically.</li>
      <li><strong>Pick your features</strong> — let the tool choose the best ones, use all, or customise.</li>
      <li><strong>Play with the sliders</strong> to get a live delivery-time prediction for your business.</li>
    </ol>
    """, unsafe_allow_html=True)

    rows = "".join(
        f"<tr><td style='border:1px solid #2a3444;padding:8px 12px;'><code>{c}</code></td>"
        f"<td style='border:1px solid #2a3444;padding:8px 12px;'>{d}</td>"
        f"<td style='border:1px solid #2a3444;padding:8px 12px;'>{e}</td></tr>"
        for c, d, e in _COLUMN_RULES)
    st.markdown(f"""
    <h3 style="font-family:'Rajdhani',sans-serif;color:#e6edf3;margin:14px 0 8px 0;">Column formatting rules</h3>
    <div style="overflow-x:auto;">
    <table style="width:100%;border-collapse:collapse;font-family:'Rajdhani',sans-serif;font-size:14px;color:#e6edf3;">
      <thead><tr>
        <th style="border:1px solid #2a3444;padding:8px 12px;text-align:left;">Column</th>
        <th style="border:1px solid #2a3444;padding:8px 12px;text-align:left;">What to enter</th>
        <th style="border:1px solid #2a3444;padding:8px 12px;text-align:left;">Example</th>
      </tr></thead>
      <tbody>{rows}</tbody>
    </table></div>
    <ul style="font-family:'Rajdhani',sans-serif;font-size:15px;color:#e6edf3;line-height:1.9;margin-top:14px;">
      <li><strong>Do not rename, add or reorder</strong> the header columns.</li>
      <li>Use plain numbers only (no units like <code>km</code> or <code>min</code>).</li>
      <li>Keep category spelling consistent ( <code>Rainy</code> , not <code>rainy</code> / <code>Rain</code> ).</li>
      <li>You need <strong>at least 30 valid rows</strong> to train a model; 200+ gives better accuracy.</li>
      <li>Blank cells are fine: they are filled automatically (median / most frequent value). Rows with a blank <strong>Delivery_Time_min</strong> are removed.</li>
    </ul>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# Prediction inputs (same look as the Test The Model tab; ranges / categories come from the user's data)
# ─────────────────────────────────────────────────────────────
def _numeric_input(col: str, series: pd.Series, key: str):
    label, help_ = INPUT_LABELS[col], INPUT_HELP[col]
    lo, hi, med = float(series.min()), float(series.max()), float(series.median())
    if hi <= lo:
        return st.number_input(label, value=lo, key=key, help=help_)
    if bool(np.allclose(series, series.round())):
        return st.slider(label, int(lo), int(hi), int(round(med)), step=1, key=key, help=help_)
    lo, hi = float(np.floor(lo * 10) / 10), float(np.ceil(hi * 10) / 10)
    return st.slider(label, lo, hi, float(min(max(round(med, 1), lo), hi)), step=0.1,
                     format="%.2f", key=key, help=help_)


def _categorical_input(col: str, encoders: dict, df: pd.DataFrame, key: str):
    classes = list(encoders[col].classes_)
    hint = mp.ORDER_HINT.get(col, [])
    options = [c for c in hint if c in classes] + [c for c in classes if c not in hint]
    default = options.index(df[col].mode()[0]) if df[col].mode()[0] in options else 0
    return st.selectbox(INPUT_LABELS[col], options, index=default, key=key, help=INPUT_HELP[col])


# ─────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────
def _hint(text: str, margin: str = "-6px 0 14px 0") -> None:
    st.markdown(f"""
    <p style="font-family:'Rajdhani',sans-serif; font-size:14px; color:#8b949e;
              margin: {margin}; line-height:1.6;">{text}</p>
    """, unsafe_allow_html=True)


def render_custom_dataset_tab(ui) -> None:
    """ui: object exposing section_heading, insight_line, kpi_card, prediction_banner."""

    # ── Upload (the only extra step compared with the Test The Model tab) ──
    ui.section_heading("", "Try With Your Own Dataset")
    _hint("Upload your own delivery records. The tool cleans them automatically, trains a model on your data "
          "and lets you predict delivery times exactly like in the <strong style=\"color:#e0c070;\">Test The Model</strong> tab.",
          "-6px 0 12px 0")

    template_bytes = _read_local(TEMPLATE_FILENAME) or mp.template_csv_text().encode("utf-8")
    sample_bytes = _read_local(SAMPLE_FILENAME)
    # compact buttons with slightly tighter spacing between them
    d1, d2, _spacer = st.columns([1.15, 1.15, 0.7], gap="small")
    with d1:
        stretch(st.download_button, "Download Empty Template CSV", data=template_bytes,
                file_name=TEMPLATE_FILENAME, mime="text/csv", key="dl_template")
    with d2:
        if sample_bytes is not None:
            stretch(st.download_button, "Download Sample Filled CSV", data=sample_bytes,
                    file_name=SAMPLE_DOWNLOAD_NAME, mime="text/csv", key="dl_sample")
        else:
            st.warning(f"Sample file '{SAMPLE_FILENAME}' was not found next to app.py.")

    with st.expander("How to Use This Tool (Tutorial & Guide)", expanded=False):
        _tutorial()

    st.markdown("<div style='height:14px;'></div>", unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload your completed CSV file", type=["csv"], key="custom_csv")

    if uploaded is None:
        st.info("Upload a CSV to get started. Not sure how? Download the sample file above and open the tutorial.")
        return

    raw = uploaded.getvalue()
    data_key = hashlib.md5(raw).hexdigest()
    try:
        df, report = _clean_upload(raw)
    except ValueError as e:
        st.error(f"{e}")
        return
    except Exception as e:                                           # noqa: BLE001
        st.error(f"Something went wrong while reading your file: {e}")
        return

    st.success(f"Your file **{uploaded.name}** is ready — {report['rows_final']:,} valid rows will be used.")
    st.divider()

    # ══════════════════════════════════════════════════
    # SAME STRUCTURE AS "TEST THE MODEL"
    # ══════════════════════════════════════════════════
    ui.section_heading("", "Predict Delivery Time")
    _hint("Choose which features the model should use, fill in the delivery details and click "
          "<strong style=\"color:#e0c070;\">Predict</strong> to get an estimated delivery time from a "
          "Random Forest trained on your data.")

    table = _selection_table(data_key, df)
    best = mp.get_best_features(table)

    mode_options = [ALL_LABEL] + ([BEST_LABEL] if best else [])
    feature_mode = st.radio(
        "Which features should the model use?", mode_options, horizontal=True,
        key=f"cust_mode_{data_key[:8]}",
        help="All features = the full Random Forest (7 inputs). Best features = a simpler Random Forest "
             "trained only on the factors that matter most in your data.")
    use_best = feature_mode == BEST_LABEL and bool(best)
    active_features = list(best) if use_best else list(mp.FEATURE_ORDER)

    # ── Train BOTH models as soon as the file is uploaded (cached, so this happens only once per file) ──
    trained_all = _train_private(data_key, tuple(mp.FEATURE_ORDER), df)
    trained_best = _train_private(data_key, tuple(best), df) if best else None
    trained = trained_best if use_best else trained_all

    teal = "font-family:'Rajdhani',sans-serif; font-size:14px; color:#2dd4bf; margin:4px 0 0 0;"
    if use_best:
        st.markdown(f"<p style=\"{teal}\">Using the <strong>{len(active_features)} best features</strong>: "
                    + ", ".join(mp.FEATURE_META[f][1] for f in active_features) + "</p>", unsafe_allow_html=True)
    else:
        st.markdown(f"<p style=\"{teal}\">Using <strong>all 7 features</strong>.</p>", unsafe_allow_html=True)

    # ── Drop-down summary (same as Test The Model) ──
    with st.expander("Feature summary — best features vs all features", expanded=False):
        dropped = [f for f in mp.FEATURE_ORDER if f not in best]
        ui.insight_line(1, "<strong>All features (7):</strong> the full model uses every column — Distance, Weather, "
                           "Traffic Level, Time of Day, Vehicle Type, Preparation Time and Courier Experience.")
        ui.insight_line(2, f"<strong>Best features ({len(best)}):</strong> "
                           + (", ".join(mp.FEATURE_META[f][1] for f in best) if best else "none found in your data")
                           + ". These are the factors with a statistically significant link to delivery time (p &lt; 0.05) "
                             "in the Pearson correlation test (numbers) or the One-Way ANOVA test (categories).")
        if dropped and best:
            ui.insight_line(3, "<strong>Left out of the best-features model:</strong> "
                               + ", ".join(mp.FEATURE_META[f][1] for f in dropped)
                               + " — on their own they change delivery time very little.")
        st.markdown("<p style='font-weight:700;font-size:16px;color:#c9a84c;margin:12px 0 6px 0;'>"
                    "How each feature was tested</p>", unsafe_allow_html=True)
        show = table[["Feature", "Test", "Statistic", "p-value", "Selected", "Why"]].copy()
        show["Feature"] = show["Feature"].map(lambda c: f"{mp.FEATURE_ICONS[c]} {mp.FEATURE_META[c][1]}")
        show["Statistic"] = show["Statistic"].round(3)
        show["p-value"] = show["p-value"].map(lambda v: "< 0.0001" if v < 1e-4 else f"{v:.4f}")
        show["Selected"] = show["Selected"].map({True: "⭐ Best", False: "—"})
        show.insert(5, "In all-features model", "✅")
        stretch(st.dataframe, show, hide_index=True)

        if trained_best is not None:
            st.markdown("<p style='font-weight:700;font-size:16px;color:#c9a84c;margin:12px 0 6px 0;'>"
                        "Accuracy comparison (same test data)</p>", unsafe_allow_html=True)
            am, bm = trained_all.metrics_rf, trained_best.metrics_rf
            stretch(st.dataframe, pd.DataFrame({
                "Model": ["All features", "Best features"],
                "Features used": [len(mp.FEATURE_ORDER), len(best)],
                "R²": [round(am["R2"], 3), round(bm["R2"], 3)],
                "MAE (min)": [round(am["MAE"], 2), round(bm["MAE"], 2)],
                "RMSE (min)": [round(am["RMSE"], 2), round(bm["RMSE"], 2)],
            }), hide_index=True)

    # ── Accuracy KPIs (available before the user clicks Predict) ──
    k1, k2 = st.columns(2)
    with k1:
        ui.kpi_card("", "R² · All Features", f"{trained_all.metrics_rf['R2']:.2f}",
                    f"All 7 features · MAE {trained_all.metrics_rf['MAE']:.2f} min")
    with k2:
        if trained_best is not None:
            ui.kpi_card("", "R² · Best Features", f"{trained_best.metrics_rf['R2']:.2f}",
                        f"Best {len(best)} features · MAE {trained_best.metrics_rf['MAE']:.2f} min")
        else:
            ui.kpi_card("", "R² · Best Features", "n/a", "No feature passed the tests")
    st.caption(f"Both models were trained on {trained_all.n_train:,} rows and tested on {trained_all.n_test:,} unseen rows."
               + (f" {trained_all.note}" if trained_all.note else ""))

    st.markdown("")

    # ── Input form — two columns (only the chosen features are shown) ──
    encoders = mp.fit_label_encoders(df)
    col_left, col_right = st.columns(2)
    raw_inputs = {}
    num_feats = [c for c in active_features if c in mp.NUMERIC_FEATURES]
    cat_feats = [c for c in active_features if c in mp.CATEGORICAL_FEATURES]

    with col_left:
        ui.section_heading("", "Numerical Inputs")
        for c in num_feats:
            raw_inputs[c] = _numeric_input(c, df[c], f"cust_{data_key[:8]}_{c}")

    with col_right:
        ui.section_heading("", "Categorical Inputs")
        for c in cat_feats:
            raw_inputs[c] = _categorical_input(c, encoders, df, f"cust_{data_key[:8]}_{c}")

    st.markdown("")

    # ── Predict button (full width) ──
    predict_clicked = st.button("🔮 Predict Delivery Time", use_container_width=True, key="cust_predict")

    if predict_clicked:
        try:
            prediction = mp.predict_one(trained, raw_inputs)
            r2 = trained.metrics_rf["R2"]
            if use_best:
                ui.prediction_banner(
                    prediction,
                    f"Predicted by Random Forest &nbsp;·&nbsp; best {len(active_features)} features "
                    f"&nbsp;·&nbsp; R² = {r2:.2f}")
            else:
                ui.prediction_banner(
                    prediction,
                    f"Predicted by Random Forest &nbsp;·&nbsp; all 7 features &nbsp;·&nbsp; R² = {r2:.2f}")

            st.markdown("<br>", unsafe_allow_html=True)
            ui.section_heading("", "Inputs Used for Prediction")
            for start in range(0, len(active_features), 4):
                chunk = active_features[start:start + 4]
                card_cols = st.columns(len(chunk))
                for card_col, feat in zip(card_cols, chunk):
                    icon, label, unit = mp.FEATURE_META[feat]
                    val = raw_inputs[feat]
                    text = f"{val:g} {unit}".strip() if feat in mp.NUMERIC_FEATURES else str(val)
                    with card_col:
                        ui.kpi_card(icon, label, text)
                st.markdown("<br>", unsafe_allow_html=True)
        except Exception as e:                                       # noqa: BLE001
            st.error(f"Prediction failed: {e}")
    else:
        st.markdown("<br>", unsafe_allow_html=True)
        with st.expander("Tips for Using the Predictor", expanded=False):
            ui.insight_line(1, "Slider ranges and dropdown options come from <strong>your own data</strong>.")
            ui.insight_line(2, "Both models (all features and best features) are trained as soon as your file is uploaded — "
                               "compare their R² scores above, then click <strong>Predict Delivery Time</strong>.")
            ui.insight_line(3, "Want fewer inputs? Switch to <strong>Use best features</strong> — only the factors that matter most "
                               "in your data are shown.")
            ui.insight_line(4, "More rows (200+) give a more accurate model.")

    # ══════════════════════════════════════════════════
    # BELOW THE PREDICTION SECTION
    # Dataset overview + data quality report
    # ══════════════════════════════════════════════════
    st.divider()

    ui.section_heading("", "Dataset Overview")
    _hint("The model behind the predictions above was trained on your uploaded dataset, summarised below.", "-6px 0 12px 0")

    d1, d2, d3 = st.columns(3)
    with d1:
        ui.kpi_card("", "Records", f"{report['rows_final']:,}", "Delivery orders")
    with d2:
        ui.kpi_card("", "Predictive Drivers", "7", "3 numeric · 4 categorical")
    with d3:
        ui.kpi_card("", "Target", "Delivery Time", "In minutes")

    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        ui.kpi_card("", "Rows Uploaded", f"{report['rows_uploaded']:,}", "In your file")
    with c2:
        ui.kpi_card("", "Duplicates Removed", f"{report['duplicates_removed']:,}", "Identical rows")
    with c3:
        ui.kpi_card("", "Rows Removed", f"{report['rows_dropped_no_target']:,}", "No usable Delivery_Time_min")
    with c4:
        ui.kpi_card("", "Blanks Filled", f"{report['blanks_filled']:,}", "Median / most frequent")

    notes = []
    if report["invalid_entries"]:
        notes.append(f"{report['invalid_entries']:,} entries such as '--', '?', 'unknown' or text inside a number "
                     "column were treated as blank cells.")
    if report["extra_columns_ignored"]:
        notes.append("Extra columns ignored: " + ", ".join(report["extra_columns_ignored"]) + ".")
    if notes:
        st.caption("" + " ".join(notes))

    st.markdown("<br>", unsafe_allow_html=True)

    ui.section_heading("", "Data Quality Report")
    _hint("Charts and plain-English insights generated directly from your "
          "data — distributions, categorical rankings, average-time bars, box plots, two-factor heatmaps, "
          "and correlations.", "-6px 0 12px 0")
    render_data_quality_report(df)

    st.caption("Your file is processed in memory only and is not saved to disk.")
    st.markdown("<div style='height:50px;'></div>", unsafe_allow_html=True)