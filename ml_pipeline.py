"""
ml_pipeline.py
==============
Shared data-cleaning, feature-selection and model-training logic for the
Food Delivery Time Predictor.

This file has NO Streamlit code in it, so the exact same functions can be used by:
  * app.py                  -> "Best features" mode of the pre-trained model
  * custom_dataset_tab.py   -> the "Try With Your Own Dataset" tab
  * a plain Python shell    -> `python ml_pipeline.py` rebuilds the best-features files

The steps mirror Food_Delivery_Time_Prediction_model.ipynb:
  cleaning (Step 6) -> statistical tests (Steps 16-17) -> label encoding (Step 19)
  -> 80/20 split with random_state=42 (Step 20) -> Decision Tree baseline (Step 21)
  -> Random Forest (Step 23) -> metrics (Steps 22 / 24) -> feature importance (Step 26).
"""
from __future__ import annotations

import json
import os
import pickle
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeRegressor

# ─────────────────────────────────────────────────────────────
# SCHEMA  (identical to the columns of the dataset / template CSV)
# ─────────────────────────────────────────────────────────────
ID_COLUMN = "Order_ID"
TARGET = "Delivery_Time_min"
NUMERIC_FEATURES = ["Distance_km", "Preparation_Time_min", "Courier_Experience_yrs"]
CATEGORICAL_FEATURES = ["Weather", "Traffic_Level", "Time_of_Day", "Vehicle_Type"]

# Column order the model is trained on (same as the notebook after dropping Order_ID)
FEATURE_ORDER = [
    "Distance_km", "Weather", "Traffic_Level", "Time_of_Day",
    "Vehicle_Type", "Preparation_Time_min", "Courier_Experience_yrs",
]
TEMPLATE_COLUMNS = [ID_COLUMN] + FEATURE_ORDER + [TARGET]

# icon, friendly name, unit
FEATURE_META = {
    "Distance_km":            ("", "Distance", "km"),
    "Weather":                ("", "Weather", ""),
    "Traffic_Level":          ("", "Traffic Level", ""),
    "Time_of_Day":            ("", "Time of Day", ""),
    "Vehicle_Type":           ("", "Vehicle Type", ""),
    "Preparation_Time_min":   ("", "Preparation Time", "min"),
    "Courier_Experience_yrs": ("", "Courier Experience", "yrs"),
}

# Icons shown only in the feature-selection table (cards stay icon-free)
FEATURE_ICONS = {
    "Distance_km": "📍", "Weather": "🌦️", "Traffic_Level": "🚦", "Time_of_Day": "🕐",
    "Vehicle_Type": "🏍️", "Preparation_Time_min": "🍳", "Courier_Experience_yrs": "🧑‍💼",
}

# Natural order of ordinal categories (used for charts and dropdowns)
ORDER_HINT = {
    "Traffic_Level": ["Low", "Medium", "High"],
    "Time_of_Day": ["Morning", "Afternoon", "Evening", "Night"],
}

MIN_ROWS = 30            # minimum valid rows needed to train a model
ALPHA = 0.05             # significance level (same as notebook Step 17)
MAX_TRAIN_ROWS = 30000   # larger uploads are randomly sampled (keeps RAM / time low)

# Text that really means "no value" — treated exactly like a blank cell
_JUNK_TOKENS = {
    "--", "-", "?", "??", "na", "n/a", "nan", "null", "none", "unknown", "invalid",
    "missing", "nil", "#n/a", "#value!", "undefined",
}

# Numeric columns whose value must be > 0 (experience may legitimately be 0)
_MUST_BE_POSITIVE = {"Distance_km", "Preparation_Time_min", TARGET}


# ─────────────────────────────────────────────────────────────
# 1. CLEANING  (notebook Step 6, plus safeguards for user-made files)
# ─────────────────────────────────────────────────────────────
def _canonical_headers(df: pd.DataFrame) -> pd.DataFrame:
    """Fix header spelling differences such as ' distance_km ' or 'weather'."""
    lookup = {c.lower(): c for c in TEMPLATE_COLUMNS}
    rename = {}
    for col in df.columns:
        key = str(col).strip().lower().replace(" ", "_")
        if key in lookup:
            rename[col] = lookup[key]
    return df.rename(columns=rename)


def clean_dataframe(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Cleans a delivery dataset exactly like the notebook does and returns
    (clean_df, report).

    Notebook steps (always applied):
      * missing Weather / Traffic_Level / Time_of_Day -> most frequent value (mode)
      * missing Courier_Experience_yrs                -> median
      * Order_ID dropped

    Extra safeguards for files made by users (do nothing on the clean training CSV):
      * entries such as '--', '?', 'unknown', 'invalid' or text inside a numeric column
        are treated as blank; impossible numbers (<= 0 distance etc.) are treated as blank
      * category spelling/casing is tidied ('rainy ' -> 'Rainy')
      * fully duplicated rows are removed
      * rows with no Delivery_Time_min are removed (we cannot learn from them)
      * blanks in the other numeric columns are filled with the median

    Raises ValueError with a human-friendly message when the file cannot be used.
    """
    df = _canonical_headers(raw.copy())
    required = FEATURE_ORDER + [TARGET]
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        raise ValueError(
            "Your file is missing these required columns: "
            + ", ".join(missing_cols)
            + ". Please start from the template CSV and keep the header row exactly as it is."
        )

    extra_cols = [c for c in df.columns if c not in TEMPLATE_COLUMNS]
    keep = ([ID_COLUMN] if ID_COLUMN in df.columns else []) + required
    df = df[keep]

    report = {
        "rows_uploaded": int(len(df)),
        "extra_columns_ignored": [str(c) for c in extra_cols],
        "duplicates_removed": 0,
        "rows_dropped_no_target": 0,
        "invalid_entries": 0,
        "invalid_by_column": {},
        "blanks_filled": 0,
        "filled_by_column": {},
        "rows_final": 0,
    }

    # ── remove exact duplicate rows ──────────────────────────
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    report["duplicates_removed"] = int(before - len(df))

    # ── numeric columns: text/junk/impossible values -> blank ─
    out = pd.DataFrame(index=df.index)
    for col in NUMERIC_FEATURES + [TARGET]:
        s = df[col].astype("string").str.strip()
        blank = (s.isna() | (s == "")).to_numpy(dtype=bool)
        junk = s.str.lower().isin(_JUNK_TOKENS).fillna(False).to_numpy(dtype=bool)
        x = pd.to_numeric(s.mask(junk), errors="coerce").astype(float)
        x = x.where(x > 0) if col in _MUST_BE_POSITIVE else x.where(x >= 0)
        invalid = int((~blank & x.isna().to_numpy()).sum())
        report["invalid_by_column"][col] = invalid
        report["invalid_entries"] += invalid
        out[col] = x

    # ── categorical columns: junk -> blank, tidy spelling ────
    for col in CATEGORICAL_FEATURES:
        s = df[col].astype("string").str.strip()
        blank = (s.isna() | (s == "")).to_numpy(dtype=bool)
        junk = s.str.lower().isin(_JUNK_TOKENS).fillna(False).to_numpy(dtype=bool)
        s = s.mask(junk).str.title()
        invalid = int((~blank & s.isna().to_numpy()).sum())
        report["invalid_by_column"][col] = invalid
        report["invalid_entries"] += invalid
        out[col] = s.astype(object).where(s.notna(), np.nan)

    # ── rows without a target cannot be used ─────────────────
    has_target = out[TARGET].notna()
    report["rows_dropped_no_target"] = int((~has_target).sum())
    out = out[has_target].reset_index(drop=True)

    if len(out) < MIN_ROWS:
        raise ValueError(
            f"Only {len(out)} usable rows were found after cleaning — at least {MIN_ROWS} are needed "
            f"to train a model (200+ gives better accuracy)."
        )

    # ── fill remaining blanks: mode (categories) / median (numbers) ──
    for col in CATEGORICAL_FEATURES:
        n_missing = int(out[col].isna().sum())
        if out[col].notna().sum() == 0:
            raise ValueError(f"Column '{col}' has no usable values — please check your file.")
        mode_value = out[col].mode()[0]
        out[col] = out[col].fillna(mode_value).astype(str)
        if n_missing:
            report["filled_by_column"][col] = (n_missing, f"most frequent: {mode_value}")
            report["blanks_filled"] += n_missing

    for col in NUMERIC_FEATURES:
        n_missing = int(out[col].isna().sum())
        if out[col].notna().sum() == 0:
            raise ValueError(f"Column '{col}' has no usable numbers — please check your file.")
        median_value = float(out[col].median())
        out[col] = out[col].fillna(median_value)
        if n_missing:
            report["filled_by_column"][col] = (n_missing, f"median: {median_value:g}")
            report["blanks_filled"] += n_missing

    out = out[FEATURE_ORDER + [TARGET]].astype({c: float for c in NUMERIC_FEATURES + [TARGET]})
    report["rows_final"] = int(len(out))
    return out, report


# ─────────────────────────────────────────────────────────────
# 2. FEATURE SELECTION  (notebook Steps 16-17 turned into a rule)
# ─────────────────────────────────────────────────────────────
def feature_selection_table(df: pd.DataFrame, alpha: float = ALPHA) -> pd.DataFrame:
    """
    Tests every feature against Delivery_Time_min:
      * numeric features      -> Pearson correlation (r, p-value)
      * categorical features  -> One-Way ANOVA (F, p-value)

    A feature is a BEST feature when its p-value is below 0.05 (statistically significant).
    """
    y = df[TARGET].to_numpy(dtype=float)
    rows = []
    for col in FEATURE_ORDER:
        if col in NUMERIC_FEATURES:
            x = df[col].to_numpy(dtype=float)
            if np.std(x) == 0 or len(x) < 3:
                r, p = 0.0, 1.0
            else:
                r, p = stats.pearsonr(x, y)
            rows.append({"Feature": col, "Type": "Numeric", "Test": "Pearson correlation",
                         "Statistic": float(r), "p-value": float(p)})
        else:
            groups = [y[(df[col] == g).to_numpy()] for g in df[col].unique()]
            groups = [g for g in groups if len(g) > 0]
            try:
                if len(groups) < 2:
                    raise ValueError("single category")
                f_stat, p = stats.f_oneway(*groups)
                if np.isnan(f_stat):
                    raise ValueError("undefined")
            except Exception:
                f_stat, p = 0.0, 1.0
            rows.append({"Feature": col, "Type": "Categorical", "Test": "One-Way ANOVA",
                         "Statistic": float(f_stat), "p-value": float(p)})

    table = pd.DataFrame(rows)
    table["Selected"] = table["p-value"] < alpha
    table["Why"] = [
        ("Significant (p < 0.0001)" if p < 1e-4 else f"Significant (p = {p:.4f})") if sel
        else f"Not statistically significant (p = {p:.2f})"
        for p, sel in zip(table["p-value"], table["Selected"])
    ]
    return table


def get_best_features(table: pd.DataFrame) -> list[str]:
    """Selected features, kept in the model's training column order."""
    chosen = set(table.loc[table["Selected"], "Feature"])
    return [c for c in FEATURE_ORDER if c in chosen]


# ─────────────────────────────────────────────────────────────
# 3. ENCODING + TRAINING  (notebook Steps 19-24)
# ─────────────────────────────────────────────────────────────
def fit_label_encoders(df: pd.DataFrame) -> dict[str, LabelEncoder]:
    encoders = {}
    for col in CATEGORICAL_FEATURES:
        le = LabelEncoder()
        le.fit(df[col].astype(str))
        encoders[col] = le
    return encoders


def encode_features(df: pd.DataFrame, encoders: dict, features: list[str]) -> pd.DataFrame:
    X = pd.DataFrame(index=df.index)
    for col in features:
        if col in CATEGORICAL_FEATURES:
            X[col] = encoders[col].transform(df[col].astype(str))
        else:
            X[col] = df[col].astype(float)
    return X[features]


def regression_metrics(y_true, y_pred) -> dict:
    mse = mean_squared_error(y_true, y_pred)
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "MSE": float(mse),
        "RMSE": float(np.sqrt(mse)),
        "R2": float(r2_score(y_true, y_pred)),
    }


@dataclass
class TrainedModel:
    model: RandomForestRegressor
    features: list
    encoders: dict
    metrics_rf: dict
    metrics_dt: dict
    importances: pd.Series            # % importance per feature, sorted
    n_train: int
    n_test: int
    sampled: bool = False
    note: str = ""
    extras: dict = field(default_factory=dict)


def train_model(df: pd.DataFrame, features: list[str], test_size: float = 0.2,
                random_state: int = 42) -> TrainedModel:
    """
    Trains the Decision Tree baseline and the Random Forest on the chosen features.
    Uses the same settings as the notebook: 80/20 split, random_state=42, default
    Random Forest (100 trees).  Only for very large uploads (> 5,000 training rows)
    each leaf must hold >= 3 rows, which keeps memory small without hurting accuracy.
    """
    if not features:
        raise ValueError("Select at least one feature to train a model.")

    encoders = fit_label_encoders(df)
    work, sampled = df, False
    if len(df) > MAX_TRAIN_ROWS:
        work, sampled = df.sample(MAX_TRAIN_ROWS, random_state=random_state), True

    X = encode_features(work, encoders, features)
    y = work[TARGET].astype(float)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    extra = {"min_samples_leaf": 3} if len(X_train) > 5000 else {}
    dt = DecisionTreeRegressor(random_state=random_state, **extra).fit(X_train, y_train)
    rf = RandomForestRegressor(random_state=random_state, n_jobs=-1, **extra).fit(X_train, y_train)

    importances = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=False) * 100
    note = ""
    if sampled:
        note += f"Your file is large, so a random sample of {MAX_TRAIN_ROWS:,} rows was used for training. "
    if extra:
        note += "Large dataset: each tree leaf holds at least 3 rows to keep the model light."
    return TrainedModel(
        model=rf, features=list(features), encoders=encoders,
        metrics_rf=regression_metrics(y_test, rf.predict(X_test)),
        metrics_dt=regression_metrics(y_test, dt.predict(X_test)),
        importances=importances, n_train=len(X_train), n_test=len(X_test),
        sampled=sampled, note=note.strip(),
    )


def predict_one(trained: TrainedModel, values: dict) -> float:
    """values = {feature_name: raw value (a number, or the category text)}"""
    row = pd.DataFrame([{f: values[f] for f in trained.features}])
    X = encode_features(row, trained.encoders, trained.features)
    return float(trained.model.predict(X)[0])


# ─────────────────────────────────────────────────────────────
# 4. DEFAULT-DATASET "BEST FEATURES" MODEL  (used by app.py)
# ─────────────────────────────────────────────────────────────
BEST_MODEL_FILE = "rf_model_best.pkl"
BEST_META_FILE = "best_features.json"
_REQUIRED_META_KEYS = {"best_features", "all_features", "selection_table", "metrics_best", "metrics_all"}


def table_to_records(table: pd.DataFrame) -> list[dict]:
    cols = ["Feature", "Type", "Test", "Statistic", "p-value", "Selected", "Why"]
    recs = table[cols].to_dict("records")
    for r in recs:
        for k, v in r.items():
            if isinstance(v, (np.floating,)):
                r[k] = float(v)
            elif isinstance(v, (np.bool_,)):
                r[k] = bool(v)
    return recs


def build_best_model_files(base_dir: str, dataset_filename: str = "Food_Delivery_Times.csv", save: bool = True):
    """Trains the best-features model on the original dataset and (optionally) saves it."""
    df, _ = clean_dataframe(pd.read_csv(os.path.join(base_dir, dataset_filename)))
    table = feature_selection_table(df)
    best = get_best_features(table)
    res_best = train_model(df, best)
    res_all = train_model(df, FEATURE_ORDER)
    meta = {
        "best_features": best,
        "all_features": FEATURE_ORDER,
        "criteria": {"alpha": ALPHA},
        "selection_table": table_to_records(table),
        "metrics_best": res_best.metrics_rf,
        "metrics_all": res_all.metrics_rf,
        "importances_best": {k: float(v) for k, v in res_best.importances.items()},
    }
    if save:
        try:
            with open(os.path.join(base_dir, BEST_MODEL_FILE), "wb") as f:
                pickle.dump(res_best.model, f)
            with open(os.path.join(base_dir, BEST_META_FILE), "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)
        except OSError:
            pass  # read-only hosting: the model simply lives in memory for this session
    return res_best.model, meta


def load_or_build_best_model(base_dir: str, dataset_filename: str = "Food_Delivery_Times.csv"):
    """
    Returns (best_model, meta).  Uses rf_model_best.pkl + best_features.json when they
    exist (created by the notebook); otherwise trains the model from the dataset once.
    """
    model_path = os.path.join(base_dir, BEST_MODEL_FILE)
    meta_path = os.path.join(base_dir, BEST_META_FILE)
    if os.path.exists(model_path) and os.path.exists(meta_path):
        try:
            with open(model_path, "rb") as f:
                model = pickle.load(f)
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            # files made with the old rule (effect-size thresholds) are rebuilt with the new one
            if _REQUIRED_META_KEYS.issubset(meta) and set(meta.get("criteria", {})) == {"alpha"}:
                return model, meta
        except Exception:
            pass  # damaged / incompatible files -> rebuild below
    return build_best_model_files(base_dir, dataset_filename, save=True)


def template_csv_text() -> str:
    """Header-only CSV (no data rows)."""
    return ",".join(TEMPLATE_COLUMNS) + "\n"


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    _, info = build_best_model_files(here)
    print("Best features :", info["best_features"])
    print("Best  model   :", {k: round(v, 3) for k, v in info["metrics_best"].items()})
    print("All-feature   :", {k: round(v, 3) for k, v in info["metrics_all"].items()})
    print(f"Saved {BEST_MODEL_FILE} and {BEST_META_FILE} in {here}")