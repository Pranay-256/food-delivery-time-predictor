"""
pipeline_engine.py

Data validation, feature ranking, diagnostics and model training for the
Food Delivery Time Prediction app.

Design principle
----------------
The *full* cleaned DataFrame (all 7 predictors + target) is the single source
of truth. Feature selection never mutates it: every training call slices a
fresh copy, so a user can untick a feature, re-tick it later, and retrain
instantly from the same cleaned data stored in ``st.session_state``.

Public API
----------
validate_and_clean_data(raw_df)                  -> (cleaned_df, cleaning_logs)
clean_dataset(raw_df)                            -> alias of the above
prepare_dataset(raw_df)                          -> alias of the above
rank_features(cleaned_df)                        -> (ranked_df, best_features)
compute_feature_diagnostics(cleaned_df, selected_features=None) -> DataFrame
train_model(cleaned_df, selected_features=None)  -> artifact dict (raises ValueError)
train_custom_model(cleaned_df, selected_features=None)
                                                 -> (artifact | None, error | None)
predict_single(artifact, input_dict)             -> float
predict_delivery_time(artifact, input_dict)      -> float   (alias, used by the UI)
predict_batch(artifact, df)                      -> np.ndarray
"""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

try:  # scipy is only needed for the optional diagnostics helper
    from scipy import stats as _scipy_stats
except Exception:  # pragma: no cover
    _scipy_stats = None

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
TARGET_COL = "Delivery_Time_min"
ID_COL = "Order_ID"

NUMERIC_PREDICTORS = ["Distance_km", "Preparation_Time_min", "Courier_Experience_yrs"]
CATEGORICAL_PREDICTORS = ["Weather", "Traffic_Level", "Time_of_Day", "Vehicle_Type"]
PREDICTORS = NUMERIC_PREDICTORS + CATEGORICAL_PREDICTORS  # the 7 valid predictors

NUMERIC_COLS = NUMERIC_PREDICTORS + [TARGET_COL]
REQUIRED_COLS = PREDICTORS + [TARGET_COL]

MIN_ROWS_FOR_TRAINING = 30
IMPORTANCE_THRESHOLD_PCT = 5.0
MIN_BEST_FEATURES = 2

ANOVA_ALPHA = 0.05          # categorical: significant if p < 0.05
CORRELATION_MIN_ABS_R = 0.20  # numeric: meaningful if |r| >= 0.20

UNSEEN_CATEGORY_CODE = -1
NO_FEATURES_MSG = "Please select at least one feature to train the model."


# --------------------------------------------------------------------------- #
# 1. Schema validation & cleaning (returns the FULL cleaned frame)
# --------------------------------------------------------------------------- #
def validate_and_clean_data(raw_df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Validate the uploaded schema and return a modeling-ready DataFrame.

    The returned frame always holds all 7 predictors + the target
    (``Order_ID`` and any unrelated columns removed, types coerced, missing
    values imputed). Feature *selection* is NOT applied here, so the caller can
    cache this frame and retrain on any subset later without re-cleaning.

    Raises
    ------
    ValueError
        If the input is not a DataFrame or required columns are missing.
    """
    if not isinstance(raw_df, pd.DataFrame):
        raise ValueError("Uploaded data could not be read as a table (DataFrame).")

    logs: list[str] = []
    df = raw_df.copy()

    # Normalise header whitespace (e.g. " Weather ")
    df.columns = [str(c).strip() for c in df.columns]

    # --- Required columns ---------------------------------------------------
    missing_cols = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing_cols:
        raise ValueError(
            "Missing required column(s): "
            + ", ".join(missing_cols)
            + ". Required columns are: "
            + ", ".join(REQUIRED_COLS)
            + "."
        )

    # --- Order_ID: always dropped, zero bias ---------------------------------
    if ID_COL in df.columns:
        df = df.drop(columns=[ID_COL])
        logs.append(f"Dropped identifier column '{ID_COL}' so it cannot bias the model.")

    # Keep only the columns the pipeline understands
    extra_cols = [c for c in df.columns if c not in REQUIRED_COLS]
    if extra_cols:
        df = df.drop(columns=extra_cols)
        logs.append(f"Ignored {len(extra_cols)} unused column(s): {', '.join(extra_cols)}.")
    df = df[REQUIRED_COLS]

    initial_rows = len(df)
    logs.append(f"Received {initial_rows} row(s) in the uploaded file.")

    # --- Numeric coercion ----------------------------------------------------
    for col in NUMERIC_COLS:
        before_missing = int(df[col].isna().sum())
        df[col] = pd.to_numeric(df[col], errors="coerce")
        inf_count = int(np.isinf(df[col]).sum())
        if inf_count:
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)
        invalid = int(df[col].isna().sum()) - before_missing
        if invalid > 0:
            logs.append(
                f"Converted {invalid} non-numeric or invalid value(s) in {col} to missing."
            )

    # --- Drop rows with missing target (never impute) -------------------------
    missing_target = int(df[TARGET_COL].isna().sum())
    if missing_target:
        df = df.dropna(subset=[TARGET_COL])
        logs.append(
            f"Dropped {missing_target} row(s) with missing Delivery Time "
            "(target values are never imputed)."
        )

    # --- Categorical imputation ----------------------------------------------
    for col in CATEGORICAL_PREDICTORS:
        series = df[col]
        # Treat blank / whitespace-only strings as missing, and trim text
        series = series.where(series.isna(), series.astype(str).str.strip())
        series = series.replace("", np.nan)
        n_missing = int(series.isna().sum())

        if n_missing:
            modes = series.dropna().mode()
            fill_value = str(modes.iloc[0]) if not modes.empty else "Unknown"
            series = series.fillna(fill_value)
            if fill_value == "Unknown":
                logs.append(
                    f"Filled {n_missing} missing {col.replace('_', ' ')} entries "
                    f"with fallback value: Unknown"
                )
            else:
                logs.append(
                    f"Filled {n_missing} missing {col.replace('_', ' ')} entries "
                    f"using most frequent value: {fill_value}"
                )
        df[col] = series.astype(str)

    # --- Numeric predictor imputation -----------------------------------------
    for col in NUMERIC_PREDICTORS:
        n_missing = int(df[col].isna().sum())
        if n_missing:
            median = df[col].median()
            if pd.isna(median):  # whole column empty
                median = 0.0
            df[col] = df[col].fillna(median)
            logs.append(
                f"Filled {n_missing} missing {col.replace('_', ' ')} value(s) "
                f"using median: {median:.2f}"
            )

    df = df.reset_index(drop=True)

    if len(df) == initial_rows and len(logs) <= 2:
        logs.append("No missing or invalid values found. Data is already clean.")
    logs.append(f"{len(df)} clean row(s) are ready for modeling.")

    return df, logs


# Descriptive aliases: all three names return the full cleaned DataFrame.
clean_dataset = validate_and_clean_data
prepare_dataset = validate_and_clean_data


# --------------------------------------------------------------------------- #
# Internal helpers
# --------------------------------------------------------------------------- #
def _available_predictors(cleaned_df: pd.DataFrame) -> list[str]:
    """Valid predictor columns actually present in `cleaned_df` (never target / ID)."""
    return [c for c in PREDICTORS if c in cleaned_df.columns]


def _resolve_features(
    cleaned_df: pd.DataFrame, selected_features: Iterable[str] | None
) -> list[str]:
    """
    Turn a user selection into a validated, de-duplicated feature list that
    preserves the caller's order. ``None`` means "all available predictors".
    """
    available = _available_predictors(cleaned_df)

    if selected_features is None:
        features = list(available)
    else:
        if isinstance(selected_features, str):
            selected_features = [selected_features]
        features = list(dict.fromkeys(selected_features))  # de-dup, keep order

    if not features:
        raise ValueError(NO_FEATURES_MSG)

    invalid = [f for f in features if f not in available]
    if invalid:
        raise ValueError(
            f"Invalid feature(s) selected: {', '.join(map(str, invalid))}. "
            f"Valid options present in the data are: {', '.join(available)}."
        )
    return features


def _fit_label_encoders(df: pd.DataFrame, features: list[str]) -> dict[str, LabelEncoder]:
    """Fit one LabelEncoder per *selected* categorical column only."""
    encoders: dict[str, LabelEncoder] = {}
    for col in features:
        if col in CATEGORICAL_PREDICTORS:
            enc = LabelEncoder()
            enc.fit(df[col].astype(str))
            encoders[col] = enc
    return encoders


def _encode_column(values: pd.Series, enc: LabelEncoder) -> pd.Series:
    """Encode with a fitted LabelEncoder; unseen categories become -1."""
    mapping = {cls: i for i, cls in enumerate(enc.classes_)}
    return values.astype(str).map(mapping).fillna(UNSEEN_CATEGORY_CODE).astype(float)


def _encode_features(
    df: pd.DataFrame, features: list[str], encoders: dict[str, LabelEncoder]
) -> pd.DataFrame:
    """
    Return a fully numeric frame containing ONLY `features`, in that exact
    order. Only categorical columns that are in `features` are encoded.
    """
    X = df[features].copy()
    for col in features:
        if col in CATEGORICAL_PREDICTORS:
            if col not in encoders:
                raise ValueError(f"No fitted encoder available for feature '{col}'.")
            X[col] = _encode_column(X[col], encoders[col])
    return X.astype(float)


def _importance_pct(model: RandomForestRegressor, n: int) -> np.ndarray:
    imp = np.asarray(model.feature_importances_, dtype=float)
    total = imp.sum()
    if total > 0:
        return imp / total * 100.0
    return np.full(n, 100.0 / n)  # degenerate case (e.g. constant target)


# --------------------------------------------------------------------------- #
# 2. Feature ranking (always over ALL 7 predictors)
# --------------------------------------------------------------------------- #
def rank_features(cleaned_df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Rank the 7 valid predictors by Random Forest importance.

    Returns
    -------
    (ranked_df, best_features)
        ranked_df: columns ['feature', 'importance_pct'], sorted high -> low,
                   importance_pct sums to 100.
        best_features: features with importance_pct >= 5.0 (at least the top 2).
    """
    if cleaned_df is None or len(cleaned_df) == 0:
        raise ValueError("Cannot rank features: the cleaned dataset is empty.")

    missing = [c for c in PREDICTORS + [TARGET_COL] if c not in cleaned_df.columns]
    if missing:
        raise ValueError(f"Cannot rank features, missing column(s): {', '.join(missing)}")

    features = list(PREDICTORS)  # never includes Order_ID
    encoders = _fit_label_encoders(cleaned_df, features)
    X = _encode_features(cleaned_df, features, encoders)
    y = cleaned_df[TARGET_COL].astype(float)

    rf = RandomForestRegressor(n_estimators=150, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(X, y)

    ranked_df = (
        pd.DataFrame({"feature": features, "importance_pct": _importance_pct(rf, len(features))})
        .sort_values("importance_pct", ascending=False, kind="mergesort")
        .reset_index(drop=True)
    )

    best_features = ranked_df.loc[
        ranked_df["importance_pct"] >= IMPORTANCE_THRESHOLD_PCT, "feature"
    ].tolist()
    if len(best_features) < MIN_BEST_FEATURES:
        best_features = ranked_df["feature"].head(MIN_BEST_FEATURES).tolist()

    return ranked_df, best_features


# --------------------------------------------------------------------------- #
# 3. Statistical feature diagnostics (all columns, selected or not)
# --------------------------------------------------------------------------- #
def compute_feature_diagnostics(
    cleaned_df: pd.DataFrame, selected_features: Iterable[str] | None = None
) -> pd.DataFrame:
    """
    Significance diagnostics for every valid predictor in `cleaned_df`,
    regardless of the current selection.

    * Numeric features    -> Pearson correlation with the target
      (meaningful when |r| >= 0.20 and p < 0.05).
    * Categorical features -> One-Way ANOVA of the target across categories
      (significant when p < 0.05); eta-squared is reported as effect size.

    Returns a DataFrame with columns:
        feature, type, test, statistic, p_value, effect_size, significant,
        selected, status
    where ``status`` is "Selected" / "Excluded", so the UI can show why
    dropping a low-impact feature does (or does not) hurt accuracy.
    """
    if cleaned_df is None or len(cleaned_df) == 0:
        raise ValueError("Cannot compute diagnostics: the cleaned dataset is empty.")
    if TARGET_COL not in cleaned_df.columns:
        raise ValueError(f"Cannot compute diagnostics: '{TARGET_COL}' column is missing.")

    available = _available_predictors(cleaned_df)
    selected = set(available if selected_features is None else selected_features)
    y = pd.to_numeric(cleaned_df[TARGET_COL], errors="coerce")

    rows: list[dict[str, Any]] = []
    for col in available:
        row: dict[str, Any] = {
            "feature": col,
            "type": "numeric" if col in NUMERIC_PREDICTORS else "categorical",
            "test": "Pearson correlation" if col in NUMERIC_PREDICTORS else "One-Way ANOVA",
            "statistic": np.nan,
            "p_value": np.nan,
            "effect_size": np.nan,
            "significant": False,
        }
        try:
            if col in NUMERIC_PREDICTORS:
                x = pd.to_numeric(cleaned_df[col], errors="coerce")
                ok = x.notna() & y.notna()
                if ok.sum() >= 3 and x[ok].nunique() > 1 and y[ok].nunique() > 1:
                    if _scipy_stats is not None:
                        r, p = _scipy_stats.pearsonr(x[ok], y[ok])
                    else:  # scipy missing: r only, no p-value
                        r, p = float(np.corrcoef(x[ok], y[ok])[0, 1]), np.nan
                    row.update(statistic=float(r), p_value=float(p), effect_size=abs(float(r)))
                    row["significant"] = bool(
                        abs(r) >= CORRELATION_MIN_ABS_R and (np.isnan(p) or p < ANOVA_ALPHA)
                    )
            elif _scipy_stats is not None:
                groups = [
                    g.dropna().to_numpy(dtype=float)
                    for _, g in y.groupby(cleaned_df[col].astype(str))
                    if g.notna().sum() > 0
                ]
                if len(groups) >= 2 and sum(len(g) for g in groups) > len(groups):
                    f_stat, p = _scipy_stats.f_oneway(*groups)
                    if np.isfinite(f_stat):
                        grand = y.dropna().mean()
                        ss_between = sum(len(g) * (g.mean() - grand) ** 2 for g in groups)
                        ss_total = float(((y.dropna() - grand) ** 2).sum())
                        eta2 = ss_between / ss_total if ss_total > 0 else 0.0
                        row.update(
                            statistic=float(f_stat), p_value=float(p), effect_size=float(eta2)
                        )
                        row["significant"] = bool(p < ANOVA_ALPHA)
        except Exception:  # noqa: BLE001 - diagnostics must never break the app
            pass

        row["selected"] = col in selected
        row["status"] = "Selected" if row["selected"] else "Excluded"
        rows.append(row)

    return (
        pd.DataFrame(rows)
        .sort_values("effect_size", ascending=False, na_position="last", kind="mergesort")
        .reset_index(drop=True)
    )


# --------------------------------------------------------------------------- #
# 4. Model training (strictly on the selected columns)
# --------------------------------------------------------------------------- #
def train_model(
    cleaned_df: pd.DataFrame,
    selected_features: list[str] | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict[str, Any]:
    """
    Train a Random Forest using ONLY `selected_features` (default: all valid
    predictors). The input `cleaned_df` is never modified.

    Returns
    -------
    artifact dict with keys
        'model', 'label_encoders' (selected categorical columns only),
        'selected_features' (exact training column order),
        'metrics' (r2, mae, rmse, n_train, n_test, row_count),
        'feature_importances' ({feature: pct}, sums to 100, sorted high -> low),
        'feature_metadata' (numeric: min/max/median; categorical: sorted classes).

    Raises
    ------
    ValueError
        No / invalid features selected, missing target, or too few rows.
    """
    if cleaned_df is None or len(cleaned_df) < MIN_ROWS_FOR_TRAINING:
        n = 0 if cleaned_df is None else len(cleaned_df)
        raise ValueError(
            f"Not enough data to train a reliable model: found {n} valid row(s), "
            f"but at least {MIN_ROWS_FOR_TRAINING} are required. "
            "Please upload a larger dataset."
        )
    if TARGET_COL not in cleaned_df.columns:
        raise ValueError(f"Cleaned data is missing the target column '{TARGET_COL}'.")

    features = _resolve_features(cleaned_df, selected_features)

    # Slice: deselected columns are removed BEFORE encoding / split / fit.
    X_raw = cleaned_df[features].copy()
    y = cleaned_df[TARGET_COL].copy().astype(float)

    label_encoders = _fit_label_encoders(X_raw, features)
    X = _encode_features(X_raw, features, label_encoders)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    model = RandomForestRegressor(
        n_estimators=200, max_depth=12, random_state=random_state, n_jobs=-1
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    metrics = {
        "r2": float(r2_score(y_test, preds)),
        "mae": float(mean_absolute_error(y_test, preds)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, preds))),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "row_count": int(len(cleaned_df)),  # kept for backward compatibility
    }

    pct = _importance_pct(model, len(features))
    feature_importances = dict(
        sorted(zip(features, map(float, pct)), key=lambda kv: kv[1], reverse=True)
    )

    # Metadata only for the selected features, so the UI renders only those.
    feature_metadata: dict[str, dict[str, Any]] = {}
    for col in features:
        if col in NUMERIC_PREDICTORS:
            feature_metadata[col] = {
                "type": "numeric",
                "min": float(cleaned_df[col].min()),
                "max": float(cleaned_df[col].max()),
                "median": float(cleaned_df[col].median()),
            }
        else:
            feature_metadata[col] = {
                "type": "categorical",
                "categories": sorted(cleaned_df[col].astype(str).unique().tolist()),
            }

    return {
        "model": model,
        "label_encoders": label_encoders,
        "selected_features": list(features),
        "metrics": metrics,
        "feature_importances": feature_importances,
        "feature_metadata": feature_metadata,
    }


def train_custom_model(
    cleaned_df: pd.DataFrame, selected_features: list[str] | None = None
) -> tuple[dict[str, Any] | None, str | None]:
    """
    Backward-compatible wrapper around :func:`train_model`.

    Returns (artifact, None) on success or (None, "error message") on failure.
    ``selected_features=None`` trains on all valid predictors.
    """
    try:
        return train_model(cleaned_df, selected_features), None
    except ValueError as exc:
        return None, str(exc)


# --------------------------------------------------------------------------- #
# 5. Inference (strictly aligned to the trained feature set)
# --------------------------------------------------------------------------- #
def _artifact_features(artifact: dict[str, Any]) -> list[str]:
    feats = artifact.get("selected_features")
    if not feats:
        raise ValueError("The model artifact has no 'selected_features'.")
    return list(feats)


def _prepare_inference_frame(artifact: dict[str, Any], frame: pd.DataFrame) -> pd.DataFrame:
    """Slice to the artifact's features (exact order), validate, and encode."""
    features = _artifact_features(artifact)
    missing = [f for f in features if f not in frame.columns]
    if missing:
        raise ValueError(
            f"Missing input value(s) for active feature(s): {', '.join(missing)}. "
            f"The model was trained on: {', '.join(features)}."
        )

    sub = frame[features].copy()  # extra columns / keys are ignored here

    num_cols = [c for c in features if c in NUMERIC_PREDICTORS]
    for col in num_cols:
        sub[col] = pd.to_numeric(sub[col], errors="coerce")
    if num_cols and sub[num_cols].isna().any().any():
        raise ValueError("Numeric inputs must be valid numbers.")
    if sub[[c for c in features if c in CATEGORICAL_PREDICTORS]].isna().any().any():
        raise ValueError("Categorical inputs must not be empty.")

    return _encode_features(sub, features, artifact.get("label_encoders", {}))


def predict_single(artifact: dict[str, Any], input_dict: dict[str, Any]) -> float:
    """
    Predict delivery time (minutes) for one set of inputs.

    Only keys in ``artifact['selected_features']`` are read, in training order;
    extra keys (e.g. from previously selected features) are ignored. A missing
    active feature raises a descriptive ValueError. Unseen categories are
    encoded as -1.
    """
    features = _artifact_features(artifact)
    missing = [f for f in features if f not in input_dict]
    if missing:
        raise ValueError(
            f"Missing input value(s) for active feature(s): {', '.join(missing)}. "
            f"The model was trained on: {', '.join(features)}."
        )
    row = pd.DataFrame([{f: input_dict[f] for f in features}], columns=features)
    X = _prepare_inference_frame(artifact, row)
    return float(artifact["model"].predict(X)[0])


def predict_batch(artifact: dict[str, Any], df: pd.DataFrame) -> np.ndarray:
    """Vectorised version of :func:`predict_single` for a DataFrame of inputs."""
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Batch input must be a pandas DataFrame.")
    X = _prepare_inference_frame(artifact, df)
    return np.asarray(artifact["model"].predict(X), dtype=float)


# UI-facing name kept for backward compatibility with app.py
predict_delivery_time = predict_single
