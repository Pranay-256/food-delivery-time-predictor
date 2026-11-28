"""
eda_engine.py
=============
Reusable Data Quality / EDA report for the Food Delivery Time Predictor.

Give it a *cleaned* pandas DataFrame with the standard schema

    Distance_km, Weather, Traffic_Level, Time_of_Day, Vehicle_Type,
    Preparation_Time_min, Courier_Experience_yrs, Delivery_Time_min

and it renders the report: summary cards, simple Seaborn charts (the same
style as the Jupyter notebook) and plain-English insights that are written
from the data, never hard-coded.

Quick start (inside app.py)
---------------------------
    from eda_engine import render_data_quality_report

    render_data_quality_report(df)

Design notes
------------
* Every ``build_*`` function is pure (no Streamlit calls) and returns a
  Matplotlib figure (or ``None``) plus an insight sentence.
* Charts are saved as PNG at ``CHART_DPI`` (200) so they stay sharp, and the
  finished PNGs are cached per DataFrame, so moving a slider on the page does
  not redraw them.
* Edge cases (empty / tiny frames, constant columns, single-category columns,
  NaNs, missing columns) never raise: the affected section shows a friendly
  message instead.
"""

from __future__ import annotations

import html
import io
import math
from typing import Any, Callable, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")  # no GUI needed - charts are rendered to PNG
from matplotlib.figure import Figure  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
import streamlit as st  # noqa: E402

# ─────────────────────────────────────────────────────────────
# THEME / SCHEMA CONSTANTS
# (colours below are only used by the page cards - the charts themselves
#  use plain Seaborn defaults, exactly like the notebook)
# ─────────────────────────────────────────────────────────────
CARD = "#161B22"
CARD_ALT = "#1C2330"
BORDER = "#2A3444"
GOLD = "#D4AF37"
GOLD_LIGHT = "#E6C65C"
TEAL = "#2DD4BF"
TEXT = "#E6EDF3"
MUTED = "#8B949E"

CHART_DPI = 200            # high resolution so the charts look crisp on screen
CHART_SIZE = (10, 6)       # same figure size as the notebook
BAR_COLOR = sns.color_palette("deep")[0]       # simple blue
HIGHLIGHT_COLOR = sns.color_palette("deep")[1]  # simple orange for the "slowest" bar

TARGET = "Delivery_Time_min"
NUMERIC_DRIVERS = ["Distance_km", "Preparation_Time_min"]
NUMERIC_COLUMNS = ["Distance_km", "Preparation_Time_min", "Courier_Experience_yrs"]
CATEGORICAL_COLUMNS = ["Weather", "Traffic_Level", "Time_of_Day", "Vehicle_Type"]
REQUIRED_COLUMNS = CATEGORICAL_COLUMNS + NUMERIC_COLUMNS + [TARGET]

LABELS = {
    "Distance_km": "Distance",
    "Preparation_Time_min": "Preparation time",
    "Courier_Experience_yrs": "Courier experience",
    "Weather": "Weather",
    "Traffic_Level": "Traffic level",
    "Time_of_Day": "Time of day",
    "Vehicle_Type": "Vehicle type",
    "Delivery_Time_min": "Delivery time",
}
AXIS_TITLES = {
    "Distance_km": "Distance (km)",
    "Preparation_Time_min": "Preparation time (min)",
    "Courier_Experience_yrs": "Courier experience (years)",
    "Delivery_Time_min": "Delivery time (min)",
}
# "each additional ___" wording for the slope sentence
UNIT_PHRASE = {
    "Distance_km": "km of distance",
    "Preparation_Time_min": "minute of preparation",
    "Courier_Experience_yrs": "year of courier experience",
}
# Why a positive relationship happens, in business language
POSITIVE_EFFECT = {
    "Distance_km": "longer trips {adv} add travel time",
    "Preparation_Time_min": "slower kitchen preparation {adv} pushes deliveries later",
    "Courier_Experience_yrs": "more experienced couriers {adv} take longer",
}
# Natural order for ordinal categories; others are sorted fastest -> slowest.
CATEGORY_ORDER = {
    "Traffic_Level": ["Low", "Medium", "High"],
    "Time_of_Day": ["Morning", "Afternoon", "Evening", "Night"],
}
# Preferred "normal conditions" baseline for delay comparisons.
PREFERRED_BASELINE = {"Weather": "Clear", "Traffic_Level": "Low"}
CATEGORY_NOUN = {
    "Weather": "{} weather",
    "Traffic_Level": "{} traffic",
    "Time_of_Day": "{} deliveries",
    "Vehicle_Type": "{} deliveries",
}

# Pairs of categorical factors shown together as average-delivery-time heatmaps.
INTERACTION_PAIRS = [
    ("Traffic_Level", "Weather"),
    ("Weather", "Vehicle_Type"),
    ("Time_of_Day", "Traffic_Level"),
]
MIN_CELL_COUNT = 5  # a heatmap cell with fewer deliveries is flagged as unreliable

MAX_SCATTER_POINTS = 5000  # plotted points are sampled above this; stats use all rows


# ─────────────────────────────────────────────────────────────
# SMALL, SAFE HELPERS
# ─────────────────────────────────────────────────────────────
def _is_num(x: Any) -> bool:
    """True for real, finite numbers (handles None / NaN / inf)."""
    try:
        return x is not None and math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def _fmt(x: Any, decimals: int = 1, suffix: str = "") -> str:
    """Format a number, or return 'n/a' when it is missing."""
    return f"{float(x):,.{decimals}f}{suffix}" if _is_num(x) else "n/a"


def _fmt_mins(x: float) -> str:
    """Whole minutes for big gaps, one decimal for small ones."""
    return f"{x:.0f}" if abs(x) >= 10 else f"{x:.1f}"


def _prepare(df: Any) -> Tuple[pd.DataFrame, List[str]]:
    """
    Return (clean_copy, missing_required_columns).

    * never mutates the caller's DataFrame
    * coerces numeric columns (bad values -> NaN, +/-inf -> NaN)
    * strips whitespace from categorical text, keeping NaN as NaN
    """
    if not isinstance(df, pd.DataFrame):
        return pd.DataFrame(), list(REQUIRED_COLUMNS)

    clean = df.copy()
    for col in NUMERIC_COLUMNS + [TARGET]:
        if col in clean.columns and isinstance(clean[col], pd.Series):
            clean[col] = pd.to_numeric(clean[col], errors="coerce").replace(
                [np.inf, -np.inf], np.nan
            )
    for col in CATEGORICAL_COLUMNS:
        if col in clean.columns and isinstance(clean[col], pd.Series):
            s = clean[col]
            clean[col] = s.where(s.isna(), s.astype(str).str.strip())
    missing = [c for c in REQUIRED_COLUMNS if c not in clean.columns]
    return clean, missing


def _safe_corr(x: pd.Series, y: pd.Series) -> Optional[float]:
    """Pearson r, or None when it cannot be computed meaningfully."""
    pair = pd.concat([x, y], axis=1).dropna()
    if len(pair) < 3:
        return None
    a = pair.iloc[:, 0].to_numpy(dtype=float)
    b = pair.iloc[:, 1].to_numpy(dtype=float)
    if np.ptp(a) == 0 or np.ptp(b) == 0:  # a constant column has no correlation
        return None
    r = float(np.corrcoef(a, b)[0, 1])
    return r if math.isfinite(r) else None


def _safe_trend(x: pd.Series, y: pd.Series) -> Optional[Tuple[float, float]]:
    """Least-squares (slope, intercept), or None if x has no variation / too few rows."""
    pair = pd.concat([x, y], axis=1).dropna()
    if len(pair) < 3:
        return None
    a = pair.iloc[:, 0].to_numpy(dtype=float)
    b = pair.iloc[:, 1].to_numpy(dtype=float)
    var = float(((a - a.mean()) ** 2).sum())
    if var == 0:
        return None
    slope = float(((a - a.mean()) * (b - b.mean())).sum() / var)
    return slope, float(b.mean() - slope * a.mean())


def _strength(r: float) -> str:
    """Everyday word for the size of a correlation."""
    a = abs(r)
    if a >= 0.7:
        return "strong"
    if a >= 0.4:
        return "moderate"
    if a >= 0.2:
        return "weak"
    return "negligible"


# ─────────────────────────────────────────────────────────────
# 1. DATASET SUMMARY
# ─────────────────────────────────────────────────────────────
def summarize_dataset(df: Any) -> Dict[str, Any]:
    """Headline numbers plus a plain-English summary sentence."""
    clean, missing = _prepare(df)
    target = clean[TARGET].dropna() if TARGET in clean.columns else pd.Series(dtype=float)
    n = int(len(target))

    out: Dict[str, Any] = {
        "rows": int(len(clean)),
        "columns": int(clean.shape[1]),
        "n_target": n,
        "mean": float(target.mean()) if n else None,
        "median": float(target.median()) if n else None,
        "p25": float(target.quantile(0.25)) if n else None,
        "p75": float(target.quantile(0.75)) if n else None,
        "missing_cells": int(clean.isna().sum().sum()) if len(clean) else 0,
        "missing_columns": missing,
    }

    if n == 0:
        text = "No delivery-time values are available, so a summary can't be produced."
    elif n == 1:
        text = (
            f"Only one delivery is recorded and it took {_fmt(out['mean'])} minutes - "
            "too little data to describe a typical range."
        )
    else:
        text = (
            f"Across {out['rows']:,} deliveries, the average delivery takes "
            f"{_fmt(out['mean'])} minutes. Most deliveries (the middle 50%) land between "
            f"{_fmt(out['p25'], 0)} and {_fmt(out['p75'], 0)} minutes, and the median "
            f"delivery takes {_fmt(out['median'])} minutes."
        )
    if out["missing_cells"] > 0:
        text += f" Note: {out['missing_cells']:,} empty cells remain in the data."
    out["summary_text"] = text
    return out


# ─────────────────────────────────────────────────────────────
# 2. TARGET DISTRIBUTION
# ─────────────────────────────────────────────────────────────
def _histogram_insight(y: pd.Series) -> str:
    n = len(y)
    if n == 0:
        return "No delivery-time values are available to analyse."
    if n < 3:
        return f"Only {n} delivery record(s) - too few to judge how spread out delivery times are."

    mean, median = float(y.mean()), float(y.median())
    std = float(y.std(ddof=0))
    lo, hi = float(y.min()), float(y.max())
    q1, q3 = float(y.quantile(0.25)), float(y.quantile(0.75))

    if std == 0:
        return f"Every delivery took exactly {_fmt(mean, 0)} minutes - there is no variation at all."

    cv = std / mean if mean > 0 else 0.0  # coefficient of variation (guarded)
    if cv < 0.25:
        spread = "tightly clustered"
    elif cv < 0.5:
        spread = "moderately spread"
    else:
        spread = "widely spread"

    text = (
        f"Delivery times are {spread}: most deliveries land between {_fmt(q1, 0)} and "
        f"{_fmt(q3, 0)} minutes, but the full range runs from {_fmt(lo, 0)} to {_fmt(hi, 0)} minutes."
    )

    if n >= 4:
        fence = q3 + 1.5 * (q3 - q1)
        slow = y[y > fence]
        if len(slow) > 0:
            text += (
                f" {len(slow)} deliveries ({len(slow) / n * 100:.1f}%) are unusually slow - "
                f"over {_fmt(fence, 0)} minutes - and the slowest took {_fmt(float(slow.max()), 0)} minutes."
            )
        else:
            text += " No unusually slow deliveries stand out."
        if median > 0 and mean > median * 1.05:
            text += " A tail of slower orders pulls the average above the typical (median) time."
    return text


def _numeric_insight(col: str, r: Optional[float], trend: Optional[Tuple[float, float]],
                     n: int, is_top: bool) -> str:
    label = LABELS.get(col, col)
    if r is None or n < 3:
        return (
            f"There isn't enough varied data to measure how {label.lower()} "
            "relates to delivery time."
        )

    strength = _strength(r)
    if strength == "negligible":
        return (
            f"{label} shows almost no link to delivery time (correlation {r:+.2f}); "
            "other factors matter far more."
        )

    if is_top and strength == "weak":
        role = "the strongest numeric driver, though only a weak one"
    elif is_top:
        role = "the strongest driver"
    else:
        role = f"a {strength} driver"

    if r > 0:
        adv = "reliably" if strength == "strong" else "generally"
        effect = POSITIVE_EFFECT.get(col, "higher values {adv} add time").format(adv=adv)
        text = f"{label} is {role} - {effect} (correlation {r:+.2f})."
    else:
        text = f"{label} is {role} - higher values tend to shorten delivery time (correlation {r:+.2f})."

    if trend is not None:
        slope = trend[0]
        phrase = UNIT_PHRASE.get(col, "unit")
        verb = "adds" if slope > 0 else "removes"
        text += f" Each additional {phrase} {verb} about {abs(slope):.1f} min of delivery time."
    return text


def analyze_numeric_driver(df: Any, col: str) -> Dict[str, Any]:
    """Correlation, trend line and sample size for one numeric column vs the target."""
    clean, _ = _prepare(df)
    result: Dict[str, Any] = {"column": col, "data": None, "r": None, "trend": None, "n": 0}
    if col not in clean.columns or TARGET not in clean.columns:
        return result
    data = clean[[col, TARGET]].dropna()
    result.update(
        data=data, n=int(len(data)),
        r=_safe_corr(data[col], data[TARGET]),
        trend=_safe_trend(data[col], data[TARGET]),
    )
    return result


# ─────────────────────────────────────────────────────────────
# 4. CATEGORICAL IMPACT
# ─────────────────────────────────────────────────────────────
def _noun(col: str, category: Any) -> str:
    return CATEGORY_NOUN.get(col, "{} deliveries").format(category)


def analyze_categorical_impact(df: Any, col: str) -> Dict[str, Any]:
    """Per-category stats and the biggest-delay comparison for one categorical column."""
    clean, _ = _prepare(df)
    res: Dict[str, Any] = {
        "column": col, "usable": False, "data": None, "stats": None, "n_categories": 0,
        "comparable": False, "top": None, "baseline": None, "fastest": None,
        "delta": 0.0, "spread": 0.0, "meaningful": False,
        "baseline_is_preferred": False, "small_groups": False,
    }
    if col not in clean.columns or TARGET not in clean.columns:
        return res
    data = clean[[col, TARGET]].dropna()
    if data.empty:
        return res

    stats = data.groupby(col)[TARGET].agg(["count", "mean", "median"])
    res.update(usable=True, data=data, stats=stats, n_categories=int(len(stats)),
               small_groups=bool(stats["count"].min() < 5))
    if len(stats) < 2:
        return res

    means = stats["mean"]
    top, fastest = means.idxmax(), means.idxmin()
    preferred = PREFERRED_BASELINE.get(col)
    baseline = preferred if preferred in means.index else fastest
    if baseline == top:  # the "normal" baseline is itself the slowest -> compare to the fastest
        baseline = fastest
    delta = float(means[top] - means[baseline])
    overall = float(data[TARGET].mean())
    threshold = max(1.0, 0.05 * abs(overall))

    res.update(
        comparable=True, top=top, baseline=baseline, fastest=fastest, delta=delta,
        spread=float(means.max() - means.min()),
        meaningful=bool(delta >= threshold),
        baseline_is_preferred=bool(preferred is not None and baseline == preferred),
    )
    return res


def _categorical_insight(res: Dict[str, Any]) -> str:
    col = res["column"]
    label = LABELS.get(col, col)
    if not res["usable"]:
        return f"{label} data is not available, so its impact can't be assessed."
    if not res["comparable"]:
        only = res["stats"].index[0]
        return f"All records share a single {label.lower()} value ({only}), so no comparison is possible."

    if res["meaningful"]:
        text = (
            f"{_noun(col, res['top'])} adds an average of +{_fmt_mins(res['delta'])} mins "
            f"compared to {_noun(col, res['baseline'])}."
        )
        if res["baseline"] != res["fastest"]:
            fast_mean = res["stats"].loc[res["fastest"], "mean"]
            text += f" The fastest group is {res['fastest']} ({_fmt(fast_mean)} min on average)."
        elif not res["baseline_is_preferred"]:
            text += f" ({res['baseline']} is the fastest group.)"
    else:
        text = (
            f"{label} makes little practical difference: average delivery times differ by only "
            f"{res['spread']:.1f} min across its groups."
        )
    if res["small_groups"]:
        text += " Some groups have fewer than 5 deliveries, so treat this as indicative."
    return text


def _order_by_mean(col: str, means: pd.Series) -> List[Any]:
    """Natural order for ordinal columns, otherwise fastest -> slowest."""
    preferred = CATEGORY_ORDER.get(col)
    if preferred and set(means.index) <= set(preferred):
        return [c for c in preferred if c in means.index]
    return means.sort_values().index.tolist()


def _gap_threshold(overall_mean: Any) -> float:
    """Smallest average-time gap (minutes) worth calling meaningful."""
    return max(1.0, 0.05 * abs(float(overall_mean))) if _is_num(overall_mean) else 1.0


def _mean_bar_insight(res: Dict[str, Any]) -> str:
    col = res["column"]
    label = LABELS.get(col, col)
    if not res["usable"]:
        return f"{label} data is not available, so its averages can't be shown."
    if not res["comparable"]:
        return (
            f"All records share a single {label.lower()} value ({res['stats'].index[0]}), "
            "so there is nothing to compare."
        )

    means = res["stats"]["mean"]
    top, low = means.idxmax(), means.idxmin()
    gap = float(means[top] - means[low])
    overall = float(res["data"][TARGET].mean())

    if gap < _gap_threshold(overall):
        text = (
            f"Average delivery time is almost the same across {label.lower()} groups - "
            f"only {gap:.1f} min separates the slowest from the fastest."
        )
    else:
        text = (
            f"{top} is the slowest {label.lower()} group at {_fmt(means[top])} min on average, "
            f"{_fmt_mins(gap)} min longer than {low}, the fastest at {_fmt(means[low])} min."
        )
        if float(means[low]) > 0:
            text += f" That is about {gap / float(means[low]) * 100:.0f}% longer."
    if res["small_groups"]:
        text += " Some groups have fewer than 5 deliveries, so treat this as indicative."
    return text


def _combo_phrase(r: str, c: str, rv: Any, cv: Any) -> str:
    return f"{LABELS[r]} = {rv} and {LABELS[c]} = {cv}"


def _interaction_insight(cells: pd.DataFrame, r: str, c: str, n_empty: int) -> str:
    reliable = cells[cells["count"] >= MIN_CELL_COUNT]
    pool = reliable if len(reliable) >= 2 else cells
    if len(pool) < 2:
        return "Not enough combinations have data to compare them."

    slow = pool.loc[pool["mean"].idxmax()]
    fast = pool.loc[pool["mean"].idxmin()]
    gap = float(slow["mean"] - fast["mean"])
    text = (
        f"The slowest combination is {_combo_phrase(r, c, slow[r], slow[c])} "
        f"(average {_fmt(slow['mean'])} min, {int(slow['count'])} deliveries); the fastest is "
        f"{_combo_phrase(r, c, fast[r], fast[c])} (average {_fmt(fast['mean'])} min, "
        f"{int(fast['count'])} deliveries) - a gap of {_fmt_mins(gap)} min."
    )
    n_sparse = int((cells["count"] < MIN_CELL_COUNT).sum())
    if pool is cells and n_sparse:
        text += f" Most combinations have fewer than {MIN_CELL_COUNT} deliveries, so treat this as indicative."
    elif n_sparse:
        text += f" {n_sparse} combination(s) with fewer than {MIN_CELL_COUNT} deliveries were left out of this comparison."
    if n_empty:
        text += f" {n_empty} combination(s) never occur in the data (blank cells)."
    return text


# ─────────────────────────────────────────────────────────────
# CHART HELPERS
# ─────────────────────────────────────────────────────────────
def _new_axes(size: Tuple[float, float] = CHART_SIZE) -> Tuple[Figure, Any]:
    """A fresh Matplotlib figure + axes (no pyplot, so it is safe in Streamlit's threads)."""
    fig = Figure(figsize=size)
    return fig, fig.subplots()


def _to_png(fig: Optional[Figure]) -> Optional[bytes]:
    """Save a figure as a high-DPI PNG and return the raw bytes."""
    if fig is None:
        return None
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=CHART_DPI, bbox_inches="tight")
    return buf.getvalue()


# ─────────────────────────────────────────────────────────────
# 2. TARGET DISTRIBUTION CHART
# ─────────────────────────────────────────────────────────────
def build_target_histogram(df: Any) -> Tuple[Optional[Figure], str]:
    """Histogram + KDE curve of Delivery_Time_min (same as the notebook) + dynamic insight."""
    clean, _ = _prepare(df)
    if TARGET not in clean.columns:
        return None, "The delivery-time column is missing, so its distribution can't be shown."
    y = clean[TARGET].dropna()
    insight = _histogram_insight(y)
    if y.empty:
        return None, insight

    fig, ax = _new_axes()
    sns.histplot(y, kde=y.nunique() > 1, bins=30 if y.nunique() >= 30 else max(1, y.nunique()), ax=ax)
    ax.set_title("Distribution of Delivery Time (min)")
    ax.set_xlabel("Delivery Time (min)")
    ax.set_ylabel("Frequency")
    return fig, insight


# ─────────────────────────────────────────────────────────────
# 3. NUMERICAL DRIVERS (scatter plots)
# ─────────────────────────────────────────────────────────────
def build_numeric_driver_plots(df: Any) -> List[Tuple[str, Optional[Figure], str]]:
    """Scatter plot + trend line + insight for Distance and Preparation time."""
    analyses = {c: analyze_numeric_driver(df, c) for c in NUMERIC_DRIVERS}
    usable = {c: a for c, a in analyses.items() if a["r"] is not None}
    top = max(usable, key=lambda c: abs(usable[c]["r"])) if usable else None

    results: List[Tuple[str, Optional[Figure], str]] = []
    for col in NUMERIC_DRIVERS:
        a = analyses[col]
        if a["data"] is None:
            results.append((col, None, f"{LABELS[col]} data is not available in this dataset."))
            continue
        insight = _numeric_insight(col, a["r"], a["trend"], a["n"], col == top)
        data = a["data"]
        if data.empty:
            results.append((col, None, insight))
            continue

        plot_data = data.sample(MAX_SCATTER_POINTS, random_state=42) if len(data) > MAX_SCATTER_POINTS else data
        fig, ax = _new_axes()
        if a["trend"] is not None:  # scatter + straight trend line
            sns.regplot(x=col, y=TARGET, data=plot_data, ci=None,
                        scatter_kws={"alpha": 0.5}, line_kws={"color": "red"}, ax=ax)
        else:                       # no variation in x -> a trend line is impossible
            sns.scatterplot(x=col, y=TARGET, data=plot_data, alpha=0.5, ax=ax)
        ax.set_title(f"{LABELS[col]} vs. Delivery Time")
        ax.set_xlabel(AXIS_TITLES[col])
        ax.set_ylabel(AXIS_TITLES[TARGET])
        results.append((col, fig, insight))
    return results


# ─────────────────────────────────────────────────────────────
# 4. CATEGORICAL BOX PLOTS
# ─────────────────────────────────────────────────────────────
def build_categorical_box_plots(df: Any) -> List[Tuple[str, Optional[Figure], str]]:
    """Box plot (viridis palette, like the notebook) and insight for each categorical column."""
    results: List[Tuple[str, Optional[Figure], str]] = []
    for col in CATEGORICAL_COLUMNS:
        res = analyze_categorical_impact(df, col)
        insight = _categorical_insight(res)
        if not res["usable"]:
            results.append((col, None, insight))
            continue

        order = [str(c) for c in _order_by_mean(col, res["stats"]["mean"])]
        data = res["data"].copy()
        data[col] = data[col].astype(str)

        fig, ax = _new_axes()
        sns.boxplot(x=col, y=TARGET, data=data, hue=col, order=order, hue_order=order,
                    palette="viridis", legend=False, ax=ax)
        ax.set_title(f"Delivery Time vs. {col}")
        ax.set_xlabel(col)
        ax.set_ylabel(AXIS_TITLES[TARGET])
        ax.tick_params(axis="x", rotation=45)
        results.append((col, fig, insight))
    return results


# ─────────────────────────────────────────────────────────────
# 4b. CATEGORY COMPARISONS (factor ranking, average bars, two-factor heatmaps)
# ─────────────────────────────────────────────────────────────
def build_categorical_mean_bars(df: Any) -> List[Tuple[str, Optional[Figure], str]]:
    """Average delivery time per category (bars + overall-average line) for each categorical column."""
    results: List[Tuple[str, Optional[Figure], str]] = []
    for col in CATEGORICAL_COLUMNS:
        res = analyze_categorical_impact(df, col)
        insight = _mean_bar_insight(res)
        if not res["usable"]:
            results.append((col, None, insight))
            continue

        stats = res["stats"]
        order = _order_by_mean(col, stats["mean"])
        means = stats.loc[order, "mean"]
        slowest = means.idxmax() if res["comparable"] else None
        overall = float(res["data"][TARGET].mean())

        plot_df = pd.DataFrame({col: [str(c) for c in order], "mean": means.to_numpy(dtype=float)})
        palette = {str(c): (HIGHLIGHT_COLOR if c == slowest else BAR_COLOR) for c in order}

        fig, ax = _new_axes()
        sns.barplot(x=col, y="mean", data=plot_df, hue=col, palette=palette, legend=False,
                    errorbar=None, ax=ax)
        ax.axhline(overall, color="gray", linestyle="--", linewidth=1.3, label=f"Overall avg {overall:.1f} min")
        for container in ax.containers:
            ax.bar_label(container, fmt="%.1f", padding=3)
        top_val = float(means.max())
        ax.set_ylim(0, max(top_val, overall) * 1.2 if top_val > 0 else 1)
        ax.set_title(f"Average Delivery Time by {LABELS[col]}")
        ax.set_xlabel(col)
        ax.set_ylabel("Average delivery time (min)")
        ax.legend(frameon=False, loc="upper left")
        results.append((col, fig, insight))
    return results


def build_categorical_ranking(df: Any) -> Tuple[Optional[Figure], str]:
    """Rank categorical factors by how far average delivery time swings between their groups."""
    rows: List[Dict[str, Any]] = []
    overall: Optional[float] = None
    for col in CATEGORICAL_COLUMNS:
        res = analyze_categorical_impact(df, col)
        if res["comparable"]:
            means = res["stats"]["mean"]
            rows.append({"col": col, "spread": res["spread"],
                         "slow": means.idxmax(), "fast": means.idxmin()})
            overall = float(res["data"][TARGET].mean())
    if not rows:
        return None, ("No categorical column has at least two groups with delivery-time data, "
                      "so the factors can't be compared.")

    rows.sort(key=lambda r: r["spread"], reverse=True)
    top, last = rows[0], rows[-1]

    plot_df = pd.DataFrame({"factor": [LABELS[r["col"]] for r in rows],
                            "gap": [r["spread"] for r in rows]})
    palette = {f: (HIGHLIGHT_COLOR if i == 0 else BAR_COLOR) for i, f in enumerate(plot_df["factor"])}

    fig, ax = _new_axes((10, 4.5))
    sns.barplot(x="gap", y="factor", data=plot_df, hue="factor", palette=palette, legend=False,
                orient="h", ax=ax)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.1f min", padding=4)
    ax.set_xlim(0, max(float(plot_df["gap"].max()) * 1.18, 1))
    ax.set_title("Impact of Each Categorical Factor")
    ax.set_xlabel("Gap between slowest and fastest group (min)")
    ax.set_ylabel("")

    if top["spread"] < _gap_threshold(overall):
        text = (f"None of the categorical factors changes average delivery time by much "
                f"(the largest gap is {top['spread']:.1f} min).")
    else:
        text = (f"{LABELS[top['col']]} makes the biggest difference: average delivery time "
                f"differs by {_fmt_mins(top['spread'])} min between {top['slow']} (slowest) and "
                f"{top['fast']} (fastest).")
        if len(rows) > 1:
            if last["spread"] < _gap_threshold(overall):
                text += f" {LABELS[last['col']]} matters least, with almost no effect ({last['spread']:.1f} min)."
            else:
                text += f" {LABELS[last['col']]} matters least, with a gap of {_fmt_mins(last['spread'])} min."
    return fig, text


def build_interaction_heatmaps(df: Any) -> List[Tuple[str, Optional[Figure], str]]:
    """Average delivery time for every pair of categories from two factors (e.g. Traffic x Weather)."""
    clean, _ = _prepare(df)
    results: List[Tuple[str, Optional[Figure], str]] = []
    for r, c in INTERACTION_PAIRS:
        key = f"{r}_x_{c}"
        if any(x not in clean.columns for x in (r, c, TARGET)):
            results.append((key, None, f"{LABELS[r]} / {LABELS[c]} data is not available."))
            continue
        data = clean[[r, c, TARGET]].dropna()
        if data.empty or data[r].nunique() < 2 or data[c].nunique() < 2:
            results.append((key, None,
                f"{LABELS[r]} and {LABELS[c]} each need at least two groups to compare them together."))
            continue

        cells = data.groupby([r, c])[TARGET].agg(["mean", "count"]).reset_index()
        row_order = _order_by_mean(r, data.groupby(r)[TARGET].mean())
        col_order = _order_by_mean(c, data.groupby(c)[TARGET].mean())
        mean_p = cells.pivot(index=r, columns=c, values="mean").reindex(index=row_order, columns=col_order)
        cnt_p = (cells.pivot(index=r, columns=c, values="count")
                 .reindex(index=row_order, columns=col_order).fillna(0))
        n_empty = int((cnt_p == 0).to_numpy().sum())

        # each cell shows the average and how many deliveries it is based on
        labels = np.array([[("" if cnt_p.iloc[i, j] == 0 else f"{mean_p.iloc[i, j]:.1f}\nn={int(cnt_p.iloc[i, j])}")
                            for j in range(len(col_order))] for i in range(len(row_order))])

        fig, ax = _new_axes((10, 6))
        sns.heatmap(mean_p.astype(float), annot=labels, fmt="", cmap="YlOrRd", linewidths=1,
                    linecolor="white", cbar_kws={"label": "Average delivery time (min)"}, ax=ax)
        ax.set_title(f"Average Delivery Time: {r} x {c}")
        ax.set_xlabel(c)
        ax.set_ylabel(r)
        ax.tick_params(axis="y", rotation=0)
        results.append((key, fig, _interaction_insight(cells, r, c, n_empty)))
    return results


# ─────────────────────────────────────────────────────────────
# 5. CORRELATION HEATMAP
# ─────────────────────────────────────────────────────────────
def build_correlation_heatmap(df: Any) -> Tuple[Optional[Figure], str]:
    """Annotated correlation heatmap of the numeric columns (coolwarm, like the notebook) + takeaway."""
    clean, _ = _prepare(df)
    candidates = [c for c in NUMERIC_COLUMNS + [TARGET] if c in clean.columns]
    data = clean[candidates].dropna() if candidates else pd.DataFrame()

    if len(data) < 3:
        return None, "Not enough complete rows (at least 3 needed) to compute correlations."

    usable = [c for c in candidates if data[c].nunique() > 1]
    excluded = [LABELS[c] for c in candidates if c not in usable]
    if len(usable) < 2:
        return None, "Too few numeric columns with varying values to compute correlations."

    corr = data[usable].corr().fillna(0.0)

    fig, ax = _new_axes((10, 8))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
    ax.set_title("Correlation Matrix of Numerical Features")

    # ---- plain-English takeaway ----
    feats = [c for c in usable if c != TARGET]
    parts: List[str] = []
    if TARGET in usable and feats:
        ranked = sorted(((c, float(corr.loc[c, TARGET])) for c in feats),
                        key=lambda kv: abs(kv[1]), reverse=True)
        top_c, top_r = ranked[0]
        if abs(top_r) >= 0.2:
            parts.append(
                f"{LABELS[top_c]} has the strongest link with delivery time "
                f"(correlation {top_r:+.2f}, {_strength(top_r)})."
            )
        else:
            parts.append("None of the numeric factors has a meaningful straight-line link with delivery time on its own.")
        weak = [LABELS[c] for c, r in ranked if abs(r) < 0.2 and not (c == top_c and abs(top_r) < 0.2)]
        if weak:
            parts.append(f"{', '.join(weak)} show{'s' if len(weak) == 1 else ''} almost no direct link to delivery time.")
    elif TARGET not in usable:
        parts.append("Delivery time does not vary in this data, so its drivers can't be ranked.")

    pairs = [(a, b, float(corr.loc[a, b])) for i, a in enumerate(feats) for b in feats[i + 1:]]
    if pairs:
        a, b, v = max(pairs, key=lambda t: abs(t[2]))
        if abs(v) >= 0.7:
            parts.append(f"{LABELS[a]} and {LABELS[b]} move closely together (correlation {v:+.2f}), so they carry overlapping information.")
        else:
            parts.append("The input factors are largely independent of each other, so each contributes its own information.")
    if excluded:
        parts.append(f"Left out (no variation): {', '.join(excluded)}.")
    parts.append("Categorical factors are covered in the charts above.")
    return fig, " ".join(parts)


# ─────────────────────────────────────────────────────────────
# BUILD + CACHE ALL CHARTS (one cache entry per DataFrame)
# ─────────────────────────────────────────────────────────────
_NO_CHART = (None, "This chart could not be drawn for this dataset.")


def _build_all_charts(df: pd.DataFrame) -> Dict[str, Tuple[Optional[bytes], str]]:
    """Draw every chart once and return {key: (png_bytes_or_None, insight_text)}."""
    assets: Dict[str, Tuple[Optional[bytes], str]] = {}

    def single(key: str, builder: Callable[[Any], Tuple[Optional[Figure], str]]) -> None:
        try:
            fig, insight = builder(df)
            assets[key] = (_to_png(fig), insight)
        except Exception as exc:  # noqa: BLE001 - one bad chart must not break the report
            assets[key] = (None, f"This chart could not be drawn ({type(exc).__name__}: {exc}).")

    def many(prefix: str, builder: Callable[[Any], List[Tuple[str, Optional[Figure], str]]]) -> None:
        try:
            for name, fig, insight in builder(df):
                assets[f"{prefix}:{name}"] = (_to_png(fig), insight)
        except Exception as exc:  # noqa: BLE001
            assets[f"{prefix}:error"] = (None, f"These charts could not be drawn ({type(exc).__name__}: {exc}).")

    single("hist", build_target_histogram)
    many("num", build_numeric_driver_plots)
    single("rank", build_categorical_ranking)
    many("mean", build_categorical_mean_bars)
    many("box", build_categorical_box_plots)
    many("inter", build_interaction_heatmaps)
    single("corr", build_correlation_heatmap)
    return assets


@st.cache_data(show_spinner=False, max_entries=4)
def _cached_charts(df: pd.DataFrame) -> Dict[str, Tuple[Optional[bytes], str]]:
    """Cached wrapper: the charts are redrawn only when the data changes."""
    return _build_all_charts(df)


# ─────────────────────────────────────────────────────────────
# STREAMLIT UI HELPERS
# ─────────────────────────────────────────────────────────────
def _section_heading(icon: str, title: str) -> None:
    st.markdown(
        f"""
        <div style="background: linear-gradient(90deg, rgba(212,175,55,0.12) 0%, transparent 100%);
                    border-left: 3px solid {GOLD}; border-radius: 0 8px 8px 0;
                    padding: 10px 18px; margin: 18px 0 10px 0;">
            <span style="font-family:'Rajdhani',sans-serif; font-weight:700; font-size:20px;
                         color:{GOLD_LIGHT}; letter-spacing:0.4px;">
                {html.escape(title)}
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _kpi_card(icon: str, label: str, value: str, sub: str = "") -> None:
    icon_html = f'<div style="font-size:26px; margin-bottom:6px;">{icon}</div>' if icon else ""
    sub_html = (
        f"<p style='font-size:12px; color:{MUTED}; margin:2px 0 0 0; "
        f"font-family:Rajdhani,sans-serif;'>{html.escape(sub)}</p>" if sub else ""
    )
    card_html = f"""
        <div style="background: linear-gradient(145deg, {CARD_ALT}, {CARD}); border: 1px solid {BORDER};
                    border-top: 2px solid {GOLD}; border-radius: 10px; padding: 16px 18px;
                    text-align: center; height: 100%; box-shadow: 0 4px 16px rgba(0,0,0,0.3);">
            {icon_html}
            <p style="font-family:'Rajdhani',sans-serif; font-size:12px; color:{MUTED};
                      margin:0 0 4px 0; text-transform:uppercase; letter-spacing:0.8px;">{html.escape(label)}</p>
            <p style="font-family:'Rajdhani',sans-serif; font-weight:700; font-size:22px;
                      color:{GOLD_LIGHT}; margin:0; line-height:1.1;">{html.escape(value)}</p>
            {sub_html}
        </div>
        """
    # drop empty / indented lines: a blank line would end the HTML block and show the rest as code
    st.markdown("\n".join(l.strip() for l in card_html.splitlines() if l.strip()), unsafe_allow_html=True)


def _insight_box(text: str) -> None:
    st.markdown(
        f"""
        <div style="background:{CARD}; border:1px solid {BORDER}; border-left:3px solid {TEAL};
                    border-radius:0 8px 8px 0; padding:10px 16px; margin:4px 0 14px 0;">
            <span style="font-family:'Rajdhani',sans-serif; font-weight:700; font-size:14px;
                         color:{TEAL}; margin-right:8px;">&#128161; INSIGHT</span>
            <span style="font-family:'Rajdhani',sans-serif; font-size:15px; color:{TEXT};
                         line-height:1.6;">{html.escape(text)}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _chart_with_insight(chart: Tuple[Optional[bytes], str]) -> None:
    """Show one PNG chart (if there is one) followed by its insight box."""
    png, insight = chart
    if png is not None:
        st.image(png)
    _insight_box(insight)


def _safe_section(name: str, func: Callable[[], None]) -> None:
    """Run one report section; a failure shows a warning instead of crashing the page."""
    try:
        func()
    except Exception as exc:  # noqa: BLE001 - deliberate: keep the rest of the report alive
        st.warning(f"The '{name}' section could not be rendered ({type(exc).__name__}: {exc}).")


def _grid(charts: Dict[str, Tuple[Optional[bytes], str]], prefix: str, names: List[str], columns: int) -> None:
    """Lay the charts named ``prefix:name`` out in rows of ``columns``."""
    items = [charts.get(f"{prefix}:{n}", _NO_CHART) for n in names]
    columns = max(1, int(columns))
    for start in range(0, len(items), columns):
        row = items[start:start + columns]
        cols = st.columns(len(row)) if columns > 1 else [st.container()]
        for holder, chart in zip(cols, row):
            with holder:
                _chart_with_insight(chart)


# ─────────────────────────────────────────────────────────────
# PUBLIC ENTRY POINT
# ─────────────────────────────────────────────────────────────
def render_data_quality_report(df: Any, *, grid_columns: int = 1) -> None:
    """
    Render the full Data Quality Report for a cleaned delivery DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        Cleaned data with the standard schema (extra columns are ignored).
    grid_columns : int
        1 = charts stacked full width (default); 2 = two charts per row.
    """
    if not isinstance(df, pd.DataFrame) or df.empty:
        st.warning("No data to analyse - please provide a non-empty DataFrame.")
        return

    _, missing = _prepare(df)
    if missing:
        st.warning(
            "These expected columns are missing, so related sections are limited: "
            + ", ".join(missing)
        )
    if len(df) < 30:
        st.info(f"Small dataset ({len(df)} rows): treat the patterns below as indicative only.")

    # Draw every chart once (cached - instant on later reruns)
    with st.spinner("Drawing charts..."):
        charts = _cached_charts(df)

    # 1 ── Dataset summary ─────────────────────────────────
    def _summary() -> None:
        _section_heading("", "Dataset Summary")
        s = summarize_dataset(df)
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            _kpi_card("", "Total Rows", f"{s['rows']:,}", "Deliveries recorded")
        with c2:
            _kpi_card("", "Columns", f"{s['columns']:,}", "Fields per delivery")
        with c3:
            _kpi_card("", "Avg Delivery Time", _fmt(s["mean"], 1, " min"), "Mean of all deliveries")
        with c4:
            rng = f"{_fmt(s['p25'], 0)}–{_fmt(s['p75'], 0)} min" if _is_num(s["p25"]) else "n/a"
            _kpi_card("", "Typical Range", rng, "25th to 75th percentile")
        st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)
        _insight_box(s["summary_text"])

    # 2 ── Target distribution ─────────────────────────────
    def _target() -> None:
        _section_heading("", "Delivery Time Distribution")
        _chart_with_insight(charts.get("hist", _NO_CHART))

    # 3 ── Numerical drivers ───────────────────────────────
    def _numeric() -> None:
        _section_heading("", "Numerical Drivers of Delivery Time")
        st.caption("Each dot = one delivery · red line = overall trend")
        _grid(charts, "num", NUMERIC_DRIVERS, grid_columns)

    # 4 ── Category comparisons ────────────────────────────
    def _cat_ranking() -> None:
        _section_heading("", "Which Categorical Factor Matters Most")
        _chart_with_insight(charts.get("rank", _NO_CHART))

    def _cat_means() -> None:
        _section_heading("", "Average Delivery Time by Category")
        st.caption("Orange bar = slowest group · dashed line = overall average")
        _grid(charts, "mean", CATEGORICAL_COLUMNS, grid_columns)

    def _categorical() -> None:
        _section_heading("", "Categorical Impact on Delivery Time")
        st.caption("Box = middle 50% of deliveries · line inside the box = median · dots = unusual deliveries")
        _grid(charts, "box", CATEGORICAL_COLUMNS, grid_columns)

    def _interactions() -> None:
        _section_heading("", "Combined Effects of Two Factors")
        st.caption("Each cell = average delivery time (min) for that combination · n = number of deliveries · blank = combination never occurs")
        _grid(charts, "inter", [f"{r}_x_{c}" for r, c in INTERACTION_PAIRS], grid_columns)

    # 5 ── Correlation heatmap ─────────────────────────────
    def _heatmap() -> None:
        _section_heading("", "Correlation Heatmap")
        _chart_with_insight(charts.get("corr", _NO_CHART))

    _safe_section("Dataset Summary", _summary)
    _safe_section("Delivery Time Distribution", _target)
    _safe_section("Numerical Drivers", _numeric)
    _safe_section("Categorical Factor Ranking", _cat_ranking)
    _safe_section("Average Delivery Time by Category", _cat_means)
    _safe_section("Categorical Impact", _categorical)
    _safe_section("Combined Effects", _interactions)
    _safe_section("Correlation Heatmap", _heatmap)