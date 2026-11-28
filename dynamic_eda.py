"""
dynamic_eda.py
==============
Charts + plain-English insights for ANY dataset that follows the template layout
(used by the "Try With Your Own Dataset" tab).

It reproduces the visuals of Food_Delivery_Time_Prediction_model.ipynb:
  Step 9   target distribution              Step 13  which categorical factor matters most
  Step 10  numeric scatter plots            Step 14  average delivery time by category
  Step 11  correlation matrix               Step 15  two-factor heatmaps
  Step 12  categorical box plots            Step 16/17 Pearson + ANOVA tests
and writes the "What we observed" text for each one from the user's own numbers
(nothing is hard-coded to the original 1,000-row dataset).

Design for speed / low RAM:
  * build_report()  is pure Python (no Streamlit) and returns PNG bytes + text, so the whole
    report can be cached once per uploaded file and re-shown instantly when sliders move.
  * every matplotlib figure is closed right after it is saved.
  * scatter plots are sampled to at most 3,000 points.
"""
from __future__ import annotations

import io

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

from ml_pipeline import (  # noqa: E402
    CATEGORICAL_FEATURES, FEATURE_META, NUMERIC_FEATURES, ORDER_HINT, TARGET,
    feature_selection_table, get_best_features,
)

# ─────────────────────────────────────────────────────────────
# Look & feel (matches the app's dark gold / teal theme)
# ─────────────────────────────────────────────────────────────
BG, GRID, TEXT, MUTED = "#161b22", "#2a3444", "#e6edf3", "#8b949e"
GOLD, TEAL, ORANGE, BLUE = "#c9a84c", "#2dd4bf", "#DD8452", "#4C72B0"
RC = {
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "axes.edgecolor": GRID, "axes.labelcolor": TEXT, "text.color": TEXT,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.titlecolor": TEXT,
    "grid.color": GRID, "axes.titleweight": "bold", "axes.titlesize": 13,
    "axes.spines.top": False, "axes.spines.right": False,
}
SMALL_CELL = 30           # heat-map cells with fewer deliveries are flagged as "hints"
MIN_CELL_FOR_RANKING = 10  # cells with fewer deliveries are ignored when naming fastest / slowest


def _png(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=105, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def _pretty(col: str) -> str:
    return FEATURE_META[col][1] if col in FEATURE_META else col.replace("_", " ")


def _unit(col: str) -> str:
    return FEATURE_META[col][2] if col in FEATURE_META else ""


def _order(df: pd.DataFrame, col: str) -> list:
    """Natural order for ordinal columns, otherwise fastest -> slowest."""
    means = df.groupby(col)[TARGET].mean().sort_values()
    hint = ORDER_HINT.get(col, [])
    first = [c for c in hint if c in means.index]
    return first + [c for c in means.index if c not in first]


def _strength(r: float) -> str:
    a = abs(r)
    if a >= 0.7:
        return "strong"
    if a >= 0.4:
        return "moderate"
    if a >= 0.2:
        return "weak"
    return "essentially no"


def _b(text) -> str:
    return f"<strong>{text}</strong>"


# ─────────────────────────────────────────────────────────────
# INDIVIDUAL CHARTS  (each returns PNG bytes)
# ─────────────────────────────────────────────────────────────
def chart_distribution(df):
    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=(9, 4.4))
        sns.histplot(df[TARGET], kde=True, bins=30, color=GOLD, edgecolor=BG, ax=ax)
        ax.lines[0].set_color(TEAL) if ax.lines else None
        ax.set(title="Distribution of Delivery Time (min)", xlabel="Delivery Time (min)", ylabel="Frequency")
        return _png(fig)


def chart_scatter(df, col):
    plot_df = df if len(df) <= 3000 else df.sample(3000, random_state=42)
    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=(9, 4.4))
        sns.regplot(x=col, y=TARGET, data=plot_df, ci=None, ax=ax,
                    scatter_kws={"alpha": 0.45, "s": 18, "color": BLUE}, line_kws={"color": "#e05252"})
        unit = _unit(col)
        ax.set(title=f"{_pretty(col)} vs. Delivery Time",
               xlabel=f"{_pretty(col)} ({unit})" if unit else _pretty(col), ylabel="Delivery Time (min)")
        return _png(fig)


def chart_corr(df):
    num = NUMERIC_FEATURES + [TARGET]
    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=(8, 5.6))
        sns.heatmap(df[num].corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax, vmin=-1, vmax=1,
                    linewidths=1, linecolor=BG, cbar_kws={"shrink": 0.8})
        ax.set_title("Correlation Matrix of Numerical Features")
        ax.tick_params(axis="x", rotation=30)
        return _png(fig)


def chart_box(df, col):
    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=(9, 4.4))
        order = _order(df, col)
        sns.boxplot(x=col, y=TARGET, data=df, order=order, hue=col, palette="viridis", legend=False, ax=ax,
                    flierprops={"markerfacecolor": MUTED, "markeredgecolor": MUTED, "markersize": 3},
                    medianprops={"color": "white", "linewidth": 1.8},
                    whiskerprops={"color": MUTED}, capprops={"color": MUTED}, boxprops={"edgecolor": MUTED})
        ax.set(title=f"Delivery Time vs. {_pretty(col)}", xlabel=_pretty(col), ylabel="Delivery Time (min)")
        ax.tick_params(axis="x", rotation=30)
        return _png(fig)


def chart_gaps(gaps: pd.Series):
    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=(9, 3.8))
        labels = [_pretty(c) for c in gaps.index]
        colors = [ORANGE] + [BLUE] * (len(gaps) - 1)
        bars = ax.barh(labels[::-1], gaps.values[::-1], color=colors[::-1])
        for bar, v in zip(bars, gaps.values[::-1]):
            ax.text(v + gaps.max() * 0.015, bar.get_y() + bar.get_height() / 2, f"{v:.1f} min",
                    va="center", fontsize=10, fontweight="bold", color=TEXT)
        ax.set_xlim(0, max(gaps.max(), 1e-9) * 1.2)
        ax.set(title="Impact of Each Categorical Factor", xlabel="Gap between slowest and fastest group (min)")
        return _png(fig)


def chart_category_means(df, col, overall):
    means = df.groupby(col)[TARGET].mean()
    means = means[_order(df, col)]
    colors = [ORANGE if m == means.max() else BLUE for m in means]
    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=(9, 4.4))
        bars = ax.bar(means.index.astype(str), means.values, color=colors)
        for bar, v in zip(bars, means.values):
            ax.text(bar.get_x() + bar.get_width() / 2, v + overall * 0.015, f"{v:.1f}",
                    ha="center", fontsize=10, fontweight="bold", color=TEXT)
        ax.axhline(overall, color=MUTED, linestyle="--", label=f"Overall avg {overall:.1f} min")
        ax.set_ylim(0, max(means.max(), overall) * 1.2)
        ax.set(title=f"Average Delivery Time by {_pretty(col)}", xlabel=_pretty(col),
               ylabel="Average delivery time (min)")
        ax.legend(frameon=False, loc="upper left", labelcolor=TEXT)
        ax.tick_params(axis="x", rotation=30)
        return _png(fig)


def chart_pair_heatmap(avg, cnt, row_col, col_col):
    labels = avg.round(1).astype(str) + "\nn=" + cnt.astype(str)
    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=(9, 4.8))
        sns.heatmap(avg, annot=labels, fmt="", cmap="YlOrRd", linewidths=1, linecolor=BG, ax=ax,
                    annot_kws={"fontsize": 9}, cbar_kws={"label": "Average delivery time (min)"})
        ax.set(title=f"Average Delivery Time: {_pretty(row_col)} × {_pretty(col_col)}",
               xlabel=_pretty(col_col), ylabel=_pretty(row_col))
        ax.tick_params(axis="y", rotation=0)
        return _png(fig)


def chart_model_comparison(metrics: dict[str, dict]) -> bytes:
    """metrics = {'Baseline Decision Tree': {...}, 'Random Forest': {...}}  (notebook Step 25)"""
    perf = pd.DataFrame(metrics).T[["MAE", "MSE", "RMSE", "R2"]]
    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=(9, 4.6))
        perf.plot(kind="bar", ax=ax, width=0.78, color=["#2F6690", "#3A7D44", "#D17A22", "#8E4585"])
        for cont in ax.containers:
            ax.bar_label(cont, fmt="%.2f", padding=3, color=TEXT, fontsize=9, fontweight="bold")
        ax.set_ylim(0, perf.to_numpy().max() * 1.18)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
        ax.set(title="Model Performance Comparison", xlabel="", ylabel="Score")
        ax.legend(title="Metric", frameon=False, labelcolor=TEXT)
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        ax.set_axisbelow(True)
        return _png(fig)


def chart_importance(importances: pd.Series) -> bytes:
    """Random-Forest feature importance (notebook Step 26): top 3 highlighted."""
    imp = importances.sort_values(ascending=True)
    cutoff = imp.sort_values(ascending=False).iloc[min(2, len(imp) - 1)]
    colors = [TEAL if v >= cutoff else GOLD for v in imp.values]
    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=(9, max(2.6, 0.6 * len(imp) + 1)))
        ax.barh([_pretty(c) for c in imp.index], imp.values, color=colors)
        for i, v in enumerate(imp.values):
            ax.text(v + imp.max() * 0.015, i, f"{v:.1f}%", va="center", fontsize=10, fontweight="bold", color=TEXT)
        ax.set_xlim(0, imp.max() * 1.18)
        ax.set_xticks([])
        for s in ("left", "bottom"):
            ax.spines[s].set_visible(False)
        ax.tick_params(left=False)
        ax.set_title("Feature Importance — Random Forest")
        return _png(fig)


# ─────────────────────────────────────────────────────────────
# REPORT BUILDER  (pure Python — cacheable)
# ─────────────────────────────────────────────────────────────
def build_report(df: pd.DataFrame) -> list[dict]:
    """
    Returns a list of sections:
      {"icon", "title", "intro", "items": [ {"type": "image", "png": bytes, "caption": str}
                                           | {"type": "insights", "lines": [html, ...]}
                                           | {"type": "table", "df": DataFrame} ]}
    """
    y = df[TARGET]
    overall = float(y.mean())
    sections: list[dict] = []

    def add(icon, title, intro, items):
        sections.append({"icon": icon, "title": title, "intro": intro, "items": items})

    # ── Step 9: target distribution ──────────────────────────
    q1, med, q3 = y.quantile([0.25, 0.5, 0.75])
    iqr = q3 - q1
    fence = q3 + 1.5 * iqr
    n_slow = int((y > fence).sum())
    lines = [
        f"The average delivery takes about {_b(f'{overall:.0f} minutes')} (the median is {med:.0f}), and the "
        f"middle 50% of deliveries land between {_b(f'{q1:.0f} and {q3:.0f} minutes')}.",
    ]
    if overall > med + 1:
        lines.append(f"The curve has a longer tail on the right: a few orders take much longer (up to "
                     f"{_b(f'{y.max():.0f} minutes')}), which pulls the average above the median. "
                     + (f"Only {n_slow} deliveries ({n_slow / len(y):.1%}) are unusually slow (over {fence:.0f} minutes)."
                        if n_slow else "None are extreme outliers."))
    elif overall < med - 1:
        lines.append(f"The curve has a longer tail on the left: some orders are very quick (as low as "
                     f"{_b(f'{y.min():.0f} minutes')}), which pulls the average below the median.")
    else:
        lines.append(f"The average and median are almost the same, so delivery times are fairly balanced "
                     f"(range {y.min():.0f}–{y.max():.0f} minutes).")
    lines.append(f"Delivery times typically vary by about {y.std():.0f} minutes either side of the average, which tells "
                 "us that several factors are at work, so a good model needs to look at more than one of them.")
    add("", "Distribution of Delivery Time",
        "How often different delivery times occur. The curve shows the overall shape.",
        [{"type": "image", "png": chart_distribution(df), "caption": "Distribution of delivery time"},
         {"type": "insights", "lines": lines}])

    # ── Step 10: numeric scatter plots ───────────────────────
    corr = {c: float(df[c].corr(y)) if df[c].std() > 0 else 0.0 for c in NUMERIC_FEATURES}
    ranked_num = sorted(NUMERIC_FEATURES, key=lambda c: -abs(corr[c]))
    items, lines = [], []
    for col in ranked_num:
        items.append({"type": "image", "png": chart_scatter(df, col), "caption": f"{_pretty(col)} vs delivery time"})
        r, name, unit = corr[col], _pretty(col), _unit(col)
        if df[col].std() > 0:
            slope = float(np.polyfit(df[col], y, 1)[0])
            lo_cut, hi_cut = df[col].quantile(0.25), df[col].quantile(0.75)
            lo_avg = y[df[col] <= lo_cut].mean()
            hi_avg = y[df[col] >= hi_cut].mean()
            u = f" {unit}" if unit else ""
            direction = "adds" if slope >= 0 else "removes"
            if abs(r) >= 0.2:
                lines.append(
                    f"{_b(name + ' vs. Delivery Time:')} correlation about {_b(f'{r:.2f}')} "
                    f"({_strength(r)} link). Each extra{' ' + unit if unit else ' unit'} {direction} roughly "
                    f"{_b(f'{abs(slope):.1f} minutes')} on average. The lowest quarter of values (≤ {lo_cut:g}{u}) "
                    f"averages {lo_avg:.0f} min, versus {hi_avg:.0f} min for the highest quarter (≥ {hi_cut:g}{u}).")
            else:
                lines.append(
                    f"{_b(name + ' vs. Delivery Time:')} correlation is only {_b(f'{r:.2f}')} — "
                    f"{_strength(r)} straight-line relationship. The lowest quarter of values averages "
                    f"{lo_avg:.0f} min and the highest quarter {hi_avg:.0f} min, so it is not a main driver.")
    top = ranked_num[0]
    if abs(corr[top]) >= 0.4:
        lines.append(f"Overall, {_b(_pretty(top))} is the main numeric driver"
                     + (f"; {_pretty(ranked_num[1])} helps explain delivery time, but only partly."
                        if abs(corr[ranked_num[1]]) >= 0.2 else "."))
    else:
        lines.append("None of the numeric columns has a strong straight-line link with delivery time here — "
                     "look at the category factors below.")
    items.append({"type": "insights", "lines": lines})
    add("", "Numeric Features vs. Delivery Time",
        "One dot per delivery; the red line is the overall trend. Rising dots mean the feature and the delivery time grow together.",
        items)

    # ── Step 11: correlation matrix ──────────────────────────
    inputs = pd.DataFrame({c: df[c] for c in NUMERIC_FEATURES}).corr().abs().to_numpy().copy()
    np.fill_diagonal(inputs, 0)
    max_in = float(np.nan_to_num(inputs).max())
    parts = ", ".join(f"{_b(_pretty(c))} ({corr[c]:.2f})" for c in ranked_num)
    lines = [f"Link with delivery time, strongest first: {parts}.",
             ("The numeric input columns are almost completely unrelated to each other (largest value "
              f"{max_in:.2f}), so each one brings its own separate information to the model."
              if max_in < 0.2 else
              f"Some numeric input columns overlap with each other (largest value {max_in:.2f}), so they partly "
              "carry the same information."),
             "This chart only covers numeric columns. The category columns are explored in the charts that follow."]
    add("", "Correlation Matrix", "Colour-coded grid of how strongly each pair of numeric columns is related (−1 … +1).",
        [{"type": "image", "png": chart_corr(df), "caption": "Correlation matrix"},
         {"type": "insights", "lines": lines}])

    # ── Step 12: box plots ───────────────────────────────────
    medians = {c: df.groupby(c)[TARGET].median() for c in CATEGORICAL_FEATURES}
    iqrs = {c: df.groupby(c)[TARGET].apply(lambda s: s.quantile(.75) - s.quantile(.25)) for c in CATEGORICAL_FEATURES}
    items, lines = [], []
    spread = {c: float(medians[c].max() - medians[c].min()) for c in CATEGORICAL_FEATURES}
    for col in CATEGORICAL_FEATURES:
        items.append({"type": "image", "png": chart_box(df, col), "caption": f"Delivery time by {_pretty(col)}"})
    for col in sorted(CATEGORICAL_FEATURES, key=lambda c: -spread[c]):
        m = medians[col]
        if spread[col] < 3:
            lines.append(f"{_b(_pretty(col) + ':')} the boxes sit at almost the same height (medians of "
                         f"{m.min():.0f}–{m.max():.0f} minutes), so this makes little visible difference.")
        else:
            txt = (f"{_b(_pretty(col) + ':')} the median is lowest for {_b(m.idxmin())} ({m.min():.0f} min) and "
                   f"highest for {_b(m.idxmax())} ({m.max():.0f} min).")
            widest = iqrs[col].idxmax()
            others = iqrs[col].drop(widest)
            if len(others) and iqrs[col][widest] >= 1.4 * others.mean():
                txt += f" The {widest} box is also much taller, meaning delivery times are far less predictable there."
            lines.append(txt)
    add("", "Category Features vs. Delivery Time (Box Plots)",
        "Each box shows the median, the middle 50% range and outliers for every category.",
        items + [{"type": "insights", "lines": lines}])

    # ── Step 13: ranking by gap ──────────────────────────────
    gaps = pd.Series(
        {c: df.groupby(c)[TARGET].mean().max() - df.groupby(c)[TARGET].mean().min() for c in CATEGORICAL_FEATURES}
    ).sort_values(ascending=False)
    lines = []
    for i, (col, g) in enumerate(gaps.items()):
        means = df.groupby(col)[TARGET].mean()
        if g < 3:
            continue
        lead = "matters most" if i == 0 else ("comes next" if i == 1 else "also matters")
        lines.append(f"{_b(_pretty(col))} {lead}: the average delivery time differs by {_b(f'{g:.1f} minutes')} "
                     f"between {means.idxmax()} (slowest) and {means.idxmin()} (fastest).")
    small = [f"{_pretty(c)} ({g:.1f} min)" for c, g in gaps.items() if g < 3]
    if small:
        lines.append(f"{_b(', '.join(small))} barely change the average delivery time.")
    if not lines:
        lines.append("None of the category columns changes the average delivery time by more than 3 minutes.")
    add("", "Which Categorical Factor Matters Most?",
        "Gap between the average delivery time of the slowest and the fastest group in each category column.",
        [{"type": "image", "png": chart_gaps(gaps), "caption": "Impact of each categorical factor"},
         {"type": "insights", "lines": lines}])

    # ── Step 14: average by category ─────────────────────────
    items, lines = [], []
    for col in CATEGORICAL_FEATURES:
        items.append({"type": "image", "png": chart_category_means(df, col, overall),
                      "caption": f"Average delivery time by {_pretty(col)}"})
        means = df.groupby(col)[TARGET].mean().sort_values()
        listing = ", ".join(f"{k} ({v:.1f})" for k, v in means.items())
        if means.max() - means.min() < 3:
            lines.append(f"{_b(_pretty(col) + ':')} every bar is almost as tall as the overall average line "
                         f"({listing}), so it does not change delivery time much on its own.")
        else:
            below = [k for k, v in means.items() if v < overall - 1]
            lines.append(f"{_b(_pretty(col) + ':')} {listing}. {_b(means.idxmin())} is the fastest at "
                         f"{means.min():.1f} min and {_b(means.idxmax())} the slowest at {means.max():.1f} min "
                         f"(overall average {overall:.1f}). "
                         + (f"Below the overall average: {', '.join(below)}." if below else ""))
    add("", "Average Delivery Time by Category",
        f"The slowest group is orange and the dashed line marks the overall average ({overall:.1f} min).",
        items + [{"type": "insights", "lines": lines}])

    # ── Step 15: two-factor heatmaps ─────────────────────────
    pairs = [("Traffic_Level", "Weather"), ("Weather", "Vehicle_Type"), ("Time_of_Day", "Traffic_Level")]
    items, lines = [], []
    for row_col, col_col in pairs:
        rows, cols = _order(df, row_col), _order(df, col_col)
        avg = df.pivot_table(index=row_col, columns=col_col, values=TARGET, aggfunc="mean").reindex(index=rows, columns=cols)
        cnt = df.pivot_table(index=row_col, columns=col_col, values=TARGET, aggfunc="count").reindex(index=rows, columns=cols).fillna(0).astype(int)
        items.append({"type": "image", "png": chart_pair_heatmap(avg, cnt, row_col, col_col),
                      "caption": f"{_pretty(row_col)} × {_pretty(col_col)}"})
        ok = avg.where(cnt >= MIN_CELL_FOR_RANKING)
        if ok.notna().sum().sum() < 2:
            continue
        stacked = ok.stack()
        (sr, sc), slow = stacked.idxmax(), stacked.max()
        (fr, fc), fast = stacked.idxmin(), stacked.min()
        txt = (f"{_b(_pretty(row_col) + ' × ' + _pretty(col_col) + ':')} the slowest combination is "
               f"{_b(f'{sr} + {sc} at {slow:.1f} minutes')} ({cnt.loc[sr, sc]} deliveries), while the fastest is "
               f"{fr} + {fc} at {fast:.1f} minutes, a gap of about {slow - fast:.0f} minutes.")
        # does the effect of the column factor change a lot between rows?  (interaction hint)
        ranges = (ok.max(axis=1) - ok.min(axis=1)).dropna()
        if len(ranges) >= 2 and ranges.max() >= 5 and ranges.max() >= 2 * max(ranges.min(), 0.5):
            txt += (f" The effect of {_pretty(col_col)} is much bigger when {_pretty(row_col)} is {_b(ranges.idxmax())} "
                    f"({ranges.max():.0f} min spread) than when it is {ranges.idxmin()} ({ranges.min():.0f} min), "
                    "so the two factors make each other worse.")
        if cnt.loc[sr, sc] < SMALL_CELL:
            txt += f" Only {cnt.loc[sr, sc]} deliveries sit in the slowest cell, so treat it as a hint rather than a firm conclusion."
        lines.append(txt)
    lines.append("Cells with fewer than 30 deliveries are shaky — treat them as hints rather than firm conclusions.")
    add("", "Combined Effect of Two Factors (Heatmaps)",
        "Each cell shows the average delivery time of one combination, with n = the number of deliveries behind it. Darker red = slower.",
        items + [{"type": "insights", "lines": lines}])

    # ── Steps 16-17: statistical tests ───────────────────────
    table = feature_selection_table(df)
    best = get_best_features(table)
    shown = table[["Feature", "Test", "Statistic", "p-value", "Selected", "Why"]].copy()
    shown["Feature"] = shown["Feature"].map(_pretty)
    shown["Statistic"] = shown["Statistic"].round(3)
    shown["p-value"] = shown["p-value"].map(lambda p: "< 0.0001" if p < 1e-4 else f"{p:.4f}")
    shown["Selected"] = shown["Selected"].map({True: "Best", False: "—"})
    lines = [f"{_b('Best features for your data:')} " + ", ".join(_pretty(c) for c in best) + ".",
             "Numeric columns are tested with Pearson correlation and category columns with One-Way ANOVA. "
             "A feature counts as <em>best</em> when the result is statistically significant (p &lt; 0.05)."]
    dropped = [c for c in table["Feature"] if c not in best]
    if dropped:
        lines.append("Not selected: " + ", ".join(_pretty(c) for c in dropped)
                     + " — they are not statistically significant on their own.")
    add("", "Statistical Tests & Best Features",
        "How each factor is tested against delivery time, and which ones the tool recommends keeping.",
        [{"type": "table", "df": shown.reset_index(drop=True)}, {"type": "insights", "lines": lines}])

    return sections


# ─────────────────────────────────────────────────────────────
# STREAMLIT RENDERER  (imports streamlit lazily so build_report works anywhere)
# ─────────────────────────────────────────────────────────────
def stretch(fn, *args, **kwargs):
    """
    Calls a Streamlit element at full width on every Streamlit version:
    new releases use width="stretch" (use_container_width is deprecated), older ones use
    use_container_width=True.
    """
    try:
        return fn(*args, width="stretch", **kwargs)
    except Exception:  # older Streamlit does not know width="stretch"
        return fn(*args, use_container_width=True, **kwargs)


def show_png(st, png: bytes, caption: str | None = None):
    stretch(st.image, png, caption=caption)


def render_report(sections: list[dict], ui) -> None:
    """ui must provide section_heading(icon, title) and insight_line(number, html)."""
    import streamlit as st

    for sec in sections:
        with st.expander(sec['title'], expanded=False):
            if sec["intro"]:
                st.markdown(f"<p style='font-size:14px;color:#8b949e;margin:0 0 10px 0;'>{sec['intro']}</p>",
                            unsafe_allow_html=True)
            n = 0
            images = [it for it in sec["items"] if it["type"] == "image"]
            for it in sec["items"]:
                if it["type"] == "image":
                    show_png(st, it["png"], None)
                elif it["type"] == "table":
                    stretch(st.dataframe, it["df"], hide_index=True)
                elif it["type"] == "insights":
                    st.markdown("<p style='font-weight:700;font-size:16px;color:#c9a84c;margin:10px 0 6px 0;'>"
                                "What we observed</p>", unsafe_allow_html=True)
                    for line in it["lines"]:
                        n += 1
                        ui.insight_line(n, line)


def render_data_quality_report_dynamic(df: pd.DataFrame, ui, data_key: str) -> None:
    """Cached wrapper: the report is built once per uploaded file, then re-shown instantly."""
    import streamlit as st

    @st.cache_data(show_spinner="Drawing charts for your data…", max_entries=2)
    def _cached(key: str, _df: pd.DataFrame):
        return build_report(_df)

    render_report(_cached(data_key, df), ui)