import json
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Recommender Results", layout="wide")

METRICS_PATH = Path(__file__).parent.parent.parent / "evaluation_metrics" / "recommender_metrics.json"

MODEL_COLORS = {
    "Popularity baseline": "#2ca02c",
    "Item k-NN": "#ff7f0e",
    "ALS": "#4c72b0",
    "Sequential warm-up": "#c44e52",
    "Item2Vec": "#9467bd",
}

METRIC_ORDER_WARM = ["MRR@10", "Recall@10", "Coverage", "Novelty@10"]
METRIC_ORDER_COLD = ["MRR@10", "Recall@10", "Coverage", "Novelty@10"]

METRIC_KIND = {
    "MRR@10": "score",
    "Recall@10": "score",
    "Coverage": "coverage",
    "Novelty@10": "novelty",
}


def load_recommender_data():
    if METRICS_PATH.exists():
        with open(METRICS_PATH, "r") as f:
            return json.load(f)
    return None


def fmt_score(value):
    if value is None or pd.isna(value):
        return "-"
    return f"{value:.4f}"


def fmt_delta(value):
    if value is None or pd.isna(value):
        return "-"
    return f"{value:+.4f}"


def fmt_pct(value):
    if value is None or pd.isna(value):
        return "-"
    return f"{value:.2%}"


def fmt_pct_point_delta(value):
    if value is None or pd.isna(value):
        return "-"
    return f"{value:+.2%} pp"


def fmt_bits(value):
    if value is None or pd.isna(value):
        return "-"
    return f"{value:.4f} bits"


def fmt_bits_delta(value):
    if value is None or pd.isna(value):
        return "-"
    return f"{value:+.4f} bits"


def relative_change(value, baseline):
    if value is None or baseline is None or pd.isna(value) or pd.isna(baseline) or baseline == 0:
        return None
    return (value - baseline) / baseline


def fmt_improvement(value):
    if value is None or pd.isna(value):
        return "-"
    return f"{value:+.1%}"


def fmt_value_by_metric(metric_name, value):
    kind = METRIC_KIND[metric_name]
    if kind == "coverage":
        return fmt_pct(value)
    if kind == "novelty":
        return fmt_bits(value)
    return fmt_score(value)


def fmt_delta_by_metric(metric_name, value):
    kind = METRIC_KIND[metric_name]
    if kind == "coverage":
        return fmt_pct_point_delta(value)
    if kind == "novelty":
        return fmt_bits_delta(value)
    return fmt_delta(value)


def style_delta_cell(value):
    if value is None or pd.isna(value):
        return ""
    if value > 0:
        return "font-weight: 700; background-color: #e8f5e9; color: #1b5e20;"
    if value < 0:
        return "font-weight: 700; background-color: #fdecea; color: #8a1c1c;"
    return "font-weight: 700; background-color: #f3f4f6; color: #374151;"


def style_model_cell(value):
    if value == "Popularity baseline":
        return "font-weight: 700;"
    return ""


def build_styled_table(display_df, raw_df, delta_cols):
    styler = display_df.style

    for display_col, raw_col in delta_cols.items():
        styler = styler.apply(
            lambda _col, rc=raw_col: [style_delta_cell(v) for v in raw_df[rc]],
            subset=[display_col],
            axis=0,
        )

    styler = styler.apply(
        lambda _col: [style_model_cell(v) for v in display_df["Model"]],
        subset=["Model"],
        axis=0,
    )

    return styler


def build_metric_chart(chart_df, metric_name, model_order):
    metric_df = chart_df[chart_df["Metric"] == metric_name].copy()
    metric_max = float(metric_df["Value"].max()) if not metric_df.empty else 1.0
    y_max = max(metric_max * 1.15, 0.05)

    if METRIC_KIND[metric_name] == "coverage":
        y_title = "Coverage"
    elif METRIC_KIND[metric_name] == "novelty":
        y_title = "Bits"
    else:
        y_title = "Score"

    return (
        alt.Chart(metric_df)
        .mark_bar()
        .encode(
            x=alt.X("Model:N", title=None, sort=model_order, axis=alt.Axis(labelAngle=0, labelLimit=140)),
            y=alt.Y("Value:Q", title=y_title, scale=alt.Scale(domain=[0, y_max])),
            color=alt.Color(
                "Model:N",
                scale=alt.Scale(
                    domain=list(MODEL_COLORS.keys()),
                    range=list(MODEL_COLORS.values()),
                ),
                legend=None,
            ),
            tooltip=[
                "Model",
                "Metric",
                alt.Tooltip("ValueLabel:N", title="Value"),
                alt.Tooltip("DeltaLabel:N", title="Change vs Baseline"),
                alt.Tooltip("ImprovementLabel:N", title="Relative Improvement"),
            ],
        )
        .properties(title=metric_name, height=320)
    )


def build_chart_df(section_metrics, models, metric_order):
    baseline = section_metrics["popularity"]
    rows = []
    for model_name, key in models:
        model_metrics = section_metrics[key]
        for metric in metric_order:
            value = model_metrics[metric]
            delta = value - baseline[metric]
            improvement = 0.0 if key == "popularity" else relative_change(value, baseline[metric])
            rows.append(
                {
                    "Model": model_name,
                    "Metric": metric,
                    "Value": value,
                    "Delta": delta,
                    "Improvement": improvement,
                    "ValueLabel": fmt_value_by_metric(metric, value),
                    "DeltaLabel": fmt_delta_by_metric(metric, delta),
                    "ImprovementLabel": fmt_improvement(improvement),
                }
            )
    return pd.DataFrame(rows)


def build_table_raw_df(section_metrics, models, metric_order):
    baseline = section_metrics["popularity"]
    rows = []
    for model_name, key in models:
        model_metrics = section_metrics[key]
        row = {"Model": model_name}
        for metric in metric_order:
            value = model_metrics[metric]
            row[metric] = value
            row[f"Delta {metric}"] = value - baseline[metric]
            row[f"Improvement {metric}"] = 0.0 if key == "popularity" else relative_change(value, baseline[metric])
        rows.append(row)
    return pd.DataFrame(rows)


def build_table_display_df(table_raw_df, metric_order):
    cols = {"Model": table_raw_df["Model"]}
    for metric in metric_order:
        cols[metric] = table_raw_df[metric].map(lambda v, m=metric: fmt_value_by_metric(m, v))
        cols[f"Improvement vs Baseline ({metric})"] = table_raw_df[f"Improvement {metric}"].map(fmt_improvement)
        cols[f"Actual Change ({metric})"] = table_raw_df[f"Delta {metric}"].map(
            lambda v, m=metric: fmt_delta_by_metric(m, v)
        )
    return pd.DataFrame(cols)


data = load_recommender_data()

if not data:
    st.error("No recommender metrics found.")
    st.warning("Please run the `recommender/recommender_namdo.ipynb` notebook to generate the evaluation results.")
    st.stop()

metrics = data["metrics"]
warm = metrics["warm_sessions"]
cold = metrics["cold_sessions"]
params = data.get("params", {})
als_params = params.get("als", {})
item2vec_params = params.get("item2vec", {})

warm_models = [
    ("Popularity baseline", "popularity"),
    ("Item k-NN", "knn"),
    ("ALS", "als"),
    ("Item2Vec", "item2vec"),
]
cold_models = [
    ("Popularity baseline", "popularity"),
    ("Sequential warm-up", "sequential_warmup"),
]

warm_baseline = warm["popularity"]
cold_baseline = cold["popularity"]

warm_chart_df = build_chart_df(warm, warm_models, METRIC_ORDER_WARM)
cold_chart_df = build_chart_df(cold, cold_models, METRIC_ORDER_COLD)

warm_table_raw_df = build_table_raw_df(warm, warm_models, METRIC_ORDER_WARM)
cold_table_raw_df = build_table_raw_df(cold, cold_models, METRIC_ORDER_COLD)

warm_table_df = build_table_display_df(warm_table_raw_df, METRIC_ORDER_WARM)
cold_table_df = build_table_display_df(cold_table_raw_df, METRIC_ORDER_COLD)

warm_delta_cols = {f"Improvement vs Baseline ({m})": f"Improvement {m}" for m in METRIC_ORDER_WARM}
cold_delta_cols = {f"Improvement vs Baseline ({m})": f"Improvement {m}" for m in METRIC_ORDER_COLD}

warm_styler = build_styled_table(warm_table_df, warm_table_raw_df, warm_delta_cols)
cold_styler = build_styled_table(cold_table_df, cold_table_raw_df, cold_delta_cols)

best_warm_mrr_model = max(
    [("ALS", warm["als"]["MRR@10"]), ("Item k-NN", warm["knn"]["MRR@10"]), ("Item2Vec", warm["item2vec"]["MRR@10"])],
    key=lambda x: x[1],
)
best_warm_novelty_model = max(
    [("ALS", warm["als"]["Novelty@10"]), ("Item k-NN", warm["knn"]["Novelty@10"]), ("Item2Vec", warm["item2vec"]["Novelty@10"])],
    key=lambda x: x[1],
)

st.title("Recommender Analysis")
st.markdown(
    """
This page compares recommender models in two settings:
**warm sessions** with multiple observed clicks and **cold-start sessions** with one observed click.
"""
)

warm_tab, cold_tab = st.tabs(["Warm Sessions", "Cold-Start Sessions"])

with warm_tab:
    st.markdown(
        f"""
For warm sessions, **{best_warm_mrr_model[0]}** has the strongest **MRR@10 ({best_warm_mrr_model[1]:.4f})**.
For personalization depth, **{best_warm_novelty_model[0]}** has the highest **Novelty@10 ({best_warm_novelty_model[1]:.4f} bits)**.
"""
    )

    warm_summary_cols = st.columns(3)
    warm_summary_cols[0].metric(
        "ALS Recall@10 improvement",
        fmt_improvement(relative_change(warm["als"]["Recall@10"], warm_baseline["Recall@10"])),
        fmt_delta(warm["als"]["Recall@10"] - warm_baseline["Recall@10"]),
    )
    warm_summary_cols[1].metric(
        "Item k-NN MRR@10 improvement",
        fmt_improvement(relative_change(warm["knn"]["MRR@10"], warm_baseline["MRR@10"])),
        fmt_delta(warm["knn"]["MRR@10"] - warm_baseline["MRR@10"]),
    )
    warm_summary_cols[2].metric(
        "Item2Vec Novelty@10 improvement",
        fmt_improvement(relative_change(warm["item2vec"]["Novelty@10"], warm_baseline["Novelty@10"])),
        fmt_bits_delta(warm["item2vec"]["Novelty@10"] - warm_baseline["Novelty@10"]),
    )
    st.caption("Improvements are relative to the popularity baseline.")

    warm_columns = st.columns(len(METRIC_ORDER_WARM))
    warm_model_order = [name for name, _ in warm_models]
    for metric_name, column in zip(METRIC_ORDER_WARM, warm_columns):
        with column:
            st.altair_chart(
                build_metric_chart(warm_chart_df, metric_name, warm_model_order),
                width="stretch",
            )

    st.subheader("Warm-Session Metrics Table")
    st.dataframe(warm_styler, width="stretch", hide_index=True)

with cold_tab:
    seq_cov_delta = cold["sequential_warmup"]["Coverage"] - cold_baseline["Coverage"]
    st.markdown(
        f"""
For cold sessions, **Sequential warm-up** is the dedicated cold-start model in this comparison
and reaches **MRR@10={cold["sequential_warmup"]["MRR@10"]:.4f}**.
"""
    )

    cold_summary_cols = st.columns(3)
    cold_summary_cols[0].metric(
        "Sequential MRR@10 improvement",
        fmt_improvement(relative_change(cold["sequential_warmup"]["MRR@10"], cold_baseline["MRR@10"])),
        fmt_delta(cold["sequential_warmup"]["MRR@10"] - cold_baseline["MRR@10"]),
    )
    cold_summary_cols[1].metric(
        "Sequential Recall@10 improvement",
        fmt_improvement(relative_change(cold["sequential_warmup"]["Recall@10"], cold_baseline["Recall@10"])),
        fmt_delta(cold["sequential_warmup"]["Recall@10"] - cold_baseline["Recall@10"]),
    )
    cold_summary_cols[2].metric(
        "Sequential coverage improvement",
        fmt_improvement(relative_change(cold["sequential_warmup"]["Coverage"], cold_baseline["Coverage"])),
        fmt_pct_point_delta(seq_cov_delta),
    )
    st.caption("Improvements are relative to the popularity baseline.")

    cold_columns = st.columns(len(METRIC_ORDER_COLD))
    cold_model_order = [name for name, _ in cold_models]
    for metric_name, column in zip(METRIC_ORDER_COLD, cold_columns):
        with column:
            st.altair_chart(
                build_metric_chart(cold_chart_df, metric_name, cold_model_order),
                width="stretch",
            )

    st.subheader("Cold-Start Metrics Table")
    st.dataframe(cold_styler, width="stretch", hide_index=True)

st.caption(
    f"ALS params: rank={als_params.get('rank', 'N/A')}, alpha={als_params.get('alpha', 'N/A')}, regParam={als_params.get('regParam', 'N/A')}. "
    f"Item2Vec params: vectorSize={item2vec_params.get('vectorSize', 'N/A')}, windowSize={item2vec_params.get('windowSize', 'N/A')}, "
    f"maxIter={item2vec_params.get('maxIter', 'N/A')}, minCount={item2vec_params.get('minCount', 'N/A')}. "
    "Evaluation uses a leave-last-out next-item split."
)
