import json
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Recommender Results", layout="wide")

METRICS_PATH = Path(__file__).parent.parent.parent / "evaluation_metrics" / "recommender_metrics.json"

MODEL_COLORS = {
    "ALS": "#4c72b0",
    "Popularity baseline": "#2ca02c",
    "Item k-NN": "#ff7f0e",
    "Sequential warm-up": "#c44e52",
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


def fmt_pct_delta(value):
    if value is None or pd.isna(value):
        return "-"
    return f"{value:+.2%}"


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

    return (
        alt.Chart(metric_df)
        .mark_bar()
        .encode(
            x=alt.X("Model:N", title=None, sort=model_order, axis=alt.Axis(labelAngle=0, labelLimit=140)),
            y=alt.Y("Value:Q", title="Score", scale=alt.Scale(domain=[0, y_max])),
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
                alt.Tooltip("Value:Q", format=".4f"),
                alt.Tooltip("Delta:Q", format="+.4f"),
            ],
        )
        .properties(title=metric_name, height=320)
    )


data = load_recommender_data()

if not data:
    st.error("No recommender metrics found.")
    st.warning("Please run the `recommender/recommender_namdo.ipynb` notebook to generate the evaluation results.")
    st.stop()

metrics = data["metrics"]
warm = metrics["warm_sessions"]
cold = metrics["cold_sessions"]
params = data.get("params", {"rank": "N/A", "alpha": "N/A", "regParam": "N/A"})

warm_baseline = warm["popularity"]
cold_baseline = cold["popularity"]
warm_rank_candidates = {
    "ALS": warm["als"]["NDCG@10"],
    "Popularity baseline": warm_baseline["NDCG@10"],
    "Item k-NN": warm["knn"]["NDCG@10"],
}
best_warm_model = max(warm_rank_candidates, key=warm_rank_candidates.get)

warm_chart_df = pd.DataFrame(
    [
        {"Model": "Popularity baseline", "Metric": "NDCG@10", "Value": warm_baseline["NDCG@10"], "Delta": 0.0},
        {"Model": "ALS", "Metric": "NDCG@10", "Value": warm["als"]["NDCG@10"], "Delta": warm["als"]["NDCG@10"] - warm_baseline["NDCG@10"]},
        {"Model": "Item k-NN", "Metric": "NDCG@10", "Value": warm["knn"]["NDCG@10"], "Delta": warm["knn"]["NDCG@10"] - warm_baseline["NDCG@10"]},
        {"Model": "Popularity baseline", "Metric": "Recall@10", "Value": warm_baseline["Recall@10"], "Delta": 0.0},
        {"Model": "ALS", "Metric": "Recall@10", "Value": warm["als"]["Recall@10"], "Delta": warm["als"]["Recall@10"] - warm_baseline["Recall@10"]},
        {"Model": "Item k-NN", "Metric": "Recall@10", "Value": warm["knn"]["Recall@10"], "Delta": warm["knn"]["Recall@10"] - warm_baseline["Recall@10"]},
        {"Model": "Popularity baseline", "Metric": "MRR@10", "Value": warm_baseline["MRR@10"], "Delta": 0.0},
        {"Model": "ALS", "Metric": "MRR@10", "Value": warm["als"]["MRR@10"], "Delta": warm["als"]["MRR@10"] - warm_baseline["MRR@10"]},
        {"Model": "Item k-NN", "Metric": "MRR@10", "Value": warm["knn"]["MRR@10"], "Delta": warm["knn"]["MRR@10"] - warm_baseline["MRR@10"]},
        {"Model": "Popularity baseline", "Metric": "Coverage", "Value": warm_baseline["Coverage"], "Delta": 0.0},
        {"Model": "ALS", "Metric": "Coverage", "Value": warm["als"]["Coverage"], "Delta": warm["als"]["Coverage"] - warm_baseline["Coverage"]},
        {"Model": "Item k-NN", "Metric": "Coverage", "Value": warm["knn"]["Coverage"], "Delta": warm["knn"]["Coverage"] - warm_baseline["Coverage"]},
    ]
)

warm_table_raw_df = pd.DataFrame(
    [
        {
            "Model": "Popularity baseline",
            "NDCG@10": warm_baseline["NDCG@10"],
            "Delta NDCG@10": 0.0,
            "Recall@10": warm_baseline["Recall@10"],
            "Delta Recall@10": 0.0,
            "MRR@10": warm_baseline["MRR@10"],
            "Delta MRR@10": 0.0,
            "Coverage": warm_baseline["Coverage"],
            "Delta Coverage": 0.0,
        },
        {
            "Model": "Item k-NN",
            "NDCG@10": warm["knn"]["NDCG@10"],
            "Delta NDCG@10": warm["knn"]["NDCG@10"] - warm_baseline["NDCG@10"],
            "Recall@10": warm["knn"]["Recall@10"],
            "Delta Recall@10": warm["knn"]["Recall@10"] - warm_baseline["Recall@10"],
            "MRR@10": warm["knn"]["MRR@10"],
            "Delta MRR@10": warm["knn"]["MRR@10"] - warm_baseline["MRR@10"],
            "Coverage": warm["knn"]["Coverage"],
            "Delta Coverage": warm["knn"]["Coverage"] - warm_baseline["Coverage"],
        },
        {
            "Model": "ALS",
            "NDCG@10": warm["als"]["NDCG@10"],
            "Delta NDCG@10": warm["als"]["NDCG@10"] - warm_baseline["NDCG@10"],
            "Recall@10": warm["als"]["Recall@10"],
            "Delta Recall@10": warm["als"]["Recall@10"] - warm_baseline["Recall@10"],
            "MRR@10": warm["als"]["MRR@10"],
            "Delta MRR@10": warm["als"]["MRR@10"] - warm_baseline["MRR@10"],
            "Coverage": warm["als"]["Coverage"],
            "Delta Coverage": warm["als"]["Coverage"] - warm_baseline["Coverage"],
        },
    ]
)

warm_table_df = pd.DataFrame(
    {
        "Model": warm_table_raw_df["Model"],
        "NDCG@10": warm_table_raw_df["NDCG@10"].map(fmt_score),
        "Delta vs Baseline (NDCG@10)": warm_table_raw_df["Delta NDCG@10"].map(fmt_delta),
        "Recall@10": warm_table_raw_df["Recall@10"].map(fmt_score),
        "Delta vs Baseline (Recall@10)": warm_table_raw_df["Delta Recall@10"].map(fmt_delta),
        "MRR@10": warm_table_raw_df["MRR@10"].map(fmt_score),
        "Delta vs Baseline (MRR@10)": warm_table_raw_df["Delta MRR@10"].map(fmt_delta),
        "Coverage": warm_table_raw_df["Coverage"].map(fmt_pct),
        "Delta vs Baseline (Coverage)": warm_table_raw_df["Delta Coverage"].map(fmt_pct_delta),
    }
)

cold_chart_df = pd.DataFrame(
    [
        {"Model": "Popularity baseline", "Metric": "NDCG@10", "Value": cold_baseline["NDCG@10"], "Delta": 0.0},
        {"Model": "Sequential warm-up", "Metric": "NDCG@10", "Value": cold["sequential_warmup"]["NDCG@10"], "Delta": cold["sequential_warmup"]["NDCG@10"] - cold_baseline["NDCG@10"]},
        {"Model": "Popularity baseline", "Metric": "Recall@10", "Value": cold_baseline["Recall@10"], "Delta": 0.0},
        {"Model": "Sequential warm-up", "Metric": "Recall@10", "Value": cold["sequential_warmup"]["Recall@10"], "Delta": cold["sequential_warmup"]["Recall@10"] - cold_baseline["Recall@10"]},
        {"Model": "Popularity baseline", "Metric": "Coverage", "Value": cold_baseline["Coverage"], "Delta": 0.0},
        {"Model": "Sequential warm-up", "Metric": "Coverage", "Value": cold["sequential_warmup"]["Coverage"], "Delta": cold["sequential_warmup"]["Coverage"] - cold_baseline["Coverage"]},
    ]
)

cold_table_raw_df = pd.DataFrame(
    [
        {
            "Model": "Popularity baseline",
            "NDCG@10": cold_baseline["NDCG@10"],
            "Delta NDCG@10": 0.0,
            "Recall@10": cold_baseline["Recall@10"],
            "Delta Recall@10": 0.0,
            "Coverage": cold_baseline["Coverage"],
            "Delta Coverage": 0.0,
        },
        {
            "Model": "Sequential warm-up",
            "NDCG@10": cold["sequential_warmup"]["NDCG@10"],
            "Delta NDCG@10": cold["sequential_warmup"]["NDCG@10"] - cold_baseline["NDCG@10"],
            "Recall@10": cold["sequential_warmup"]["Recall@10"],
            "Delta Recall@10": cold["sequential_warmup"]["Recall@10"] - cold_baseline["Recall@10"],
            "Coverage": cold["sequential_warmup"]["Coverage"],
            "Delta Coverage": cold["sequential_warmup"]["Coverage"] - cold_baseline["Coverage"],
        },
    ]
)

cold_table_df = pd.DataFrame(
    {
        "Model": cold_table_raw_df["Model"],
        "NDCG@10": cold_table_raw_df["NDCG@10"].map(fmt_score),
        "Delta vs Baseline (NDCG@10)": cold_table_raw_df["Delta NDCG@10"].map(fmt_delta),
        "Recall@10": cold_table_raw_df["Recall@10"].map(fmt_score),
        "Delta vs Baseline (Recall@10)": cold_table_raw_df["Delta Recall@10"].map(fmt_delta),
        "Coverage": cold_table_raw_df["Coverage"].map(fmt_pct),
        "Delta vs Baseline (Coverage)": cold_table_raw_df["Delta Coverage"].map(fmt_pct_delta),
    }
)

warm_styler = build_styled_table(
    warm_table_df,
    warm_table_raw_df,
    {
        "Delta vs Baseline (NDCG@10)": "Delta NDCG@10",
        "Delta vs Baseline (Recall@10)": "Delta Recall@10",
        "Delta vs Baseline (MRR@10)": "Delta MRR@10",
        "Delta vs Baseline (Coverage)": "Delta Coverage",
    },
)

cold_styler = build_styled_table(
    cold_table_df,
    cold_table_raw_df,
    {
        "Delta vs Baseline (NDCG@10)": "Delta NDCG@10",
        "Delta vs Baseline (Recall@10)": "Delta Recall@10",
        "Delta vs Baseline (Coverage)": "Delta Coverage",
    },
)

st.title("Recommender Analysis")
st.markdown(
    """
This page compares the recommender models in two settings:
**warm sessions** with multiple observed clicks, and **cold-start sessions** with the sequential fix.
"""
)

warm_tab, cold_tab = st.tabs(["Warm Sessions", "Cold-Start Sessions"])

with warm_tab:
    st.markdown(
        f"""
For sessions with multiple interactions, **{best_warm_model}** is the strongest ranking model here on
**NDCG@10 = {warm_rank_candidates[best_warm_model]:.4f}**.
The chart below compares **ALS**, **Popularity baseline**, and **Item k-NN** across all warm-session metrics.
"""
    )

    warm_metric_order = ["NDCG@10", "Recall@10", "MRR@10", "Coverage"]
    warm_columns = st.columns(len(warm_metric_order))
    for metric_name, column in zip(warm_metric_order, warm_columns):
        with column:
            st.altair_chart(
                build_metric_chart(warm_chart_df, metric_name, ["Popularity baseline", "Item k-NN", "ALS"]),
                use_container_width=True,
            )

    st.markdown(
        f"""
- **ALS** improves the warm-session baseline on **NDCG@10** by **{warm["als"]["NDCG@10"] - warm_baseline["NDCG@10"]:+.4f}** and on **Recall@10** by **{warm["als"]["Recall@10"] - warm_baseline["Recall@10"]:+.4f}**.
- **Item k-NN** currently has **{warm["knn"]["Coverage"]:.2%} coverage** with a **{warm["knn"]["NDCG@10"] - warm_baseline["NDCG@10"]:+.4f}** change in NDCG@10 and **{warm["knn"]["Recall@10"] - warm_baseline["Recall@10"]:+.4f}** in Recall@10 vs the warm-session baseline.
"""
    )

    st.subheader("Warm-Session Metrics Table")
    st.dataframe(warm_styler, use_container_width=True, hide_index=True)

with cold_tab:
    st.markdown(
        f"""
For cold-start sessions, the sequential fallback is mainly valuable for **coverage**:
it reaches **{cold["sequential_warmup"]["Coverage"]:.2%}**, which is a **{cold["sequential_warmup"]["Coverage"] - cold_baseline["Coverage"]:+.2%}** change vs the popularity baseline.
"""
    )

    cold_metric_order = ["NDCG@10", "Recall@10", "Coverage"]
    cold_columns = st.columns(len(cold_metric_order))
    for metric_name, column in zip(cold_metric_order, cold_columns):
        with column:
            st.altair_chart(
                build_metric_chart(cold_chart_df, metric_name, ["Popularity baseline", "Sequential warm-up"]),
                use_container_width=True,
            )

    st.markdown(
        f"""
- **Sequential warm-up** changes **NDCG@10** by **{cold["sequential_warmup"]["NDCG@10"] - cold_baseline["NDCG@10"]:+.4f}** and **Recall@10** by **{cold["sequential_warmup"]["Recall@10"] - cold_baseline["Recall@10"]:+.4f}** vs the cold-start baseline.
- The biggest gain is in **coverage**, moving from **{cold_baseline["Coverage"]:.2%}** to **{cold["sequential_warmup"]["Coverage"]:.2%}**.
"""
    )

    st.subheader("Cold-Start Metrics Table")
    st.dataframe(cold_styler, use_container_width=True, hide_index=True)

st.caption(
    f"Model parameters: rank={params['rank']}, alpha={params['alpha']}, regParam={params['regParam']}. "
    "Evaluation performed with a leave-one-out next-item split (last click per session held out)."
)
