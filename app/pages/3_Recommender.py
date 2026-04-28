import streamlit as st
import pandas as pd
import altair as alt

st.set_page_config(page_title="Recommender Results", layout="wide")

ACCENT = "#5b7fbd"
SUCCESS = "#28a745"

st.title("Recommender Analysis")
st.markdown("""
This page showcases the results of the Collaborative Filtering models and the custom 
**Sequential Warm-up** fix implemented for cold-start sessions.
""")

st.header("Key Performance Improvements")
c1, c2, c3 = st.columns(3)

with c1:
    st.metric("ALS vs Baseline", "+121%", delta="Improvement in NDCG@10", delta_color="normal")
with c2:
    st.metric("Sequential vs Baseline", "+148%", delta="Improvement in NDCG@10", delta_color="normal")
with c3:
    st.metric("Catalogue Coverage", "97.7%", delta="ALS & Sequential Models", delta_color="off")

st.divider()

tab1, tab2, tab3 = st.tabs(["Warm Session Results", "Cold-Start Fix", "Item Popularity"])

with tab1:
    st.subheader("ALS (Collaborative Filtering) vs Popularity Baseline")
    st.write("For sessions with multiple interactions, the ALS model significantly outperforms the popularity baseline.")
    
    warm_data = pd.DataFrame({
        "Model": ["Popularity", "Popularity", "ALS", "ALS"],
        "Metric": ["NDCG@10", "Recall@10", "NDCG@10", "Recall@10"],
        "Value": [0.0606, 0.1213, 0.1338, 0.2767]
    })
    
    chart_warm = alt.Chart(warm_data).mark_bar().encode(
        x=alt.X("Model:N", title=None),
        y=alt.Y("Value:Q", title="Score"),
        color=alt.Color("Model:N", scale=alt.Scale(range=[ACCENT, SUCCESS])),
        column=alt.Column("Metric:N", title=None)
    ).properties(width=300, height=300)
    
    st.altair_chart(chart_warm)

with tab2:
    st.subheader("Solving Cold-Start with Sequential Warm-up")
    st.info("""
    **Challenge:** ALS cannot provide recommendations for users with only one interaction (cold-start).
    
    **Solution:** We implemented a **Session-Sequential Warm-up** using an item-to-item transition matrix 
    derived from existing session flows.
    """)
    
    cold_data = pd.DataFrame({
        "Model": ["Popularity", "Popularity", "Sequential", "Sequential"],
        "Metric": ["NDCG@10", "Recall@10", "NDCG@10", "Recall@10"],
        "Value": [0.0918, 0.1831, 0.2280, 0.4386]
    })
    
    chart_cold = alt.Chart(cold_data).mark_bar().encode(
        x=alt.X("Model:N", title=None),
        y=alt.Y("Value:Q", title="Score"),
        color=alt.Color("Model:N", scale=alt.Scale(range=[ACCENT, SUCCESS])),
        column=alt.Column("Metric:N", title=None)
    ).properties(width=300, height=300)
    
    st.altair_chart(chart_cold)
    
    st.success(f"Sequential warm-up provides **97.7% coverage** for cold sessions, where ALS provided **0%**.")

with tab3:
    st.subheader("Top 10 Most Popular Items")
    st.write("The following items drive the majority of clicks in the shop.")
    
    top_items = pd.DataFrame({
        "item_id": ["B4", "A2", "A11", "P1", "B10", "A4", "A15", "A5", "A10", "A1"],
        "views": [3579, 3013, 2789, 2681, 2566, 2522, 2489, 2354, 2280, 2265]
    })
    
    bar_items = alt.Chart(top_items).mark_bar(color=ACCENT).encode(
        x=alt.X("views:Q", title="Views"),
        y=alt.Y("item_id:N", sort="-x", title="Item ID"),
        tooltip=["item_id", "views"]
    ).properties(height=400)
    
    st.altair_chart(bar_items, use_container_width=True)

st.divider()
st.markdown("---")
st.caption("Model parameters: rank=20, alpha=80.0, regParam=1.0. Evaluation performed on a 20% holdout set.")
