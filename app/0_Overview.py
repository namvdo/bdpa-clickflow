import streamlit as st
import pandas as pd
import altair as alt
from pathlib import Path

from lookups import country_names, category_names

st.set_page_config(
    page_title="bdpa-clickflow",
    layout="wide",
)

ACCENT = "#5b7fbd"

output_dir = Path(__file__).parent.parent / "output"
cleaned_path = output_dir / "cleaned.parquet"
sessions_path = output_dir / "sessions.parquet"

st.title("Overview")
st.caption("UCI Clickstream Data for Online Shopping (2008)")

if not cleaned_path.exists() or not sessions_path.exists():
    st.write("Pipeline outputs not found. Run the scripts in pipeline/ first.")
    st.stop()

clicks = pd.read_parquet(cleaned_path)
sessions = pd.read_parquet(sessions_path)

# top-line numbers
total_clicks = len(clicks)
total_sessions = len(sessions)
n_countries = sessions["country"].nunique()
date_min = pd.to_datetime(sessions["date"]).min().date()
date_max = pd.to_datetime(sessions["date"]).max().date()
above_avg_rate = sessions["bought"].mean()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total clicks", f"{total_clicks:,}")
c2.metric("Total sessions", f"{total_sessions:,}")
c3.metric("Countries", n_countries)
c4.metric("Sessions with above-avg item", f"{above_avg_rate:.1%}")

st.write(f"Date range: {date_min} to {date_max}")

st.divider()

# sessions over time
st.subheader("Sessions over time")
by_day = (
    sessions.assign(date=pd.to_datetime(sessions["date"]))
    .groupby("date")
    .size()
    .reset_index(name="sessions")
)

line = (
    alt.Chart(by_day)
    .mark_line(color=ACCENT)
    .encode(
        x=alt.X("date:T", title=None),
        y=alt.Y("sessions:Q", title="sessions"),
        tooltip=["date:T", "sessions:Q"],
    )
    .properties(height=260)
)
st.altair_chart(line, use_container_width=True)

st.divider()

# top countries and top categories side by side
left, right = st.columns(2)

with left:
    st.subheader("Top countries by sessions")
    # cast and strip in case parquet stored country as float-like or with whitespace
    country_col = sessions["country"].astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    top_countries = (
        country_col.value_counts()
        .head(10)
        .rename_axis("country_code")
        .reset_index(name="sessions")
    )
    top_countries["country"] = top_countries["country_code"].map(country_names).fillna("unknown")

    bar_c = (
        alt.Chart(top_countries)
        .mark_bar(color=ACCENT)
        .encode(
            x=alt.X("sessions:Q", title="sessions"),
            y=alt.Y("country:N", sort="-x", title=None),
            tooltip=["country:N", "sessions:Q"],
        )
        .properties(height=320)
    )
    st.altair_chart(bar_c, use_container_width=True)

with right:
    st.subheader("Clicks by category")
    cat_col = clicks["page_1_main_category"].astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    by_cat = (
        cat_col.value_counts()
        .rename_axis("cat_code")
        .reset_index(name="clicks")
    )
    by_cat["category"] = by_cat["cat_code"].map(category_names).fillna("unknown")

    bar_cat = (
        alt.Chart(by_cat)
        .mark_bar(color=ACCENT)
        .encode(
            x=alt.X("clicks:Q", title="clicks"),
            y=alt.Y("category:N", sort="-x", title=None),
            tooltip=["category:N", "clicks:Q"],
        )
        .properties(height=220)
    )
    st.altair_chart(bar_cat, use_container_width=True)

st.divider()

# distribution of clicks per session
st.subheader("Clicks per session")
# clip the long tail so the chart stays readable
clipped = sessions["n_clicks"].clip(upper=50)

hist = (
    alt.Chart(pd.DataFrame({"n_clicks": clipped}))
    .mark_bar(color=ACCENT)
    .encode(
        x=alt.X("n_clicks:Q", bin=alt.Bin(maxbins=40), title="clicks per session (clipped at 50)"),
        y=alt.Y("count():Q", title="sessions"),
    )
    .properties(height=260)
)
st.altair_chart(hist, use_container_width=True)

median_clicks = int(sessions["n_clicks"].median())
mean_clicks = sessions["n_clicks"].mean()
st.write(f"Median session length: {median_clicks} clicks. Mean: {mean_clicks:.1f}.")
