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
n_products = clicks["page_2_clothing_model"].nunique()
n_categories = clicks["page_1_main_category"].nunique()
date_min = pd.to_datetime(sessions["date"]).min().date()
date_max = pd.to_datetime(sessions["date"]).max().date()
above_avg_rate = sessions["bought"].mean()
missing_total = int(clicks.isna().sum().sum())

# bought is derived from price_2 (above-avg flag), so this is a proxy not a real purchase
avg_clicks_per_session = sessions["n_clicks"].mean()
avg_session_price = sessions["avg_price"].mean()
multi_cat_share = (sessions["n_categories"] > 1).mean()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total clicks", f"{total_clicks:,}")
c2.metric("Total sessions", f"{total_sessions:,}")
c3.metric("Countries", n_countries)
c4.metric("Products", f"{n_products:,}")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Avg clicks / session", f"{avg_clicks_per_session:.2f}")
c2.metric("Avg session price", f"{avg_session_price:.2f}")
c3.metric("Multi-category sessions", f"{multi_cat_share:.1%}")
c4.metric("Sessions viewing premium item", f"{above_avg_rate:.1%}")

st.caption(
    f"Date range: {date_min} to {date_max}. "
    f"Categories: {n_categories}. "
    f"Missing values in cleaned data: {missing_total}. "
    "Premium item = above category average price (proxy for purchase intent)."
)

st.divider()

# sessions over time
st.subheader("Sessions over time")
sessions_dt = sessions.assign(date=pd.to_datetime(sessions["date"]))
by_day = (
    sessions_dt.groupby("date")
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

# day of week pattern
st.subheader("Activity by day of week")
dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
by_dow = (
    sessions_dt.assign(dow=sessions_dt["date"].dt.day_name())
    .groupby("dow")
    .size()
    .reset_index(name="sessions")
)

dow_bar = (
    alt.Chart(by_dow)
    .mark_bar(color=ACCENT)
    .encode(
        x=alt.X("dow:N", sort=dow_order, title=None),
        y=alt.Y("sessions:Q", title="sessions"),
        tooltip=["dow:N", "sessions:Q"],
    )
    .properties(height=240)
)
st.altair_chart(dow_bar, use_container_width=True)

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

# top products
st.subheader("Top viewed products")
top_products = (
    clicks["page_2_clothing_model"]
    .value_counts()
    .head(10)
    .rename_axis("product")
    .reset_index(name="clicks")
)

bar_p = (
    alt.Chart(top_products)
    .mark_bar(color=ACCENT)
    .encode(
        x=alt.X("clicks:Q", title="clicks"),
        y=alt.Y("product:N", sort="-x", title="clothing model"),
        tooltip=["product:N", "clicks:Q"],
    )
    .properties(height=320)
)
st.altair_chart(bar_p, use_container_width=True)

st.caption(
    "A handful of products receive a large share of clicks. The recommender page uses this skew as a popularity baseline."
)

st.divider()

# distribution of clicks per session
st.subheader("Clicks per session")

median_clicks = int(sessions["n_clicks"].median())
mean_clicks = sessions["n_clicks"].mean()
max_clicks = int(sessions["n_clicks"].max())
single_click_share = (sessions["n_clicks"] == 1).mean()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Median", f"{median_clicks}")
c2.metric("Mean", f"{mean_clicks:.1f}")
c3.metric("Max", f"{max_clicks}")
c4.metric("Single-click sessions", f"{single_click_share:.1%}")

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

st.divider()

# pointer to the rest of the app
st.subheader("What's in this app")
st.markdown(
    """
    - **EDA Insights** — region activity, category and colour preferences, price behaviour, session depth, and a category association rule.
    - **User-Entity Behavior modeling** — frequential analysis, event- and session-level clustering experiments, and cluster profiling.
    - **Recommender Analysis** — ALS, item k-NN, and a sequential cold-start fallback compared against a popularity baseline.
    """
)