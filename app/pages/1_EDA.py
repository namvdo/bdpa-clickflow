import streamlit as st
import pandas as pd
import altair as alt
from pathlib import Path

st.set_page_config(
    page_title="EDA Insights",
    layout="wide",
)

ACCENT = "#5b7fbd"


# =========================
# Paths
# =========================

output_dir = Path(__file__).parent.parent.parent / "output"
cleaned_path = output_dir / "cleaned.parquet"

if not cleaned_path.exists():
    st.write("Pipeline output not found. Run the scripts in pipeline/ first.")
    st.stop()


# =========================
# Load data
# =========================

clicks = pd.read_parquet(cleaned_path)


# =========================
# Helper mappings
# =========================

eu_codes = [
    2,
    3,
    8,
    9,
    10,
    11,
    14,
    15,
    16,
    17,
    18,
    21,
    22,
    23,
    24,
    25,
    27,
    30,
    34,
    35,
    36,
    37,
    41,
]

non_eu_europe_codes = [7, 19, 28, 31, 32, 33, 38, 39]
outside_europe_codes = [1, 4, 5, 6, 20, 26, 40, 42]

category_map = {
    "1": "trousers",
    "2": "skirts",
    "3": "blouses",
    "4": "sale",
}

colour_map = {
    "1": "beige",
    "2": "black",
    "3": "blue",
    "4": "brown",
    "5": "burgundy",
    "6": "gray",
    "7": "green",
    "8": "navy blue",
    "9": "many colors",
    "10": "olive",
    "11": "pink",
    "12": "red",
    "13": "violet",
    "14": "white",
}

plot_colour_map = {
    "beige": "beige",
    "black": "black",
    "blue": "blue",
    "brown": "brown",
    "burgundy": "#800020",
    "gray": "gray",
    "green": "green",
    "navy blue": "navy",
    "many colors": "gold",
    "olive": "olive",
    "pink": "pink",
    "red": "red",
    "violet": "violet",
    "white": "white",
}


# =========================
# Data preparation
# =========================

clicks = clicks.copy()

# Normalize columns because parquet may store codes as int, float, or string
clicks["country_code"] = (
    clicks["country"].astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
)

clicks["category_code"] = (
    clicks["page_1_main_category"]
    .astype(str)
    .str.strip()
    .str.replace(r"\.0$", "", regex=True)
)

clicks["colour_code"] = (
    clicks["colour"].astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
)

clicks["category_name"] = clicks["category_code"].map(category_map).fillna("unknown")
clicks["colour_name"] = clicks["colour_code"].map(colour_map).fillna("unknown")

clicks["country_int"] = clicks["country_code"].astype(int)


def assign_region(country):
    if country == 29:
        return "Poland"
    elif country in eu_codes:
        return "Other_EU"
    elif country in non_eu_europe_codes:
        return "Europe_non_EU"
    elif country in outside_europe_codes:
        return "Outside_Europe"
    else:
        return "Unidentified_or_Domain"


clicks["region"] = clicks["country_int"].apply(assign_region)

clicks["price_2_int"] = clicks["price_2"].astype(int)


# Session-level dataframe
sessions = (
    clicks.groupby(["session_id", "region"])
    .agg(
        num_clicks=("session_id", "size"),
        num_unique_products=("page_2_clothing_model", "nunique"),
        num_unique_categories=("category_name", "nunique"),
        avg_price_viewed=("price", "mean"),
        max_page=("page", "max"),
        max_order=("order", "max"),
        share_above_avg_price=("price_2_int", "mean"),
    )
    .reset_index()
)


# =========================
# Page title
# =========================

st.title("Exploratory Data Analysis")
st.caption(
    "Clickstream behaviour, product popularity, colour preference, price behaviour, and session depth."
)

st.markdown("""
    The goal of this section is to understand the structure of the clickstream data
    before moving to machine learning and recommender systems.
    """)


# =========================
# 1. Dataset overview
# =========================

st.divider()
st.subheader("Dataset overview")

total_clicks = len(clicks)
total_sessions = clicks["session_id"].nunique()
num_products = clicks["page_2_clothing_model"].nunique()
num_categories = clicks["category_name"].nunique()
num_countries = clicks["country_code"].nunique()

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total clicks", f"{total_clicks:,}")
c2.metric("Sessions", f"{total_sessions:,}")
c3.metric("Products", f"{num_products:,}")
c4.metric("Categories", f"{num_categories}")
c5.metric("Countries", f"{num_countries}")

missing_values = clicks.isna().sum().sum()
st.write(f"Total missing values in cleaned data: **{missing_values:,}**")


# =========================
# 2. Region activity
# =========================

st.divider()
st.subheader("Region activity")

region_activity = (
    clicks.groupby("region")
    .agg(
        num_clicks=("session_id", "size"),
        num_sessions=("session_id", "nunique"),
    )
    .reset_index()
)

region_activity["avg_clicks_per_session"] = (
    region_activity["num_clicks"] / region_activity["num_sessions"]
).round(2)

region_activity = region_activity.sort_values("num_sessions", ascending=False)

left, right = st.columns([2, 1])

with left:
    chart = (
        alt.Chart(region_activity)
        .mark_bar(color=ACCENT)
        .encode(
            x=alt.X("num_sessions:Q", title="Number of sessions"),
            y=alt.Y("region:N", sort="-x", title=None),
            tooltip=[
                "region:N",
                "num_sessions:Q",
                "num_clicks:Q",
                "avg_clicks_per_session:Q",
            ],
        )
        .properties(height=300)
    )
    st.altair_chart(chart, use_container_width=True)

with right:
    st.dataframe(region_activity, use_container_width=True)

st.info(
    "Key insight: the dataset is strongly dominated by Poland. "
    "This matters because later ML and recommender models may mainly learn Polish user behaviour."
)


# =========================
# 3. Category popularity
# =========================

st.divider()
st.subheader("Product category popularity")

category_popularity = (
    clicks.groupby("category_name")
    .agg(
        num_clicks=("session_id", "size"),
        num_sessions=("session_id", "nunique"),
    )
    .reset_index()
    .sort_values("num_clicks", ascending=False)
)

chart = (
    alt.Chart(category_popularity)
    .mark_bar(color=ACCENT)
    .encode(
        x=alt.X("num_clicks:Q", title="Number of clicks"),
        y=alt.Y("category_name:N", sort="-x", title=None),
        tooltip=["category_name:N", "num_clicks:Q", "num_sessions:Q"],
    )
    .properties(height=260)
)

st.altair_chart(chart, use_container_width=True)

st.info(
    "Key insight: trousers are the most viewed category, while sale, blouses, and skirts "
    "also have substantial activity."
)


# =========================
# 4. Category popularity by region
# =========================

st.divider()
st.subheader("Category popularity by region")

category_by_region = (
    clicks.groupby(["region", "category_name"]).size().reset_index(name="num_clicks")
)

chart = (
    alt.Chart(category_by_region)
    .mark_bar()
    .encode(
        x=alt.X("region:N", title="Region"),
        y=alt.Y("num_clicks:Q", title="Number of clicks"),
        color=alt.Color("category_name:N", title="Category"),
        tooltip=["region:N", "category_name:N", "num_clicks:Q"],
    )
    .properties(height=360)
)

st.altair_chart(chart, use_container_width=True)

st.info(
    "Key insight: Poland has strong interest in sale items, while Other_EU is more dominated "
    "by trousers and blouses. This supports region-aware recommendations."
)


# =========================
# 5. Top products
# =========================

st.divider()
st.subheader("Top viewed products")

top_products = (
    clicks.groupby("page_2_clothing_model")
    .agg(
        num_clicks=("session_id", "size"),
        num_sessions=("session_id", "nunique"),
    )
    .reset_index()
    .sort_values("num_clicks", ascending=False)
    .head(10)
)

chart = (
    alt.Chart(top_products)
    .mark_bar(color=ACCENT)
    .encode(
        x=alt.X("num_clicks:Q", title="Number of clicks"),
        y=alt.Y("page_2_clothing_model:N", sort="-x", title="Clothing model"),
        tooltip=["page_2_clothing_model:N", "num_clicks:Q", "num_sessions:Q"],
    )
    .properties(height=330)
)

st.altair_chart(chart, use_container_width=True)

st.info(
    "Key insight: a small number of products receive much more attention. "
    "These products can be used as a simple popularity-based recommendation baseline."
)


# =========================
# 6. Colour popularity
# =========================

st.divider()
st.subheader("Colour popularity")

colour_popularity = (
    clicks.groupby("colour_name")
    .agg(
        num_clicks=("session_id", "size"),
        num_sessions=("session_id", "nunique"),
    )
    .reset_index()
    .sort_values("num_clicks", ascending=False)
)

colour_domain = list(plot_colour_map.keys())
colour_range = list(plot_colour_map.values())

chart = (
    alt.Chart(colour_popularity)
    .mark_bar(strokeWidth=1.5)
    .encode(
        x=alt.X("colour_name:N", sort="-y", title="Colour"),
        y=alt.Y("num_clicks:Q", title="Number of clicks"),
        color=alt.Color(
            "colour_name:N",
            scale=alt.Scale(domain=colour_domain, range=colour_range),
            legend=None,
        ),
        stroke=alt.condition(
            alt.datum.colour_name == "white",
            alt.value("black"),
            alt.value(None),
        ),
        tooltip=["colour_name:N", "num_clicks:Q", "num_sessions:Q"],
    )
    .properties(height=320)
)

st.altair_chart(chart, use_container_width=True)

st.info(
    "Key insight: black and blue dominate overall colour preferences. "
    "Colour is therefore a useful product attribute for personalization."
)


# =========================
# 7. Top colours per region
# =========================

st.divider()
st.subheader("Top colours by region")

selected_region = st.selectbox(
    "Select region",
    sorted(clicks["region"].unique()),
    index=(
        sorted(clicks["region"].unique()).index("Poland")
        if "Poland" in sorted(clicks["region"].unique())
        else 0
    ),
)

top_colours_region = (
    clicks[clicks["region"] == selected_region]
    .groupby("colour_name")
    .size()
    .reset_index(name="num_clicks")
    .sort_values("num_clicks", ascending=False)
    .head(5)
)

chart = (
    alt.Chart(top_colours_region)
    .mark_bar(strokeWidth=1.5)
    .encode(
        x=alt.X("colour_name:N", sort="-y", title="Colour"),
        y=alt.Y("num_clicks:Q", title="Number of clicks"),
        color=alt.Color(
            "colour_name:N",
            scale=alt.Scale(domain=colour_domain, range=colour_range),
            legend=None,
        ),
        stroke=alt.condition(
            alt.datum.colour_name == "white",
            alt.value("black"),
            alt.value(None),
        ),
        tooltip=["colour_name:N", "num_clicks:Q"],
    )
    .properties(height=280)
)

st.altair_chart(chart, use_container_width=True)


# =========================
# 8. Price distribution
# =========================

st.divider()
st.subheader("Price behaviour")

price_min = clicks["price"].min()
price_q1 = clicks["price"].quantile(0.25)
price_median = clicks["price"].median()
price_q3 = clicks["price"].quantile(0.75)
price_max = clicks["price"].max()
price_mean = clicks["price"].mean()
above_avg_share = clicks["price_2_int"].mean()

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Min price", f"{price_min:.0f}")
c2.metric("Q1", f"{price_q1:.0f}")
c3.metric("Median", f"{price_median:.0f}")
c4.metric("Q3", f"{price_q3:.0f}")
c5.metric("Max price", f"{price_max:.0f}")

st.write(f"Average viewed price: **{price_mean:.2f}**")
st.write(f"Share of above-average price views: **{above_avg_share:.1%}**")

price_chart = (
    alt.Chart(clicks[["price"]])
    .mark_bar(color=ACCENT)
    .encode(
        x=alt.X("price:Q", bin=alt.Bin(maxbins=30), title="Price"),
        y=alt.Y("count():Q", title="Number of clicks"),
        tooltip=["count():Q"],
    )
    .properties(height=300)
)

st.altair_chart(price_chart, use_container_width=True)

st.info(
    "Key insight: viewed products are mostly in the medium price range. "
    "The average and median viewed prices are close, so interest is not only driven by cheap products."
)


# =========================
# 9. Above-average price preference by region
# =========================

st.divider()
st.subheader("Above-average price views by region")

price_pref = (
    clicks.groupby("region")
    .agg(
        share_above_avg_price=("price_2_int", "mean"),
        avg_price=("price", "mean"),
        num_clicks=("session_id", "size"),
    )
    .reset_index()
    .sort_values("share_above_avg_price", ascending=False)
)

chart = (
    alt.Chart(price_pref)
    .mark_bar(color=ACCENT)
    .encode(
        x=alt.X(
            "share_above_avg_price:Q",
            title="Share above average price",
            axis=alt.Axis(format="%"),
        ),
        y=alt.Y("region:N", sort="-x", title=None),
        tooltip=[
            "region:N",
            alt.Tooltip("share_above_avg_price:Q", format=".1%"),
            alt.Tooltip("avg_price:Q", format=".2f"),
            "num_clicks:Q",
        ],
    )
    .properties(height=280)
)

st.altair_chart(chart, use_container_width=True)


# =========================
# 10. Session depth
# =========================

st.divider()
st.subheader("Session depth")

avg_clicks = sessions["num_clicks"].mean()
median_clicks = sessions["num_clicks"].median()
avg_max_page = sessions["max_page"].mean()
median_max_page = sessions["max_page"].median()
page1_only_share = (sessions["max_page"] == 1).mean()

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Avg clicks/session", f"{avg_clicks:.2f}")
c2.metric("Median clicks/session", f"{median_clicks:.0f}")
c3.metric("Avg max page", f"{avg_max_page:.2f}")
c4.metric("Median max page", f"{median_max_page:.0f}")
c5.metric("Page-1 only", f"{page1_only_share:.1%}")

page_depth = (
    sessions.groupby("max_page")
    .size()
    .reset_index(name="num_sessions")
    .sort_values("max_page")
)

chart = (
    alt.Chart(page_depth)
    .mark_bar(color=ACCENT)
    .encode(
        x=alt.X("max_page:O", title="Maximum page reached"),
        y=alt.Y("num_sessions:Q", title="Number of sessions"),
        tooltip=["max_page:O", "num_sessions:Q"],
    )
    .properties(height=300)
)

st.altair_chart(chart, use_container_width=True)

st.info(
    "Key insight: many users leave early. Almost 40% of sessions stop at page 1, "
    "so recommendations should appear early in the browsing session."
)


# =========================
# 11. Clicks per session
# =========================

st.divider()
st.subheader("Clicks per session")

clicks_session_chart = (
    alt.Chart(pd.DataFrame({"num_clicks": sessions["num_clicks"].clip(upper=50)}))
    .mark_bar(color=ACCENT)
    .encode(
        x=alt.X(
            "num_clicks:Q",
            bin=alt.Bin(maxbins=40),
            title="Clicks per session, clipped at 50",
        ),
        y=alt.Y("count():Q", title="Number of sessions"),
    )
    .properties(height=300)
)

st.altair_chart(clicks_session_chart, use_container_width=True)


# =========================
# 12. Unique products per session
# =========================

st.divider()
st.subheader("Unique products per session")

unique_products_chart = (
    alt.Chart(
        pd.DataFrame(
            {"num_unique_products": sessions["num_unique_products"].clip(upper=50)}
        )
    )
    .mark_bar(color=ACCENT)
    .encode(
        x=alt.X(
            "num_unique_products:Q",
            bin=alt.Bin(maxbins=40),
            title="Unique products per session, clipped at 50",
        ),
        y=alt.Y("count():Q", title="Number of sessions"),
    )
    .properties(height=300)
)

st.altair_chart(unique_products_chart, use_container_width=True)


# =========================
# 13. Category association rule
# =========================

st.divider()
st.subheader("Example category association rule")

st.markdown("""
    Example rule for Poland:

    **blouses → trousers**
    """)

# Unique category basket per session
category_basket = clicks[["session_id", "region", "category_name"]].drop_duplicates()

# Compute selected rule directly
poland_basket = category_basket[category_basket["region"] == "Poland"]

num_poland_sessions = poland_basket["session_id"].nunique()

blouse_sessions = set(
    poland_basket.loc[poland_basket["category_name"] == "blouses", "session_id"]
)

trouser_sessions = set(
    poland_basket.loc[poland_basket["category_name"] == "trousers", "session_id"]
)

both_sessions = blouse_sessions.intersection(trouser_sessions)

support = len(both_sessions) / num_poland_sessions
confidence = len(both_sessions) / len(blouse_sessions)
lift = confidence / (len(trouser_sessions) / num_poland_sessions)

rule_df = pd.DataFrame(
    {
        "Rule": ["blouses → trousers"],
        "Support": [support],
        "Confidence": [confidence],
        "Lift": [lift],
    }
)

c1, c2, c3 = st.columns(3)
c1.metric("Support", f"{support:.4f}")
c2.metric("Confidence", f"{confidence:.4f}")
c3.metric("Lift", f"{lift:.4f}")

st.dataframe(rule_df, use_container_width=True)

st.markdown(f"""
    - **Support**: {support:.2%} of Polish sessions contain both blouses and trousers.
    - **Confidence**: among Polish sessions that contain blouses, {confidence:.2%} also contain trousers.
    - **Lift**: blouses and trousers appear together {lift:.2f}× more often than expected by chance.
    """)


# =========================
# 14. Final EDA summary
# =========================

st.divider()
st.subheader("EDA summary")

top_region = region_activity.iloc[0]
top_category = category_popularity.iloc[0]
top_colour = colour_popularity.iloc[0]

st.markdown(f"""
    - The dataset contains **{total_clicks:,} clicks** and **{total_sessions:,} sessions**.
    - The most active region is **{top_region["region"]}** with **{top_region["num_sessions"]:,} sessions**.
    - The most viewed category is **{top_category["category_name"]}** with **{top_category["num_clicks"]:,} clicks**.
    - The most viewed colour is **{top_colour["colour_name"]}** with **{top_colour["num_clicks"]:,} clicks**.
    - The median session length is **{median_clicks:.0f} clicks**.
    - About **{page1_only_share:.1%}** of sessions stop at the first page.
    """)

st.success(
    "Main takeaway: region, category, colour, price, and early-session behaviour are important signals "
    "for the later ML and recommender system parts."
)
