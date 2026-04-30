import streamlit as st
import pandas as pd
import altair as alt
from pathlib import Path

st.set_page_config(
    page_title="EDA Insights",
    layout="wide",
)

ACCENT = "#ff0000"

# =========================
# Paths
# =========================

output_dir = Path(__file__).parent.parent.parent / "output"
cleaned_path = output_dir / "cleaned.parquet"
figures_path = Path(__file__).parent / "figures"

if not cleaned_path.exists():
    st.write("Pipeline output not found. Run the scripts in pipeline/ first.")
    st.stop()

# =========================
# Load data
# =========================

clicks = pd.read_parquet(cleaned_path)

# =========================
# Constants / mappings
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
# Preprocessing
# =========================

clicks = clicks.copy()

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
        return "Other EU"
    elif country in non_eu_europe_codes:
        return "Europe non EU"
    elif country in outside_europe_codes:
        return "Outside Europe"
    else:
        return "Unidentified"


clicks["region"] = clicks["country_int"].apply(assign_region)
clicks["price_2_int"] = clicks["price_2"].astype(int)

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
st.caption("Clickstream-Based User Behavior Analytics")

# =========================
# 1. Region-based activity
# =========================

st.divider()
st.header("1. Region-based activity: the dataset is dominated by Poland")

region_activity = (
    clicks.groupby("region")
    .agg(
        clicks=("session_id", "size"),
        sessions=("session_id", "nunique"),
    )
    .reset_index()
)

region_order = [
    "Poland",
    "Other EU",
    "Unidentified",
    "Europe non EU",
    "Outside Europe",
]

region_activity["region"] = pd.Categorical(
    region_activity["region"],
    categories=region_order,
    ordered=True,
)

region_activity = region_activity.sort_values("region")

left, right = st.columns([1, 2])

with left:
    st.markdown(f"""
        - Poland: **{int(region_activity.loc[region_activity["region"] == "Poland", "sessions"].iloc[0]):,} sessions**, **{int(region_activity.loc[region_activity["region"] == "Poland", "clicks"].iloc[0]):,} clicks**
        - Other EU: **{int(region_activity.loc[region_activity["region"] == "Other EU", "sessions"].iloc[0]):,} sessions**, **{int(region_activity.loc[region_activity["region"] == "Other EU", "clicks"].iloc[0]):,} clicks**
        - Unidentified: **{int(region_activity.loc[region_activity["region"] == "Unidentified", "sessions"].iloc[0]):,} sessions**
        - Europe non EU: **{int(region_activity.loc[region_activity["region"] == "Europe non EU", "sessions"].iloc[0]):,} sessions**
        - Outside Europe: **{int(region_activity.loc[region_activity["region"] == "Outside Europe", "sessions"].iloc[0]):,} sessions**
        """)

with right:
    chart = (
        alt.Chart(region_activity)
        .mark_bar(color=ACCENT)
        .encode(
            x=alt.X("region:N", sort=region_order, title="Region"),
            y=alt.Y("sessions:Q", title="Number of sessions"),
            tooltip=["region:N", "sessions:Q", "clicks:Q"],
        )
        .properties(height=350)
    )
    st.altair_chart(chart, use_container_width=True)

st.info(
    "Main point: most activity comes from Poland, so later ML and recommender-system results "
    "may be strongly influenced by Polish user behavior."
)

# =========================
# 2. Category popularity
# =========================

st.divider()
st.header("2. Category popularity: trousers are most viewed, but all categories matter")

category_popularity = (
    clicks.groupby("category_name")
    .agg(
        clicks=("session_id", "size"),
        sessions=("session_id", "nunique"),
    )
    .reset_index()
    .sort_values("clicks", ascending=False)
)

left, right = st.columns([1, 2])

with left:
    for _, row in category_popularity.iterrows():
        st.markdown(f"- {row['category_name']}: **{int(row['clicks']):,} clicks**")

with right:
    chart = (
        alt.Chart(category_popularity)
        .mark_bar(color=ACCENT)
        .encode(
            x=alt.X("category_name:N", sort="-y", title="Category"),
            y=alt.Y("clicks:Q", title="Number of clicks"),
            tooltip=["category_name:N", "clicks:Q", "sessions:Q"],
        )
        .properties(height=350)
    )
    st.altair_chart(chart, use_container_width=True)

st.info(
    "Main point: trousers are the leading category, but sale, blouses, and skirts also have strong activity."
)

# =========================
# 3. Regional category differences
# =========================

st.divider()
st.header("3. Regional category differences")

category_by_region = (
    clicks.groupby(["region", "category_name"]).size().reset_index(name="clicks")
)

poland_cat = category_by_region[category_by_region["region"] == "Poland"].sort_values(
    "clicks", ascending=False
)

other_eu_cat = category_by_region[
    category_by_region["region"] == "Other EU"
].sort_values("clicks", ascending=False)

left, middle, right = st.columns([1, 1, 2])

with left:
    st.subheader("Poland")
    for _, row in poland_cat.iterrows():
        st.markdown(f"- {row['category_name']}: **{int(row['clicks']):,}**")

with middle:
    st.subheader("Other EU")
    for _, row in other_eu_cat.iterrows():
        st.markdown(f"- {row['category_name']}: **{int(row['clicks']):,}**")

with right:
    plot_df = category_by_region[
        category_by_region["region"].isin(["Poland", "Other EU"])
    ]

    chart = (
        alt.Chart(plot_df)
        .mark_bar()
        .encode(
            x=alt.X("region:N", title="Region"),
            y=alt.Y("clicks:Q", title="Number of clicks"),
            color=alt.Color("category_name:N", title="Category"),
            tooltip=["region:N", "category_name:N", "clicks:Q"],
        )
        .properties(height=350)
    )
    st.altair_chart(chart, use_container_width=True)

st.info(
    "Main point: sale is much more important in Poland, while Other EU is more dominated by trousers and blouses."
)

# =========================
# 4. Product-level popularity
# =========================


st.divider()
st.header("4. Product-level popularity: top products can guide recommendation")

top_products = (
    clicks.groupby("page_2_clothing_model")
    .agg(
        clicks=("session_id", "size"),
        sessions=("session_id", "nunique"),
    )
    .reset_index()
    .sort_values("clicks", ascending=False)
    .head(10)
)

left, right = st.columns([1, 2])

with left:
    for _, row in top_products.head(5).iterrows():
        st.markdown(
            f"- {row['page_2_clothing_model']}: **{int(row['clicks']):,} clicks**"
        )

with right:
    chart = (
        alt.Chart(top_products)
        .mark_bar(color=ACCENT)
        .encode(
            x=alt.X("page_2_clothing_model:N", sort="-y", title="Clothing model"),
            y=alt.Y("clicks:Q", title="Number of clicks"),
            tooltip=["page_2_clothing_model:N", "clicks:Q", "sessions:Q"],
        )
        .properties(height=350)
    )
    st.altair_chart(chart, use_container_width=True)

st.info(
    "Main point: top products provide a simple popularity-based recommendation baseline."
)

# =========================
# 4.1 Zipf-like product frequency analysis
# =========================

st.divider()
st.header("4.1 Product frequency analysis: popularity follows a Zipf-like pattern")

st.markdown("""
    Product clicks are not evenly distributed. A small number of clothing models receive many clicks,
    while most products receive relatively few. This kind of long-tail behavior is common in
    recommendation problems.
    """)

col1, col2, col3, col4 = st.columns(4)

col1.metric("Zipf R²", "0.923")
col2.metric("Alpha", "0.95")
col3.metric("50% of click volume", "48 items")
col4.metric("90% sessions", "< 30 clicks")

st.info(
    "Main point: popularity is a strong baseline for recommendation, but it can also bias the system "
    "toward already popular products and ignore long-tail items."
)

st.markdown("### Main conclusions")

st.markdown("""
    - Product clicks approximately follow a **Zipf-like distribution**.
    - The fitted Zipf model explains the ranking trend well, with **R² ≈ 0.92**.
    - A small number of products account for a large share of clicks.
    - Around **48 products account for 50% of all clicks**, showing strong concentration.
    - The tail contains many low-frequency products, which may be harder for ML/recommender models to learn.
    - Session lengths are short: **90% of sessions contain fewer than 30 clicks**.
    """)

st.markdown("### Visual analysis")

images = {
    "Zipf fit — original scale": "zipfs_curve.png",
    "Zipf fit — log-log scale": "zipfs_fit_in_log_space.png",
    "Residuals vs rank": "residuals_vs_rank.png",
    "Cumulative click probability": "cumulative_clicks.png",
    "Session length distribution": "probability_+_cumulative_distribution_of_session_lengths.png",
}

selected = st.radio(
    "Select figure",
    list(images.keys()),
    horizontal=True,
)

selected_image_path = figures_path / images[selected]

if selected_image_path.exists():
    st.image(selected_image_path, use_container_width=True)
else:
    st.warning(
        f"Figure not found: {selected_image_path}. "
        "Make sure the figures are inside streamlit_app/pages/figures/."
    )

with st.expander("Interpretation"):
    st.markdown("""
        The frequency distribution confirms that user attention is concentrated on a limited set of products.
        The Zipf model fits the overall trend well, especially for the main ranked products.

        The residual plot shows stronger deviations in the tail, where rare products have very low click counts.
        This means rare products are noisier and may need filtering, grouping, or separate treatment.

        For the recommender system, this is important because a simple popularity-based model may perform well,
        but it may over-recommend popular products and reduce product diversity.
        """)

# =========================
# 5. Colour preference
# =========================

st.divider()
st.header("5. Colour preference: black and blue dominate")

colour_popularity = (
    clicks.groupby("colour_name")
    .agg(
        clicks=("session_id", "size"),
        sessions=("session_id", "nunique"),
    )
    .reset_index()
    .sort_values("clicks", ascending=False)
)

colour_domain = list(plot_colour_map.keys())
colour_range = list(plot_colour_map.values())

chart = (
    alt.Chart(colour_popularity)
    .mark_bar(strokeWidth=1.5)
    .encode(
        x=alt.X("colour_name:N", sort="-y", title="Colour"),
        y=alt.Y("clicks:Q", title="Number of clicks"),
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
        tooltip=["colour_name:N", "clicks:Q", "sessions:Q"],
    )
    .properties(height=430)
)

st.altair_chart(chart, use_container_width=True)

top_two_colours = colour_popularity.head(2)
st.markdown(f"""
    - {top_two_colours.iloc[0]["colour_name"]}: **{int(top_two_colours.iloc[0]["clicks"]):,} clicks**
    - {top_two_colours.iloc[1]["colour_name"]}: **{int(top_two_colours.iloc[1]["clicks"]):,} clicks**
    """)

st.info(
    "Main point: black and blue dominate colour preference, so colour is a useful product feature for personalization."
)

# =========================
# 6. Price behavior
# =========================

st.divider()
st.header("6. Price behavior: users mostly view medium-price items")

price_min = clicks["price"].min()
price_q1 = clicks["price"].quantile(0.25)
price_median = clicks["price"].median()
price_q3 = clicks["price"].quantile(0.75)
price_max = clicks["price"].max()
price_mean = clicks["price"].mean()
above_avg_share = clicks["price_2_int"].mean()

left, right = st.columns([1, 2])

with left:
    st.markdown(f"""
        - min price: **{price_min:.0f}**
        - Q1: **{price_q1:.0f}**
        - median: **{price_median:.0f}**
        - Q3: **{price_q3:.0f}**
        - max: **{price_max:.0f}**
        - average: **{price_mean:.1f}**
        - share above-average price views: **{above_avg_share:.1%}**
        """)

with right:
    chart = (
        alt.Chart(clicks[["price"]])
        .mark_bar(color=ACCENT)
        .encode(
            x=alt.X("price:Q", bin=alt.Bin(maxbins=30), title="Price"),
            y=alt.Y("count():Q", title="Number of clicks"),
            tooltip=["count():Q"],
        )
        .properties(height=350)
    )
    st.altair_chart(chart, use_container_width=True)

st.info(
    "Main point: users mostly view medium-price items. Interest is not driven only by very cheap products."
)

# =========================
# 7. Session depth
# =========================

st.divider()
st.header("7. Session depth: many users leave early!")

avg_clicks = sessions["num_clicks"].mean()
median_clicks = sessions["num_clicks"].median()
avg_max_page = sessions["max_page"].mean()
median_max_page = sessions["max_page"].median()
page1_only_share = (sessions["max_page"] == 1).mean()

page_depth = (
    sessions.groupby("max_page")
    .size()
    .reset_index(name="sessions")
    .sort_values("max_page")
)

left, right = st.columns([1, 2])

with left:
    st.markdown(f"""
        - Average clicks per session: **{avg_clicks:.2f}**
        - Page-1-only sessions: **{page1_only_share:.1%}**
        - Median clicks per session: **{median_clicks:.0f}**
        - Average max page: **{avg_max_page:.2f}**
        - Median max page: **{median_max_page:.0f}**
        """)

with right:
    chart = (
        alt.Chart(page_depth)
        .mark_bar(color=ACCENT)
        .encode(
            x=alt.X("max_page:O", title="Maximum page reached in session"),
            y=alt.Y("sessions:Q", title="Number of sessions"),
            tooltip=["max_page:O", "sessions:Q"],
        )
        .properties(height=350)
    )
    st.altair_chart(chart, use_container_width=True)

st.info(
    "Main point: many users leave early, so recommendations should appear early in the browsing session."
)

# =========================
# 8. Category associations
# =========================

st.divider()
st.header("8. Category associations: suggesting recommender systems")

st.markdown("""
    Here, each session is treated as a basket of viewed product categories.
    Association rules help us understand which categories are commonly viewed together.
    """)

# Unique category basket per session
category_basket = clicks[["session_id", "region", "category_name"]].drop_duplicates()

# =========================
# Build category association rules
# =========================

category_pairs = []

for (region, session_id), group in category_basket.groupby(["region", "session_id"]):
    categories = sorted(group["category_name"].unique())

    for i in range(len(categories)):
        for j in range(len(categories)):
            if i != j:
                category_pairs.append(
                    {
                        "region": region,
                        "session_id": session_id,
                        "antecedent": categories[i],
                        "consequent": categories[j],
                    }
                )

category_pairs_df = pd.DataFrame(category_pairs)

if category_pairs_df.empty:
    st.warning("No category pairs found.")
else:
    # Count pair sessions
    pair_counts = (
        category_pairs_df.groupby(["region", "antecedent", "consequent"])
        .agg(pair_sessions=("session_id", "nunique"))
        .reset_index()
    )

    # Count total sessions per region
    region_session_counts = (
        category_basket.groupby("region")
        .agg(num_sessions=("session_id", "nunique"))
        .reset_index()
    )

    # Count sessions containing each category
    category_session_counts = (
        category_basket.groupby(["region", "category_name"])
        .agg(category_sessions=("session_id", "nunique"))
        .reset_index()
    )

    # Join counts for antecedent
    rules_df = pair_counts.merge(region_session_counts, on="region", how="left")

    rules_df = rules_df.merge(
        category_session_counts.rename(
            columns={
                "category_name": "antecedent",
                "category_sessions": "antecedent_sessions",
            }
        ),
        on=["region", "antecedent"],
        how="left",
    )

    # Join counts for consequent
    rules_df = rules_df.merge(
        category_session_counts.rename(
            columns={
                "category_name": "consequent",
                "category_sessions": "consequent_sessions",
            }
        ),
        on=["region", "consequent"],
        how="left",
    )

    # Calculate support, confidence, lift
    rules_df["support"] = rules_df["pair_sessions"] / rules_df["num_sessions"]

    rules_df["confidence"] = rules_df["pair_sessions"] / rules_df["antecedent_sessions"]

    rules_df["consequent_probability"] = (
        rules_df["consequent_sessions"] / rules_df["num_sessions"]
    )

    rules_df["lift"] = rules_df["confidence"] / rules_df["consequent_probability"]

    rules_df["rule"] = rules_df["antecedent"] + " → " + rules_df["consequent"]

    # =========================
    # Main selected rule: Poland, blouses → trousers
    # =========================

    selected_rule = rules_df[
        (rules_df["region"] == "Poland")
        & (rules_df["antecedent"] == "blouses")
        & (rules_df["consequent"] == "trousers")
    ].iloc[0]

    st.subheader("Main example rule for Poland: blouses → trousers")

    c1, c2, c3 = st.columns(3)
    c1.metric("Support", f"{selected_rule['support']:.4f}")
    c2.metric("Confidence", f"{selected_rule['confidence']:.4f}")
    c3.metric("Lift", f"{selected_rule['lift']:.4f}")

    st.markdown(f"""
        - **Support**: how often both appear together  
          **{selected_rule["support"]:.2%}** of Polish sessions contain both.

        - **Confidence**: how often trousers appear when blouses appear  
          **{selected_rule["confidence"]:.2%}** of blouse sessions also contain trousers.

        - **Lift**: strength beyond random co-occurrence  
          **{selected_rule["lift"]:.2f}×** more likely than expected by chance.
        """)

    # =========================
    # Table of other extracted category rules
    # =========================

    st.subheader("Other extracted category association rules")

    selected_region_for_rules = st.selectbox(
        "Select region for association rules",
        sorted(rules_df["region"].unique()),
        index=(
            sorted(rules_df["region"].unique()).index("Poland")
            if "Poland" in sorted(rules_df["region"].unique())
            else 0
        ),
        key="category_rule_region",
    )

    display_rules = (
        rules_df[rules_df["region"] == selected_region_for_rules]
        .sort_values(["support", "lift"], ascending=[False, False])
        .copy()
    )

    display_rules = display_rules[
        [
            "rule",
            "pair_sessions",
            "support",
            "confidence",
            "lift",
        ]
    ]

    display_rules["support"] = display_rules["support"].round(4)
    display_rules["confidence"] = display_rules["confidence"].round(4)
    display_rules["lift"] = display_rules["lift"].round(4)

    st.dataframe(
        display_rules,
        use_container_width=True,
        hide_index=True,
    )

    st.info(
        "Support shows how common the pair is, confidence shows how often the consequent appears "
        "when the antecedent appears, and lift shows whether the co-occurrence is stronger than random chance."
    )

st.success(
    "EDA takeaway: region, category, product popularity, colour, price, and early-session behavior "
    "provide useful signals for the machine learning and recommender-system stages."
)
