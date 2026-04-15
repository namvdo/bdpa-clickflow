# Task 2 — Exploratory Analysis and Business Insights

This section describes the main steps carried out in the exploratory analysis of the cleaned clickstream dataset. The goal is to summarize user browsing behavior, compare geographic groups, and extract business insights from sessions, products, prices, and co-view patterns.

## 1. Session-level feature engineering

The click-level dataset is aggregated to the session level. For each session, the following summary features are computed:

- number of clicks
- number of distinct products viewed
- number of distinct main categories visited
- average viewed price
- deepest page reached
- maximum click order within the session
- share of viewed items that are above their category-average price

This produces a compact session-level dataset for later analysis.

## 2. Session statistics by region

The session-level table is further aggregated by region. For each region, the analysis computes:

- total number of sessions
- average clicks per session
- average number of unique products per session
- average session-level viewed price
- average share of above-average-price products
- share of sessions that remain only on page 1

This provides a high-level summary of engagement and browsing intensity across regions.

## 3. Main category analysis by region

The dataset is grouped by `region` and `page_1_main_category` to count how often each main category is viewed in each region. The results are reshaped into a pivot table and visualized as a grouped bar chart. This shows regional differences in product category interest.

## 4. Product popularity analysis

The dataset is grouped by `region` and `page_2_clothing_model` to identify the most frequently viewed product models in each region. In addition, the top 10 most viewed product models overall are computed and visualized. This helps identify the most popular items in the store.

## 5. Colour preference analysis

The dataset is grouped by `region` and `colour` to count colour frequencies. A window function is then used to rank colours within each region, and the top 5 colours per region are selected. This highlights the most preferred product colours in each geographic segment.

## 6. Price preference analysis

The analysis measures price-related behavior by region using two indicators:

- the share of viewed products whose price is above the category average
- the average viewed price

This shows whether users from different regions tend to browse relatively more expensive items.

## 7. Page-depth analysis

Using the session-level table, page-depth behavior is summarized by region. The following metrics are computed:

- average maximum page reached
- share of sessions that stay only on page 1

This indicates how deeply users browse the website and whether some groups are more likely to stop early.

## 8. Basket construction for session-based product analysis

A reduced dataset called `basket_df` is created containing unique combinations of:

- session ID
- region
- product model

Duplicate product views within the same session are removed. This table is used to study co-viewed products within sessions.

## 9. Product pair generation within sessions

A self-join is applied to `basket_df` so that product pairs viewed in the same session can be generated. Each resulting row represents a unique pair of products that co-occurred within one session. This is the basis for association-style analysis.

## 10. Pair, session, and item counts

Three supporting count tables are created:

- `pair_counts`: how many sessions contain a given product pair
- `session_counts`: total number of sessions per region
- `item_counts`: how many sessions contain each individual product

These counts are needed to compute association measures such as support, confidence, and lift.

## 11. Association rule metrics

For each product pair, the following association-style measures are computed:

- **Support**: how often the pair appears among all sessions in the region
- **Confidence**: how often one product appears when the other is present
- **Lift**: how much stronger the co-occurrence is compared to random expectation

These measures are used to identify meaningful product relationships and potential recommendation patterns.

## 12. Output of ranked product association rules

The final rules table is sorted by support so that the most common co-viewed product pairs appear first. This provides interpretable business insight into browsing patterns and product combinations.

## Summary

In this analysis, the cleaned clickstream data was aggregated at both click level and session level. We examined regional differences in session counts, category preferences, product popularity, colour choices, price behavior, and browsing depth. Finally, we constructed session-based product baskets and performed an association-style analysis of co-viewed products using support, confidence, and lift. Together, these steps provide a business-oriented view of customer browsing behavior in the online store.