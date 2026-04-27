# Clickstream Recommender System

Predictive pipeline for e-shop product recommendations using a 165k-row clickstream dataset.

### 1. Feature Aggregation

Converts raw clicks into implicit interaction scores. Multiple clicks on a single item are weighted as stronger preference signals for the ALS optimizer.

### 2. Temporal Evaluation Split

Uses a **Leave-Last-Out** strategy per session:

- **Train**: n-1 interactions.
- **Test**: The final interaction (ground truth).

### 3. Dual-Track Modeling

- **Warm Sessions (15,664)**: Matrix factorization (ALS) learns latent factors for personalized ranking.
- **Cold Sessions (3,320)**: Session-Sequential Warm-up using item-to-item transition probabilities to mitigate the lack of user history.

## Performance Evaluation

| Metric               | Popularity Baseline | ALS Model              | Delta      |
| :------------------- | :------------------ | :--------------------- | :--------- |
| **NDCG@10**          | 0.0606              | **0.1338**             | **+120%**  |
| **Recall@10**        | 0.1213              | **0.2767**             | **+128%**  |
| **Catalog Coverage** | 4.61% (10 items)    | **97.70% (212 items)** | **+93.1%** |
