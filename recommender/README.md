# Clickstream Recommender

Next-click recommender on the 165k-row e-shop clickstream. Three models, all evaluated under a leave-last-out split.

## Setup

- **Split**: hold the last click of each session as ground truth; train on the rest.
- **Warm sessions** (15,664, $\geq 2$ training clicks): scored by ALS and item k-NN.
- **Cold sessions** (3,320, exactly 1 training click): scored by the same item-item similarity matrix using the single observed click as the seed.
- **Baseline**: popularity ranker (top-10 most-clicked items, same list for everyone).

## Models

- **ALS** (Spark MLlib, implicit feedback): rank=20, $\lambda$=1.0, $\alpha$=80. Picked by grid search on NDCG@10.
- **Item k-NN** (Sarwar et al., co-interaction): cosine similarity over $\ell_2$-normalised item columns of the session-item matrix, top-50 neighbours per seed. Score for a candidate item is the weighted sum of similarities to items in the user's history.
- **Cold fallback**: same k-NN matrix, restricted to the single seed item.

## Results (top-10)

**Warm sessions (n=15,664):**

| Metric    | Popularity | Item k-NN  | ALS        |
| :-------- | :--------- | :--------- | :--------- |
| NDCG@10   | 0.0512     | 0.1105     | **0.1110** |
| MRR@10    | 0.0360     | **0.0793** | 0.0775     |
| Recall@10 | 0.1027     | 0.2138     | **0.2221** |
| Coverage  | 23.04%     | **98.16%** | 97.70%     |

ALS and item k-NN are basically tied. Both roughly double the baseline on every ranking metric. With 91% of (session, item) pairs at click count one, the ALS confidence weight collapses to "viewed or not", which is essentially what cosine similarity in k-NN already captures.

**Cold sessions (n=3,320):**

| Metric    | Popularity | Item-sim fallback | $\Delta$ |
| :-------- | :--------- | :---------------- | :------- |
| NDCG@10   | 0.0807     | **0.1903**        | +136%    |
| Recall@10 | 0.1623     | **0.3434**        | +112%    |
| Coverage  | 5.07%      | **98.16%**        | +93.1 pp |

The cold fallback is the largest jump in the system. The single observed item is a clean seed for the lookup, and the matrix already encodes that signal from warm sessions — no extra training needed.
