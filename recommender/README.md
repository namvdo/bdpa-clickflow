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

Improvement is measured relative to the popularity baseline:
`(model score - popularity score) / popularity score`. The value in parentheses is the actual numeric change from the baseline; coverage change is shown in percentage points.

**Warm sessions (n=15,664):**

| Metric    | Popularity | Item k-NN | Improvement vs popularity | ALS | Improvement vs popularity |
| :-------- | :--------- | :-------- | :------------------------ | :-- | :------------------------ |
| NDCG@10   | 0.0512     | 0.1105    | +115.8% (+0.0593)         | **0.1110** | **+116.8% (+0.0598)** |
| MRR@10    | 0.0360     | **0.0793** | **+120.3% (+0.0433)**    | 0.0775 | +115.3% (+0.0415) |
| Recall@10 | 0.1027     | 0.2138    | +108.2% (+0.1111)         | **0.2221** | **+116.3% (+0.1194)** |
| Coverage  | 23.04%     | **98.16%** | **+326.0% (+75.12 pp)**  | 97.70% | +324.0% (+74.66 pp) |

ALS and item k-NN are basically tied. ALS has the largest warm-session NDCG@10 and Recall@10 lift, while item k-NN slightly leads on MRR@10 and coverage. With 91% of (session, item) pairs at click count one, the ALS confidence weight collapses to "viewed or not", which is essentially what cosine similarity in k-NN already captures.

**Cold sessions (n=3,320):**

| Metric    | Popularity | Item-sim fallback | Improvement vs popularity |
| :-------- | :--------- | :---------------- | :------------------------ |
| NDCG@10   | 0.0807     | **0.1903**        | **+135.8% (+0.1096)**     |
| Recall@10 | 0.1623     | **0.3434**        | **+111.6% (+0.1811)**     |
| Coverage  | 5.07%      | **98.16%**        | **+1,836.1% (+93.09 pp)** |

The cold fallback is the largest jump in the system. The single observed item is a clean seed for the lookup, and the matrix already encodes that signal from warm sessions — no extra training needed.
