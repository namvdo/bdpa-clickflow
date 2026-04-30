# Clickstream Recommender

Next-click recommender on the 165k-row e-shop clickstream. Four models are evaluated under a leave-last-out split.

## Setup

- **Split**: hold the last click of each session as ground truth; train on the rest.
- **Warm sessions** (15,664, at least 2 training clicks): scored by ALS, item k-NN, and Item2Vec.
- **Cold sessions** (3,320, exactly 1 training click): scored by the item-similarity fallback and Item2Vec from the single observed click.
- **Baseline**: popularity ranker, using the top-10 most-clicked training items after removing already-seen items.

## Models

- **Popularity**: global item popularity from the training split.
- **ALS** (Spark MLlib, implicit feedback): rank=20, lambda=1.0, alpha=80.
- **Item k-NN** (Sarwar et al., co-interaction): cosine similarity over l2-normalised item columns of the session-item matrix, top-50 neighbours per seed. Score for a candidate item is the weighted sum of similarities to items in the user's history.
- **Item2Vec / Word2Vec**: Spark Word2Vec with vectorSize=32, windowSize=3, maxIter=20, minCount=1. It learns item embeddings from local product context inside ordered sessions. Recommendations average the embeddings of the products clicked so far, then rank candidate products by cosine similarity to that session vector.
- **Cold fallback**: same k-NN matrix, restricted to the single seed item.

## Metrics

- **MRR@10**: reciprocal rank of the held-out next click.
- **Recall@10**: whether the held-out next click appears in the top 10.
- **Coverage**: share of the 217-item catalogue recommended at least once.
- **Novelty@10**: mean self-information, `avg(-log2(p(item)))`, over recommended items, where `p(item)` is training-set item popularity. Higher novelty means the model is recommending less obvious items.

Improvement is measured relative to the popularity baseline. For coverage, the delta is percentage points; for novelty, the delta is bits.

## Results

**Warm sessions (n=15,664):**

| Metric | Popularity | Item k-NN | Improvement | ALS | Improvement | Item2Vec | Improvement |
| :-- | --: | --: | --: | --: | --: | --: | --: |
| MRR@10 | 0.0360 | **0.0793** | **+120.3% (+0.0433)** | 0.0775 | +115.3% (+0.0415) | 0.0665 | +84.7% (+0.0305) |
| Recall@10 | 0.1027 | 0.2138 | +108.2% (+0.1111) | **0.2221** | **+116.3% (+0.1194)** | 0.1934 | +88.3% (+0.0907) |
| Coverage | 23.04% | 98.16% | +75.12 pp | 97.70% | +74.66 pp | **100.00%** | **+76.96 pp** |
| Novelty@10 | 6.0467 | 7.1015 | +1.0548 bits | 7.4114 | +1.3647 bits | **7.6799** | **+1.6332 bits** |

ALS and item k-NN remain the strongest accuracy models, but Item2Vec gives the highest novelty and full catalogue coverage while still beating the popularity baseline on MRR and recall. That is the useful signal here: the sequence-aware model is not only re-ranking the head of the catalogue.

**Cold sessions (n=3,320):**

| Metric | Popularity | Item-sim fallback | Improvement | Item2Vec | Improvement |
| :-- | --: | --: | --: | --: | --: |
| MRR@10 | 0.0561 | **0.1437** | **+156.1% (+0.0876)** | 0.1104 | +96.8% (+0.0543) |
| Recall@10 | 0.1623 | **0.3434** | **+111.6% (+0.1811)** | 0.3310 | +103.9% (+0.1687) |
| Coverage | 5.07% | 98.16% | +93.09 pp | **100.00%** | **+94.93 pp** |
| Novelty@10 | 6.0038 | 7.1905 | +1.1867 bits | **7.3400** | **+1.3362 bits** |

The item-similarity fallback is still best on cold-session accuracy, while Item2Vec is close on recall and leads on novelty/coverage. For one-click sessions, both non-popularity models turn the observed item into a useful seed instead of defaulting to globally popular products.
