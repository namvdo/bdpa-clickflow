# Clickstream Recommender

Next-click recommender on the 165k-row e-shop clickstream. Four models are evaluated under a leave-last-out split.

## Setup

- **Split**: a temporal train/test split. Sessions starting before month 7 are for training; sessions starting in month 7 or 8 are for testing. Within the test set, the last click is held out as ground truth.
- **Warm sessions** (17,031 training, 5,585 evaluable test): scored by ALS (fold-in), item k-NN, and Item2Vec.
- **Cold sessions** (3,320): scored by the item-similarity fallback.
- **Baseline**: popularity ranker, using the top-10 most-clicked training items after removing already-seen items.

## Models

- **Popularity**: global item popularity from the training split.
- **ALS** (Spark MLlib, implicit feedback): rank=20, lambda=1.0, alpha=80.
- **Item k-NN** (Sarwar et al., co-interaction): cosine similarity over l2-normalised item columns of the session-item matrix, top-50 neighbours per seed. Score for a candidate item is the weighted sum of similarities to items in the user's history.
- **Item2Vec / Word2Vec**: Spark Word2Vec with vectorSize=32, windowSize=3, maxIter=20, minCount=1. It learns item embeddings from local product context inside ordered sessions. Recommendations use **exponential weighted pooling** (decay=0.8) to give more importance to recent clicks, then rank candidate products by cosine similarity to that session vector.
- **Cold fallback**: same k-NN matrix, restricted to the single seed item.

## Metrics

- **MRR@10**: reciprocal rank of the held-out next click.
- **Recall@10**: whether the held-out next click appears in the top 10.
- **Coverage**: share of the 217-item catalogue recommended at least once.
- **Novelty@10**: mean self-information, `avg(-log2(p(item)))`, over recommended items, where `p(item)` is training-set item popularity. Higher novelty means the model is recommending less obvious items.

Improvement is measured relative to the popularity baseline. For coverage, the delta is percentage points; for novelty, the delta is bits.

## Results

**Warm sessions (n=5,585):**

| Metric | Popularity | Item k-NN | Improvement | ALS | Improvement | Item2Vec | Improvement |
| :-- | --: | --: | --: | --: | --: | --: | --: |
| MRR@10 | 0.0415 | 0.0894 | +115.4% (+0.0479) | **0.0867** | +108.9% (+0.0452) | 0.0747 | +80.0% (+0.0332) |
| Recall@10 | 0.1192 | 0.2312 | +93.9% (+0.1120) | **0.2337** | **+96.1% (+0.1145)** | 0.2015 | +69.0% (+0.0823) |
| Coverage | 22.12% | 98.16% | +76.04 pp | 97.24% | +75.12 pp | **100.00%** | **+77.88 pp** |
| Novelty@10 | 6.0566 | 7.1015 | +1.0449 bits | 7.3699 | +1.3133 bits | **7.6799** | **+1.6233 bits** |

ALS and item k-NN remain the strongest accuracy models, but Item2Vec gives the highest novelty and full catalogue coverage while still beating the popularity baseline on MRR and recall. That is the useful signal here: the sequence-aware model is not only re-ranking the head of the catalogue.

**Cold sessions (n=3,320):**

| Metric | Popularity | Item-sim fallback | Improvement |
| :-- | --: | --: | --: |
| MRR@10 | 0.0561 | **0.1437** | **+156.1% (+0.0876)** |
| Recall@10 | 0.1623 | **0.3434** | **+111.6% (+0.1811)** |
| Coverage | 5.07% | **98.16%** | **+93.09 pp** |
| Novelty@10 | 6.0038 | **7.1905** | **+1.1867 bits** |

For one-click sessions, the item-similarity fallback turns the observed item into a strong seed instead of defaulting to globally popular products.
