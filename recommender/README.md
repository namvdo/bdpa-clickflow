# Clickstream Recommender

Next-click recommender on the 165k-row e-shop clickstream. Four models are evaluated under a leave-last-out split.

## Setup

- **Split**: a temporal train/test split. Sessions starting before month 7 are for training; sessions starting in month 7 or 8 are for testing. Within the test set, the last click is held out as ground truth.
- **Warm sessions** (17,031 training, 5,585 evaluable test): scored by ALS (fold-in), item k-NN, and Item2Vec.
- **Cold sessions** (3,320): scored by the item-similarity fallback and Item2Vec.
- **Baseline**: popularity ranker, using the top-10 most-clicked training items after removing already-seen items.

## Models

- **Popularity**: global item popularity from the training split.
- **ALS** (Spark MLlib, implicit feedback): rank=20, lambda=0.1, alpha=80.
- **Item k-NN** (Sarwar et al., co-interaction): cosine similarity over l2-normalised item columns of the session-item matrix, top-50 neighbours per seed. Score for a candidate item is the weighted sum of similarities to items in the user's history.
- **Item2Vec / Word2Vec**: Spark Word2Vec with vectorSize=32, windowSize=5, maxIter=20, minCount=5. It learns item embeddings from local product context inside ordered sessions. Recommendations use **exponential weighted pooling** (decay=0.8) to give more importance to recent clicks, then rank candidate products by cosine similarity to that session vector.
- **Cold fallback**: same k-NN matrix, restricted to the single seed item.

## Metrics

- **MRR@10**: reciprocal rank of the held-out next click.
- **Recall@10**: whether the held-out next click appears in the top 10.
- **Coverage**: share of the 217-item catalogue recommended at least once.
- **Novelty@10**: mean self-information, `avg(-log2(p(item)))`, over recommended items, where `p(item)` is training-set item popularity. Higher novelty means the model is recommending less obvious items.

Improvement is measured relative to the popularity baseline. For coverage, the delta is percentage points; for novelty, the delta is bits.

## Results

**Warm sessions (n=5,585):**

| Metric     | Popularity |  Item k-NN |       Improvement |    ALS |       Improvement |   Item2Vec |           Improvement |
| :--------- | ---------: | ---------: | ----------------: | -----: | ----------------: | ---------: | --------------------: |
| MRR@10     |     0.0415 |     0.0966 | +132.8% (+0.0551) | 0.0867 | +108.9% (+0.0452) | **0.1036** | **+149.6% (+0.0621)** |
| Recall@10  |     0.1192 |     0.2460 | +106.4% (+0.1268) | 0.2337 |  +96.1% (+0.1145) | **0.2806** | **+135.4% (+0.1614)** |
| Coverage   |     22.12% | **98.16%** |     **+76.04 pp** | 97.24% |         +75.12 pp | **98.16%** |         **+76.04 pp** |
| Novelty@10 |     6.0566 |     7.1585 |      +1.1019 bits | 7.3699 |      +1.3133 bits | **7.6716** |      **+1.6150 bits** |

Item2Vec and item k-NN are the strongest models for both accuracy and personalization. Item2Vec, in particular, provides the highest MRR, recall, and novelty, effectively capturing local sequence context while maintaining high catalogue coverage.

**Cold sessions (n=3,320):**

| Metric     | Popularity | Item-sim fallback |           Improvement |
| :--------- | ---------: | ----------------: | --------------------: |
| MRR@10     |     0.0545 |        **0.1450** | **+166.1% (+0.0905)** |
| Recall@10  |     0.1639 |        **0.3434** | **+109.5% (+0.1795)** |
| Coverage   |      5.07% |        **98.16%** |         **+93.09 pp** |
| Novelty@10 |     6.0233 |        **7.2209** |      **+1.1976 bits** |

For one-click sessions, the item-similarity fallback turns the observed item into a strong seed instead of defaulting to globally popular products.
