# Clickstream Analysis for Online Shopping — Big Data Project

> Master's course project — Big Data (University of Oulu)
> Tools: PySpark, Spark MLlib, Docker, Streamlit (webapp)

---

## Project Overview

This project analyzes clickstream data from an online clothing store using a Big Data pipeline built on Apache Spark, creating a scalable backbone for potential implementations. We process raw user click logs, engineer session-level features, and apply machine learning to understand and predict user behavior. A recommender system surfaces product suggestions from browsing patterns and a shared Streamlit app presents all findings interactively.

- 165,474 click records | 14 features | No missing values
- 5 months of data (April–August 2008)
- Clothing store for pregnant women | Country-level IP data included

## Dataset Info

| Field     | Details                                                                                                                                  |
| --------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| Source    | [Clickstream Data for Online Shopping — UCI ML Repository](https://archive.ics.uci.edu/dataset/553/clickstream+data+for+online+shopping) |
| Instances | 165,474                                                                                                                                  |
| Features  | 14 (year, month, day, order, country, session ID, page, page 2, colour, location, model photography, price, price 2, page)               |
| Format    | CSV (semicolon `;` separated)                                                                                                            |
| License   | CC BY 4.0                                                                                                                                |

---

## Architecture

```
Raw CSV (UCI Dataset)
        │
        ▼
Task 1 — Spark Ingestion & Pipeline
  └── Clean → Sessionize → Feature Engineering → Parquet output
        │
        ├──▶ Task 2 — EDA & Business Insights (PySpark SQL + Matplotlib)
        │
        ├──▶ Task 3 — ML & Behavior Prediction (Spark MLlib)
        │
        └──▶ Task 4 — Recommender System (Spark MLlib ALS)
                │
                ▼
        Shared Streamlit App (app/)
```

---

## Task Division

Each task will have the primary reporting owner, while others should contribute as well. Task 1 is exeptionally encoraged to be completed by all team members.

| Task         | Description                              | Owner              |
| ------------ | ---------------------------------------- | ------------------ |
| Task 1       | Data Engineering & Spark Pipeline        | Sajjad Ghaeminejad |
| Report draft | Core report structure                    | All members        |
| Task 2       | Exploratory Analysis & Business Insights | Hamid              |
| Task 3       | ML & Behavior Prediction                 | Leo                |
| Task 4       | Recommender System                       | Nam                |
| Shared       | Streamlit Integration App                | Sajjad Ghaeminejad |
| Report       | Report writing                           | All Members        |
| Presentation | 15 minutes Power Point                   | All Members        |

## Description of primary tasks

We will use the clickstream dataset to study how users behave while browsing an online shop,
and then try to predict what kind of session they are having based on their early clicks. This project is more concrete and easier to explain: we want to understand user behavior and see whether early browsing actions can
help predict what the user will do later. In simple terms, we first group clicks into sessions and
turn raw click data into meaningful features such as number of clicks, time spent, categories visited,
products viewed, and price-related behavior. Then we analyze patterns in the sessions and build
a prediction/recommendation model. For example, we can try to predict whether a session will become an engaged
shopping session, whether the user is likely to end up in a certain product category, or whether the
session shows strong purchase intent.

Basic ideas of what can be done can be found in the aticle [Łapczyński M., Białowąs S.: DISCOVERING PATTERNS OF USERS’ BEHAVIOUR IN AN E-SHOP – COMPARISON OF CONSUMER BUYING BEHAVIOURSIN POLAND AND OTHER EUROPEAN COUNTRIES](https://cdn.discordapp.com/attachments/1483904494776483933/1484514576010706974/13_M.Lapczynski_S.Bia_owas_Discovering_patterns_of_users_.pdf?ex=69ca5eef&is=69c90d6f&hm=d9366cd5b286517215119358401fa740aa3917fb03d5acef06041f4e92c1b27b&)

Below is preliminary list of tasks to be carried out. Throughout the project implementation list may be updated, as well as the tasks.

### Task 1.

Basic exploration of data, it's content and potential transformations for the future steps. The output of the stage is mainly data card:

1. Total available features (columns), and their description, useless features and why
2. Data types and casting.
3. Consitency of the data - do we have malformed values, do we have missing values? Datetime parsing etc.
4. Basic aggregations (session level for instance).
5. Basic plotting.
6. df.info()

This stage is very imporant since it's purpose is to create a unified preprocessing backbone for future steps, which can be shared by all teammates. It is important that each teammember understands the data, and potential future steps.

### Task 2.

This stage exculusively focuses on business insights, for instance, some geography based aggregations, purchasing volumes, most prefferd products etc. It also may partially overlap with ML task in terms in sequence analysis and associations rules.

Potential Outputs:

1. Barplots
2. (Scatter)Plots
3. Histograms, pair plots, box plots, heatmaps

### Task 3.

This task goes deeper into ML side, potentially exploring advanced unsupervisied ML and clustering methods, or other applicable choices.

Outputs:

1. Robust models
2. Test-validation results, performance metrics.
3. Potential plotting as in task 2.
4. Applicable literature

### Task 4: Recommender System

#### Setup

- **Split**: a temporal train/test split. Sessions starting before month 7 are for training; sessions starting in month 7 or 8 are for testing. Within the test set, the last click is held out as ground truth.
- **Warm sessions** (17,031 training, 5,585 evaluable test): scored by ALS (fold-in), item k-NN, and Item2Vec.
- **Cold sessions** (3,320): scored by the item-similarity fallback.
- **Baseline**: popularity ranker, using the top-10 most-clicked training items after removing already-seen items.

#### Models

- **Popularity**: global item popularity from the training split.
- **ALS** (Spark MLlib, implicit feedback): rank=20, lambda=1.0, alpha=80.
- **Item k-NN** (Sarwar et al., co-interaction): cosine similarity over l2-normalised item columns of the session-item matrix, top-50 neighbours per seed. Score for a candidate item is the weighted sum of similarities to items in the user's history.
- **Item2Vec / Word2Vec**: Spark Word2Vec with vectorSize=32, windowSize=3, maxIter=20, minCount=1. It learns item embeddings from local product context inside ordered sessions. Recommendations average the embeddings of the products clicked so far, then rank candidate products by cosine similarity to that session vector (used for warm-session evaluation).
- **Cold fallback**: same k-NN matrix, restricted to the single seed item.

#### Metrics

- **MRR@10**: reciprocal rank of the held-out next click.
- **Recall@10**: whether the held-out next click appears in the top 10.
- **Coverage**: share of the 217-item catalogue recommended at least once.
- **Novelty@10**: mean self-information, `avg(-log2(p(item)))`, over recommended items, where `p(item)` is training-set item popularity. Higher novelty means the model is recommending less obvious items.

Improvement is measured relative to the popularity baseline. For coverage, the delta is percentage points; for novelty, the delta is bits.

#### Results

**Warm sessions (n=15,664):**

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

### Report and Presentation

Teammates contribute to producing the final project artifacts.

## Suggested Timeline

It is expected that specified tasks are carried out almost in parallel manner. Some tasks may require more research, and may overlap in one or another way.

| Task         | Description                              | Tentative Deadline |
| ------------ | ---------------------------------------- | ------------------ |
| Task 1       | Data Engineering & Spark Pipeline        | April 6            |
| Report draft | Partially written report                 | April 6            |
| Task 2       | Exploratory Analysis & Business Insights | April 12           |
| Task 3       | ML & Behavior Prediction                 | April 16           |
| Task 4       | Recommender System                       | April 21           |
| Shared       | Streamlit Integration App                | April 25           |
| Report       | Report writing                           | **May 3** final    |
| Presentation | 15 minutes Power Point                   | **April 30** final |

---

## Repo Structure

```
bdpa-clickflow/
│
├── README.md                   # This file !
├── docker-compose.yml          # Spin up Spark environment
├── requirements.txt            # Requirements! => python dependencies
│
├── data/                       #
│
├── docs/                       # Project documentation
│   ├── project_idea.pdf
│   └── architecture.png
│
├── pipeline/             # Task 1 - Data Engineering & Spark Pipeline
├── eda/                  # Task 2 - Exploratory Analysis & Business Insights
├── ml/                   # Task 3 - ML & Behavior Prediction
├── recommender/          # Task 4 - Recommender System
│
└── app/                        # Shared Streamlit App
```

---

## Getting Started

### 1. Clone the repo

```bash
git clone <https://github.com/namvdo/bdpa-clickflow>
cd bdpa-clickflow
```

### 2. Download the dataset

Follow the instructions in [`data/README.md`](data/README.md) to download and place the dataset.

### 3. Start the environment

```bash
docker-compose up
```

This spins up a Spark environment accessible to all tasks.

### 4. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the Streamlit app

```bash
streamlit run app/0_Overview.py
```

---

## Tech Stack

to be added!

---

## Dataset Citation

```
Clickstream Data for Online Shopping [Dataset]. (2019).
UCI Machine Learning Repository. https://doi.org/10.24432/C5QK7X
```

---

## Contributing (Agile Workflow)

1. Each person works in their own task folder
2. Open a GitHub Issue for your task and assign it to yourself
3. Create a branch: `git checkout -b task1-pipeline`
4. Push your work and open a Pull Request when ready
5. Everyone reviews before merging to `main`
