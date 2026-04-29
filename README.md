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

- **Split**: hold the last click of each session as ground truth; train on the rest.
- **Warm sessions** (15,664, $\geq 2$ training clicks): scored by ALS and item k-NN.
- **Cold sessions** (3,320, exactly 1 training click): scored by the same item-item similarity matrix using the single observed click as the seed.
- **Baseline**: popularity ranker (top-10 most-clicked items, same list for everyone).

#### Models

- **ALS** (Spark MLlib, implicit feedback): rank=20, $\lambda$=1.0, $\alpha$=80. Picked by grid search on NDCG@10.
- **Item k-NN** (Sarwar et al., co-interaction): cosine similarity over $\ell_2$-normalised item columns of the session-item matrix, top-50 neighbours per seed. Score for a candidate item is the weighted sum of similarities to items in the user's history.
- **Cold fallback**: same k-NN matrix, restricted to the single seed item.

#### Results (top-10)

Improvement is measured relative to the popularity baseline:
`(model score - popularity score) / popularity score`. The value in parentheses is the actual numeric change from the baseline; coverage change is shown in percentage points.

**Warm sessions (n=15,664):**

| Metric    | Popularity | Item k-NN  | Improvement vs popularity | ALS        | Improvement vs popularity |
| :-------- | :--------- | :--------- | :------------------------ | :--------- | :------------------------ |
| NDCG@10   | 0.0512     | 0.1105     | +115.8% (+0.0593)         | **0.1110** | **+116.8% (+0.0598)**     |
| MRR@10    | 0.0360     | **0.0793** | **+120.3% (+0.0433)**     | 0.0775     | +115.3% (+0.0415)         |
| Recall@10 | 0.1027     | 0.2138     | +108.2% (+0.1111)         | **0.2221** | **+116.3% (+0.1194)**     |
| Coverage  | 23.04%     | **98.16%** | **+326.0% (+75.12 pp)**   | 97.70%     | +324.0% (+74.66 pp)       |

ALS and item k-NN are basically tied. ALS has the largest warm-session NDCG@10 and Recall@10 lift, while item k-NN slightly leads on MRR@10 and coverage. With 91% of (session, item) pairs at click count one, the ALS confidence weight collapses to "viewed or not", which is essentially what cosine similarity in k-NN already captures.

**Cold sessions (n=3,320):**

| Metric    | Popularity | Item-sim fallback | Improvement vs popularity |
| :-------- | :--------- | :---------------- | :------------------------ |
| NDCG@10   | 0.0807     | **0.1903**        | **+135.8% (+0.1096)**     |
| Recall@10 | 0.1623     | **0.3434**        | **+111.6% (+0.1811)**     |
| Coverage  | 5.07%      | **98.16%**        | **+1,836.1% (+93.09 pp)** |

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
streamlit run app/main.py
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
