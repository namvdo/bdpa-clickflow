# Clickstream Analysis for Online Shopping — Big Data Project

> Master's course project — Big Data (University of Oulu)
> Tools: PySpark, Spark MLlib, Docker, Streamlit

---

## Project Overview

This project analyzes clickstream data from an online clothing store using a Big Data pipeline built on Apache Spark. We process raw user click logs, engineer session-level features, and apply machine learning to understand and predict user behavior. A recommender system surfaces product suggestions from browsing patterns and a shared Streamlit app presents all findings interactively.

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

| Task   | Description                              | Owner              |
| ------ | ---------------------------------------- | ------------------ |
| Task 1 | Data Engineering & Spark Pipeline        | Sajjad Ghaeminejad |
| Task 2 | Exploratory Analysis & Business Insights | Hamid              |
| Task 3 | ML & Behavior Prediction                 | Leo                |
| Task 4 | Recommender System                       | Nam                |
| Shared | Streamlit Integration App                | Sajjad Ghaeminejad |

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
