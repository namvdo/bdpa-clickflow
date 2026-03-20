# BDPA Clickflow

**Analysing Clickstream Data for Online Shopping** — Big Data Processing course project.

We analyse the [UCI Clickstream Dataset](https://archive.ics.uci.edu/dataset/553/clickstream+data+for+online+shopping) (165K sessions, 14 features) from an online maternity-clothing store using PySpark.

---

## Quick Start

> **Prerequisites:** Python 3.10–3.12 (PySpark doesn't support 3.13+), Java 11+

```bash
# 1. Clone & enter the repo
git clone <repo-url> && cd bdpa-clickflow

# 2. Create a virtual environment (use python3.12 specifically)
python3.12 -m venv venv
source venv/bin/activate        # macOS / Linux
# venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download the dataset
python scripts/download_data.py

# 5. Start Jupyter (optional)
jupyter notebook
```

To verify PySpark works:

```python
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("clickflow").getOrCreate()
df = spark.read.csv("data/e-shop clothing 2008.csv", header=True, inferSchema=True, sep=";")
df.printSchema()
df.show(5)
spark.stop()
```

---

## Project Structure

```
bdpa-clickflow/
├── data/               # Dataset files (git-ignored, download locally)
├── notebooks/          # Jupyter notebooks for exploration & analysis
├── scripts/
│   └── download_data.py
├── src/                # Shared Python modules
├── requirements.txt
└── README.md
```

- **`data/`** — Not committed to git. Each person downloads via the script.
- **`notebooks/`** — Individual exploration notebooks.
- **`src/`** — Shared helper code (cleaning functions, Spark utilities, etc.).
- **`scripts/`** — Automation scripts.

---

## Dataset Info

| Field | Details |
|-------|---------|
| Source | [UCI ML Repository #553](https://archive.ics.uci.edu/dataset/553/clickstream+data+for+online+shopping) |
| Instances | 165,474 |
| Features | 14 (year, month, day, order, country, session ID, page, page 2, colour, location, model photography, price, price 2, page) |
| Format | CSV (semicolon `;` separated) |
| License | CC BY 4.0 |

---
## Citation

```
Clickstream Data for Online Shopping [Dataset]. (2019).
UCI Machine Learning Repository. https://doi.org/10.24432/C5QK7X.
```
