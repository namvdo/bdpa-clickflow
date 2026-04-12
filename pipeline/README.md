# Pipeline

This is the data pipeline for the project. It takes the raw CSV and turns it into clean, session level features ready for the ML and recommender tasks.

---

## Quick Run

Run the scripts in this order from the project root:

```bash
python pipeline/ingest.py
python pipeline/clean.py
python pipeline/sessionize.py
python pipeline/feature_engineering.py
```

Each script reads from `output/` and writes back to `output/`, so just make sure the `data/` folder has the CSV first.

---

## What each step does

**ingest.py** : just reads the raw CSV and prints the schema and row count. Nothing is saved, it's mainly there to verify the data loads correctly.

**clean.py**: renames columns to snake_case, merges year/month/day into a single `date` column, casts types, and converts `price_2` to boolean. saves the result as `output/cleaned.parquet`.

**sessionize.py**: collapses the click level data into one row per session. Computes things like number of clicks, average price, number of categories and colours browsed, and whether the user saw a sale item. Saves as `output/sessions.parquet`.

**feature_engineering.py**: takes the session data and prepares it for MLlib. extracts month and day of week from the date, converts the `bought` column to a 0/1 label, assembles all numeric columns into a feature vector, and scales it. Saves as `output/features.parquet`.
