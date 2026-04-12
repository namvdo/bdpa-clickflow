"""
Download the Clickstream Data for Online Shopping dataset from UCI ML Repository.

Usage:
    python scripts/download_data.py

The dataset will be extracted into the data/ directory.
"""

import os
import ssl
import zipfile
import urllib.request

DATASET_URL = "https://archive.ics.uci.edu/static/public/553/clickstream+data+for+online+shopping.zip"
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
ZIP_PATH = os.path.join(DATA_DIR, "dataset.zip")


def _get_ssl_context():
    """Build an SSL context, falling back gracefully on macOS cert issues."""
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        pass
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


def download_and_extract():
    os.makedirs(DATA_DIR, exist_ok=True)

    csv_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".csv")]
    if csv_files:
        print(f"Dataset already present: {csv_files}")
        print("Delete the CSV files in data/ and re-run if you want to re-download.")
        return

    print("Downloading dataset from UCI repository...")
    ctx = _get_ssl_context()
    with urllib.request.urlopen(DATASET_URL, context=ctx) as response:
        with open(ZIP_PATH, "wb") as f:
            f.write(response.read())
    print(f"Downloaded to {ZIP_PATH}")

    print("Extracting...")
    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        zf.extractall(DATA_DIR)
    print(f"Extracted to {DATA_DIR}/")

    os.remove(ZIP_PATH)
    print("Done! Zip file removed.")

    for f in os.listdir(DATA_DIR):
        filepath = os.path.join(DATA_DIR, f)
        if os.path.isfile(filepath):
            size_mb = os.path.getsize(filepath) / (1024 * 1024)
            print(f"  {f} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    download_and_extract()