"""config.py - paths and constants. No hard-coded absolute paths (PRD NFR: maintainability)."""
import os
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
RAW = ROOT / "datasets"
PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"
IMAGES = ROOT / "images"
SQL_DIR = ROOT / "sql"
PBI_DATA = ROOT / "powerbi" / "data"
# SQLite copy used for local testing and the Streamlit app (set CREDORA_DB_PATH if the project folder is read-only / synced)
DB_PATH = pathlib.Path(os.environ.get("CREDORA_DB_PATH", ROOT / "data" / "credora.db"))
SEED = 42
TABLES = ["customers", "credit_history", "loan_applications", "transactions"]
for _d in (PROCESSED, MODELS, REPORTS, IMAGES, PBI_DATA):
    _d.mkdir(parents=True, exist_ok=True)
