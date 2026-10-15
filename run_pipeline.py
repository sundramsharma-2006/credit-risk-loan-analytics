"""Run the whole pipeline end to end (PRD FR-12):   python run_pipeline.py
validate + clean -> SQL mirror -> features -> default model -> risk score -> exports -> EDA -> Power BI project -> tests.
Needs a writable database location: set CREDORA_DB_PATH if the project folder is synced or read-only."""
import subprocess
import sys
import time


def step(name, fn):
    t = time.time(); print(f"\n=== {name} ==="); fn(); print(f"[{name}: {time.time() - t:.0f}s]")


def main():
    from src import data_load, eda, export, features, risk_score, train
    step("1 validate, clean, build database mirror", data_load.main)
    step("2 features", features.main)
    step("3 default model (4 algorithms, grouped CV)", train.run)
    step("4 risk scores + bands", risk_score.run)
    step("5 exports for PostgreSQL / Power BI", lambda: print(export.run()))
    step("6 EDA charts", eda.run)
    step("7 Power BI project", lambda: subprocess.run([sys.executable, "powerbi/build_pbip.py"], check=True))
    step("8 tests", lambda: subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests"], check=True))
    print("\nDone. Dashboard: streamlit run dashboard/app.py")


if __name__ == "__main__":
    main()
