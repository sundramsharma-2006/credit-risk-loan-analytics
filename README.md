# Credora Finance - Credit Risk & Loan Approval Analytics Platform

End-to-end analytics and machine-learning build on a **synthetic** retail-lending book (5,000 customers, 5,000 bureau records, 7,500 loan applications, 40,079 transactions): validated relational database, ten analytical SQL queries, EDA, leakage-aware features, a default-prediction model (4 algorithms), a 0-100 customer risk score, and dashboards (Streamlit + Power BI project). Built against `documentation/PRD.docx`.

> **Verification status - read first.** Built and tested in a sandbox *without* PostgreSQL, Streamlit, Power BI, scikit-learn, XGBoost or SHAP.

| Component | Status |
|---|---|
| Validation + cleaning (23 checks) | Run and tested |
| SQL: schema + 10 analytical queries + 8 dashboard queries | Run and tested on **SQLite** (schema DDL taken verbatim from the PRD's PostgreSQL file; 3 PostgreSQL-only constructs translated mechanically). **PostgreSQL execution not verified here** - run it yourself |
| Features, 4 models, risk score, EDA (24 charts), 5 notebooks | Run and tested (40 unit tests) |
| ML backend | NumPy re-implementation of scikit-learn's API (`src/models/mini_sklearn.py`); real scikit-learn is used automatically when installed. Gradient boosting stands in for XGBoost; occlusion explanations stand in for SHAP |
| Streamlit dashboard | Data layer (`dashboard/services.py`) tested; **page itself compile-checked only, never rendered** |
| Power BI project (`powerbi/credora_dashboard.pbip`) | **Generated, JSON/field references checked, never opened in Power BI.** Manual build steps in `powerbi/README_BUILD.md` |

## Headline results (and their limits)
* **The PRD success targets were not met.** Default model (Logistic Regression, chosen by grouped CV): test ROC-AUC **0.73** (95% CI 0.69-0.77) vs target 0.80; recall at the business threshold **0.66** vs target 0.70 (precision 0.36). Not tuned against the test set.
* **Debt-to-income does most of the work**: alone it gives AUC 0.70; default rate rises from 9% (DTI < 0.3) to 42% (DTI > 1.2). Credit score is second.
* **Risk bands work directionally**: observed default rate 22% (Low) / 40% (Medium) / 51% (High) among customers with a loan outcome. High band = 988 of 5,000 customers.
* **Biggest caveat - selection bias**: repayment is only observed for approved loans (median DTI 0.71); rejected applications (median DTI 2.39) are scored by extrapolation and cannot be validated.
* **Data finding**: two e-mail addresses are shared by different customers, which would make the schema's `UNIQUE(email)` reject a PostgreSQL bulk load. Use the cleaned `data/processed/customers.csv` (`sql/load_data.sql`).

## Quick start
```bash
pip install -r requirements.txt
export CREDORA_DB_PATH=/tmp/credora.db      # any writable path (SQLite copy)
python run_pipeline.py                      # ~5 minutes: validate -> features -> models -> scores -> EDA -> Power BI project -> tests
python -m unittest discover -s tests -v
streamlit run dashboard/app.py              # untested page - see status table
```
**PostgreSQL:** create a database, run the DDL part of `sql/credit_risk_platform.sql` (sections 1-5; skip the 6-row seed INSERTs), then `sql/load_data.sql`, then the ten queries (section 7), then `sql/scored_customers.sql` and its two `\copy` lines.

## Layout (follows the PRD)
```
datasets/        four raw CSVs            sql/        PostgreSQL schema, load, scoring tables, dashboard queries
data/processed/  cleaned tables, features, scores, model outputs, validation + cleaning logs
src/             config, data_load, features, modeling, train, risk_score, export, eda, models/ (mini_sklearn)
notebooks/       00 database+SQL, 01 EDA, 02 features, 03 modeling, 04 risk scoring (executed)
models/          default_model.pkl, risk_model.pkl, model_card.md
dashboard/       app.py (Streamlit), services.py (tested data layer)
powerbi/         PBIP project, data/, DAX_measures.md, README_BUILD.md, build_pbip.py
documentation/   PRD.docx, data_dictionary.md, Final_Project_Report (docx/pdf)
tests/           test_sql (12), test_models (18), test_dashboard (10)
```

## Design decisions worth defending
* **Split by customer**, not by row: 1,094 of 3,403 disbursed rows share a customer with another loan, so a random split leaks (naive split AUC 0.755 vs 0.731).
* **Leakage control**: post-decision fields excluded; transactions aggregated to customer grain before the join; contact details, gender and marital status excluded (no fairness audit was run).
* **Calibrated probabilities** (Platt on out-of-fold predictions) so `risk_score = round(100 x P)` means roughly "x% of similar loans default"; thresholds chosen on training folds only.
* **Customer score**: highest probability over a customer's applications (conservative); customers who never applied (1,096) use a weaker profile-only model and are flagged in `scored_how`.

## Known data limitations
One bureau snapshot (2024-06-30) vs applications spanning 2022-2024 (possible temporal leakage); only 4-12 transactions per customer; 935 applications are dated before `customer_since`; age is computed as of 2026; 173 applications are still "Under Review".

## Next steps
Run SQL on PostgreSQL; render and fix the dashboard pages and open the PBIP; install scikit-learn/XGBoost/SHAP and re-run; run a fair-lending/bias review; investigate reject inference; evaluate on a genuinely later time window if more data arrives.
