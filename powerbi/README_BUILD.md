# Power BI dashboard - build guide

**Status (read first):** `credora_dashboard.pbip` was *generated* by `build_pbip.py` using the same PBIP/TMDL format as the Zomato project (which opened in Power BI Desktop). It has **not been opened in Power BI in the build sandbox**. JSON validity and field references were checked by script; visual layout was not seen. If Power BI rejects something, use the manual steps below - they take about 45 minutes. The Streamlit dashboard (`dashboard/app.py`) covers the same eight components.

## Open the generated project
1. Install Power BI Desktop; enable **File -> Options -> Preview features -> Power BI Project (.pbip) save option** and **TMDL**.
2. Double-click `credora_dashboard.pbip`.
3. If the data path is wrong: **Transform data -> Edit parameters -> DataPath** = full path of `powerbi/data`.
4. **Home -> Refresh**. Then **File -> Save as .pbix** if you want a single file.

## Pages (all 8 PRD components)
| Page | PRD components |
|---|---|
| 1 Executive Overview | approval rate, default rate, monthly applications, approval vs rejection, risk distribution (+ KPI cards, slicers: loan type, state, risk category) |
| 2 Credit & Risk | default rate by loan type, credit-score bands, credit score by application month |
| 3 Geography | applications / approval / default / disbursed value by state |
| 4 High-Risk Watchlist | filterable table (risk category, state, employment) |
| 5 Model Performance | model comparison, feature importance, honest AUC note |

## Manual build (if the PBIP fails)
1. **Get data** -> Text/CSV: load the five files in `powerbi/data` (`dim_date`, `dim_customer`, `fact_loans`, `ml_model_comparison`, `ml_feature_importance`). Set `application_date`/`Date`/`MonthStart` to Date type.
2. **Model view**: `fact_loans[application_date]` -> `dim_date[Date]` and `fact_loans[customer_id]` -> `dim_customer[customer_id]`, both many-to-one, single direction.
3. **New measure**: paste each measure from `DAX_measures.md`.
4. Build each page from the table above; apply `credora_theme.json` (View -> Themes -> Browse for themes).

## Data notes that affect how to read the visuals
* Bureau data is **one snapshot (2024-06-30)**: "credit score trend" compares applicant cohorts by application month; it is not a time series of the same customers.
* `Default Rate` only covers disbursed loans (3,403 of 7,500 applications). Scores for non-disbursed applications are extrapolations.
* Contact details (e-mail, phone, names) and gender are deliberately not in the model.
