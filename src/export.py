"""export.py - write the files that PostgreSQL (\\copy) and Power BI read from the model outputs.
 data/processed/scored_customers_db.csv / scored_applications_db.csv   exact column order of the scored_* tables
 powerbi/data/*.csv                                                    dimension / fact / model tables for the Power BI model
Run: python -m src.export   (after src.train and src.risk_score)"""
import json

import numpy as np
import pandas as pd

from src.config import PBI_DATA, PROCESSED, REPORTS
from src.features import load_tables


def run():
    sc = pd.read_csv(PROCESSED / "scored_customers.csv"); sa = pd.read_csv(PROCESSED / "scored_applications.csv")
    sc[["customer_id", "default_probability", "risk_score", "risk_category", "scored_how", "top_risk_factors"]].to_csv(PROCESSED / "scored_customers_db.csv", index=False)
    sa[["application_id", "customer_id", "default_probability", "scored_how"]].to_csv(PROCESSED / "scored_applications_db.csv", index=False)

    t = load_tables(); c, h, l = t["customers"], t["credit_history"], t["loan_applications"]
    band = pd.cut(h.credit_score, [0, 579, 649, 699, 749, 1000], labels=["1 Very Poor (<580)", "2 Poor (580-649)", "3 Fair (650-699)", "4 Good (700-749)", "5 Excellent (750+)"])
    dc = c[["customer_id", "gender", "age", "employment_type", "education", "residence_type", "annual_income", "city", "state", "kyc_status"]].merge(
        h[["customer_id", "credit_score", "credit_utilization_ratio", "num_late_payments_30d", "num_defaults_prior"]].assign(credit_band=band.astype(str).values), on="customer_id").merge(
        sc[["customer_id", "default_probability", "risk_score", "risk_category", "scored_how", "top_risk_factors"]], on="customer_id", how="left")
    dc = dc.drop(columns=["gender"]); dc.to_csv(PBI_DATA / "dim_customer.csv", index=False)                        # gender not needed for any visual (fair-lending: not exposed)
    f = l.merge(sa[["application_id", "default_probability"]].rename(columns={"default_probability": "model_default_probability"}), on="application_id", how="left")
    f["application_month"] = f.application_date.str[:7] + "-01"
    for col in ["approval_status"]: f[f"Is{col.split('_')[0].title()}"] = 0
    f["IsApproved"] = (f.approval_status == "Approved").astype(int); f["IsRejected"] = (f.approval_status == "Rejected").astype(int); f["IsUnderReview"] = (f.approval_status == "Under Review").astype(int)
    f = f.drop(columns=["IsApproval"], errors="ignore"); f["approved_amount_crore"] = (f.approved_amount / 1e7).round(4)
    f.to_csv(PBI_DATA / "fact_loans.csv", index=False)
    d = pd.date_range("2022-07-01", "2024-06-30"); pd.DataFrame({"Date": d.strftime("%Y-%m-%d"), "Year": d.year, "MonthNo": d.month, "MonthStart": d.to_period("M").to_timestamp().strftime("%Y-%m-%d"),
                                                                 "YearMonth": d.strftime("%Y-%m")}).to_csv(PBI_DATA / "dim_date.csv", index=False)
    pd.read_csv(PROCESSED / "model_comparison.csv")[["Model", "CV_AUC", "Accuracy", "Precision", "Recall", "F1", "ROC_AUC"]].round(4).to_csv(PBI_DATA / "ml_model_comparison.csv", index=False)
    pd.read_csv(PROCESSED / "feature_importance.csv").head(15).round(5).to_csv(PBI_DATA / "ml_feature_importance.csv", index=False)
    return {p.name: len(pd.read_csv(p)) for p in sorted(PBI_DATA.glob("*.csv"))}


if __name__ == "__main__":
    print(json.dumps(run(), indent=1))
