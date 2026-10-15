"""features.py - modelling tables (PRD section 13.2/13.3).

Grain rules (the PRD's own warning): transactions are aggregated to ONE row per customer before joining, so no
application row is duplicated.  Two feature tables are produced:

  application_features()  one row per loan application  (7,500)  -> default model (FR-04), underwriting view
  customer_features()     one row per customer           (5,000)  -> customer risk score (FR-05), watchlist

Excluded on purpose (leakage / privacy / fairness):
  approval_status, approved_amount, decision_date, disbursed_flag   known only after the decision (PRD 13.3)
  email, phone, names, ids                                          PII / identifiers (PRD NFR)
  gender, marital_status                                            not in the PRD feature list; fair-lending review is future work
  city/state                                                        kept for the dashboard only (high-cardinality, adds noise)
Known caveat (cannot be fixed with this data): the bureau snapshot (2024-06-30) and the 6-month transaction window
(Jan-Jun 2024) are measured AFTER many applications, so they may partly reflect later repayment behaviour.
"""
import numpy as np
import pandas as pd

from src.config import PROCESSED
from src.data_load import load_raw

ASOF = pd.Timestamp("2024-06-30")
CATEGORICAL_APP = ["loan_type", "employment_type", "residence_type", "education"]
CATEGORICAL_CUST = ["employment_type", "residence_type", "education"]
TARGET = "loan_default"


def load_tables():
    try:
        return {k: pd.read_csv(PROCESSED / f"{k}.csv") for k in ["customers", "credit_history", "loan_applications", "transactions"]}
    except FileNotFoundError:
        return load_raw()


def transaction_features(x):
    """One row per customer from the 6-month transaction log."""
    cr = x.transaction_type == "Credit"
    g = pd.DataFrame({
        "txn_count": x.groupby("customer_id").size(),
        "total_inflow": x[cr].groupby("customer_id").amount.sum(),
        "total_outflow": x[~cr].groupby("customer_id").amount.sum(),
        "emi_outflow": x[x.category == "EMI"].groupby("customer_id").amount.sum(),
        "avg_balance": x.groupby("customer_id").balance_after.mean(),
        "min_balance": x.groupby("customer_id").balance_after.min(),
    }).fillna(0.0)
    g["net_cash_flow"] = g.total_inflow - g.total_outflow
    g["inflow_outflow_ratio"] = g.total_inflow / g.total_outflow.replace(0, np.nan)          # NaN when no outflow -> median-imputed
    g["emi_share_of_outflow"] = g.emi_outflow / g.total_outflow.replace(0, np.nan)
    return g.reset_index()


def customer_base(t):
    c, h, x = t["customers"], t["credit_history"], t["transactions"]
    d = c.merge(h, on="customer_id").merge(transaction_features(x), on="customer_id", how="left")
    d["tenure_years"] = (ASOF - pd.to_datetime(d.customer_since)).dt.days / 365.25
    d["debt_to_limit"] = d.total_outstanding_debt / d.total_credit_limit.replace(0, np.nan)
    d["credit_age_years"] = d.oldest_account_age_months / 12.0
    d["late_payments_total"] = d.num_late_payments_30d + d.num_late_payments_90d
    return d


NUM_CUST = ["age", "dependents", "annual_income", "years_at_residence", "tenure_years", "credit_score", "num_open_accounts", "num_credit_inquiries_6m",
            "credit_utilization_ratio", "num_late_payments_30d", "num_late_payments_90d", "num_defaults_prior", "bankruptcies", "total_credit_limit",
            "total_outstanding_debt", "credit_age_years", "debt_to_limit", "late_payments_total", "txn_count", "total_inflow", "total_outflow",
            "emi_outflow", "avg_balance", "min_balance", "net_cash_flow", "inflow_outflow_ratio", "emi_share_of_outflow"]
NUM_APP = NUM_CUST + ["requested_amount", "loan_term_months", "interest_rate", "applicant_monthly_income", "existing_emi", "debt_to_income_ratio",
                      "collateral_flag", "loan_to_income"]


def application_features(t=None):
    t = t or load_tables(); l = t["loan_applications"]; cb = customer_base(t)
    d = l.merge(cb, on="customer_id", how="left", suffixes=("", "_c"))
    d["loan_to_income"] = d.requested_amount / d.annual_income
    d["application_month"] = d.application_date.str[:7]
    return d


def customer_features(t=None):
    """Customer grain. Target = 1 if the customer defaulted on ANY disbursed loan, 0 if all disbursed loans repaid, NaN if never disbursed."""
    t = t or load_tables(); l = t["loan_applications"]; d = customer_base(t)
    dis = l[l.disbursed_flag == 1].groupby("customer_id").loan_default.max().rename("ever_default")
    n_app = l.groupby("customer_id").size().rename("n_applications"); n_dis = (l.disbursed_flag.groupby(l.customer_id).sum()).rename("n_disbursed")
    return d.merge(dis, on="customer_id", how="left").merge(n_app, on="customer_id", how="left").merge(n_dis, on="customer_id", how="left").fillna({"n_applications": 0, "n_disbursed": 0})


def design_matrix(df, num, cat, medians=None, columns=None):
    """Median-impute numerics, winsorise them at the training 1st/99th percentile (loan_to_income reaches 100x and inflow/outflow 169x,
    which would otherwise dominate linear models), 'Unknown' for missing categoricals, one-hot encode (FR-06).
    `medians` is the fitted-parameter dict returned by a previous call on TRAINING data (keeps the old keyword name)."""
    if medians is None:
        medians = dict(median=df[num].median(), lo=df[num].quantile(0.01), hi=df[num].quantile(0.99))
    X = df[num].astype(float).clip(lower=medians["lo"], upper=medians["hi"], axis=1).fillna(medians["median"])
    C = pd.get_dummies(df[cat].fillna("Unknown").astype(str), dtype=float)
    X = pd.concat([X.reset_index(drop=True), C.reset_index(drop=True)], axis=1); X.index = df.index
    if columns is not None:
        X = X.reindex(columns=columns, fill_value=0.0)
    return X, medians


def main():
    t = load_tables(); a = application_features(t); c = customer_features(t)
    a.to_csv(PROCESSED / "application_features.csv", index=False); c.to_csv(PROCESSED / "customer_features.csv", index=False)
    print("application grain", a.shape, "| disbursed with target:", int(a[TARGET].notna().sum()), "| default rate", round(a[TARGET].mean(), 4))
    print("customer grain   ", c.shape, "| customers with a disbursed loan:", int(c.ever_default.notna().sum()), "| ever-default rate", round(c.ever_default.mean(), 4))
    print("missing after join:", a[NUM_APP].isna().sum()[a[NUM_APP].isna().sum() > 0].to_dict())


if __name__ == "__main__":
    main()
