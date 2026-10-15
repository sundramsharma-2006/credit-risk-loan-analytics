"""services.py - data layer for the Streamlit dashboard (no Streamlit imports, so it is unit-tested).
Every dashboard number comes from the cleaned tables + model outputs in data/processed; nothing is typed in."""
import json
import pathlib
import sys

import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from src.config import PROCESSED, REPORTS  # noqa: E402

RISK_ORDER = ["Low Risk", "Medium Risk", "High Risk"]
BANDS = ["1 Very Poor (<580)", "2 Poor (580-649)", "3 Fair (650-699)", "4 Good (700-749)", "5 Excellent (750+)"]


def load():
    c = pd.read_csv(PROCESSED / "customers.csv"); h = pd.read_csv(PROCESSED / "credit_history.csv"); l = pd.read_csv(PROCESSED / "loan_applications.csv")
    sc = pd.read_csv(PROCESSED / "scored_customers.csv"); sa = pd.read_csv(PROCESSED / "scored_applications.csv")
    cust = c[["customer_id", "city", "state", "employment_type", "annual_income", "kyc_status"]].merge(h[["customer_id", "credit_score", "credit_utilization_ratio"]], on="customer_id").merge(
        sc[["customer_id", "default_probability", "risk_score", "risk_category", "scored_how", "top_risk_factors"]], on="customer_id")
    cust["credit_band"] = pd.cut(cust.credit_score, [0, 579, 649, 699, 749, 1000], labels=BANDS).astype(str)
    loans = l.merge(cust[["customer_id", "city", "state", "credit_score", "risk_category", "risk_score"]], on="customer_id").merge(
        sa[["application_id", "default_probability"]].rename(columns={"default_probability": "model_default_probability"}), on="application_id")
    loans["month"] = loans.application_date.str[:7]
    return dict(customers=cust, loans=loans)


def apply_filters(d, loan_types=None, states=None, risk=None, months=None):
    """Return (loans, customers) after the dashboard filters. Loan filters also restrict customers via customer_id where it makes sense."""
    L, C = d["loans"], d["customers"]
    if loan_types: L = L[L.loan_type.isin(loan_types)]
    if states: L = L[L.state.isin(states)]; C = C[C.state.isin(states)]
    if risk: L = L[L.risk_category.isin(risk)]; C = C[C.risk_category.isin(risk)]
    if months: L = L[(L.month >= months[0]) & (L.month <= months[1])]
    return L, C


def kpis(L, C):
    dec = L[L.approval_status != "Under Review"]; dis = L[L.disbursed_flag == 1]
    return dict(applications=len(L), approval_rate_pct=100 * (dec.approval_status == "Approved").mean() if len(dec) else float("nan"),
                default_rate_pct=100 * dis.loan_default.mean() if len(dis) else float("nan"), disbursed_crore=dis.approved_amount.sum() / 1e7,
                high_risk_customers=int((C.risk_category == "High Risk").sum()), avg_risk_score=float(C.risk_score.mean()) if len(C) else float("nan"))


def approval_trend(L):
    g = L.groupby("month").apply(lambda x: pd.Series({"approved": (x.approval_status == "Approved").sum(), "rejected": (x.approval_status == "Rejected").sum(),
                                                       "approval_rate_pct": 100 * (x.approval_status == "Approved").sum() / max((x.approval_status != "Under Review").sum(), 1)}), include_groups=False)
    return g.reset_index()


def default_by_loan_type(L):
    dis = L[L.disbursed_flag == 1]; return dis.groupby("loan_type").agg(disbursed_loans=("loan_default", "size"), defaults=("loan_default", "sum"), default_rate_pct=("loan_default", lambda s: 100 * s.mean())).reset_index().sort_values("default_rate_pct", ascending=False)


def risk_distribution(C):
    return C.groupby("risk_category").size().reindex(RISK_ORDER).fillna(0).astype(int).rename("customers").reset_index()


def credit_band_counts(C):
    return C.groupby("credit_band").size().reindex(BANDS).fillna(0).astype(int).rename("customers").reset_index()


def score_by_application_month(L):
    return L.groupby("month").agg(applicants=("customer_id", "nunique"), avg_credit_score=("credit_score", "mean")).reset_index()


def state_table(L):
    def f(x):
        dec = x[x.approval_status != "Under Review"]; dis = x[x.disbursed_flag == 1]
        return pd.Series({"applications": len(x), "approval_rate_pct": 100 * (dec.approval_status == "Approved").mean() if len(dec) else float("nan"),
                          "default_rate_pct": 100 * dis.loan_default.mean() if len(dis) else float("nan"), "disbursed_crore": dis.approved_amount.sum() / 1e7})
    return L.groupby("state").apply(f, include_groups=False).reset_index().sort_values("applications", ascending=False)


def monthly_applications(L):
    return L.groupby("month").size().rename("applications").reset_index()


def approval_vs_rejection(L):
    return L.pivot_table(index="month", columns="approval_status", values="application_id", aggfunc="count").fillna(0).reset_index()


def watchlist(C, risk=("High Risk",), states=None, n=100):
    W = C[C.risk_category.isin(risk)]
    if states: W = W[W.state.isin(states)]
    return W.sort_values("risk_score", ascending=False).head(n)[["customer_id", "city", "state", "employment_type", "credit_score", "risk_score", "risk_category", "top_risk_factors"]]


def applicant(d, customer_id):
    """Underwriter view: customer score + every application with its model P(default)."""
    c = d["customers"][d["customers"].customer_id == customer_id]
    if c.empty: return None, None
    a = d["loans"][d["loans"].customer_id == customer_id][["application_id", "application_date", "loan_type", "requested_amount", "debt_to_income_ratio", "approval_status", "model_default_probability", "loan_default"]]
    return c.iloc[0].to_dict(), a


def model_summary():
    out = {}
    for n in ["default_model", "risk_score"]:
        p = REPORTS / f"{n}_summary.json"; out[n] = json.loads(p.read_text()) if p.exists() else None
    return out
