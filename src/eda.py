"""eda.py - 24 exploratory charts -> images/eda_XX_*.png and reports/eda_findings.json (every number computed, none typed in).
Run: python -m src.eda"""
import json
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.config import IMAGES, REPORTS
from src.features import application_features, customer_base, load_tables

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid"); BLUE, RED, GREY = "#0B5FA5", "#C23B3B", "#8A94A6"


def _save(fig, name):
    fig.tight_layout(); fig.savefig(IMAGES / name, dpi=105); plt.close(fig)


def run():
    t = load_tables(); a = application_features(t); cb = customer_base(t); F = {}
    d = a[a.loan_default.notna()].copy(); d["default"] = d.loan_default.astype(int); dec = a[a.approval_status != "Under Review"]
    F["disbursed_default_rate_pct"] = round(100 * d.default.mean(), 1); F["approval_rate_decided_pct"] = round(100 * (dec.approval_status == "Approved").mean(), 1)
    a["month"] = a.application_date.str[:7]; a["approved"] = (a.approval_status == "Approved").astype(int)

    def bar_rate(df, col, title, name, order=None, rot=0, target="default"):
        g = df.groupby(col)[target].agg(["mean", "size"]); g = g.loc[order] if order else g.sort_values("mean")
        fig, ax = plt.subplots(figsize=(7.5, 4)); ax.bar(g.index.astype(str), g["mean"] * 100, color=BLUE); ax.axhline(df[target].mean() * 100, color=RED, ls="--", label="overall"); ax.set(title=title, ylabel="%"); ax.legend(); plt.setp(ax.get_xticklabels(), rotation=rot, ha="right" if rot else "center"); _save(fig, name); return g

    g = bar_rate(d, "loan_type", "Default rate by loan type (disbursed)", "eda_01_default_by_loan_type.png"); F["default_rate_by_loan_type_pct"] = (g["mean"] * 100).round(1).to_dict()
    g = a.groupby("loan_type").approved.mean().sort_values() * 100; fig, ax = plt.subplots(figsize=(7.5, 4)); ax.bar(g.index, g.values, color=BLUE); ax.axhline(100 * dec.approved.mean() if "approved" in dec else 0, color=RED, ls="--"); ax.set(title="Approval rate by loan type (all applications)", ylabel="%"); _save(fig, "eda_02_approval_by_loan_type.png"); F["approval_rate_by_loan_type_pct"] = g.round(1).to_dict()
    fig, ax = plt.subplots(figsize=(8, 4)); sns.kdeplot(data=d, x="credit_score", hue="default", common_norm=False, ax=ax); ax.set_title("Credit score: repaid (0) vs defaulted (1)"); _save(fig, "eda_03_credit_score_by_default.png")
    d["score_band"] = pd.cut(d.credit_score, [299, 579, 649, 699, 749, 900], labels=["<580", "580-649", "650-699", "700-749", "750+"]); g = bar_rate(d, "score_band", "Default rate by credit-score band", "eda_04_default_by_score_band.png", order=["<580", "580-649", "650-699", "700-749", "750+"]); F["default_rate_by_score_band_pct"] = (g["mean"] * 100).round(1).to_dict()
    fig, ax = plt.subplots(figsize=(8, 4)); sns.kdeplot(data=d, x="debt_to_income_ratio", hue="default", common_norm=False, ax=ax, clip=(0, 3)); ax.set_title("Debt-to-income: repaid vs defaulted"); _save(fig, "eda_05_dti_by_default.png")
    d["dti_band"] = pd.cut(d.debt_to_income_ratio, [0, .3, .5, .8, 1.2, 3.01], labels=["<0.3", "0.3-0.5", "0.5-0.8", "0.8-1.2", ">1.2"]); g = bar_rate(d, "dti_band", "Default rate by debt-to-income band", "eda_06_default_by_dti_band.png", order=["<0.3", "0.3-0.5", "0.5-0.8", "0.8-1.2", ">1.2"]); F["default_rate_by_dti_band_pct"] = (g["mean"] * 100).round(1).to_dict(); F["dti_band_counts"] = g["size"].to_dict()
    g = bar_rate(d, "employment_type", "Default rate by employment type", "eda_07_default_by_employment.png", rot=20); F["default_rate_by_employment_pct"] = (g["mean"] * 100).round(1).to_dict(); F["employment_counts"] = g["size"].to_dict()
    d["income_band"] = pd.qcut(d.annual_income, 5, labels=["Q1 (lowest)", "Q2", "Q3", "Q4", "Q5 (highest)"]); g = bar_rate(d, "income_band", "Default rate by income quintile", "eda_08_default_by_income.png", order=["Q1 (lowest)", "Q2", "Q3", "Q4", "Q5 (highest)"])
    m = a.groupby("month").size(); fig, ax = plt.subplots(figsize=(11, 4)); ax.bar(m.index, m.values, color=BLUE); plt.setp(ax.get_xticklabels(), rotation=90); ax.set(title="Monthly loan applications"); _save(fig, "eda_09_monthly_applications.png"); F["applications_per_month_min_max"] = [int(m.min()), int(m.max())]
    mm = a[a.approval_status != "Under Review"].groupby("month").approved.mean() * 100; fig, ax = plt.subplots(figsize=(11, 4)); ax.plot(mm.index, mm.values, marker="o", color=BLUE); ax.set_ylim(0, 100); plt.setp(ax.get_xticklabels(), rotation=90); ax.set(title="Approval rate by month (decided applications)", ylabel="%"); _save(fig, "eda_10_approval_trend.png")
    F["approval_rate_monthly_range_pct"] = [round(float(mm.min()), 1), round(float(mm.max()), 1)]
    st = a.pivot_table(index="month", columns="approval_status", values="application_id", aggfunc="count").fillna(0); fig, ax = plt.subplots(figsize=(11, 4)); st.plot.area(ax=ax, stacked=True, color=["#2E9E5B", RED, "#E0A526"]); ax.set(title="Approval vs rejection (stacked, applications per month)"); plt.setp(ax.get_xticklabels(), rotation=90); _save(fig, "eda_11_approval_vs_rejection.png")
    fig, ax = plt.subplots(figsize=(9, 4.2)); sns.boxplot(data=a, x="loan_type", y="requested_amount", showfliers=False, ax=ax, color=BLUE); ax.set_yscale("log"); ax.set(title="Requested amount by loan type (log scale)"); _save(fig, "eda_12_amount_by_type.png")
    s = a.groupby("state").agg(apps=("application_id", "size"), approved=("approved", "mean")); ds = d.groupby("state").default.mean(); s["default"] = ds; s = s.sort_values("apps")
    fig, ax = plt.subplots(figsize=(8, 5.5)); ax.barh(s.index, s.apps, color=BLUE); ax.set(title="Loan applications by state"); _save(fig, "eda_13_applications_by_state.png"); F["states"] = int(len(s)); F["approval_rate_state_range_pct"] = [round(float(s.approved.min() * 100), 1), round(float(s.approved.max() * 100), 1)]; F["default_rate_state_range_pct"] = [round(float(s.default.min() * 100), 1), round(float(s.default.max() * 100), 1)]
    cv = a[a.disbursed_flag == 1].groupby("city").approved_amount.sum().sort_values().tail(12) / 1e7; fig, ax = plt.subplots(figsize=(8, 4.5)); ax.barh(cv.index, cv.values, color=BLUE); ax.set(title="Top-12 cities by disbursed value (Rs crore)"); _save(fig, "eda_14_disbursed_by_city.png")
    d["util_band"] = pd.cut(d.credit_utilization_ratio, [0, .3, .5, .7, .9, 1.0], labels=["<30%", "30-50%", "50-70%", "70-90%", ">90%"]); g = bar_rate(d, "util_band", "Default rate by credit utilisation", "eda_15_default_by_utilization.png", order=["<30%", "30-50%", "50-70%", "70-90%", ">90%"]); F["default_rate_by_utilization_pct"] = (g["mean"] * 100).round(1).to_dict()
    d["late30"] = d.num_late_payments_30d.clip(upper=4); g = bar_rate(d, "late30", "Default rate by 30-day late payments (4 = 4+)", "eda_16_default_by_late_payments.png", order=sorted(d.late30.unique())); F["default_rate_by_late30_pct"] = {int(k): round(v * 100, 1) for k, v in g["mean"].items()}
    fig, ax = plt.subplots(1, 2, figsize=(11, 4)); sns.boxplot(data=d, x="default", y="avg_balance", showfliers=False, ax=ax[0], color=BLUE); ax[0].set_title("Average balance"); sns.boxplot(data=d, x="default", y="net_cash_flow", showfliers=False, ax=ax[1], color=BLUE); ax[1].set_title("Net cash flow (6 months)"); _save(fig, "eda_17_transaction_behaviour.png")
    num = ["default", "credit_score", "debt_to_income_ratio", "credit_utilization_ratio", "num_late_payments_30d", "num_defaults_prior", "annual_income", "requested_amount", "interest_rate", "age", "avg_balance", "net_cash_flow", "emi_share_of_outflow", "loan_to_income"]
    cc = d[num].corr(); fig, ax = plt.subplots(figsize=(10, 8)); sns.heatmap(cc, annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=ax, annot_kws={"size": 7}); ax.set_title("Correlation matrix (disbursed loans)"); _save(fig, "eda_18_correlation.png")
    cd = cc["default"].drop("default").sort_values(key=abs, ascending=False); F["top_corr_with_default"] = cd.head(6).round(3).to_dict()
    g = bar_rate(cb.assign(approved=0).merge(a.groupby("customer_id").approved.max().rename("ever_approved"), on="customer_id", how="left").fillna({"ever_approved": 0}), "kyc_status", "Share of customers ever approved, by KYC status", "eda_19_kyc_vs_approval.png", target="ever_approved"); F["ever_approved_by_kyc_pct"] = (g["mean"] * 100).round(1).to_dict()
    d["age_band"] = pd.cut(d.age, [17, 25, 35, 45, 55, 75], labels=["18-25", "26-35", "36-45", "46-55", "56+"]); g = bar_rate(d, "age_band", "Default rate by age band", "eda_20_default_by_age.png", order=["18-25", "26-35", "36-45", "46-55", "56+"])
    d["emi_band"] = pd.qcut(d.emi_share_of_outflow, 4, duplicates="drop"); g = bar_rate(d, "emi_band", "Default rate by EMI share of outflow (quartiles)", "eda_21_default_by_emi_share.png", rot=20)
    fig, ax = plt.subplots(figsize=(8, 4)); sns.boxplot(data=d, x="loan_type", y="interest_rate", hue="default", showfliers=False, ax=ax); ax.set_title("Interest rate by loan type and outcome"); _save(fig, "eda_22_interest_by_type.png")
    nl = int((a.groupby("customer_id").size().reindex(cb.customer_id).isna()).sum()); apps = a.groupby("customer_id").size().reindex(cb.customer_id).fillna(0).astype(int).value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(7, 4)); ax.bar(apps.index.astype(str), apps.values, color=BLUE); ax.set(title="Customers by number of applications", xlabel="applications"); _save(fig, "eda_23_applications_per_customer.png"); F["customers_without_application"] = nl
    dd = a[a.disbursed_flag == 0]; fig, ax = plt.subplots(figsize=(8, 4)); sns.kdeplot(a[a.disbursed_flag == 1].debt_to_income_ratio, label="disbursed", ax=ax, clip=(0, 3)); sns.kdeplot(dd.debt_to_income_ratio, label="not disbursed", ax=ax, clip=(0, 3)); ax.legend(); ax.set_title("DTI: disbursed vs not disbursed (selection effect)"); _save(fig, "eda_24_selection_effect.png")
    F["median_dti_disbursed"] = round(float(a[a.disbursed_flag == 1].debt_to_income_ratio.median()), 2); F["median_dti_not_disbursed"] = round(float(dd.debt_to_income_ratio.median()), 2)
    F["multi_loan_customer_rows_pct"] = round(100 * float(d.customer_id.duplicated(keep=False).mean()), 1); F["n_charts"] = len(list(IMAGES.glob("eda_*.png")))
    (REPORTS / "eda_findings.json").write_text(json.dumps(F, indent=1, default=float)); return F


if __name__ == "__main__":
    print(json.dumps(run(), indent=1, default=float))
