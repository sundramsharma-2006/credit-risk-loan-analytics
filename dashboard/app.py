"""Credora Finance - Credit Risk & Loan Approval dashboard (Streamlit).   Run from the project root:  streamlit run dashboard/app.py
All eight PRD components are on the page: approval rate, default rate, risk distribution, credit-score view, geography, monthly applications,
approval-vs-rejection trend and a filterable high-risk list.  NOTE: written and compile-checked in a sandbox without Streamlit; the data layer
(dashboard/services.py) is unit-tested, the page itself has not been rendered."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import pandas as pd
import streamlit as st

from dashboard import services as S

st.set_page_config(page_title="Credora Finance - Credit Risk", page_icon="C", layout="wide")


@st.cache_data
def get():
    return S.load()


d = get()
st.title("Credora Finance - Credit Risk & Loan Approval Analytics")
st.caption("Synthetic retail-lending dataset (5,000 customers, 7,500 applications, Jul 2022 - Jun 2024). Decision support only: the model advises, underwriters decide.")

with st.sidebar:
    st.header("Filters")
    lt = st.multiselect("Loan type", sorted(d["loans"].loan_type.unique())); stt = st.multiselect("State", sorted(d["customers"].state.unique()))
    rk = st.multiselect("Risk category", S.RISK_ORDER); months = sorted(d["loans"].month.unique()); mr = st.select_slider("Application month", options=months, value=(months[0], months[-1]))
L, C = S.apply_filters(d, lt, stt, rk, mr); k = S.kpis(L, C)
c = st.columns(6)
c[0].metric("Applications", f"{k['applications']:,}"); c[1].metric("Approval rate", f"{k['approval_rate_pct']:.1f}%"); c[2].metric("Default rate (disbursed)", f"{k['default_rate_pct']:.1f}%")
c[3].metric("Disbursed value", f"Rs {k['disbursed_crore']:,.1f} cr"); c[4].metric("High-risk customers", f"{k['high_risk_customers']:,}"); c[5].metric("Avg risk score", f"{k['avg_risk_score']:.1f}")
t1, t2, t3, t4, t5 = st.tabs(["Approvals & volume", "Credit & risk", "Geography", "High-risk watchlist", "Applicant lookup & model"])
with t1:
    a, b = st.columns(2)
    with a: st.subheader("Approval rate by month"); st.line_chart(S.approval_trend(L).set_index("month")["approval_rate_pct"])
    with b: st.subheader("Monthly applications"); st.bar_chart(S.monthly_applications(L).set_index("month"))
    st.subheader("Approval vs rejection"); st.area_chart(S.approval_vs_rejection(L).set_index("month"))
with t2:
    a, b = st.columns(2)
    with a: st.subheader("Default rate by loan type"); st.bar_chart(S.default_by_loan_type(L).set_index("loan_type")["default_rate_pct"])
    with b: st.subheader("Customer risk distribution"); st.bar_chart(S.risk_distribution(C).set_index("risk_category"))
    a, b = st.columns(2)
    with a: st.subheader("Credit-score bands"); st.bar_chart(S.credit_band_counts(C).set_index("credit_band"))
    with b:
        st.subheader("Average applicant credit score by application month"); st.line_chart(S.score_by_application_month(L).set_index("month")["avg_credit_score"])
        st.caption("Bureau data is a single snapshot (2024-06-30), so this compares applicant cohorts - it is not a time series of the same customers.")
with t3:
    st.subheader("Loan distribution by state"); stt_tab = S.state_table(L); st.bar_chart(stt_tab.set_index("state")["applications"]); st.dataframe(stt_tab.round(1), use_container_width=True)
with t4:
    rsel = st.multiselect("Risk category", S.RISK_ORDER, default=["High Risk"], key="wl_risk"); st.dataframe(S.watchlist(C, tuple(rsel) or ("High Risk",), stt or None, 200), use_container_width=True)
    st.caption("Risk factors = how much each feature moves the model score versus a typical customer (occlusion). Contact details are not shown.")
with t5:
    cid = st.text_input("Customer ID (e.g. CUST100001)", "CUST100001").strip(); cust, apps = S.applicant(d, cid)
    if cust is None: st.warning("Customer not found.")
    else:
        m = st.columns(3); m[0].metric("Risk score (0-100)", int(cust["risk_score"])); m[1].metric("Risk category", cust["risk_category"]); m[2].metric("Credit score", int(cust["credit_score"]))
        st.write("Top risk factors:", cust["top_risk_factors"]); st.caption(f"Scored how: {cust['scored_how']}"); st.dataframe(apps, use_container_width=True)
    ms = S.model_summary()["default_model"]
    if ms:
        st.subheader("Model performance (held-out customers)"); t = ms["test_default_threshold"]; st.write(f"Champion: {ms['champion']} | ROC-AUC {t['ROC_AUC']:.3f} | PRD target 0.80 not met | recall at business threshold {ms['test_at_recall70_threshold']['Recall']:.2f} (target 0.70 not met)")
        st.image(str(pathlib.Path(__file__).resolve().parents[1] / "images" / "ml_default_evaluation.png"))
