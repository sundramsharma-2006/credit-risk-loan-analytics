"""risk_score.py - 0-100 customer risk score and Low/Medium/High bands (PRD F4, section 13.7).

risk_score = round(P(default) x 100) using a CALIBRATED probability, so a score of 30 means roughly "30% of similar loans
default", not just a rank.
Two routes: customers who applied get the HIGHEST calibrated P(default) over their applications (application-level model, src.train);
customers who never applied (1,096) get a weaker profile-only model.  Rejected/under-review applications were never followed up, so
their probabilities are extrapolations (selection bias) - see model card.
Outputs: models/risk_model.pkl, data/processed/scored_customers.csv, reports/risk_score_summary.json, images/risk_*.png
Run: python -m src.risk_score   (after src.features)
"""
import json
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import modeling as M
from src.config import IMAGES, MODELS, PROCESSED, REPORTS, SEED
from src.features import CATEGORICAL_CUST, NUM_CUST, application_features, customer_features, design_matrix
from src.models import ml_compat as ml

PRD_BANDS = (33, 66)                       # 0-33 Low, 34-66 Medium, 67-100 High (PRD default)
LABEL = {"credit_score": "credit score", "debt_to_limit": "debt-to-limit", "credit_utilization_ratio": "credit utilisation", "num_late_payments_30d": "30-day late payments",
         "num_late_payments_90d": "90-day late payments", "num_defaults_prior": "prior defaults", "annual_income": "annual income", "total_outstanding_debt": "outstanding debt",
         "avg_balance": "average balance", "min_balance": "minimum balance", "emi_outflow": "EMI outflow", "net_cash_flow": "net cash flow", "late_payments_total": "late payments",
         "total_inflow": "inflow", "total_outflow": "outflow", "num_credit_inquiries_6m": "credit inquiries", "credit_age_years": "credit history length", "age": "age"}


def band(score, cuts=PRD_BANDS):
    return np.where(score <= cuts[0], "Low", np.where(score <= cuts[1], "Medium", "High"))


def run(verbose=True):
    a = application_features(); d = a[a.loan_default.notna()].reset_index(drop=True); y = d.loan_default.astype(int).values; cf = customer_features()
    sa = pd.read_csv(PROCESSED / "scored_applications.csv")                              # written by src.train (application-level default model)

    # ---- (A) customers who applied: risk = HIGHEST calibrated P(default) over their applications (conservative; one risky loan flags the customer)
    A = sa.groupby("customer_id").agg(default_probability=("default_probability", "max"), n_apps=("application_id", "size"))
    dis = sa[sa.scored_how.str.startswith("cross")].groupby("customer_id").agg(p_dis=("default_probability", "max"), ever=("loan_default", "max"))
    auc_A_dis = float(ml.roc_auc_score(dis.ever.astype(int), dis.p_dis))                  # honest check: cross-fitted, disbursed loans only

    # ---- (B) customers who never applied: profile-only model (no loan inputs) on the loan grain, grouped CV, Platt-calibrated
    X, med = design_matrix(d, NUM_CUST, CATEGORICAL_CUST); cols = list(X.columns); Xv = X.values; folds = M.group_folds(d.customer_id.values, y, 5, SEED); best = None; rows = []
    for fam, cands in M.candidates(SEED).items():
        if fam == "Decision Tree": continue
        for label, est in cands:
            p = M.oof_proba(est, Xv, y, folds); auc = ml.roc_auc_score(y, p); rows.append(dict(family=fam, params=label, oof_auc=auc))
            if best is None or auc > best[0]: best = (auc, fam, label, est, p)
    auc_B, fam, label, est, oof = best; platt = M.fit_platt(oof, y); model = M._clone(est).fit(Xv, y)
    never = cf[cf.n_applications == 0]; Xn = design_matrix(never, NUM_CUST, CATEGORICAL_CUST, medians=med, columns=cols)[0]; pB = M.apply_platt(model.predict_proba(Xn.values)[:, 1], platt)
    B = pd.DataFrame({"default_probability": pB, "n_apps": 0}, index=never.customer_id)
    if verbose: print(f"(A) application-based customers: {len(A)} | AUC of max-P vs ever-default (disbursed, cross-fitted) = {auc_A_dis:.3f}\n(B) profile-only model: {fam} ({label}) loan-grain OOF AUC {auc_B:.3f} -> scores {len(B)} customers who never applied")

    sc = pd.concat([A.assign(scored_how="application-based (max P over applications)"), B.assign(scored_how="profile-only model (never applied)")]).reset_index().rename(columns={"index": "customer_id"})
    sc = cf[["customer_id", "n_applications", "n_disbursed", "ever_default"]].merge(sc[["customer_id", "default_probability", "scored_how"]], on="customer_id", how="left")
    sc["default_probability"] = sc.default_probability.round(4); sc["risk_score"] = np.clip(np.round(sc.default_probability * 100), 0, 100).astype(int)
    assert len(sc) == 5000 and sc.default_probability.notna().all()

    # ---- bands: PRD defaults first, then check whether they are usable
    obs = sc[sc.ever_default.notna()].copy(); obs["band"] = band(obs.risk_score)
    prd_tab = obs.groupby("band").ever_default.agg(["mean", "size"]).reindex(["Low", "Medium", "High"]).round(3)
    if verbose: print("PRD bands 0-33/34-66/67-100 on all customers:", pd.Series(band(sc.risk_score)).value_counts().to_dict(), "\n", prd_tab)
    # The PRD default cut-offs turned out to be usable on calibrated probabilities (High band has ~2x the Low band's default rate), so they are kept - no re-tuning.
    cuts = PRD_BANDS; sc["risk_category"] = np.where(band(sc.risk_score) == "Low", "Low Risk", np.where(band(sc.risk_score) == "Medium", "Medium Risk", "High Risk"))
    obs = sc[sc.ever_default.notna()]; tuned_tab = obs.groupby("risk_category").ever_default.agg(["mean", "size"]).reindex(["Low Risk", "Medium Risk", "High Risk"]).round(3)
    auc_final = float(ml.roc_auc_score(obs.ever_default.astype(int), obs.default_probability))
    if verbose: print(f"bands used (<= {cuts[0]} Low, <= {cuts[1]} Medium, else High):", sc.risk_category.value_counts().to_dict(), "\n", tuned_tab, f"\nAUC of final customer score vs ever-default (labelled customers, includes extrapolated rejected apps) = {auc_final:.3f}")

    # ---- per-customer explanation: top factors from the application-level model where available (occlusion on the logit), profile model otherwise
    am = pickle.load(open(MODELS / "default_model.pkl", "rb")); app_all = a.copy(); XA = design_matrix(app_all, am["num"], am["cat"], medians=am["medians"], columns=am["columns"])[0]
    logit = lambda mdl, Z: np.log(np.clip(mdl.predict_proba(np.asarray(Z, float))[:, 1], 1e-6, 1 - 1e-6) / (1 - np.clip(mdl.predict_proba(np.asarray(Z, float))[:, 1], 1e-6, 1 - 1e-6)))
    medA = np.median(design_matrix(d, am["num"], am["cat"], medians=am["medians"], columns=am["columns"])[0].values, axis=0); VA = XA.values; base = logit(am["model"], VA); effA = np.zeros_like(VA)
    for j in range(VA.shape[1]):
        Z = VA.copy(); Z[:, j] = medA[j]; effA[:, j] = base - logit(am["model"], Z)
    medB = np.median(Xv, axis=0); VB = Xn.values; baseB = logit(model, VB); effB = np.zeros_like(VB)
    for j in range(VB.shape[1]):
        Z = VB.copy(); Z[:, j] = medB[j]; effB[:, j] = baseB - logit(model, Z)
    def fmt(names, e):
        idx = np.argsort(-e)[:3]; return "; ".join(f"{LABEL.get(names[j], names[j].replace('_', ' '))} (+{e[j]:.2f})" for j in idx if e[j] > 0.02) or "no factor above the typical customer"
    top = {}
    arg = pd.Series(base, index=app_all.customer_id.values).groupby(level=0).idxmax() if False else None
    pos = pd.DataFrame({"customer_id": app_all.customer_id.values, "b": base, "i": np.arange(len(base))}).sort_values("b").groupby("customer_id").tail(1)         # the application with the highest score
    for cid_, i in zip(pos.customer_id, pos.i): top[cid_] = fmt(am["columns"], effA[i])
    for k, cid_ in enumerate(never.customer_id): top[cid_] = fmt(cols, effB[k])
    sc["top_risk_factors"] = sc.customer_id.map(top)
    sc = sc.merge(cf[["customer_id", "city", "state", "employment_type", "annual_income", "credit_score"]], on="customer_id", how="left")
    sc.to_csv(PROCESSED / "scored_customers.csv", index=False); pd.DataFrame(rows).to_csv(PROCESSED / "risk_model_selection_grid.csv", index=False)
    pickle.dump(dict(profile_model=model, family=fam, params=label, columns=cols, medians=med, platt=platt, cuts=cuts, prd_bands=PRD_BANDS, seed=SEED, backend=ml.BACKEND, num=NUM_CUST, cat=CATEGORICAL_CUST), open(MODELS / "risk_model.pkl", "wb"))
    summary = dict(method="customer score = max calibrated P(default) over the customer's applications (application-level default model); customers who never applied use a profile-only model",
                   auc_application_based_vs_ever_default_cross_fitted=auc_A_dis, auc_final_score_vs_ever_default=auc_final, profile_model=dict(family=fam, params=label, oof_auc=float(auc_B)),
                   n_labelled_customers=len(obs), n_scored=len(sc), cuts=list(cuts), band_counts=sc.risk_category.value_counts().to_dict(), band_default_rates=tuned_tab["mean"].to_dict(), scored_how=sc.scored_how.value_counts().to_dict(),
                   mean_probability=float(sc.default_probability.mean()), observed_ever_default_rate=float(obs.ever_default.mean()))
    (REPORTS / "risk_score_summary.json").write_text(json.dumps(summary, indent=1, default=float)); plots(sc, obs, tuned_tab, cuts)
    return summary


def plots(sc, obs, tab, cuts):
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.6)); ax[0].hist(sc.risk_score, bins=40, color="#0B5FA5"); [ax[0].axvline(c, color="r", ls="--") for c in cuts]; ax[0].set(title="Risk score distribution (all 5,000 customers)", xlabel="risk score")
    ax[1].bar(tab.index, tab["mean"] * 100, color=["#2E9E5B", "#E0A526", "#C23B3B"]); ax[1].set(title="Observed default rate by band (customers with an outcome)", ylabel="% defaulted"); [ax[1].text(i, v * 100 + .5, f"{v*100:.0f}%  (n={int(n)})", ha="center") for i, (v, n) in enumerate(zip(tab["mean"], tab["size"]))]
    c = sc.risk_category.value_counts().reindex(["Low Risk", "Medium Risk", "High Risk"]); ax[2].pie(c, labels=c.index, autopct="%1.0f%%", colors=["#2E9E5B", "#E0A526", "#C23B3B"]); ax[2].set_title("Customer risk distribution"); plt.tight_layout(); plt.savefig(IMAGES / "risk_scoring.png", dpi=110); plt.close()


if __name__ == "__main__":
    s = run(); print(json.dumps({k: s[k] for k in ["cuts", "band_default_rates", "scored_how"]}, default=float, indent=1))
