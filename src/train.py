"""train.py - loan-default model (PRD F3, section 13).
Trains Logistic Regression, Decision Tree, Random Forest and Gradient Boosting (XGBoost stand-in) on DISBURSED loans only, with
a customer-grouped 80/20 split, grouped 5-fold CV for tuning/selection (test set touched once), class-weighting for the ~23%
default rate, Platt calibration and two operating thresholds chosen on out-of-fold predictions.
Outputs: models/default_model.pkl, data/processed/{model_comparison,model_selection_grid,feature_importance,scored_applications}.csv,
reports/default_model_summary.json, images/ml_*.png.   Run: python -m src.train
"""
import json
import pickle
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import modeling as M
from src.config import IMAGES, MODELS, PROCESSED, REPORTS, SEED
from src.features import CATEGORICAL_APP, NUM_APP, TARGET, application_features, design_matrix
from src.models import ml_compat as ml

BUREAU = ["credit_score", "num_open_accounts", "num_credit_inquiries_6m", "credit_utilization_ratio", "num_late_payments_30d", "num_late_payments_90d",
          "num_defaults_prior", "bankruptcies", "total_credit_limit", "total_outstanding_debt", "credit_age_years", "debt_to_limit", "late_payments_total"]
TXN = ["txn_count", "total_inflow", "total_outflow", "emi_outflow", "avg_balance", "min_balance", "net_cash_flow", "inflow_outflow_ratio", "emi_share_of_outflow"]


def predict_fn(model):
    return lambda X: model.predict_proba(np.asarray(X, float))[:, 1]


def run(verbose=True):
    t0 = time.time(); a = application_features(); a["row_id"] = np.arange(len(a)); d = a[a[TARGET].notna()].reset_index(drop=True)
    y = d[TARGET].astype(int).values; groups = d.customer_id.values
    tr, te = M.group_split(groups, y, 0.2, SEED); Xdf, med = design_matrix(d.iloc[tr], NUM_APP, CATEGORICAL_APP); cols = list(Xdf.columns)
    Xtr = Xdf.values; Xte = design_matrix(d.iloc[te], NUM_APP, CATEGORICAL_APP, medians=med, columns=cols)[0].values; ytr, yte = y[tr], y[te]
    assert not set(groups[tr]) & set(groups[te]), "customer leakage between train and test"
    folds = M.group_folds(groups[tr], ytr, 5, SEED)
    if verbose: print(f"backend={ml.BACKEND} | disbursed loans={len(d)} default rate={y.mean():.3f} | train={len(tr)} test={len(te)} | features={len(cols)}")

    # ---- tune each family by grouped CV AUC; test metrics at the 0.5 threshold with class weights
    grid_rows, comp, fitted = [], [], {}
    comp.append(dict(Model="Baseline: predict no default", Stage="baseline", CV_AUC=0.5, **M.metrics(yte, np.zeros(len(yte)) + 1e-9, 0.5) | {"ROC_AUC": 0.5, "PR_AUC": yte.mean()}))
    for fam, cands in M.candidates(SEED).items():
        best = None
        for label, est in cands:
            aucs = [ml.roc_auc_score(ytr[v], M._clone(est).fit(Xtr[t], ytr[t]).predict_proba(Xtr[v])[:, 1]) for t, v in folds]
            grid_rows.append(dict(family=fam, params=label, cv_auc_mean=np.mean(aucs), cv_auc_std=np.std(aucs)))
            if best is None or np.mean(aucs) > best[0]: best = (np.mean(aucs), np.std(aucs), label, est)
        cv, sd, label, est = best; m = M._clone(est).fit(Xtr, ytr); p = m.predict_proba(Xte)[:, 1]; fitted[fam] = (m, label)
        comp.append(dict(Model=fam, Stage=label, CV_AUC=cv, CV_AUC_std=sd, **M.metrics(yte, p, 0.5)))
        if verbose: print(f"{fam:38s} {label:22s} CV AUC {cv:.3f}+-{sd:.3f} | test AUC {comp[-1]['ROC_AUC']:.3f} recall@0.5 {comp[-1]['Recall']:.2f} precision {comp[-1]['Precision']:.2f}")
    comp = pd.DataFrame(comp); pd.DataFrame(grid_rows).to_csv(PROCESSED / "model_selection_grid.csv", index=False)

    # ---- champion by CV ROC-AUC (never by the test set)
    fam = comp[comp.Stage != "baseline"].sort_values("CV_AUC", ascending=False).iloc[0].Model; model, label = fitted[fam]
    oof = M.oof_proba(model, Xtr, ytr, folds); platt = M.fit_platt(oof, ytr); oof_c = M.apply_platt(oof, platt)
    t_f1, t_rec = M.thresholds(oof_c, ytr, 0.70); p_raw = model.predict_proba(Xte)[:, 1]; p_te = M.apply_platt(p_raw, platt)
    m_f1, m_rec, m_def = M.metrics(yte, p_te, t_f1), M.metrics(yte, p_te, t_rec), M.metrics(yte, p_te, 0.5)
    if verbose: print(f"\nchampion: {fam} ({label}) | calibrated test AUC {m_def['ROC_AUC']:.3f}\n  F1-max threshold {t_f1:.3f}: recall {m_f1['Recall']:.2f} precision {m_f1['Precision']:.2f} F1 {m_f1['F1']:.2f}\n  recall>=0.70 threshold {t_rec:.3f}: recall {m_rec['Recall']:.2f} precision {m_rec['Precision']:.2f} F1 {m_rec['F1']:.2f}")

    # ---- diagnostics that test whether the headline number can be trusted
    diag = {}
    Xn, Xnt, yn, ynt = ml.train_test_split(np.vstack([Xtr, Xte]), np.r_[ytr, yte], test_size=0.2, random_state=SEED, stratify=np.r_[ytr, yte])
    diag["naive_random_split_auc"] = ml.roc_auc_score(ynt, M._clone(model).fit(Xn, yn).predict_proba(Xnt)[:, 1])          # same-customer rows in both sets
    yr = d.iloc[te].application_date.str[:4].values; diag["test_auc_by_application_year"] = {k: float(ml.roc_auc_score(yte[yr == k], p_te[yr == k])) for k in sorted(set(yr)) if len(set(yte[yr == k])) == 2 and (yr == k).sum() > 60}
    diag["test_rows_by_application_year"] = {k: int((yr == k).sum()) for k in sorted(set(yr))}
    for name, drop in [("without_bureau_features", BUREAU), ("without_bureau_and_transaction_features", BUREAU + TXN), ("application_features_only", [c for c in NUM_APP if c in BUREAU + TXN or c in ["age", "dependents", "annual_income", "years_at_residence", "tenure_years"]])]:
        keep = [i for i, c in enumerate(cols) if c not in drop]; diag[f"test_auc_{name}"] = ml.roc_auc_score(yte, M._clone(model).fit(Xtr[:, keep], ytr).predict_proba(Xte[:, keep])[:, 1])
    diag["single_feature_auc"] = {c: float(max(ml.roc_auc_score(yte, Xte[:, cols.index(c)]), 1 - ml.roc_auc_score(yte, Xte[:, cols.index(c)]))) for c in ["debt_to_income_ratio", "credit_score", "loan_to_income", "annual_income"]}
    rng = np.random.RandomState(SEED); cust_te = groups[te]; uc = np.unique(cust_te); idx_by = {c: np.where(cust_te == c)[0] for c in uc}; bs = []
    for _ in range(300):                                    # customer-level bootstrap: rows of one customer stay together
        ii = np.concatenate([idx_by[c] for c in rng.choice(uc, len(uc))])
        if len(set(yte[ii])) == 2: bs.append(ml.roc_auc_score(yte[ii], p_te[ii]))
    diag["test_auc_95ci_customer_bootstrap"] = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]; diag["test_defaults"] = int(yte.sum())
    if verbose: print("diagnostics:", json.dumps(diag, indent=1, default=float))

    # ---- importance (permutation AUC drop on the test set) + built-in importances
    imp = M.permutation_importance(predict_fn(model), pd.DataFrame(Xte, columns=cols), yte, n_repeats=5, seed=SEED).rename("auc_drop_when_shuffled").to_frame()
    if hasattr(model, "feature_importances_"): imp["builtin_importance"] = model.feature_importances_
    elif hasattr(model, "model_"): imp["abs_coef_standardised"] = np.abs(model.coef_.ravel())
    imp = imp.sort_values("auc_drop_when_shuffled", ascending=False); imp.index.name = "feature"; imp.reset_index().to_csv(PROCESSED / "feature_importance.csv", index=False)
    comp.to_csv(PROCESSED / "model_comparison.csv", index=False)
    if verbose: print("\ntop drivers:\n", imp.head(8).round(4).to_string())

    # ---- score every application: disbursed loans use cross-fitted (honest) predictions; non-disbursed use the champion
    Xall, _ = design_matrix(d, NUM_APP, CATEGORICAL_APP, medians=med, columns=cols); folds_all = M.group_folds(groups, y, 5, SEED)
    oof_all = M.apply_platt(M.oof_proba(model, Xall.values, y, folds_all), platt)
    rest = a[a[TARGET].isna()]; Xr = design_matrix(rest, NUM_APP, CATEGORICAL_APP, medians=med, columns=cols)[0].values
    p_rest = M.apply_platt(model.predict_proba(Xr)[:, 1], platt)
    sc = pd.concat([pd.DataFrame({"application_id": d.application_id, "customer_id": d.customer_id, "default_probability": oof_all, "scored_how": "cross-fitted (loan was disbursed)"}),
                    pd.DataFrame({"application_id": rest.application_id, "customer_id": rest.customer_id, "default_probability": p_rest, "scored_how": "extrapolated (loan not disbursed - outcome never observed)"})])
    sc = sc.merge(a[["application_id", "loan_type", "approval_status", "application_date", "requested_amount", TARGET]], on="application_id"); sc["default_probability"] = sc.default_probability.round(4)
    sc.to_csv(PROCESSED / "scored_applications.csv", index=False)

    bundle = dict(model=model, family=fam, params=label, columns=cols, medians=med, platt=platt, threshold_f1=t_f1, threshold_recall70=t_rec, seed=SEED, backend=ml.BACKEND,
                  num=NUM_APP, cat=CATEGORICAL_APP)
    MODELS.mkdir(exist_ok=True); pickle.dump(bundle, open(MODELS / "default_model.pkl", "wb"))
    summary = dict(backend=ml.BACKEND, champion=fam, params=label, n_disbursed=len(d), n_train=len(tr), n_test=len(te), default_rate=float(y.mean()), n_features=len(cols), cv_auc=float(comp[comp.Model == fam].CV_AUC.iloc[0]),
                   test_default_threshold=m_def, threshold_f1=t_f1, test_at_f1_threshold=m_f1, threshold_recall70=t_rec, test_at_recall70_threshold=m_rec, diagnostics=diag,
                   targets=dict(roc_auc_ge_0_80=bool(m_def["ROC_AUC"] >= 0.80), recall_ge_0_70=bool(m_rec["Recall"] >= 0.70)), top_features=imp.head(10).auc_drop_when_shuffled.round(4).to_dict(),
                   calibration=dict(platt=platt, mean_calibrated_oof=float(oof_c.mean()), actual_rate=float(ytr.mean())), seconds=round(time.time() - t0, 1))
    (REPORTS / "default_model_summary.json").write_text(json.dumps(summary, indent=1, default=float)); plots(comp, yte, p_te, m_rec, t_rec, imp, p_raw, ytr, oof_c)
    return summary


def plots(comp, yte, p, m_rec, t_rec, imp, p_raw, ytr, oof_c):
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.8)); fpr, tpr, _ = ml.roc_curve(yte, p); ax[0].plot(fpr, tpr, color="#0B5FA5", lw=2, label=f"AUC {ml.roc_auc_score(yte, p):.3f}"); ax[0].plot([0, 1], [0, 1], "k--"); ax[0].set(title="ROC curve (held-out customers)", xlabel="False positive rate", ylabel="Recall"); ax[0].legend()
    cm = np.array([[m_rec["TN"], m_rec["FP"]], [m_rec["FN"], m_rec["TP"]]]); ax[1].imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2): ax[1].text(j, i, f"{cm[i, j]}", ha="center", va="center", fontsize=15, color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax[1].set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Pred repay", "Pred default"], yticklabels=["Actual repay", "Actual default"], title=f"Confusion matrix @ threshold {t_rec:.2f} (recall target 0.70)")
    c = comp[comp.Stage != "baseline"]; ax[2].barh(c.Model, c.CV_AUC, xerr=c.CV_AUC_std, color="#0B5FA5"); ax[2].axvline(0.80, color="r", ls="--", label="PRD target 0.80"); ax[2].set(xlim=(0.5, 0.85), title="Grouped 5-fold CV ROC-AUC"); ax[2].legend(); plt.tight_layout(); plt.savefig(IMAGES / "ml_default_evaluation.png", dpi=110); plt.close()
    fig, ax = plt.subplots(1, 2, figsize=(14, 5)); i = imp.head(12)[::-1]; ax[0].barh(i.index, i.auc_drop_when_shuffled, color="#0B5FA5"); ax[0].set_title("Top drivers of default (AUC drop when shuffled)")
    bins = pd.qcut(oof_c, 10, duplicates="drop"); g = pd.DataFrame({"p": oof_c, "y": ytr}).groupby(bins, observed=True).mean(); ax[1].plot(g.p, g.y, "o-", color="#0B5FA5"); ax[1].plot([0, .7], [0, .7], "k--"); ax[1].set(title="Calibration (train out-of-fold, deciles)", xlabel="predicted P(default)", ylabel="observed default rate"); plt.tight_layout(); plt.savefig(IMAGES / "ml_default_drivers_calibration.png", dpi=110); plt.close()


if __name__ == "__main__":
    s = run(); print(json.dumps({k: s[k] for k in ["champion", "targets", "seconds"]}, default=float))
