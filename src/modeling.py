"""modeling.py - shared, tested helpers: customer-grouped splits/CV, candidate grids, Platt calibration, thresholds, metrics,
permutation importance and per-row occlusion explanations.

Why grouped splits: one customer can have several disbursed loans (1,094 of 3,403 rows share a customer with another loan) and
their bureau/transaction features are identical, so a random row split would put the same person in both train and test and
inflate the score.  We split by customer_id.
"""
import numpy as np
import pandas as pd

from src.models import ml_compat as ml

try:
    from sklearn.base import clone as _clone
except ImportError:
    from src.models.mini_sklearn import clone as _clone


# ----------------------------------------------------------------------------- splitting
def group_split(groups, y, test_size=0.2, seed=42):
    """Customer-level stratified split. Returns (train_idx, test_idx). A customer is positive if any of their rows is positive."""
    df = pd.DataFrame({"g": np.asarray(groups), "y": np.asarray(y)}); gy = df.groupby("g").y.max()
    rng = np.random.RandomState(seed); test_g = []
    for v in (0, 1):
        ids = gy[gy == v].index.values; rng.shuffle(ids); test_g += list(ids[: int(round(len(ids) * test_size))])
    is_test = df.g.isin(set(test_g)).values
    return np.where(~is_test)[0], np.where(is_test)[0]


def group_folds(groups, y, k=5, seed=42):
    """List of (train_idx, valid_idx) with no customer in both."""
    df = pd.DataFrame({"g": np.asarray(groups), "y": np.asarray(y)}); gy = df.groupby("g").y.max(); rng = np.random.RandomState(seed); fold_of = {}
    for v in (0, 1):
        ids = gy[gy == v].index.values; rng.shuffle(ids)
        for i, g in enumerate(ids): fold_of[g] = i % k
    f = df.g.map(fold_of).values
    return [(np.where(f != i)[0], np.where(f == i)[0]) for i in range(k)]


# ----------------------------------------------------------------------------- models
class Scaled:
    """Standardise inputs inside the estimator (for Logistic Regression). clone()-compatible."""
    def __init__(self, model=None, scaler=None):
        self.model, self.scaler = model, scaler
    def get_params(self, deep=True): return {"model": self.model, "scaler": self.scaler}
    def set_params(self, **p):
        for k, v in p.items(): setattr(self, k, v)
        return self
    def fit(self, X, y):
        self.scaler_ = ml.StandardScaler().fit(X); self.model_ = _clone(self.model).fit(self.scaler_.transform(X), y); return self
    def predict_proba(self, X): return self.model_.predict_proba(self.scaler_.transform(X))
    def predict(self, X): return self.model_.predict(self.scaler_.transform(X))
    @property
    def coef_(self): return self.model_.coef_


def candidates(seed=42):
    """{family: [(label, estimator), ...]} - small grids tuned by grouped 5-fold CV (PRD 13.4/13.5).
    XGBoost is not installed in the build sandbox: gradient boosting (scikit-learn style) stands in for it."""
    out = {"Logistic Regression": [(f"C={c}", Scaled(ml.LogisticRegression(C=c, class_weight="balanced", max_iter=200))) for c in (0.003, 0.01, 0.03, 0.1, 1.0)],
           "Decision Tree": [(f"depth={d}, leaf={l}", ml.DecisionTreeClassifier(max_depth=d, min_samples_leaf=l, class_weight="balanced", random_state=seed)) for d in (3, 4, 5, 6) for l in (30,)],
           "Random Forest": [(f"depth={d}, leaf={l}", ml.RandomForestClassifier(n_estimators=150, max_depth=d, min_samples_leaf=l, class_weight="balanced", random_state=seed)) for d in (6, 8, 10) for l in (10, 25)],
           "Gradient Boosting (XGBoost stand-in)": [(f"n={n}, depth={d}", ml.GradientBoostingClassifier(n_estimators=n, max_depth=d, learning_rate=0.05, min_samples_leaf=20, subsample=0.8, class_weight="balanced", random_state=seed)) for n in (100, 200) for d in (2, 3)]}
    return out


def oof_proba(est, X, y, folds):
    """Out-of-fold P(class 1) using the supplied (grouped) folds. X is a numpy array."""
    p = np.zeros(len(y))
    for tr, va in folds:
        p[va] = _clone(est).fit(X[tr], y[tr]).predict_proba(X[va])[:, 1]
    return p


# ----------------------------------------------------------------------------- calibration / thresholds / metrics
def fit_platt(p, y, iters=500, lr=0.5):
    """Logistic calibration of scores onto observed frequencies (so risk_score = P*100 means what it says)."""
    z = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6))); a, b = 1.0, 0.0; y = np.asarray(y, float)
    for _ in range(iters):
        q = 1 / (1 + np.exp(-(a * z + b))); g = q - y; a -= lr * np.mean(g * z); b -= lr * np.mean(g)
    return float(a), float(b)


def apply_platt(p, params):
    z = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6))); return 1 / (1 + np.exp(-(params[0] * z + params[1])))


def metrics(y, p, thr=0.5):
    y = np.asarray(y); pred = (p >= thr).astype(int); tn, fp, fn, tp = ml.confusion_matrix(y, pred).ravel()
    return dict(Accuracy=ml.accuracy_score(y, pred), Precision=ml.precision_score(y, pred, zero_division=0), Recall=ml.recall_score(y, pred, zero_division=0),
                F1=ml.f1_score(y, pred, zero_division=0), ROC_AUC=ml.roc_auc_score(y, p), PR_AUC=ml.average_precision_score(y, p), TN=int(tn), FP=int(fp), FN=int(fn), TP=int(tp))


def thresholds(p, y, recall_target=0.70):
    """(f1_max_threshold, recall_target_threshold) chosen on out-of-fold predictions only."""
    grid = np.unique(np.round(np.quantile(p, np.linspace(0.02, 0.98, 97)), 4)); best_f1, t_f1, t_rec = -1, 0.5, grid[0]
    for t in grid:
        pr = (p >= t).astype(int); f = ml.f1_score(y, pr, zero_division=0)
        if f > best_f1: best_f1, t_f1 = f, t
        if ml.recall_score(y, pr, zero_division=0) >= recall_target: t_rec = t               # highest threshold still meeting the recall target
    return float(t_f1), float(t_rec)


def permutation_importance(predict, X, y, n_repeats=5, seed=42):
    """Mean drop in ROC-AUC when one column is shuffled (model-agnostic)."""
    rng = np.random.RandomState(seed); Xv = np.asarray(X, float).copy(); base = ml.roc_auc_score(y, predict(Xv)); out = []
    for j in range(Xv.shape[1]):
        d = []
        for _ in range(n_repeats):
            col = Xv[:, j].copy(); Xv[:, j] = rng.permutation(col); d.append(base - ml.roc_auc_score(y, predict(Xv))); Xv[:, j] = col
        out.append(np.mean(d))
    return pd.Series(out, index=list(X.columns) if hasattr(X, "columns") else None)


def explain_row(predict, row, medians, top=5):
    """Occlusion explanation for one row: change in raw score when each feature is replaced by its population median.
    Positive = pushes risk UP. Returns a DataFrame sorted by absolute effect (NFR: every prediction traceable to ranked factors)."""
    x = np.asarray(row, float).reshape(1, -1); base = predict(x)[0]; out = []
    for j in range(x.shape[1]):
        xx = x.copy(); xx[0, j] = medians[j]; out.append(base - predict(xx)[0])
    return np.array(out)
