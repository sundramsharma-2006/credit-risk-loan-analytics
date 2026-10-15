"""ml_compat.py - one import point for ML tooling: real scikit-learn when installed, otherwise the NumPy fallback."""
try:
    from sklearn.linear_model import LinearRegression, LogisticRegression
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
    from sklearn.ensemble import (RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier as _SKGBC, GradientBoostingRegressor)
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV, StratifiedKFold, KFold
    from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix, average_precision_score,
                                 precision_recall_curve, mean_absolute_error, mean_squared_error, r2_score, silhouette_score)
    from sklearn.utils.class_weight import compute_sample_weight
    import sklearn
    BACKEND = f"scikit-learn {sklearn.__version__}"

    class GradientBoostingClassifier(_SKGBC):            # scikit-learn's GB has no class_weight -> support it through sample weights
        def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3, min_samples_leaf=1, class_weight=None, random_state=None):
            super().__init__(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth, min_samples_leaf=min_samples_leaf, random_state=random_state)
            self.class_weight = class_weight

        def fit(self, X, y, sample_weight=None):
            if self.class_weight == "balanced":
                sample_weight = compute_sample_weight("balanced", y)
            return super().fit(X, y, sample_weight=sample_weight)
except ImportError:                                       # pragma: no cover - offline fallback
    from src.models.mini_sklearn import *                 # noqa: F401,F403
    from src.models.mini_sklearn import KMeans, PCA, silhouette_score
    BACKEND = "mini_sklearn (NumPy fallback, scikit-learn API subset)"
    try:
        precision_recall_curve
    except NameError:
        import numpy as _np

        def precision_recall_curve(y, s):
            y = _np.asarray(y); o = _np.argsort(-_np.asarray(s)); y = y[o]; tp = _np.cumsum(y); k = _np.arange(1, len(y) + 1)
            return tp / k, tp / max(y.sum(), 1), _np.asarray(s)[o]
