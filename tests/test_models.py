"""Model/feature/scoring tests (run after src.train and src.risk_score): python -m unittest discover -s tests -v"""
import pickle
import unittest

import numpy as np
import pandas as pd

from src import features as fe
from src import modeling as M
from src.config import MODELS, PROCESSED
from src.risk_score import band


class Splits(unittest.TestCase):
    def setUp(self):
        rng = np.random.RandomState(0); self.g = rng.randint(0, 300, 1000).astype(str); self.y = (rng.rand(1000) < 0.25).astype(int)

    def test_group_split_has_no_customer_overlap(self):
        tr, te = M.group_split(self.g, self.y, 0.2, 1); self.assertFalse(set(self.g[tr]) & set(self.g[te])); self.assertEqual(len(tr) + len(te), 1000)

    def test_group_folds_partition_and_no_overlap(self):
        folds = M.group_folds(self.g, self.y, 5, 1); seen = np.concatenate([v for _, v in folds]); self.assertEqual(sorted(seen), list(range(1000)))
        for tr, va in folds: self.assertFalse(set(self.g[tr]) & set(self.g[va]))

    def test_split_is_reproducible(self):
        a = M.group_split(self.g, self.y, 0.2, 7); b = M.group_split(self.g, self.y, 0.2, 7); self.assertTrue((a[1] == b[1]).all())


class Calibration(unittest.TestCase):
    def test_platt_matches_base_rate(self):
        rng = np.random.RandomState(1); y = (rng.rand(5000) < 0.23).astype(int); raw = np.clip(0.5 + 0.1 * (y - 0.23) + rng.randn(5000) * 0.1, 0.01, 0.99)    # inflated, weakly informative scores
        c = M.apply_platt(raw, M.fit_platt(raw, y)); self.assertAlmostEqual(c.mean(), y.mean(), delta=0.01)

    def test_recall_threshold_meets_target_on_the_data_it_was_chosen_on(self):
        rng = np.random.RandomState(2); y = (rng.rand(3000) < 0.25).astype(int); p = np.clip(0.2 + 0.25 * y + rng.randn(3000) * 0.15, 0, 1)
        _, t = M.thresholds(p, y, 0.70); rec = ((p >= t) & (y == 1)).sum() / y.sum(); self.assertGreaterEqual(rec, 0.70)


class Bands(unittest.TestCase):
    def test_prd_bands(self):
        self.assertEqual(list(band(np.array([0, 33, 34, 66, 67, 100]))), ["Low", "Low", "Medium", "Medium", "High", "High"])


class Features(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.t = fe.load_tables(); cls.a = fe.application_features(cls.t); cls.c = fe.customer_features(cls.t)

    def test_no_row_fan_out(self):
        self.assertEqual(len(self.a), 7500); self.assertTrue(self.a.application_id.is_unique); self.assertEqual(len(self.c), 5000); self.assertTrue(self.c.customer_id.is_unique)

    def test_transaction_aggregate_is_customer_grain(self):
        self.assertTrue(fe.transaction_features(self.t["transactions"]).customer_id.is_unique)

    def test_no_leaky_or_pii_columns_in_model_inputs(self):
        X, _ = fe.design_matrix(self.a[self.a.loan_default.notna()], fe.NUM_APP, fe.CATEGORICAL_APP); bad = ("approval_status", "approved_amount", "decision_date", "disbursed", "email", "phone", "first_name", "last_name", "gender", "marital", "loan_default")
        self.assertEqual([c for c in X.columns if c.startswith(bad)], [])

    def test_winsorising_uses_training_bounds_only(self):
        d = self.a[self.a.loan_default.notna()]; X, par = fe.design_matrix(d.iloc[:1000], fe.NUM_APP, fe.CATEGORICAL_APP); Y, _ = fe.design_matrix(d.iloc[1000:], fe.NUM_APP, fe.CATEGORICAL_APP, medians=par, columns=list(X.columns))
        self.assertLessEqual(Y["loan_to_income"].max(), par["hi"]["loan_to_income"] + 1e-9); self.assertEqual(list(Y.columns), list(X.columns))


class Artifacts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.b = pickle.load(open(MODELS / "default_model.pkl", "rb")); cls.sc = pd.read_csv(PROCESSED / "scored_customers.csv"); cls.sa = pd.read_csv(PROCESSED / "scored_applications.csv")

    def test_model_predicts_probabilities(self):
        a = fe.application_features(); X, _ = fe.design_matrix(a.head(200), self.b["num"], self.b["cat"], medians=self.b["medians"], columns=self.b["columns"])
        p = self.b["model"].predict_proba(X.values)[:, 1]; self.assertTrue(((p >= 0) & (p <= 1)).all())

    def test_every_application_scored(self):
        self.assertEqual(len(self.sa), 7500); self.assertTrue(self.sa.default_probability.between(0, 1).all()); self.assertTrue(self.sa.application_id.is_unique)

    def test_every_customer_scored_once_with_valid_range(self):
        self.assertEqual(len(self.sc), 5000); self.assertTrue(self.sc.customer_id.is_unique); self.assertTrue(self.sc.risk_score.between(0, 100).all())
        self.assertEqual(set(self.sc.risk_category), {"Low Risk", "Medium Risk", "High Risk"})

    def test_category_consistent_with_score(self):
        exp = np.where(self.sc.risk_score <= 33, "Low Risk", np.where(self.sc.risk_score <= 66, "Medium Risk", "High Risk")); self.assertTrue((exp == self.sc.risk_category).all())

    def test_score_equals_rounded_probability(self):
        self.assertTrue((np.abs(self.sc.risk_score - np.round(self.sc.default_probability * 100)) <= 1).all())

    def test_every_score_has_an_explanation(self):
        self.assertTrue(self.sc.top_risk_factors.notna().all())

    def test_bands_are_ordered_by_observed_default_rate(self):
        o = self.sc[self.sc.ever_default.notna()].groupby("risk_category").ever_default.mean(); self.assertLess(o["Low Risk"], o["Medium Risk"]); self.assertLess(o["Medium Risk"], o["High Risk"])

    def test_scored_disbursed_loans_are_cross_fitted(self):
        d = self.sa[self.sa.loan_default.notna()]; self.assertTrue(d.scored_how.str.startswith("cross-fitted").all())


if __name__ == "__main__":
    unittest.main()
