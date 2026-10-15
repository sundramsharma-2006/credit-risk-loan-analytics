"""SQL-layer tests: python -m unittest discover -s tests -v   (set CREDORA_DB_PATH to a writable path)"""
import sqlite3
import unittest

import numpy as np
import pandas as pd

from src import data_load as dl


class SqlLayer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = dl.load_raw(); cls.t, cls.log = dl.clean(cls.raw); cls.con = dl.build_sqlite(tables=cls.t); cls.res = dl.run_analytics(cls.con)
        cls.c, cls.h, cls.l, cls.x = (cls.t[k] for k in dl.TABLES)

    def test_cleaned_data_passes_all_checks(self):
        v = dl.validate(self.t); self.assertTrue(v.passed.all(), v[~v.passed].check.tolist())

    def test_raw_email_issue_is_detected(self):
        v = dl.validate(self.raw); self.assertEqual(v[~v.passed].check.tolist(), ["customers.email is unique (schema has UNIQUE)"])

    def test_row_counts(self):
        n = {tb: self.con.execute(f"SELECT COUNT(*) FROM {tb}").fetchone()[0] for tb in dl.TABLES}
        self.assertEqual(n, {"customers": 5000, "credit_history": 5000, "loan_applications": 7500, "transactions": 40079})

    def test_ten_queries_run_and_return_rows(self):
        self.assertEqual(len(self.res), 10)
        bad = {k: v["error"] for k, v in self.res.items() if v["error"] or len(v["df"]) == 0}
        self.assertEqual(bad, {})

    def test_constraints_reject_bad_rows(self):
        bad = ["INSERT INTO credit_history (credit_id,customer_id,as_of_date,credit_score) VALUES ('X1','CUST100001','2024-06-30',950)",
               "INSERT INTO credit_history (credit_id,customer_id,as_of_date,credit_score,credit_utilization_ratio) VALUES ('X2','CUST100002','2024-06-30',600,1.5)",
               "INSERT INTO transactions (transaction_id,customer_id,transaction_date,transaction_type,amount) VALUES ('T1','NOPE','2024-01-01','Debit',5)",
               "INSERT INTO transactions (transaction_id,customer_id,transaction_date,transaction_type,amount) VALUES ('T2','CUST100001','2024-01-01','Refund',5)",
               "INSERT INTO loan_applications (application_id,customer_id,application_date,loan_type,approval_status) VALUES ('A1','CUST100001','2024-01-01','Home','Maybe')",
               "INSERT INTO loan_applications (application_id,customer_id,application_date,loan_type,approval_status,decision_date) VALUES ('A2','CUST100001','2024-01-05','Home','Approved','2024-01-01')"]
        for s in bad:
            with self.assertRaises(sqlite3.IntegrityError, msg=s):
                self.con.execute(s)

    def test_q71_credit_bands_match_pandas(self):
        q = self.res["7.1"]["df"].set_index("credit_band").customers; h = self.h
        b = pd.cut(h.credit_score, [0, 579, 649, 699, 749, 1000], labels=["Very Poor (300-579)", "Poor (580-649)", "Fair (650-699)", "Good (700-749)", "Excellent (750-900)"]).value_counts()
        for k in q.index: self.assertEqual(q[k], b[k])
        self.assertEqual(q.sum(), 5000)

    def test_q72_approval_rate_overall(self):
        d = self.res["7.2"]["df"]; self.assertEqual(d.applications.sum(), 7500); self.assertEqual(d.approved.sum(), 3403)

    def test_q75_default_rate_by_loan_type(self):
        d = self.res["7.5"]["df"].set_index("loan_type"); p = self.l.dropna(subset=["loan_default"]).groupby("loan_type").loan_default.mean() * 100
        for k in d.index: self.assertAlmostEqual(d.loc[k, "default_rate_pct"], round(p[k], 2), places=2)
        self.assertEqual(d.disbursed_loans.sum(), 3403)

    def test_q76_median_matches_pandas(self):
        d = self.res["7.6"]["df"].set_index("loan_type"); p = self.l.groupby("loan_type").requested_amount.median()
        for k in d.index: self.assertAlmostEqual(d.loc[k, "median_amount"], round(p[k]), delta=1)

    def test_q77_disbursed_value(self):
        d = self.res["7.7"]["df"]; self.assertAlmostEqual(d.disbursed_amount_mn.sum(), self.l.approved_amount.sum() / 1e6, delta=0.3)

    def test_q78_risk_categories_cover_everyone(self):
        d = self.res["7.8"]["df"]; self.assertEqual(d.customers.sum(), 5000); self.assertEqual(set(d.risk_category), {"Low Risk", "Medium Risk", "High Risk"})

    def test_q710_driver_default_rates(self):
        d = self.res["7.10"]["df"].set_index("driver"); m = self.l.merge(self.h, on="customer_id").dropna(subset=["loan_default"])
        self.assertAlmostEqual(d.loc["credit_score<600", "default_rate_pct"], round(100 * m[m.credit_score < 600].loan_default.mean(), 2), places=2)
        self.assertEqual(d.loc["dti>0.5", "n"], int((m.debt_to_income_ratio > 0.5).sum()))


if __name__ == "__main__":
    unittest.main()
