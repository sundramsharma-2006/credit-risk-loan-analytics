"""Dashboard data-layer tests (run after src.train, src.risk_score): python -m unittest discover -s tests -v"""
import unittest

import pandas as pd

from dashboard import services as S
from src import data_load as dl


class DashboardServices(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from src import export
        export.run(); cls.d = S.load(); t, _ = dl.clean(); cls.con = dl.build_sqlite(tables=t); dl.add_scores(cls.con); cls.q = dl.run_dashboard_queries(cls.con)

    def test_kpis_unfiltered(self):
        k = S.kpis(*S.apply_filters(self.d)); self.assertEqual(k["applications"], 7500); self.assertAlmostEqual(k["approval_rate_pct"], 46.4, places=1); self.assertAlmostEqual(k["default_rate_pct"], 23.4, places=1)
        self.assertAlmostEqual(k["disbursed_crore"], 441.2, places=1)

    def test_d1_matches_service_trend(self):
        sql = self.q["D1"]["df"].set_index("month"); svc = S.approval_trend(self.d["loans"]).set_index("month")
        self.assertTrue((sql.approved == svc.approved).all()); self.assertTrue(((sql.approval_rate_pct - svc.approval_rate_pct).abs() < 0.06).all())

    def test_d2_matches_service(self):
        sql = self.q["D2"]["df"].set_index("loan_type").default_rate_pct; svc = S.default_by_loan_type(self.d["loans"]).set_index("loan_type").default_rate_pct
        for k in sql.index: self.assertAlmostEqual(sql[k], svc[k], delta=0.06)

    def test_d3_risk_distribution_covers_all_customers(self):
        self.assertEqual(int(self.q["D3"]["df"].customers.sum()), 5000); self.assertEqual(int(S.risk_distribution(self.d["customers"]).customers.sum()), 5000)

    def test_d5_state_applications_sum(self):
        self.assertEqual(int(self.q["D5"]["df"].applications.sum()), 7500); self.assertEqual(self.q["D5"]["df"].shape[0], 13)

    def test_d8_only_high_risk_sorted(self):
        w = self.q["D8"]["df"]; self.assertTrue((w.risk_category == "High Risk").all()); self.assertTrue(w.risk_score.is_monotonic_decreasing)

    def test_filters_reduce_rows_and_combine(self):
        L, C = S.apply_filters(self.d, ["Home"], None, ["High Risk"]); self.assertTrue((L.loan_type == "Home").all()); self.assertTrue((L.risk_category == "High Risk").all()); self.assertLess(len(L), 7500)

    def test_watchlist_state_filter(self):
        st_ = self.d["customers"].state.iloc[0]; w = S.watchlist(self.d["customers"], ("High Risk",), [st_], 50); self.assertTrue((w.state == st_).all())

    def test_no_pii_columns_in_dashboard_tables(self):
        for df in (self.d["customers"], self.d["loans"]):
            self.assertFalse({"email", "phone", "first_name", "last_name"} & set(df.columns))

    def test_applicant_lookup(self):
        c, a = S.applicant(self.d, "CUST100001"); self.assertEqual(c["customer_id"], "CUST100001"); self.assertGreaterEqual(len(a), 1); self.assertIsNone(S.applicant(self.d, "NOPE")[0])


if __name__ == "__main__":
    unittest.main()
