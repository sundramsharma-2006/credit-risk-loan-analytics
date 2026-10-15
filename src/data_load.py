"""data_load.py - ingestion, validation and a SQLite mirror of the PostgreSQL schema (PRD F1, FR-01/02/03).

* load_raw()        read the four CSVs
* validate()        integrity / business-rule checks -> DataFrame (all must pass)
* build_sqlite()    create the schema from sql/credit_risk_platform.sql (sections 1-5), bulk-load the CSVs with
                    foreign keys ON, so constraint violations are really rejected
* run_analytics()   run the ten analytical queries *from the PostgreSQL file itself* after a mechanical dialect
                    translation (DATE_TRUNC / TO_CHAR / PERCENTILE_CONT / ::date)
PostgreSQL itself is not available in the build sandbox: the original .sql file is delivered unchanged for PostgreSQL.
CLI:  python -m src.data_load
"""
import re
import sqlite3
import statistics

import pandas as pd

from src.config import DB_PATH, PROCESSED, RAW, SQL_DIR, TABLES

SQL_FILE = SQL_DIR / "credit_risk_platform.sql"


def load_raw(path=RAW):
    return {t: pd.read_csv(path / f"{t}.csv") for t in TABLES}


def validate(t=None):
    """Return a DataFrame of (check, passed, detail). Every PRD data-integrity rule is a row."""
    t = t or load_raw(); c, h, l, x = t["customers"], t["credit_history"], t["loan_applications"], t["transactions"]
    ids = set(c.customer_id); rows = []

    def chk(name, ok, detail=""):
        rows.append((name, bool(ok), str(detail)))
    for tb, pk in [("customers", "customer_id"), ("credit_history", "credit_id"), ("loan_applications", "application_id"), ("transactions", "transaction_id")]:
        chk(f"{tb}.{pk} is unique", t[tb][pk].is_unique)
    chk("credit_history: exactly one row per customer", h.customer_id.is_unique and set(h.customer_id) == ids)
    for tb in ["credit_history", "loan_applications", "transactions"]:
        chk(f"no orphan foreign keys in {tb}", t[tb].customer_id.isin(ids).all(), f"orphans={(~t[tb].customer_id.isin(ids)).sum()}")
    chk("credit_score between 300 and 900", h.credit_score.between(300, 900).all())
    chk("credit_utilization_ratio between 0 and 1", h.credit_utilization_ratio.between(0, 1).all())
    chk("approval_status in allowed set", l.approval_status.isin(["Approved", "Rejected", "Under Review"]).all())
    chk("loan_default in {0,1} or NULL", l.loan_default.dropna().isin([0, 1]).all())
    dd = l.dropna(subset=["decision_date"])
    chk("decision_date >= application_date", (pd.to_datetime(dd.decision_date) >= pd.to_datetime(dd.application_date)).all())
    chk("customers.email is unique (schema has UNIQUE)", c.email.dropna().is_unique, f"duplicate emails={int(c.email.dropna().duplicated().sum())}")
    chk("loan_default present exactly for disbursed loans", (l.loan_default.notna() == (l.disbursed_flag == 1)).all())
    chk("only approved loans are disbursed", (l[l.disbursed_flag == 1].approval_status == "Approved").all())
    chk("approved_amount <= requested_amount", (l.approved_amount <= l.requested_amount).all())
    chk("transaction type in {Credit, Debit}", x.transaction_type.isin(["Credit", "Debit"]).all())
    chk("no negative amounts", (x.amount >= 0).all() and (l.requested_amount > 0).all())
    chk("kyc_status in allowed set", c.kyc_status.isin(["Verified", "Pending", "Rejected"]).all())
    chk("no missing values outside decision_date/loan_default/email", t["customers"].drop(columns=["email"]).isna().sum().sum() + h.isna().sum().sum() + x.isna().sum().sum() + l.drop(columns=["decision_date", "loan_default"]).isna().sum().sum() == 0)
    # documented data observations (informational, not failures)
    early = l.merge(c[["customer_id", "customer_since"]], on="customer_id")
    n_early = int((pd.to_datetime(early.customer_since) > pd.to_datetime(early.application_date)).sum())
    rows.append(("INFO applications dated before customer_since", True, f"{n_early} of {len(l)} - kept; documented in data_dictionary.md"))
    rows.append(("INFO bureau snapshot date is after most applications", True, f"as_of_date={h.as_of_date.unique().tolist()}, applications {l.application_date.min()}..{l.application_date.max()} - temporal-leakage risk, see model card"))
    return pd.DataFrame(rows, columns=["check", "passed", "detail"])


def clean(t=None):
    """Minimal, logged cleaning. The raw data is already tidy (no nulls/dups/orphans); the only blocking issue found is
    two e-mail addresses shared by different people, which would violate the schema's UNIQUE(email) and make the
    PostgreSQL bulk load fail. The later customer's e-mail is set to NULL (e-mail is not used for modelling)."""
    t = {k: v.copy() for k, v in (t or load_raw()).items()}; log = []
    c = t["customers"]; dup = c.email.notna() & c.email.duplicated(keep="first")
    if dup.any():
        log.append(dict(table="customers", issue="e-mail shared by two different customers (violates UNIQUE)", rows=int(dup.sum()),
                        action="kept first customer's e-mail, set later one to NULL", ids=",".join(c.loc[dup, "customer_id"])))
        c.loc[dup, "email"] = None
    t["customers"] = c; return t, pd.DataFrame(log)


# ----------------------------------------------------------------------------- SQLite mirror
def _ddl_from_file():
    txt = SQL_FILE.read_text(encoding="utf-8")
    ddl = txt.split("-- 6. SEED DATA")[0]                       # sections 0-5: DROP/CREATE TABLE/CREATE INDEX
    ddl = ddl.replace(" CASCADE;", ";")                          # SQLite has no DROP ... CASCADE
    return ddl[: ddl.rfind("-- ====")]                           # trim the banner that starts section 6


def connect(path=None):
    con = sqlite3.connect(str(path or DB_PATH)); con.execute("PRAGMA foreign_keys = ON")
    con.create_aggregate("median", 1, _Median); return con


class _Median:
    def __init__(self): self.v = []
    def step(self, x):
        if x is not None: self.v.append(x)
    def finalize(self): return statistics.median(self.v) if self.v else None


def build_sqlite(path=None, tables=None):
    path = path or DB_PATH
    if path.exists():
        path.unlink()
    t = tables or load_raw(); con = connect(path); con.executescript(_ddl_from_file())
    con.execute("PRAGMA foreign_keys = ON")
    for tb in TABLES:                                            # parents first
        df = t[tb].astype(object).where(t[tb].notna(), None)
        cols = ", ".join(df.columns); q = ",".join("?" * df.shape[1])
        con.executemany(f"INSERT INTO {tb} ({cols}) VALUES ({q})", df.itertuples(index=False, name=None))
    con.commit(); return con


def to_sqlite_dialect(sql):
    sql = re.sub(r"DATE_TRUNC\('month',\s*(\w+)\)::date", r"substr(\1, 1, 7) || '-01'", sql)
    sql = re.sub(r"TO_CHAR\(([\w.]+),\s*'YYYY-MM'\)", r"substr(\1, 1, 7)", sql)
    sql = re.sub(r"PERCENTILE_CONT\(0\.5\)\s+WITHIN GROUP\s*\(ORDER BY (\w+)\)", r"median(\1)", sql)
    return sql


SCORE_SQL = SQL_DIR / "scored_customers.sql"


def add_scores(con):
    """Create scored_* tables from sql/scored_customers.sql and load data/processed/scored_*_db.csv (run src.export first)."""
    ddl = SCORE_SQL.read_text(encoding="utf-8").split("-- Bulk load")[0]; con.executescript(ddl); con.execute("PRAGMA foreign_keys = ON")
    for tb in ["scored_customers", "scored_applications"]:
        df = pd.read_csv(PROCESSED / f"{tb}_db.csv"); df = df.astype(object).where(df.notna(), None); cols = ", ".join(df.columns)
        con.executemany(f"INSERT INTO {tb} ({cols}) VALUES ({','.join('?' * df.shape[1])})", df.itertuples(index=False, name=None))
    con.commit(); return con


def dashboard_queries(path=SCORE_SQL):
    """{'D1': (title, sql), ...} from sql/scored_customers.sql"""
    txt = path.read_text(encoding="utf-8").split("-- DASHBOARD QUERIES")[1]; parts = re.split(r"^-- (D\d)\s+(.*)$", txt, flags=re.M); out = {}
    for i in range(1, len(parts), 3):
        body = re.split(r"^-- ={5,}", parts[i + 2], flags=re.M)[0]; out[parts[i]] = (parts[i + 1].strip(), body.strip().rstrip(";") + ";")
    return out


def run_dashboard_queries(con):
    return {k: dict(title=t_, df=pd.read_sql_query(to_sqlite_dialect(q_), con)) for k, (t_, q_) in dashboard_queries().items()}


def analytical_queries(path=SQL_FILE):
    """{'7.1': (title, sql), ...} parsed from the section-7 comments of the PostgreSQL file."""
    txt = path.read_text(encoding="utf-8").split("-- 7. ANALYTICAL QUERIES")[1]
    parts = re.split(r"^-- (7\.\d+)\s+(.*)$", txt, flags=re.M); out = {}
    for i in range(1, len(parts), 3):
        body = re.split(r"^-- ={5,}", parts[i + 2], flags=re.M)[0]
        out[parts[i]] = (parts[i + 1].strip(), body.strip().rstrip(";") + ";")
    return out


def run_analytics(con=None):
    own = con is None; con = con or connect(); res = {}
    for k, (title, sql) in analytical_queries().items():
        try:
            res[k] = dict(title=title, df=pd.read_sql_query(to_sqlite_dialect(sql), con), error=None)
        except Exception as e:                                    # noqa: BLE001 - report every failure
            res[k] = dict(title=title, df=None, error=str(e))
    if own: con.close()
    return res


def main():
    raw = load_raw(); v0 = validate(raw); print("RAW data:", f"{int(v0.passed.sum())}/{len(v0)} checks passed;", "failed:", v0[~v0.passed].check.tolist())
    t, log = clean(raw); v = validate(t); PROCESSED.mkdir(parents=True, exist_ok=True)
    v.to_csv(PROCESSED / "validation_report.csv", index=False); log.to_csv(PROCESSED / "cleaning_log.csv", index=False)
    for k in TABLES: t[k].to_csv(PROCESSED / f"{k}.csv", index=False)
    print(log.to_string(index=False)); print(v.to_string(index=False)); print(f"\nAFTER cleaning: {int(v.passed.sum())}/{len(v)} checks passed")
    con = build_sqlite(tables=t); print("rows loaded:", {tb: con.execute(f"SELECT COUNT(*) FROM {tb}").fetchone()[0] for tb in TABLES})
    r = run_analytics(con); print({k: (len(x['df']) if x['df'] is not None else x['error']) for k, x in r.items()})


if __name__ == "__main__":
    main()
