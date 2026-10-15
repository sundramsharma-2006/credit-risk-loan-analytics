-- =====================================================================
-- Credora Finance | Scoring tables + dashboard queries (PostgreSQL 14+; also runs on SQLite via src.data_load)
-- Run AFTER credit_risk_platform.sql (DDL) and load_data.sql, then load the model output (see bottom).
-- =====================================================================
DROP TABLE IF EXISTS scored_applications;
DROP TABLE IF EXISTS scored_customers;

CREATE TABLE scored_customers (
    customer_id         VARCHAR(12)  PRIMARY KEY,
    default_probability NUMERIC(5,4) NOT NULL CHECK (default_probability BETWEEN 0 AND 1),
    risk_score          SMALLINT     NOT NULL CHECK (risk_score BETWEEN 0 AND 100),
    risk_category       VARCHAR(12)  NOT NULL CHECK (risk_category IN ('Low Risk','Medium Risk','High Risk')),
    scored_how          VARCHAR(60),
    top_risk_factors    TEXT,
    CONSTRAINT fk_scored_customer FOREIGN KEY (customer_id) REFERENCES customers(customer_id) ON DELETE CASCADE
);

CREATE TABLE scored_applications (
    application_id      VARCHAR(12)  PRIMARY KEY,
    customer_id         VARCHAR(12)  NOT NULL,
    default_probability NUMERIC(5,4) NOT NULL CHECK (default_probability BETWEEN 0 AND 1),
    scored_how          VARCHAR(80),
    CONSTRAINT fk_scored_app FOREIGN KEY (application_id) REFERENCES loan_applications(application_id) ON DELETE CASCADE
);
CREATE INDEX idx_scored_cust_cat  ON scored_customers (risk_category);
CREATE INDEX idx_scored_cust_score ON scored_customers (risk_score);

-- Bulk load (psql, from the project root):
--   \copy scored_customers   FROM 'data/processed/scored_customers_db.csv'   CSV HEADER;
--   \copy scored_applications FROM 'data/processed/scored_applications_db.csv' CSV HEADER;

-- =====================================================================
-- DASHBOARD QUERIES (one per PRD section-14 component)
-- =====================================================================
-- D1  Loan approval rate: KPI + monthly trend (decided applications only; 'Under Review' excluded from the denominator)
SELECT TO_CHAR(application_date,'YYYY-MM') AS month,
       SUM(CASE WHEN approval_status='Approved' THEN 1 ELSE 0 END) AS approved,
       SUM(CASE WHEN approval_status='Rejected' THEN 1 ELSE 0 END) AS rejected,
       ROUND(100.0*SUM(CASE WHEN approval_status='Approved' THEN 1 ELSE 0 END)
             / NULLIF(SUM(CASE WHEN approval_status<>'Under Review' THEN 1 ELSE 0 END),0),1) AS approval_rate_pct
FROM loan_applications GROUP BY 1 ORDER BY 1;

-- D2  Default rate: KPI + by loan type (disbursed loans)
SELECT loan_type, COUNT(*) AS disbursed_loans, SUM(loan_default) AS defaults,
       ROUND(100.0*AVG(loan_default),1) AS default_rate_pct
FROM loan_applications WHERE disbursed_flag=1 GROUP BY loan_type ORDER BY default_rate_pct DESC;

-- D3  Customer risk distribution (Low / Medium / High)
SELECT risk_category, COUNT(*) AS customers, ROUND(AVG(risk_score),1) AS avg_risk_score
FROM scored_customers GROUP BY risk_category ORDER BY MIN(risk_score);

-- D4  Credit score view (bureau data is ONE snapshot dated 2024-06-30, so this is a distribution / cohort view, not a true time series)
SELECT TO_CHAR(la.application_date,'YYYY-MM') AS application_month, COUNT(DISTINCT la.customer_id) AS applicants,
       ROUND(AVG(ch.credit_score),1) AS avg_credit_score
FROM loan_applications la JOIN credit_history ch ON ch.customer_id = la.customer_id GROUP BY 1 ORDER BY 1;

-- D5  Geographic loan distribution (state level)
SELECT c.state, COUNT(*) AS applications,
       ROUND(100.0*SUM(CASE WHEN la.approval_status='Approved' THEN 1 ELSE 0 END)/NULLIF(SUM(CASE WHEN la.approval_status<>'Under Review' THEN 1 ELSE 0 END),0),1) AS approval_rate_pct,
       ROUND(100.0*AVG(la.loan_default),1) AS default_rate_pct,
       ROUND(SUM(la.approved_amount)/10000000.0,2) AS disbursed_crore
FROM loan_applications la JOIN customers c ON c.customer_id = la.customer_id GROUP BY c.state ORDER BY applications DESC;

-- D6  Monthly loan applications
SELECT TO_CHAR(application_date,'YYYY-MM') AS month, COUNT(*) AS applications FROM loan_applications GROUP BY 1 ORDER BY 1;

-- D7  Approval vs rejection trend
SELECT TO_CHAR(application_date,'YYYY-MM') AS month, approval_status, COUNT(*) AS applications
FROM loan_applications GROUP BY 1,2 ORDER BY 1,2;

-- D8  High-risk customer watchlist (filter by risk category and state in the dashboard)
SELECT s.customer_id, c.city, c.state, c.employment_type, ch.credit_score, s.risk_score, s.risk_category, s.top_risk_factors
FROM scored_customers s JOIN customers c ON c.customer_id = s.customer_id JOIN credit_history ch ON ch.customer_id = s.customer_id
WHERE s.risk_category = 'High Risk' ORDER BY s.risk_score DESC LIMIT 100;
