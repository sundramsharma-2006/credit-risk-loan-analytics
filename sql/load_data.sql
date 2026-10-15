-- Bulk load for PostgreSQL 14+ (run from the project root in psql, AFTER the DDL part of credit_risk_platform.sql).
-- Use the CLEANED customers file: the raw file has two e-mail addresses shared by different customers, which violates
-- the schema's UNIQUE(email) and makes the load fail (see data/processed/cleaning_log.csv).
-- Do NOT also run the 6-row seed INSERTs of credit_risk_platform.sql, or the \copy below will hit duplicate keys.
\copy customers          FROM 'data/processed/customers.csv'          CSV HEADER;
\copy credit_history     FROM 'datasets/credit_history.csv'           CSV HEADER;
\copy loan_applications  FROM 'datasets/loan_applications.csv'        CSV HEADER;
\copy transactions       FROM 'datasets/transactions.csv'             CSV HEADER;

-- sanity checks (expected: 5000 / 5000 / 7500 / 40079)
SELECT 'customers' t, COUNT(*) FROM customers UNION ALL SELECT 'credit_history', COUNT(*) FROM credit_history
UNION ALL SELECT 'loan_applications', COUNT(*) FROM loan_applications UNION ALL SELECT 'transactions', COUNT(*) FROM transactions;
