# Data dictionary - Credora Finance

Four CSVs in `datasets/`; `customer_id` joins all of them. Source definitions are in `documentation/PRD.docx` (section 11); this file adds what was **observed in the data** and the derived tables.

## Raw tables
| Table | Rows | Key | Notes observed in the data |
|---|---|---|---|
| customers | 5,000 | customer_id | no nulls; ages 21-74 (age is computed as of mid-2026, not 2024); gender only Male/Female; kyc_status only Verified/Pending; 2 pairs of customers share an e-mail (see below); `customer_since` is after the application date for 935 applications |
| credit_history | 5,000 | credit_id (1:1 customer) | one snapshot, `as_of_date` = 2024-06-30 for everyone; utilisation matches outstanding/limit (corr 0.9999) |
| loan_applications | 7,500 | application_id | Jul 2022 - Jun 2024; 3,924 Rejected, 3,403 Approved (all disbursed), 173 Under Review; `loan_default` present exactly for the 3,403 disbursed loans (23.4% default); `decision_date` null for the 173 under review |
| transactions | 40,079 | transaction_id | 4-12 rows per customer (mean 8) in Jan-Jun 2024; categories include `Transfer In`/`Transfer Out`/`ATM Withdrawal` (PRD lists `Transfer`/`ATM`) |

## Quality checks (23, all pass after cleaning - `data/processed/validation_report.csv`)
Primary keys unique; 1:1 credit history; no orphan foreign keys; value ranges (score 300-900, utilisation 0-1); approval-status set; `loan_default` only for disbursed loans; only approved loans disbursed; approved <= requested; `decision_date >= application_date`; no negative amounts; **`customers.email` unique - FAILS on raw data** (two e-mails shared by different people). Fix in `data/processed/customers.csv`: later customer's e-mail set to NULL (logged in `cleaning_log.csv`). Use this file for the PostgreSQL load (`sql/load_data.sql`), otherwise the schema's UNIQUE(email) rejects the load.

## Derived tables
| Table / file | Grain | Contents |
|---|---|---|
| `data/processed/application_features.csv` | application (7,500) | customer + bureau + transaction aggregates + loan terms + ratios (`debt_to_limit`, `loan_to_income`, `net_cash_flow`, `emi_share_of_outflow` ...) |
| `data/processed/customer_features.csv` | customer (5,000) | same without loan terms; `ever_default` (1 if any disbursed loan defaulted; null if none disbursed) |
| `scored_applications` | application | `default_probability` (cross-fitted for disbursed loans; extrapolated otherwise), `scored_how` |
| `scored_customers` | customer | `default_probability`, `risk_score` 0-100, `risk_category`, `scored_how`, `top_risk_factors` |

Excluded from modelling: contact details and names (PII), gender and marital status (fair-lending review not done), identifiers, and all post-decision fields (`approval_status`, `approved_amount`, `decision_date`, `disbursed_flag`).
