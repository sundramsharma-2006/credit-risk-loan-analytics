# Model card - Credora default model and risk score

**Version 1.0 · trained 2026-10-11 · seed 42 · backend: mini_sklearn (NumPy fallback, scikit-learn API subset)**

## Intended use
Decision support for underwriters and risk managers: a second opinion on the probability that a disbursed loan defaults, and a 0-100 risk score per customer. **It must not auto-approve or auto-reject.** Not validated for pricing, credit-line assignment, collections or any regulated use.

## Models
| | Default model (application level) | Customer risk score |
|---|---|---|
| Output | calibrated P(default) per loan application | `risk_score` = round(100 x P), band Low (0-33) / Medium (34-66) / High (67-100) |
| Algorithm | Logistic Regression (C=0.003), chosen by grouped 5-fold CV ROC-AUC from 4 families | customers who applied: highest P over their applications; never applied: profile-only Logistic Regression |
| Training data | 3,403 disbursed loans (outcome observed); 2,722 train / 681 test, **split by customer** | same |
| Features | 55 (bureau, demographics, 6-month transaction aggregates, loan terms) | same, minus loan terms for the profile model |

## Performance (held-out customers, 157 defaults in 681 loans)
| Algorithm | Tuned setting | CV ROC-AUC (5-fold, grouped) | Test ROC-AUC |
|---|---|---|---|
| Logistic Regression | C=0.003 | 0.737 ± 0.013 | 0.731 |
| Decision Tree | depth=3, leaf=30 | 0.701 ± 0.014 | 0.718 |
| Random Forest | depth=8, leaf=10 | 0.723 ± 0.011 | 0.720 |
| Gradient Boosting (XGBoost stand-in) | n=100, depth=2 | 0.728 ± 0.011 | 0.735 |

Champion test ROC-AUC **0.731** (95% CI 0.69-0.77, customer bootstrap), PR-AUC 0.448 (base rate 0.234).
Operating points (thresholds fixed on training out-of-fold predictions, test set touched once):

| Threshold | Precision | Recall | F1 | Confusion (TN/FP/FN/TP) |
|---|---|---|---|---|
| 0.5 (calibrated) | 0.549 | 0.178 | 0.269 | 501/23/129/28 |
| 0.243 (max F1) | 0.367 | 0.586 | 0.451 | 365/159/65/92 |
| 0.224 (recall target 0.70) | 0.360 | 0.656 | 0.465 | 341/183/54/103 |

**PRD success targets: ROC-AUC >= 0.80 - NOT met (0.73); recall >= 0.70 - NOT met on the test set (0.66; it was 0.70 on the out-of-fold data used to choose the threshold). We did not tune against the test set to reach them.**

At the recall-target threshold the model flags 183 repaid loans for the 103 defaulters it catches (precision 0.36, 54 defaulters missed) - a trade-off for the business to price, not a technical detail.

## Risk score bands vs what actually happened (customers with an observed loan outcome)
| Band | Customers (all 5,000) | Observed default rate (customers with a disbursed loan) |
|---|---|---|
| Low Risk | 2,201 | 22.2% |
| Medium Risk | 1,811 | 39.9% |
| High Risk | 988 | 50.7% |

The PRD cut-offs separate outcomes, so they were kept without re-tuning. Customer-score AUC against "ever defaulted": 0.716 using disbursed loans only (cross-fitted); 0.664 when extrapolated scores of rejected applications are included.

## Drivers (permutation importance, drop in test ROC-AUC when a feature is shuffled)
| Feature | AUC drop |
|---|---|
| debt_to_income_ratio | 0.0506 |
| credit_score | 0.0098 |
| num_late_payments_30d | 0.0065 |
| requested_amount | 0.0064 |
| loan_type_Auto | 0.0038 |
| loan_to_income | 0.0038 |
| education_Post Graduate | 0.0035 |
| num_open_accounts | 0.0029 |

Debt-to-income dominates. The PRD expected credit score, DTI, utilisation, prior defaults and employment type to lead; once DTI and score are in, utilisation and prior defaults rank lower (credit score correlates -0.31 with utilisation, -0.37 with prior defaults and -0.49 with 30-day late payments, so much of their signal is already in the score). SHAP was not installable here; per-customer reasons use an occlusion method (feature replaced by the population median, change in log-odds), stored in `scored_customers.top_risk_factors`.

## Checks that the number can be trusted
* Split by customer (no customer in both sets). A naive random row split gives 0.755 - customer overlap would have inflated the result.
* Removing bureau features (measured after most applications): AUC 0.702; removing bureau + transactions: 0.706. The result does not depend on them, but temporal leakage cannot be excluded.
* AUC by application year (test): 2022 0.77, 2023 0.71, 2024 0.75 (150-370 rows each; no clear pattern).
* One feature alone (DTI) gives AUC 0.695; the full model adds ~0.04.
* Calibration: Platt scaling on out-of-fold predictions; mean calibrated probability 0.235 = observed rate 0.235 on training folds.

## Limitations (important)
1. **Selection bias / reject inference.** Outcomes exist only for approved-and-disbursed loans (median DTI 0.71); rejected applications have median DTI 2.39. Scores for non-disbursed applications (mean P 0.53 vs 0.24 for disbursed) are extrapolations and cannot be validated.
2. **Temporal mismatch.** Bureau snapshot is 2024-06-30 and transactions cover Jan-Jun 2024, but applications run Jul 2022 - Jun 2024.
3. **Small test set** (157 defaults): the AUC confidence interval is about ±0.04.
4. **Thin transaction data:** 4-12 transactions per customer.
5. **Data quirks:** 935 of 7,500 applications are dated before `customer_since`; two customers share an e-mail with another customer (fixed by NULLing the later one).
6. **Fairness:** gender and marital status are excluded from features; **no bias audit was run** (PRD marks it as future work). Age, income and geography are in or near the feature set and could act as proxies.
7. Gradient boosting stands in for XGBoost, and a NumPy re-implementation of scikit-learn's API trained the models (scikit-learn, XGBoost and SHAP were not installable in the build sandbox). Results may shift slightly with the real libraries.

## Reproducibility
`python run_pipeline.py` (fixed seed 42). Artefacts: `models/default_model.pkl`, `models/risk_model.pkl`; outputs `data/processed/scored_*.csv`; metrics `reports/*_summary.json`.
