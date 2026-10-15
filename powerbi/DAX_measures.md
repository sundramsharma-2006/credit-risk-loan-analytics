# DAX measures (generated from powerbi/build_pbip.py)

All measures are already in the generated semantic model; this file is for reference or manual rebuilds.

`Approval Rate` excludes 'Under Review' from the denominator (decided applications only). `Default Rate` uses disbursed loans only (the only rows with an observed outcome).


## Table `fact_loans`

**Applications**  (format `#,0`, folder KPIs)
```dax
Applications = COUNTROWS ( fact_loans )
```

**Approved**  (format `#,0`, folder KPIs)
```dax
Approved = SUM ( fact_loans[IsApproved] )
```

**Rejected**  (format `#,0`, folder KPIs)
```dax
Rejected = SUM ( fact_loans[IsRejected] )
```

**Approval Rate**  (format `0.0%`, folder KPIs)
```dax
Approval Rate = DIVIDE ( [Approved], [Approved] + [Rejected] )
```

**Disbursed Loans**  (format `#,0`, folder KPIs)
```dax
Disbursed Loans = CALCULATE ( COUNTROWS ( fact_loans ), fact_loans[disbursed_flag] = 1 )
```

**Defaults**  (format `#,0`, folder KPIs)
```dax
Defaults = CALCULATE ( COUNTROWS ( fact_loans ), fact_loans[loan_default] = 1 )
```

**Default Rate**  (format `0.0%`, folder KPIs)
```dax
Default Rate = DIVIDE ( [Defaults], [Disbursed Loans] )
```

**Disbursed Value (Cr)**  (format `#,0.0`, folder KPIs)
```dax
Disbursed Value (Cr) = SUM ( fact_loans[approved_amount_crore] )
```

**Avg Requested Amount**  (format `#,0`, folder KPIs)
```dax
Avg Requested Amount = AVERAGE ( fact_loans[requested_amount] )
```

**Avg Applicant Credit Score**  (format `0`, folder Credit)
```dax
Avg Applicant Credit Score = AVERAGEX ( fact_loans, RELATED ( dim_customer[credit_score] ) )
```

**Avg Model Default Probability**  (format `0.0%`, folder Model)
```dax
Avg Model Default Probability = AVERAGE ( fact_loans[model_default_probability] )
```


## Table `dim_customer`

**Customers**  (format `#,0`, folder Risk)
```dax
Customers = COUNTROWS ( dim_customer )
```

**High Risk Customers**  (format `#,0`, folder Risk)
```dax
High Risk Customers = CALCULATE ( COUNTROWS ( dim_customer ), dim_customer[risk_category] = "High Risk" )
```

**Avg Risk Score**  (format `0.0`, folder Risk)
```dax
Avg Risk Score = AVERAGE ( dim_customer[risk_score] )
```

**Avg Credit Score**  (format `0`, folder Credit)
```dax
Avg Credit Score = AVERAGE ( dim_customer[credit_score] )
```


## Table `ml_feature_importance`

**Driver Importance**  (format `0.000`, folder Model)
```dax
Driver Importance = SUM ( ml_feature_importance[auc_drop_when_shuffled] )
```
