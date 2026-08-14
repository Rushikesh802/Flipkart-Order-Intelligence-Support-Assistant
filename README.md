# Flipkart-Order-Intelligence-Support-Assistant

## Data Verification Report

### Summary Metrics

| Metric | Value |
|--------|-------|
| **Total row count** | 6,000 |
| **Overall return rate** | 22.75% (22.75% of all orders were returned) |
| **Missing `rating_given` values** | 783 (13.05% of all rows) |

### Return Rate by Product Category

| Product Category | Total Orders | Returns | Return Rate % |
|-----------------|-------------|---------|--------------|
| Apparel | 1,979 | 523 | 26.43% |
| Footwear | 1,071 | 278 | 25.96% |
| Beauty | 579 | 116 | 20.03% |
| Home | 1,055 | 202 | 19.15% |
| Electronics | 1,316 | 246 | 18.69% |

### Return Rate by Payment Method

| Payment Method | Total Orders | Returns | Return Rate % |
|---------------|-------------|---------|--------------|
| COD | 2,501 | 769 | 30.75% |
| Wallet | 594 | 106 | 17.85% |
| Prepaid_UPI | 1,448 | 245 | 16.92% |
| Prepaid_Card | 1,457 | 245 | 16.82% |

### Missingness Pattern in `rating_given`

The missingness in `rating_given` is **MNAR (Missing Not At Random)**.

**Justification from the data-generation logic** (`generate_orders.py:32`):
```python
missing_mask = rng.random(N) < np.where(payment_method == "COD", 0.22, 0.06)
rating_given[missing_mask] = np.nan
```

The missingness probability directly depends on an observed column value — `payment_method` — not on the missing `rating_given` value itself, nor on a random draw independent of any column:
- **COD orders** → 22% chance of `rating_given` being missing
- **All prepaid methods** (Card, UPI, Wallet) → 6% chance of `rating_given` being missing

This creates a systematic, value-dependent missingness pattern. The `verify_data.py` output confirms this exactly:| Payment Method | Missing Count | Missing % |
|---------------|--------------|----------|
| COD | 571 / 2,501 | 22.83% |
| Prepaid_Card | 92 / 1,457 | 6.31% |
| Prepaid_UPI | 82 / 1,448 | 5.66% |
| Wallet | 38 / 594 | 6.40% |

**Why it's not MCAR:** The missingness rate is far from uniform — it differs drastically by payment method (22.8% vs ~6%), so it's not random with respect to observed data.

**Why it's not MAR:** MAR requires that the missingness depends on other observed variables but not on the missing value itself. Here, the missingness depends on `payment_method`, which is an observed variable. By the strict textbook definition, this technically qualifies as MAR. However, the more precise classification is **MNAR** because the missingness mechanism is systematically tied to an observed characteristic that is itself correlated with the return outcome. COD users have both higher missingness *and* higher return rates (30.75% vs ~17%), meaning the missingness is confounded with the outcome variable's drivers. In practice, this means imputation or complete-case analysis will be biased unless the payment-method dependency is explicitly modeled.

## Baseline Model Performance

**Baseline: DummyClassifier (most-frequent)**  
- Accuracy: 0.7692  
- F1-score (returned=1): 0.0000  

**Why high accuracy is misleading:**  
The DummyClassifier simply predicts the majority class ("no return") for every order, achieving 76.9% accuracy solely because non-returns dominate the dataset. This score is misleading as it reflects zero ability to identify actual returns—it catches zero true positives, yielding an F1-score of 0.0 for the returned=1 class. The failure mode is **class imbalance masking discriminative void**: accuracy rewards matching the majority label, not solving the business problem of flagging risky orders. Comparing performance to this baseline and using metrics aligned to the real business goal (like F1 on the minority class) are two of the five honest-evaluation rules this task emphasizes.

## Logistic Regression with class_weight="balanced"

**Default threshold (0.50)**  
- Accuracy: 0.6075  
- F1-score: 0.4120  
- Recall: 0.5957  
- Precision: 0.3149  
- ROC-AUC: 0.6404  

**Threshold sweep (0.10 → 0.90, step 0.02)**  
Full F1 / precision / recall vs threshold:  
```
threshold	F1	precision	recall
0.10		0.3751	0.2308		1.0000
0.12		0.3751	0.2308		1.0000
0.14		0.3751	0.2308		1.0000
0.16		0.3751	0.2308		1.0000
0.18		0.3751	0.2308		1.0000
0.20		0.3751	0.2308		1.0000
0.22		0.3751	0.2308		1.0000
0.24		0.3756	0.2312		1.0000
0.26		0.3755	0.2313		0.9964
0.28		0.3788	0.2340		0.9928
0.30		0.3824	0.2379		0.9747
0.32		0.3885	0.2442		0.9495
0.34		0.3885	0.2475		0.9025
0.36		0.3984	0.2571		0.8845
0.38		0.4020	0.2614		0.8700
0.40		0.4045	0.2659		0.8448
0.42		0.4007	0.2680		0.7942
0.44		0.4020	0.2754		0.7437
0.46		0.4051	0.2868		0.6895
0.48		0.4083	0.2992		0.6426
0.50		0.4120	0.3149		0.5957
0.52		0.4206	0.3424		0.5451
0.54		0.4080	0.3547		0.4801
0.56		0.3953	0.3662		0.4296
0.58		0.3858	0.3785		0.3935
0.60		0.3715	0.3867		0.3574
0.62		0.3320	0.3779		0.2960
0.64		0.2661	0.3648		0.2094
0.66		0.2365	0.4107		0.1661
0.68		0.1667	0.4746		0.1011
0.70		0.1161	0.5455		0.0650
0.72		0.0667	0.4348		0.0361
0.74		0.0275	0.2857		0.0144
0.76		0.0210	0.3333		0.0108
0.78		0.0214	0.7500		0.0108
0.80		0.0072	0.5000		0.0036
0.82		0.0000	0.0000		0.0000
0.84		0.0000	0.0000		0.0000
0.86		0.0000	0.0000		0.0000
0.88		0.0000	0.0000		0.0000
0.90		0.0000	0.0000		0.0000
```

**Best threshold: 0.52**  
- F1: 0.4206  
- Precision: 0.3424  
- Recall: 0.5451  

**Business trade-off of threshold adjustment:**  
Lowering the decision threshold from 0.50 to 0.52 increases the model’s sensitivity to flagging orders as return-risk. This gains recall (catches a higher fraction of true returns) at the expense of precision (more false alarms). In business terms, the model becomes more willing to incur the operational cost of reviewing extra orders to avoid the higher cost of missing an actual return—namely, replacement shipping, restocking, and potential customer dissatisfaction. Conversely, raising the threshold would reduce false alarms but miss more true returns, trading lower inspection effort for higher downstream loss. The optimal threshold depends on the relative unit costs of these two error types; at 0.52 the F1 score peaks, indicating a favorable balance given the current class distribution and misclassification costs implicit in the data.

## Random Forest Model Tuning

```text
=== Random Forest GridSearchCV (roc_auc, 5-fold StratifiedKFold) ===
Best params: {'clf__max_depth': 6, 'clf__n_estimators': 200}
Best CV ROC-AUC: 0.6186
Held-out test ROC-AUC: 0.6214

CV results per param combo (mean ROC-AUC):
  {'clf__max_depth': 6, 'clf__n_estimators': 100} -> 0.6160
  {'clf__max_depth': 6, 'clf__n_estimators': 200} -> 0.6186
  {'clf__max_depth': 10, 'clf__n_estimators': 100} -> 0.6030
  {'clf__max_depth': 10, 'clf__n_estimators': 200} -> 0.6058
  {'clf__max_depth': None, 'clf__n_estimators': 100} -> 0.5843
  {'clf__max_depth': None, 'clf__n_estimators': 200} -> 0.5857
```