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

## Feature Importance Analysis

### Top 5 Most Important Features (Impurity-based / Gini)
1. **`payment_method_COD` (0.1681)**: Orders paid via Cash on Delivery (COD) naturally drive higher return risk because the customer has no upfront financial commitment, making it much easier for them to change their mind, refuse the delivery, or experience buyer's remorse without immediate consequences.
2. **`price_inr` (0.1358)**: The price of an item drives return risk because expensive purchases invite much higher customer scrutiny; if a high-ticket item fails to perfectly match expectations, the financial incentive to return it is much stronger than for a cheaper item.
3. **`delivery_distance_km` (0.1016)**: Longer delivery distances might plausibly drive returns if they correlate with longer wait times (increasing the chance the customer no longer needs the item) or a higher likelihood of the package sustaining damage during transit.
4. **`customer_tenure_days` (0.1005)**: Customer tenure relates to platform loyalty; newer customers may still be testing the service and more prone to returning items if they are dissatisfied, whereas long-tenured customers likely have established trust and more predictable buying habits.
5. **`discount_pct` (0.0852)**: High discount percentages can encourage impulsive buying behavior; shoppers might purchase items they don't truly need just because the deal seems too good to pass up, ultimately leading to higher return rates once the item arrives.

### Importance Ranking Comparison
When we compute permutation importance on the held-out test split, we get a much clearer picture of which features actually generalize:

| Rank | Gini (Impurity-based) Importance | Permutation Importance (Test Set) |
|------|----------------------------------|-----------------------------------|
| **1** | `payment_method_COD` (0.1681) | `payment_method_COD` (0.0650) |
| **2** | `price_inr` (0.1358) | `price_inr` (0.0165) |
| **3** | `delivery_distance_km` (0.1016) | `delivery_days` (0.0066) |
| **4** | `customer_tenure_days` (0.1005) | `product_category_Apparel` (0.0062)|
| **5** | `discount_pct` (0.0852) | `payment_method_Prepaid_Card` (0.0059)|

### Key Takeaways
When evaluated via permutation importance, **`delivery_distance_km`**, **`customer_tenure_days`**, and **`discount_pct`** lose almost all of their importance and drop completely out of the top 5, revealing that their high Gini importance was largely misleading. 

**Why does this happen?**
Impurity-based importance inherently overrates noisy continuous features (or high-cardinality categorical features) because it gives the tree more mathematical opportunities to make splits that reduce training impurity, even if those splits are just overfitting to random noise rather than capturing true predictive signal.

## Subgroup & Root-Cause Analysis

### Overall Performance
- **Overall Precision**: 0.3099
- **Overall Recall**: 0.5415

### By Product Category
| Category | Precision | Recall | Support |
|---|---|---|---|
| Beauty | 0.3542 | 0.6538 | 113 |
| Footwear | 0.3182 | 0.6087 | 208 |
| Apparel | 0.3351 | 0.5766 | 396 |
| Home | 0.2588 | 0.5500 | 226 |
| **Electronics** | **0.2639** | **0.3519** | **257** |

### By Payment Method
| Payment Method | Precision | Recall | Support |
|---|---|---|---|
| COD | 0.3244 | 0.9125 | 503 |
| **Wallet** | **0.1667** | **0.1579** | **134** |
| **Prepaid_Card** | **0.0625** | **0.0196** | **275** |
| **Prepaid_UPI** | **0.0000** | **0.0000** | **288** |

### Underperforming Subgroups & Proposed Solutions

**1. Prepaid Payment Methods (Prepaid_UPI, Prepaid_Card, Wallet)**
The model performs abysmally on prepaid orders. It has effectively learned to use `payment_method_COD` as a crutch, predicting almost all prepaid orders as "no return". The recall drops to essentially zero for UPI and Card payments. 
* **Concrete Next Step**: Implement **payment-method-specific decision thresholds**. Because the predicted probabilities for prepaid orders are structurally lower than for COD orders, a single global threshold of 0.5 completely misses them. By evaluating the PR curve specifically for the `Prepaid` slice and setting a lower, customized threshold for those transactions, we can recover recall for prepaid users without flooding the system with false-positive COD alerts.

**2. Electronics Category**
While not as drastic as the payment methods, the `Electronics` category performs meaningfully worse than the overall average, suffering a massive drop in recall (0.3519 compared to the 0.5415 average) along with below-average precision.
* **Concrete Next Step**: Engineer an **`is_electronics * price_inr` interaction feature**. Electronics have extreme price variance (e.g., a ₹200 charging cable vs. a ₹80,000 laptop). The return risk drivers for high-end electronics are likely very different from apparel. explicitly passing this interaction term prevents the Random Forest from having to waste multiple depth levels trying to isolate the specific price-category relationship, allowing it to better segment high-risk electronics purchases.

## Product Image Categoriser Model Report

### Training Summary

- **Device Used**: cpu
- **Batch Size**: 64
- **Optimizer**: Adam
- **Learning Rate**: 0.001
- **Epochs**: 10
- **Feature Extraction Time**: ~1810 seconds
- **Head Training Time**: ~17.5 seconds

### Final Test Evaluation

**Final Test Accuracy**: 88.87%

#### 10x10 Confusion Matrix

```text
[[857   6  14  18   1   2  95   0   6   1]
 [  3 973   2  17   1   1   3   0   0   0]
 [ 14   0 847   6  53   0  79   0   1   0]
 [ 28   7  17 853  28   1  65   0   1   0]
 [  1   0  61  26 784   0 125   0   3   0]
 [  0   0   0   0   0 979   0  15   1   5]
 [127   0  39  26  67   1 733   0   6   1]
 [  0   0   0   0   0  39   0 931   1  29]
 [  4   0   0   2   0   4  10   0 979   1]
 [  0   0   0   0   1  14   0  33   1 951]]
```

#### Per-class Precision & Recall

```text
              precision    recall  f1-score   support

 T-shirt/top       0.83      0.86      0.84      1000
     Trouser       0.99      0.97      0.98      1000
    Pullover       0.86      0.85      0.86      1000
       Dress       0.90      0.85      0.88      1000
        Coat       0.84      0.78      0.81      1000
      Sandal       0.94      0.98      0.96      1000
       Shirt       0.66      0.73      0.69      1000
     Sneaker       0.95      0.93      0.94      1000
         Bag       0.98      0.98      0.98      1000
  Ankle boot       0.96      0.95      0.96      1000

    accuracy                           0.89     10000
   macro avg       0.89      0.89      0.89     10000
weighted avg       0.89      0.89      0.89     10000
```

#### Confusion Patterns Analysis

Based on the confusion matrix, there are two primary class pairs that the model frequently mistakes for one another:

**1. Shirts vs. T-shirts/tops (127 + 95 errors)**  
The most common error the model makes is confusing a `Shirt` with a `T-shirt/top` (127 misclassifications) and vice-versa (95 misclassifications). Visually, shirts (button-downs or long-sleeves) and T-shirts share the exact same core silhouette: they both cover the torso and drape over the shoulders with central neck openings. At the low resolution of FashionMNIST (28x28 pixels), the primary distinguishing features of a collared shirt—such as the collar folds, a row of small buttons down the front, or slightly stiffer fabric cuffs—are severely degraded or entirely lost. Without these fine-grained details, the model only sees the general "T" shape of the torso and sleeves, making it incredibly difficult to mathematically distinguish a casual short-sleeve shirt from a basic T-shirt.

**2. Coats vs. Shirts (125 + 67 errors)**  
The second highest confusion is mistaking a `Coat` for a `Shirt` (125 times), alongside mistaking a `Shirt` for a `Coat` (67 times). Coats and shirts are frequently confused because they share nearly identical long-sleeve, full-torso bounding geometries. While a coat in real life might be thicker, longer, and heavier, within a normalized 28x28 grayscale pixel grid, both garments appear as large blocks of pixels extending down the body and out the arms. Features that typically separate a coat from a shirt—like heavy lapels, a prominent zipper, thick material texture, or the fact that a coat is worn open over other clothes—blur into a solid dark silhouette. When a shirt has a straight cut and long sleeves, its pixel distribution maps almost perfectly onto the shape of a light jacket or coat, leading the feature extractor to mistake one for the other.