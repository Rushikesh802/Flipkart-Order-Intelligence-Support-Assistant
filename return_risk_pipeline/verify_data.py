import pandas as pd
import numpy as np

import os

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
data_path = os.path.join(ROOT_DIR, "data", "orders_dataset.csv")
if not os.path.exists(data_path):
    data_path = os.path.join(ROOT_DIR, "orders_dataset.csv")

# Load the dataset
df = pd.read_csv(data_path)

print('='*70)
print("DATA VERIFICATION REPORT")
print('='*70)

# 1. Total row count
print(f"\n1. TOTAL ROW COUNT: {len(df):,}")

# 2. Overall return rate
overall_return_rate = df['returned'].mean()
print(f"\n2. OVERALL RETURN RATE: {overall_return_rate:.4f} ({overall_return_rate*100:.2f}%)")

# 3. Missing rating_given
missing_rating = df['rating_given'].isna().sum()
missing_pct = (missing_rating / len(df)) * 100
print(f"\n3. MISSING 'rating_given' VALUES:")
print(f"   - Count: {missing_rating:,}")
print(f"   - Percentage: {missing_pct:.2f}%")

# 4. Return rate by product_category
print("\n" + "="*70)
print("4. RETURN RATE BY PRODUCT_CATEGORY")
print("="*70)
return_by_category = df.groupby('product_category').agg({
    'returned': ['sum', 'count', 'mean']
}).round(4)
return_by_category.columns = ['Returns', 'Total Orders', 'Return Rate']
return_by_category['Return Rate %'] = (return_by_category['Return Rate'] * 100).round(2)
return_by_category = return_by_category[['Total Orders', 'Returns', 'Return Rate %']]
return_by_category.index.name = 'Product Category'
print(return_by_category.to_string())

# 5. Return rate by payment_method
print("\n" + "="*70)
print("5. RETURN RATE BY PAYMENT_METHOD")
print("="*70)
return_by_payment = df.groupby('payment_method').agg({
    'returned': ['sum', 'count', 'mean']
}).round(4)
return_by_payment.columns = ['Returns', 'Total Orders', 'Return Rate']
return_by_payment['Return Rate %'] = (return_by_payment['Return Rate'] * 100).round(2)
return_by_payment = return_by_payment[['Total Orders', 'Returns', 'Return Rate %']]
return_by_payment.index.name = 'Payment Method'
print(return_by_payment.to_string())

# 6. Detailed missingness by payment_method
print("\n" + "="*70)
print("6. MISSINGNESS IN 'rating_given' BY PAYMENT_METHOD")
print("="*70)
for method in sorted(df['payment_method'].unique()):
    subset = df[df['payment_method'] == method]
    missing = subset['rating_given'].isna().sum()
    total = len(subset)
    missing_pct = (missing / total) * 100
    print(f"{method:20s}: {missing:4d} / {total:4d} missing ({missing_pct:5.2f}%)")

print(f"\nOverall missing: {missing_rating:,} / {len(df):,} ({missing_pct:.2f}%)")
