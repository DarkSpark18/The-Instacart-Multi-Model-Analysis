# check_features.py
# Run this to diagnose the feature store

import pandas as pd

# Load feature store
df = pd.read_parquet("data/features/feature_store.parquet")

print("\n" + "="*70)
print("FEATURE STORE DIAGNOSTIC")
print("="*70 + "\n")

print(f"Shape: {df.shape}")
print(f"Columns: {len(df.columns)}\n")

# Check target columns
print("TARGET COLUMNS:")
print("-" * 70)

targets = ['add_to_cart_order', 'reordered', 'churn', 'label']
for target in targets:
    if target in df.columns:
        print(f"\n{target}:")
        print(f"  Unique values: {df[target].nunique()}")
        print(f"  Value counts:\n{df[target].value_counts()}")
        print(f"  Nulls: {df[target].isna().sum()}")
    else:
        print(f"\n❌ {target}: NOT FOUND")

# Check for user_id (needed for ranking models)
print("\n\nGROUPING COLUMN:")
print("-" * 70)
if 'user_id' in df.columns:
    print(f"user_id: ✅ Present")
    print(f"  Unique users: {df['user_id'].nunique()}")
else:
    print(f"user_id: ❌ NOT FOUND (needed for ranking models)")

print("\n" + "="*70 + "\n")