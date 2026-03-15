"""
Feature Builder for Instacart Market Basket Analysis - FINAL VERSION
"""

import pandas as pd
import numpy as np
from pathlib import Path

DATA_PATH = Path("data/processed/instacart_processed.parquet")
OUTPUT_PATH = Path("data/features/feature_store.parquet")


def load_data():
    print("Loading processed dataset...")
    df = pd.read_parquet(DATA_PATH)
    print("Dataset shape:", df.shape)
    return df


# -------------------------
# USER FEATURES
# -------------------------

def build_user_features(df):

    print("Building user features...")

    user = df.groupby("user_id").agg(
        user_total_orders=("order_number", "max"),
        user_total_products=("product_id", "count"),
        user_unique_products=("product_id", "nunique"),
        user_avg_cart_size=("add_to_cart_order", "mean"),
        user_max_cart_size=("add_to_cart_order", "max"),
        user_reorder_ratio=("reordered", "mean"),
        user_avg_days_between_orders=("days_since_prior_order", "mean"),
        user_order_frequency=("order_number", "count")
    ).reset_index()

    fav_hour = (
        df.groupby(["user_id", "order_hour_of_day"])
        .size()
        .reset_index(name="count")
        .sort_values(["user_id", "count"], ascending=False)
        .drop_duplicates("user_id")
    )[["user_id", "order_hour_of_day"]]

    fav_hour.rename(columns={"order_hour_of_day": "user_favorite_hour"}, inplace=True)
    user = user.merge(fav_hour, on="user_id", how="left")

    fav_day = (
        df.groupby(["user_id", "order_dow"])
        .size()
        .reset_index(name="count")
        .sort_values(["user_id", "count"], ascending=False)
        .drop_duplicates("user_id")
    )[["user_id", "order_dow"]]

    fav_day.rename(columns={"order_dow": "user_favorite_dow"}, inplace=True)
    user = user.merge(fav_day, on="user_id", how="left")

    return user


# -------------------------
# PRODUCT FEATURES
# -------------------------

def build_product_features(df):

    print("Building product features...")

    product = df.groupby("product_id").agg(
        product_total_orders=("order_id", "count"),
        product_unique_users=("user_id", "nunique"),
        product_reorder_rate=("reordered", "mean"),
        product_avg_cart_position=("add_to_cart_order", "mean")
    ).reset_index()

    product["product_popularity_rank"] = product["product_total_orders"].rank(
        method="dense", ascending=False
    )

    peak_hour = (
        df.groupby(["product_id", "order_hour_of_day"])
        .size()
        .reset_index(name="count")
        .sort_values(["product_id", "count"], ascending=False)
        .drop_duplicates("product_id")
    )[["product_id", "order_hour_of_day"]]

    peak_hour.rename(columns={"order_hour_of_day": "product_hour_peak"}, inplace=True)

    product = product.merge(peak_hour, on="product_id", how="left")

    return product


# -------------------------
# USER PRODUCT FEATURES
# -------------------------

def build_user_product_features(df):

    print("Building user-product features...")

    up = df.groupby(["user_id", "product_id"]).agg(
        user_product_orders=("order_id", "count"),
        user_product_first_order=("order_number", "min"),
        user_product_last_order=("order_number", "max"),
        user_product_avg_cart_position=("add_to_cart_order", "mean"),
        user_product_reorder_rate=("reordered", "mean")
    ).reset_index()

    up["user_product_order_rate"] = (
        up["user_product_orders"] /
        up["user_product_last_order"]
    )

    up["orders_since_last_purchase"] = (
        up["user_product_last_order"] -
        up["user_product_first_order"]
    )

    return up


# -------------------------
# CATEGORY FEATURES
# -------------------------

def build_category_features(df):

    print("Building category features...")

    dept = df.groupby("department_id").agg(
        department_popularity=("order_id", "count"),
        department_reorder_rate=("reordered", "mean")
    ).reset_index()

    aisle = df.groupby("aisle_id").agg(
        aisle_popularity=("order_id", "count")
    ).reset_index()

    return dept, aisle


# -------------------------
# TIME FEATURES
# -------------------------

def add_time_features(df):

    print("Creating time features...")

    df["is_weekend"] = df["order_dow"].isin([0, 6]).astype(int)
    df["is_morning"] = df["order_hour_of_day"].between(6, 11).astype(int)
    df["is_evening"] = df["order_hour_of_day"].between(17, 22).astype(int)

    df["hour_sin"] = np.sin(2 * np.pi * df["order_hour_of_day"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["order_hour_of_day"] / 24)

    return df


# -------------------------
# CREATE TARGET VARIABLES
# -------------------------

def create_targets(user_features):
    
    print("Creating churn target...")
    
    # Create churn based on user behavior patterns
    # Use percentile-based approach to ensure we have both classes
    
    user_churn = user_features.copy()
    
    # Calculate churn risk score
    # Low total orders OR low reorder ratio = higher churn risk
    median_orders = user_churn['user_total_orders'].median()
    median_reorder = user_churn['user_reorder_ratio'].median()
    
    # Churn = 1 if user is in bottom 40% of orders OR bottom 40% of reorder ratio
    user_churn["churn"] = (
        (user_churn["user_total_orders"] < user_churn["user_total_orders"].quantile(0.4)) |
        (user_churn["user_reorder_ratio"] < user_churn["user_reorder_ratio"].quantile(0.4))
    ).astype(int)
    
    churn_counts = user_churn['churn'].value_counts()
    print(f"  Churn distribution: {churn_counts.to_dict()}")
    print(f"  Churn rate: {churn_counts[1] / len(user_churn) * 100:.1f}%")
    
    return user_churn[["user_id", "churn"]]


# -------------------------
# CREATE BINARY TARGETS FROM RAW DATA
# -------------------------

def create_binary_targets(df):
    """
    Create proper binary targets from raw transaction data
    """
    
    print("Creating binary reorder and label targets from raw data...")
    
    # For each user-product pair, take the LAST transaction's reordered value
    latest_transactions = df.sort_values('order_number').groupby(['user_id', 'product_id']).tail(1)
    
    binary_targets = latest_transactions[['user_id', 'product_id', 'reordered']].copy()
    
    # Ensure it's binary
    binary_targets['reordered_binary'] = (binary_targets['reordered'] > 0).astype(int)
    binary_targets['label_binary'] = binary_targets['reordered_binary'].copy()
    
    print(f"  Reordered distribution: {binary_targets['reordered_binary'].value_counts().to_dict()}")
    print(f"  Label distribution: {binary_targets['label_binary'].value_counts().to_dict()}")
    
    return binary_targets[['user_id', 'product_id', 'reordered_binary', 'label_binary']]


# -------------------------
# MERGE FEATURES
# -------------------------

def merge_all_features(df, user, product, up, dept, aisle, churn_target, binary_targets):

    print("Merging feature tables...")

    feature_df = up.merge(user, on="user_id", how="left")
    feature_df = feature_df.merge(product, on="product_id", how="left")

    product_meta = df[["product_id", "department_id", "aisle_id"]].drop_duplicates()

    feature_df = feature_df.merge(product_meta, on="product_id", how="left")

    feature_df = feature_df.merge(dept, on="department_id", how="left")
    feature_df = feature_df.merge(aisle, on="aisle_id", how="left")
    
    # Add churn target
    feature_df = feature_df.merge(churn_target, on="user_id", how="left")
    feature_df["churn"] = feature_df["churn"].fillna(0).astype(int)
    
    # Add time features aggregated at user-product level
    time_cols = ["is_weekend", "is_morning", "is_evening", "hour_sin", "hour_cos"]
    time_features = df.groupby(["user_id", "product_id"])[time_cols].mean().reset_index()
    feature_df = feature_df.merge(time_features, on=["user_id", "product_id"], how="left")
    
    # Add basket size target (keep as continuous - it's for regression)
    basket_size = df.groupby(["user_id", "product_id"])["add_to_cart_order"].mean().reset_index()
    feature_df = feature_df.merge(basket_size, on=["user_id", "product_id"], how="left")
    
    # Add BINARY reorder and label targets
    feature_df = feature_df.merge(binary_targets, on=["user_id", "product_id"], how="left")
    
    # Use the binary versions as the final targets
    feature_df['reordered'] = feature_df['reordered_binary'].fillna(0).astype(int)
    feature_df['label'] = feature_df['label_binary'].fillna(0).astype(int)
    
    # Drop the temporary columns
    feature_df = feature_df.drop(['reordered_binary', 'label_binary'], axis=1)

    print("Final feature shape:", feature_df.shape)

    return feature_df


# -------------------------
# SAVE FEATURE STORE
# -------------------------

def save_features(feature_df):

    print("Saving feature store...")
    
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    feature_df.to_parquet(OUTPUT_PATH, index=False)
    feature_df.to_csv(OUTPUT_PATH.with_suffix(".csv"), index=False)

    print("Saved to:", OUTPUT_PATH)


# -------------------------
# MAIN PIPELINE
# -------------------------

def main():

    df = load_data()

    df = add_time_features(df)

    user = build_user_features(df)
    product = build_product_features(df)
    up = build_user_product_features(df)

    dept, aisle = build_category_features(df)
    
    churn_target = create_targets(user)
    binary_targets = create_binary_targets(df)

    feature_df = merge_all_features(df, user, product, up, dept, aisle, churn_target, binary_targets)

    save_features(feature_df)

    print("\n" + "="*70)
    print("FEATURE ENGINEERING COMPLETED!")
    print("="*70)
    print(f"\nFeatures available: {feature_df.shape[1]}")
    print(f"Samples: {feature_df.shape[0]}")
    
    print(f"\n📊 TARGET DISTRIBUTIONS:")
    print(f"  add_to_cart_order (regression): mean={feature_df['add_to_cart_order'].mean():.2f}")
    print(f"  reordered (binary): {feature_df['reordered'].value_counts().to_dict()}")
    print(f"  churn (binary): {feature_df['churn'].value_counts().to_dict()}")
    print(f"  label (binary): {feature_df['label'].value_counts().to_dict()}")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()