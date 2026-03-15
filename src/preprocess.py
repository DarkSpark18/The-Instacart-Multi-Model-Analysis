import pandas as pd
from pathlib import Path


# ---------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------

# Set sample size (adjust based on your RAM)
# 5000 users = ~2-3 GB RAM
# 10000 users = ~4-6 GB RAM
# 50000 users = ~20+ GB RAM
# None = Full dataset (requires 64+ GB RAM)

SAMPLE_SIZE = 5000  # Change this or set to None for full dataset


# ---------------------------------------------------
# OUTPUT PATH
# ---------------------------------------------------

PROCESSED_PATH = Path("data/processed")
PROCESSED_PATH.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------
# PREPROCESS FUNCTION
# ---------------------------------------------------

def preprocess_data(data, sample_size=SAMPLE_SIZE):

    print("\n==============================")
    print("PREPROCESSING DATA")
    print("==============================\n")

    orders = data["orders"]
    products = data["products"]
    aisles = data["aisles"]
    departments = data["departments"]
    prior = data["prior"]
    train = data["train"]

    # ---------------------------------------------------
    # BASIC CLEANING
    # ---------------------------------------------------

    print("Checking missing values...\n")

    print("Orders null values:\n", orders.isnull().sum(), "\n")
    print("Products null values:\n", products.isnull().sum(), "\n")

    # Fill missing days_since_prior_order
    orders["days_since_prior_order"] = orders["days_since_prior_order"].fillna(0)

    # ---------------------------------------------------
    # SAMPLING (IF ENABLED)
    # ---------------------------------------------------

    if sample_size is not None:
        print(f"🔧 Sampling {sample_size} users for memory efficiency...\n")
        
        # Sample users BEFORE merging to prevent memory errors
        all_user_ids = orders["user_id"].unique()
        sample_user_ids = all_user_ids[:sample_size]
        
        print(f"Total users available: {len(all_user_ids)}")
        print(f"Sampled users: {len(sample_user_ids)}\n")
        
        # Filter orders to only include sampled users
        orders = orders[orders["user_id"].isin(sample_user_ids)].copy()
        
        # Filter prior orders to only include sampled users' orders
        sampled_order_ids = orders["order_id"].unique()
        prior = prior[prior["order_id"].isin(sampled_order_ids)].copy()
        
        print(f"Filtered orders shape: {orders.shape}")
        print(f"Filtered prior orders shape: {prior.shape}\n")
    else:
        print("⚠️  Processing FULL dataset (requires significant RAM)...\n")

    # ---------------------------------------------------
    # MERGE PRODUCT METADATA
    # ---------------------------------------------------

    print("Merging product metadata...\n")

    product_info = products.merge(aisles, on="aisle_id", how="left")
    product_info = product_info.merge(departments, on="department_id", how="left")

    print("Product info shape:", product_info.shape)

    # ---------------------------------------------------
    # MERGE ORDERS WITH PRIOR PRODUCTS
    # ---------------------------------------------------

    print("\nMerging orders with prior products...\n")

    orders_products = prior.merge(
        orders,
        on="order_id",
        how="left"
    )

    print("Orders + products shape:", orders_products.shape)

    # ---------------------------------------------------
    # MERGE WITH PRODUCT INFORMATION
    # ---------------------------------------------------

    print("\nAdding product information...\n")

    full_data = orders_products.merge(
        product_info,
        on="product_id",
        how="left"
    )

    print("Full dataset shape:", full_data.shape)

    # ---------------------------------------------------
    # DATASET SUMMARY
    # ---------------------------------------------------

    print("\nDATA SUMMARY\n")

    print("Unique users:", full_data["user_id"].nunique())
    print("Unique products:", full_data["product_id"].nunique())
    print("Unique orders:", full_data["order_id"].nunique())

    # ---------------------------------------------------
    # SAVE PROCESSED DATA
    # ---------------------------------------------------

    print("\nSaving processed dataset...\n")

    output_file = PROCESSED_PATH / "instacart_processed.parquet"

    full_data.to_parquet(output_file, engine="pyarrow", compression="snappy", index=False)
    
    # Only save CSV for small datasets (CSV is much larger)
    if sample_size is None or sample_size > 20000:
        print("Skipping CSV export for large dataset (use parquet instead)")
    else:
        full_data.to_csv(PROCESSED_PATH / "instacart_processed.csv", index=False)

    print(f"Processed data saved at: {output_file}\n")

    print("==============================")
    print("PREPROCESSING COMPLETE")
    print("==============================\n")

    return full_data