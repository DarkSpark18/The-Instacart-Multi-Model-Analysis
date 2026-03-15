import pandas as pd
from pathlib import Path

# ---------------------------------------------------
# PATH TO PROCESSED DATA
# ---------------------------------------------------

DATA_PATH = Path("data/processed/instacart_processed.parquet")


def show_data():

    print("\n==============================")
    print("SHOWING PROCESSED DATA")
    print("==============================\n")

    if not DATA_PATH.exists():
        print("❌ Processed parquet file not found.")
        print("Run preprocessing first.")
        return

    print("Loading parquet dataset...\n")

    df = pd.read_parquet(DATA_PATH)

    # ---------------------------------------------------
    # BASIC INFO
    # ---------------------------------------------------

    print("DATASET SHAPE")
    print(df.shape, "\n")

    print("COLUMNS")
    print(df.columns.tolist(), "\n")

    print("DATA TYPES")
    print(df.dtypes, "\n")

    # ---------------------------------------------------
    # PREVIEW DATA
    # ---------------------------------------------------

    print("FIRST 10 ROWS")
    print(df.head(10), "\n")

    print("LAST 10 ROWS")
    print(df.tail(10), "\n")

    # ---------------------------------------------------
    # QUICK STATS
    # ---------------------------------------------------

    print("BASIC STATISTICS")
    print(df.describe(include="all"), "\n")

    # ---------------------------------------------------
    # UNIQUE COUNTS
    # ---------------------------------------------------

    print("UNIQUE COUNTS")

    if "user_id" in df.columns:
        print("Users:", df["user_id"].nunique())

    if "product_id" in df.columns:
        print("Products:", df["product_id"].nunique())

    if "order_id" in df.columns:
        print("Orders:", df["order_id"].nunique())

    print("\n==============================")
    print("DATA PREVIEW COMPLETE")
    print("==============================\n")


if __name__ == "__main__":
    show_data()