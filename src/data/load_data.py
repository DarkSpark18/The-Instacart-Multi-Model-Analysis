import pandas as pd
from pathlib import Path

from src.data.preprocess import preprocess_data


# ---------------------------------------------------
# DATA PATH
# ---------------------------------------------------

DATA_PATH = Path("data/raw")


# ---------------------------------------------------
# LOAD DATA FUNCTION
# ---------------------------------------------------

def load_data():

    print("\n==============================")
    print("LOADING INSTACART DATASET")
    print("==============================\n")

    # Load datasets
    orders = pd.read_csv(DATA_PATH / "orders.csv")
    products = pd.read_csv(DATA_PATH / "products.csv")
    aisles = pd.read_csv(DATA_PATH / "aisles.csv")
    departments = pd.read_csv(DATA_PATH / "departments.csv")
    order_products_prior = pd.read_csv(DATA_PATH / "order_products__prior.csv")
    order_products_train = pd.read_csv(DATA_PATH / "order_products__train.csv")

    print("Datasets loaded successfully\n")

    # ---------------------------------------------------
    # BASIC DATA VALIDATION
    # ---------------------------------------------------

    print("DATASET OVERVIEW\n")

    print(f"Orders shape: {orders.shape}")
    print(f"Products shape: {products.shape}")
    print(f"Aisles shape: {aisles.shape}")
    print(f"Departments shape: {departments.shape}")
    print(f"Prior orders shape: {order_products_prior.shape}")
    print(f"Train orders shape: {order_products_train.shape}\n")

    print("Unique Users:", orders["user_id"].nunique())
    print("Unique Products:", products["product_id"].nunique())
    print("Unique Orders:", orders["order_id"].nunique(), "\n")

    # ---------------------------------------------------
    # CREATE DATA DICTIONARY
    # ---------------------------------------------------

    data = {
        "orders": orders,
        "products": products,
        "aisles": aisles,
        "departments": departments,
        "prior": order_products_prior,
        "train": order_products_train
    }

    print("Sending data to preprocessing pipeline...\n")

    # ---------------------------------------------------
    # SEND TO PREPROCESSING
    # ---------------------------------------------------

    processed_data = preprocess_data(data)  

    return processed_data
