"""
Feature Selector

This file defines which features are used by each model.

Models:
1. Basket Size Prediction
2. Reorder Prediction
3. Next Product Prediction
4. Churn Prediction
5. Recommendation System
"""

# =============================
# TARGET VARIABLES
# =============================

BASKET_TARGET = "add_to_cart_order"
REORDER_TARGET = "reordered"


# =============================
# COMMON FEATURES
# =============================

USER_FEATURES = [
    "user_total_orders",
    "user_total_products",
    "user_unique_products",
    "user_avg_cart_size",
    "user_max_cart_size",
    "user_reorder_ratio",
    "user_avg_days_between_orders",
    "user_order_frequency",
    "user_favorite_hour",
    "user_favorite_dow"
]

PRODUCT_FEATURES = [
    "product_total_orders",
    "product_unique_users",
    "product_reorder_rate",
    "product_avg_cart_position",
    "product_popularity_rank",
    "product_hour_peak"
]

USER_PRODUCT_FEATURES = [
    "user_product_orders",
    "user_product_first_order",
    "user_product_last_order",
    "user_product_avg_cart_position",
    "user_product_reorder_rate",
    "user_product_order_rate",
    "orders_since_last_purchase"
]

CATEGORY_FEATURES = [
    "department_popularity",
    "department_reorder_rate",
    "aisle_popularity"
]

TIME_FEATURES = [
    "is_weekend",
    "is_morning",
    "is_evening",
    "hour_sin",
    "hour_cos"
]


# =============================
# MODEL FEATURE SETS
# =============================

# Basket Size Model
BASKET_MODEL_FEATURES = (
    USER_FEATURES +
    TIME_FEATURES
)


# Reorder Prediction Model
REORDER_MODEL_FEATURES = (
    USER_FEATURES +
    PRODUCT_FEATURES +
    USER_PRODUCT_FEATURES +
    TIME_FEATURES
)


# Next Product Prediction Model
NEXT_PRODUCT_FEATURES = (
    USER_FEATURES +
    PRODUCT_FEATURES +
    USER_PRODUCT_FEATURES +
    TIME_FEATURES +
    CATEGORY_FEATURES
)


# Churn Prediction Model
CHURN_MODEL_FEATURES = (
    USER_FEATURES +
    TIME_FEATURES
)


# Recommendation System Model
RECOMMENDATION_FEATURES = (
    PRODUCT_FEATURES +
    USER_PRODUCT_FEATURES +
    CATEGORY_FEATURES
)


# =============================
# FEATURE REGISTRY
# =============================

FEATURE_REGISTRY = {
    "basket_model": BASKET_MODEL_FEATURES,
    "reorder_model": REORDER_MODEL_FEATURES,
    "next_product_model": NEXT_PRODUCT_FEATURES,
    "churn_model": CHURN_MODEL_FEATURES,
    "recommendation_model": RECOMMENDATION_FEATURES
}


# =============================
# HELPER FUNCTION
# =============================

def get_features(model_name: str):
    """
    Return feature list for a given model
    """

    if model_name not in FEATURE_REGISTRY:
        raise ValueError(f"Model '{model_name}' not found in registry")

    return FEATURE_REGISTRY[model_name]