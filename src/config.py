"""
Configuration file for Instacart Multi-Model ML Engine

Contains all project-level configurations including:
- Data paths
- Model hyperparameters
- Training settings
- Feature engineering parameters
"""

from pathlib import Path


# =============================
# PROJECT PATHS
# =============================

PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
FEATURES_DIR = DATA_DIR / "features"
PREDICTIONS_DIR = DATA_DIR / "predictions"

MODELS_DIR = PROJECT_ROOT / "models"
TRAINED_MODELS_DIR = MODELS_DIR / "trained"

LOGS_DIR = PROJECT_ROOT / "logs"


# =============================
# DATA FILES
# =============================

PROCESSED_DATA_FILE = PROCESSED_DATA_DIR / "instacart_processed.parquet"
FEATURE_STORE_FILE = FEATURES_DIR / "feature_store.parquet"
PREDICTIONS_FILE = PREDICTIONS_DIR / "predictions.parquet"


# =============================
# DATA PROCESSING
# =============================

# Sample size for development (set to None for full dataset)
SAMPLE_USERS = 5000

# Missing value strategy
MISSING_VALUE_STRATEGY = "fillna"  # Options: "fillna", "drop"


# =============================
# FEATURE ENGINEERING
# =============================

# Time-based features
TIME_FEATURES_ENABLED = True

# Category aggregations
CATEGORY_AGG_ENABLED = True

# User-product interactions
USER_PRODUCT_FEATURES_ENABLED = True


# =============================
# MODEL TRAINING
# =============================

# Train-test split
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Cross-validation
CV_FOLDS = 5
CV_ENABLED = False  # Set to True for CV during training


# =============================
# MODEL HYPERPARAMETERS
# =============================

# LightGBM Settings
LGBM_PARAMS = {
    "n_estimators": 600,
    "learning_rate": 0.05,
    "num_leaves": 64,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
    "verbose": -1
}

# XGBoost Settings
XGB_PARAMS = {
    "n_estimators": 600,
    "learning_rate": 0.05,
    "max_depth": 8,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
    "verbosity": 0
}

# CatBoost Settings
CATBOOST_PARAMS = {
    "iterations": 800,
    "depth": 8,
    "learning_rate": 0.05,
    "random_state": RANDOM_STATE,
    "verbose": 0
}


# =============================
# LOGGING
# =============================

LOG_LEVEL = "INFO"  # Options: DEBUG, INFO, WARNING, ERROR
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"


# =============================
# EVALUATION
# =============================

# Metrics to compute
COMPUTE_CLASSIFICATION_METRICS = True
COMPUTE_REGRESSION_METRICS = True
COMPUTE_RANKING_METRICS = True

# Ranking evaluation
RANKING_K = 10  # Top K for NDCG calculation


# =============================
# CHURN DEFINITION
# =============================

CHURN_DAYS_THRESHOLD = 30  # Days without order to consider churn
CHURN_MIN_ORDERS = 5  # Minimum orders to not be considered churned


# =============================
# SYSTEM
# =============================

# Number of CPU cores to use (-1 for all)
N_JOBS = -1

# Enable GPU (if available)
USE_GPU = False


# =============================
# HELPER FUNCTIONS
# =============================

def create_directories():
    """Create all necessary directories if they don't exist."""
    
    directories = [
        DATA_DIR,
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        FEATURES_DIR,
        PREDICTIONS_DIR,
        MODELS_DIR,
        TRAINED_MODELS_DIR,
        LOGS_DIR
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
    
    print("All directories created successfully.")


if __name__ == "__main__":
    create_directories()
    print("\n✅ Configuration loaded successfully")
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Data Directory: {DATA_DIR}")
    print(f"Models Directory: {MODELS_DIR}")
"""
Configuration file for Instacart Multi-Model ML Engine

Contains all project-level configurations including:
- Data paths
- Model hyperparameters
- Training settings
- Feature engineering parameters
"""

from pathlib import Path


# =============================
# PROJECT PATHS
# =============================

PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
FEATURES_DIR = DATA_DIR / "features"
PREDICTIONS_DIR = DATA_DIR / "predictions"

MODELS_DIR = PROJECT_ROOT / "models"
TRAINED_MODELS_DIR = MODELS_DIR / "trained"

LOGS_DIR = PROJECT_ROOT / "logs"


# =============================
# DATA FILES
# =============================

PROCESSED_DATA_FILE = PROCESSED_DATA_DIR / "instacart_processed.parquet"
FEATURE_STORE_FILE = FEATURES_DIR / "feature_store.parquet"
PREDICTIONS_FILE = PREDICTIONS_DIR / "predictions.parquet"


# =============================
# DATA PROCESSING
# =============================

# Sample size for development (set to None for full dataset)
SAMPLE_USERS = 5000

# Missing value strategy
MISSING_VALUE_STRATEGY = "fillna"  # Options: "fillna", "drop"


# =============================
# FEATURE ENGINEERING
# =============================

# Time-based features
TIME_FEATURES_ENABLED = True

# Category aggregations
CATEGORY_AGG_ENABLED = True

# User-product interactions
USER_PRODUCT_FEATURES_ENABLED = True


# =============================
# MODEL TRAINING
# =============================

# Train-test split
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Cross-validation
CV_FOLDS = 5
CV_ENABLED = False  # Set to True for CV during training


# =============================
# MODEL HYPERPARAMETERS
# =============================

# LightGBM Settings
LGBM_PARAMS = {
    "n_estimators": 600,
    "learning_rate": 0.05,
    "num_leaves": 64,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
    "verbose": -1
}

# XGBoost Settings
XGB_PARAMS = {
    "n_estimators": 600,
    "learning_rate": 0.05,
    "max_depth": 8,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
    "verbosity": 0
}

# CatBoost Settings
CATBOOST_PARAMS = {
    "iterations": 800,
    "depth": 8,
    "learning_rate": 0.05,
    "random_state": RANDOM_STATE,
    "verbose": 0
}


# =============================
# LOGGING
# =============================

LOG_LEVEL = "INFO"  # Options: DEBUG, INFO, WARNING, ERROR
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"


# =============================
# EVALUATION
# =============================

# Metrics to compute
COMPUTE_CLASSIFICATION_METRICS = True
COMPUTE_REGRESSION_METRICS = True
COMPUTE_RANKING_METRICS = True

# Ranking evaluation
RANKING_K = 10  # Top K for NDCG calculation


# =============================
# CHURN DEFINITION
# =============================

CHURN_DAYS_THRESHOLD = 30  # Days without order to consider churn
CHURN_MIN_ORDERS = 5  # Minimum orders to not be considered churned


# =============================
# SYSTEM
# =============================

# Number of CPU cores to use (-1 for all)
N_JOBS = -1

# Enable GPU (if available)
USE_GPU = False


# =============================
# HELPER FUNCTIONS
# =============================

def create_directories():
    """Create all necessary directories if they don't exist."""
    
    directories = [
        DATA_DIR,
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        FEATURES_DIR,
        PREDICTIONS_DIR,
        MODELS_DIR,
        TRAINED_MODELS_DIR,
        LOGS_DIR
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
    
    print("All directories created successfully.")


if __name__ == "__main__":
    create_directories()
    print("\n✅ Configuration loaded successfully")
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Data Directory: {DATA_DIR}")
    print(f"Models Directory: {MODELS_DIR}")
