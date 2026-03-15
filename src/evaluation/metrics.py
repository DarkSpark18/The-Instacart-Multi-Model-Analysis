"""
Evaluation Metrics for Multi-Model ML System

Provides standardized metrics for:
- Classification models
- Regression models
- Ranking models
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score
)


# -------------------------
# CLASSIFICATION METRICS
# -------------------------

def classification_metrics(y_true, y_pred, y_proba=None):
    """
    Calculate comprehensive classification metrics.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        y_proba: Predicted probabilities (optional)
    
    Returns:
        Dictionary of metrics
    """
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='binary', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='binary', zero_division=0),
        'f1_score': f1_score(y_true, y_pred, average='binary', zero_division=0)
    }
    
    if y_proba is not None:
        try:
            metrics['roc_auc'] = roc_auc_score(y_true, y_proba)
        except:
            metrics['roc_auc'] = None
    
    return metrics


# -------------------------
# REGRESSION METRICS
# -------------------------

def regression_metrics(y_true, y_pred):
    """
    Calculate comprehensive regression metrics.
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
    
    Returns:
        Dictionary of metrics
    """
    
    metrics = {
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mae': mean_absolute_error(y_true, y_pred),
        'r2_score': r2_score(y_true, y_pred)
    }
    
    return metrics


# -------------------------
# RANKING METRICS
# -------------------------

def ndcg_score(y_true, y_pred, k=10):
    """
    Calculate Normalized Discounted Cumulative Gain.
    
    Args:
        y_true: Ground truth relevance scores
        y_pred: Predicted scores
        k: Number of top results to consider
    
    Returns:
        NDCG score
    """
    
    # Sort predictions
    order = np.argsort(y_pred)[::-1]
    y_true_sorted = np.take(y_true, order[:k])
    
    # Calculate DCG
    gains = 2 ** y_true_sorted - 1
    discounts = np.log2(np.arange(len(y_true_sorted)) + 2)
    dcg = np.sum(gains / discounts)
    
    # Calculate IDCG
    ideal_order = np.argsort(y_true)[::-1]
    y_true_ideal = np.take(y_true, ideal_order[:k])
    ideal_gains = 2 ** y_true_ideal - 1
    idcg = np.sum(ideal_gains / discounts)
    
    if idcg == 0:
        return 0.0
    
    return dcg / idcg


def ranking_metrics(y_true, y_pred, k=10):
    """
    Calculate comprehensive ranking metrics.
    
    Args:
        y_true: Ground truth relevance scores
        y_pred: Predicted scores
        k: Number of top results to consider
    
    Returns:
        Dictionary of metrics
    """
    
    metrics = {
        f'ndcg@{k}': ndcg_score(y_true, y_pred, k=k)
    }
    
    return metrics


# -------------------------
# PRINT METRICS
# -------------------------

def print_metrics(metrics, model_name="Model"):
    """
    Pretty print evaluation metrics.
    
    Args:
        metrics: Dictionary of metrics
        model_name: Name of the model
    """
    
    print(f"\n{'='*50}")
    print(f"{model_name} Evaluation Metrics")
    print(f"{'='*50}")
    
    for metric_name, value in metrics.items():
        if value is not None:
            print(f"{metric_name:20s}: {value:.4f}")
        else:
            print(f"{metric_name:20s}: N/A")
    
    print(f"{'='*50}\n")
