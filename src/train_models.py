# src/train_models.py

import os
import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from src.model_registry import ModelRegistry
from src.feature_selector import get_features


class ModelTrainer:

    def __init__(self):

        self.registry = ModelRegistry()

        self.data_path = "data/features/feature_store.parquet"

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s"
        )

    # ---------------------------------------------------
    # LOAD DATA
    # ---------------------------------------------------

    def load_data(self):

        if not os.path.exists(self.data_path):

            raise FileNotFoundError(
                f"\nDataset not found.\n"
                f"Expected: {self.data_path}\n"
                f"Run feature pipeline first."
            )

        logging.info(f"Loading dataset from {self.data_path}")

        df = pd.read_parquet(self.data_path)

        logging.info(f"Dataset shape: {df.shape}")

        return df

    # ---------------------------------------------------
    # TRAIN SINGLE MODEL
    # ---------------------------------------------------

    def train_model(self, model_name, df):

        logging.info(f"Training model: {model_name}")

        model = self.registry.get(model_name)

        features = get_features(model_name)

        target = model.target

        if target not in df.columns:
            raise ValueError(f"Target column '{target}' not found in dataset")

        # Prepare features
        X = df[features].fillna(0)
        y = df[target]
        
        # Fix target values based on model type
        if model.problem_type == "classification":
            # Ensure binary targets are 0/1
            y = (y > 0).astype(int)
            
            # Check if we have both classes
            unique_vals = y.unique()
            if len(unique_vals) < 2:
                logging.warning(f"{model_name}: Only one class present in target. Skipping.")
                return {"status": "skipped", "reason": "single_class", "class_present": unique_vals[0]}

        # Split data
        if model.problem_type == "classification":
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
        
        # Handle ranking models specially
        if model.problem_type == "ranking":
            # For ranking models, we need to provide group information
            # Group by user_id if available
            if "user_id" not in df.columns:
                logging.error(f"{model_name}: user_id not found for grouping")
                return {"status": "failed", "reason": "no_user_id"}
            
            # Build the model first
            model.build_model()
            
            # Create groups for train/test
            train_indices = X_train.index
            test_indices = X_test.index
            
            # Get user_id for each row
            train_users = df.loc[train_indices, "user_id"]
            test_users = df.loc[test_indices, "user_id"]
            
            # Count items per user (group sizes)
            train_groups = train_users.value_counts().sort_index().values
            test_groups = test_users.value_counts().sort_index().values
            
            # Sort by user_id to match group order
            train_order = train_users.argsort()
            test_order = test_users.argsort()
            
            X_train_sorted = X_train.iloc[train_order]
            y_train_sorted = y_train.iloc[train_order]
            X_test_sorted = X_test.iloc[test_order]
            y_test_sorted = y_test.iloc[test_order]
            
            # Train with groups
            try:
                model.model.fit(
                    X_train_sorted, 
                    y_train_sorted,
                    group=train_groups
                )
                
                logging.info("Training completed (ranking model).")
                
                # Save the model
                model.save()
                
                return {"status": "success", "model_type": "ranking"}
                
            except Exception as e:
                logging.error(f"Ranking model training failed: {e}")
                return {"status": "failed", "error": str(e)}
        else:
            # Regular training for classification/regression
            model.train(X_train, y_train)

            results = model.evaluate(
                X_test,
                y_test,
                task=model.problem_type
            )

            model.save()

            return results

    # ---------------------------------------------------
    # TRAIN ALL MODELS
    # ---------------------------------------------------

    def train_all(self):

        df = self.load_data()

        results = {}

        for model_name in self.registry.list_models():

            try:

                results[model_name] = self.train_model(model_name, df)

            except Exception as e:

                logging.error(f"{model_name} training failed: {str(e)}")
                results[model_name] = {"status": "failed", "error": str(e)}

        return results


# ---------------------------------------------------
# MAIN
# ---------------------------------------------------

if __name__ == "__main__":

    trainer = ModelTrainer()

    results = trainer.train_all()

    print("\n" + "="*70)
    print("TRAINING RESULTS")
    print("="*70)
    for model_name, metrics in results.items():
        print(f"\n{model_name}:")
        if isinstance(metrics, dict):
            for metric, value in metrics.items():
                if isinstance(value, (int, float)) and metric not in ['class_present']:
                    print(f"  {metric}: {value:.4f}")
                else:
                    print(f"  {metric}: {value}")
        else:
            print(f"  {metrics}")
    print("\n" + "="*70 + "\n")