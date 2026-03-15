# src/models/base_model.py

import os
import joblib
import logging
import numpy as np

from abc import ABC, abstractmethod
from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error


class BaseModel(ABC):
    """
    Base class for all machine learning models in the project.
    Provides common functionality for training, prediction,
    evaluation, saving, and loading models.
    """

    def __init__(self, name: str):
        self.name = name
        self.model = None

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s"
        )

    # ------------------------------------------------------------------
    # Model Definition (must be implemented by child classes)
    # ------------------------------------------------------------------

    @abstractmethod
    def build_model(self):
        """
        Each child model must implement this method
        to initialize the ML algorithm.
        """
        pass

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, X_train, y_train):
        """
        Train the model.
        """
        logging.info(f"Training model: {self.name}")

        if self.model is None:
            self.build_model()

        self.model.fit(X_train, y_train)

        logging.info("Training completed.")

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, X):
        """
        Generate predictions.
        """
        if self.model is None:
            raise ValueError("Model is not trained or loaded.")

        return self.model.predict(X)

    def predict_proba(self, X):
        """
        Probability predictions for classification models.
        """
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)

        raise AttributeError("Model does not support probability prediction.")

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self, X_test, y_test, task="classification"):
        """
        Evaluate the model performance.

        task options:
        - classification
        - regression
        """

        preds = self.predict(X_test)

        if task == "classification":
            acc = accuracy_score(y_test, preds)

            try:
                proba = self.predict_proba(X_test)[:, 1]
                auc = roc_auc_score(y_test, proba)
            except:
                auc = None

            results = {
                "accuracy": acc,
                "roc_auc": auc
            }

        elif task == "regression":
            rmse = np.sqrt(mean_squared_error(y_test, preds))

            results = {
                "rmse": rmse
            }

        else:
            raise ValueError("Invalid task type.")

        logging.info(f"Evaluation Results: {results}")
        return results

    # ------------------------------------------------------------------
    # Saving Model
    # ------------------------------------------------------------------

    def save(self, path="models/trained"):
        """
        Save trained model to disk.
        """

        os.makedirs(path, exist_ok=True)

        file_path = os.path.join(path, f"{self.name}.joblib")

        joblib.dump(self.model, file_path)

        logging.info(f"Model saved at: {file_path}")

    # ------------------------------------------------------------------
    # Loading Model
    # ------------------------------------------------------------------

    def load(self, path="models/trained"):
        """
        Load trained model from disk.
        """

        file_path = os.path.join(path, f"{self.name}.joblib")

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Model file not found: {file_path}")

        self.model = joblib.load(file_path)

        logging.info(f"Model loaded from: {file_path}")