import logging

from src.models.basket_size_model import BasketSizeModel
from src.models.reorder_probability_model import ReorderProbabilityModel
from src.models.next_product_model import NextProductModel
from src.models.churn_prediction_model import ChurnPredictionModel
from src.models.recommendation_model import RecommendationModel


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


class ModelRegistry:
    """
    Central registry for all ML models in the project.
    Handles model discovery, creation, and management.
    """

    def __init__(self):

        self.models = {
            "basket_model": BasketSizeModel,
            "reorder_model": ReorderProbabilityModel,
            "next_product_model": NextProductModel,
            "churn_model": ChurnPredictionModel,
            "recommendation_model": RecommendationModel
        }

    # -------------------------------------------------------
    # Get Single Model
    # -------------------------------------------------------

    def get(self, model_name: str):
        """
        Instantiate and return a model by name.
        """

        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found in registry.")

        logging.info(f"Loading model: {model_name}")

        return self.models[model_name]()

    # -------------------------------------------------------
    # List All Models
    # -------------------------------------------------------

    def list_models(self):
        """
        Return all available model names.
        """

        return list(self.models.keys())

    # -------------------------------------------------------
    # Instantiate All Models
    # -------------------------------------------------------

    def create_all_models(self):
        """
        Instantiate all models in the registry.
        """

        logging.info("Creating all models...")

        return {
            name: model_class()
            for name, model_class in self.models.items()
        }

    # -------------------------------------------------------
    # Register New Model
    # -------------------------------------------------------

    def register(self, name: str, model_class):
        """
        Dynamically add a new model to the registry.
        """

        if name in self.models:
            raise ValueError(f"Model '{name}' already exists.")

        self.models[name] = model_class

        logging.info(f"Model '{name}' registered successfully.")

    # -------------------------------------------------------
    # Remove Model
    # -------------------------------------------------------

    def unregister(self, name: str):
        """
        Remove a model from the registry.
        """

        if name not in self.models:
            raise ValueError(f"Model '{name}' not found.")

        del self.models[name]

        logging.info(f"Model '{name}' removed from registry.")
