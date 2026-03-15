from catboost import CatBoostClassifier

from src.base_model import BaseModel
from src.feature_selector import get_features


class ReorderProbabilityModel(BaseModel):
    """
    Predict whether a product will be reordered.
    """

    def __init__(self):

        super().__init__("reorder_model")

        self.problem_type = "classification"
        self.target = "reordered"

        self.features = get_features("reorder_model")

    def build_model(self):

        self.model = CatBoostClassifier(
            iterations=800,
            depth=8,
            learning_rate=0.05,
            loss_function="Logloss",
            eval_metric="AUC",
            random_state=42,
            verbose=0
        )
