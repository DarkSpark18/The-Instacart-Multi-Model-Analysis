from xgboost import XGBClassifier

from src.base_model import BaseModel
from src.feature_selector import get_features


class ChurnPredictionModel(BaseModel):
    """
    Predict if a user will stop ordering.
    """

    def __init__(self):

        super().__init__("churn_model")

        self.problem_type = "classification"
        self.target = "churn"

        self.features = get_features("churn_model")

    def build_model(self):

        self.model = XGBClassifier(
            n_estimators=600,
            learning_rate=0.05,
            max_depth=8,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
            verbosity=0,
            use_label_encoder=False  # Suppress warning
        )