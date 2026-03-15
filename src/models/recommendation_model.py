from lightgbm import LGBMRanker

from src.models.base_model import BaseModel
from src.features.feature_selector import get_features


class RecommendationModel(BaseModel):
    """
    Product recommendation ranking model.
    """

    def __init__(self):

        super().__init__("recommendation_model")

        self.problem_type = "ranking"
        self.target = "label"

        self.features = get_features("recommendation_model")

    def build_model(self):

        self.model = LGBMRanker(
            objective="lambdarank",
            metric="ndcg",
            n_estimators=600,
            learning_rate=0.05,
            num_leaves=64,
            random_state=42,
            verbose=-1
        )
