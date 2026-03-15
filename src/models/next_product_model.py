from lightgbm import LGBMRanker

from src.models.base_model import BaseModel
from src.features.feature_selector import get_features


class NextProductModel(BaseModel):
    """
    Rank products that a user is most likely
    to add in the next order.
    """

    def __init__(self):

        super().__init__("next_product_model")

        self.problem_type = "ranking"
        self.target = "label"

        self.features = get_features("next_product_model")

    def build_model(self):

        self.model = LGBMRanker(
            objective="lambdarank",
            metric="ndcg",
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=64,
            random_state=42,
            verbose=-1
        )
