from lightgbm import LGBMRegressor

from src.base_model import BaseModel
from src.feature_selector import get_features


class BasketSizeModel(BaseModel):
    """
    Predict how many products a user will add
    in the next order.
    """

    def __init__(self):

        super().__init__("basket_model")

        self.problem_type = "regression"
        self.target = "add_to_cart_order"

        self.features = get_features("basket_model")

    def build_model(self):

        self.model = LGBMRegressor(
            n_estimators=600,
            learning_rate=0.05,
            num_leaves=64,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
