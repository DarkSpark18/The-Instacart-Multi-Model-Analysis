# src/predict.py

import os
import logging
import pandas as pd

from src.models.model_registry import ModelRegistry
from src.features.feature_selector import get_features


class ModelPredictor:

    def __init__(self):

        self.registry = ModelRegistry()

        # FIXED: Changed to match where features are saved
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
                f"Dataset not found: {self.data_path}"
            )

        df = pd.read_parquet(self.data_path)

        logging.info(f"Dataset loaded: {df.shape}")

        return df

    # ---------------------------------------------------
    # RUN PREDICTIONS
    # ---------------------------------------------------

    def predict_model(self, model_name, df):

        logging.info(f"Predicting with {model_name}")

        model = self.registry.get(model_name)

        model.load()

        features = get_features(model_name)

        X = df[features].fillna(0)

        preds = model.predict(X)

        df[f"{model_name}_prediction"] = preds

        return df

    # ---------------------------------------------------
    # PREDICT ALL
    # ---------------------------------------------------

    def predict_all(self):

        df = self.load_data()

        for model_name in self.registry.list_models():

            try:

                df = self.predict_model(model_name, df)

            except Exception as e:

                logging.error(f"{model_name} prediction failed: {str(e)}")

        return df

    # ---------------------------------------------------
    # SAVE
    # ---------------------------------------------------

    def save(self, df):

        os.makedirs("data/predictions", exist_ok=True)

        output_path = "data/predictions/predictions.parquet"

        df.to_parquet(output_path)

        logging.info(f"Predictions saved to {output_path}")


if __name__ == "__main__":

    predictor = ModelPredictor()

    df = predictor.predict_all()

    predictor.save(df)

    print("\n" + "="*50)
    print("PREDICTIONS GENERATED SUCCESSFULLY")
    print("="*50)
    print(f"\nPrediction columns added:")
    pred_cols = [col for col in df.columns if col.endswith("_prediction")]
    for col in pred_cols:
        print(f"  - {col}")
