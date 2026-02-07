import mlflow
import pickle
import logging

MODEL_NAME = "fake_news_gru"
MAX_LEN = 200  # must match training

def load_production_model():
    """
    Load GRU model + tokenizer from MLflow Model Registry (Production stage)

    Returns:
        model: Keras GRU model
        tokenizer: fitted Keras tokenizer
        max_len: int
        model_version: str
    """

    try:
        # --------------------------
        # Load model from Production
        # --------------------------
        model_uri = f"models:/{MODEL_NAME}@production"
        model = mlflow.keras.load_model(model_uri)

        # --------------------------
        # Load tokenizer artifact
        # --------------------------
        tokenizer_uri = (
            f"models:/{MODEL_NAME}/production/tokenizer/tokenizer.pkl"
        )

        tokenizer_path = mlflow.artifacts.download_artifacts(tokenizer_uri)

        with open(tokenizer_path, "rb") as f:
            tokenizer = pickle.load(f)

        logging.info("GRU model and tokenizer loaded successfully from MLflow")

        return model, tokenizer, MAX_LEN, "production"

    except Exception as e:
        logging.error(f"Failed to load GRU model from MLflow: {e}")
        raise RuntimeError(
            "Model loading failed. "
            "Ensure a model version exists in the Production stage."
        )
