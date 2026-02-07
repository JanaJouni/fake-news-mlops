import mlflow
import pickle
import logging
import os

MODEL_NAME = "fake_news_gru_v1S"
MAX_LEN = 200

LOCAL_TOKENIZER_PATH = "backend/artifacts/tokenizer.pkl"

def load_production_model():
    """
    Load GRU model from MLflow (Production)
    Load tokenizer locally (stable)
    """

    try:
        # ---- Load model from MLflow Registry ----
        model_uri = f"models:/{MODEL_NAME}@production"
        model = mlflow.keras.load_model(model_uri)

        # ---- Load tokenizer locally ----
        if not os.path.exists(LOCAL_TOKENIZER_PATH):
            raise FileNotFoundError(
                f"Tokenizer not found at {LOCAL_TOKENIZER_PATH}"
            )

        with open(LOCAL_TOKENIZER_PATH, "rb") as f:
            tokenizer = pickle.load(f)

        logging.info("GRU model loaded from MLflow")
        logging.info("Tokenizer loaded locally")

        return model, tokenizer, MAX_LEN, "@production"

    except Exception as e:
        logging.error(f"Failed to load GRU model/tokenizer: {e}")
        raise RuntimeError("Model loading failed")
