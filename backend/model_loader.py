import mlflow

MODEL_NAME = "fake_news_detector_v1"

def load_production_model():
    model_uri = f"models:/{MODEL_NAME}@production"
    model = mlflow.sklearn.load_model(model_uri)
    return model, "production"
