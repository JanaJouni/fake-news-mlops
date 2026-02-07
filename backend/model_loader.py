import mlflow

MODEL_NAME = "fake_news_detector"

def load_production_model():
    model_uri = f"models:/{MODEL_NAME}@production"
    model = mlflow.sklearn.load_model(model_uri)
    return model, "production"
