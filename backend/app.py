from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from backend.model_loader import load_production_model
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Fake News Detection API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model at startup
try:
    model, model_version = load_production_model()
except Exception as e:
    model = None
    model_version = None
    print("Model loading failed:", e)


class NewsRequest(BaseModel):
    text: str


@app.get("/health")
def health():
    if model is None:
        return {"status": "error", "model_loaded": False}
    return {
        "status": "ok",
        "model_loaded": True,
        "model_version": model_version
    }


@app.post("/predict")
def predict(request: NewsRequest):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    prediction = model.predict([request.text])[0]
    probabilities = model.predict_proba([request.text])[0]
    confidence = float(max(probabilities))

    return {
        "label": prediction,
        "confidence": round(confidence, 4),
        "model_version": model_version
    }
