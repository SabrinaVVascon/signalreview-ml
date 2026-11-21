"""API REST para classificar avaliacoes em portugues."""
from functools import lru_cache
from pathlib import Path
import json

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "artifacts" / "model.joblib"
METRICS_PATH = BASE_DIR / "artifacts" / "metrics.json"

app = FastAPI(
    title="SignalReview ML",
    description="Classificacao de sentimentos de avaliacoes em portugues com TF-IDF e Machine Learning.",
    version="0.1.0",
)


class PredictionRequest(BaseModel):
    texto: str = Field(min_length=3, max_length=5000, examples=["Produto excelente e entrega rapida!"])


class PredictionResponse(BaseModel):
    sentimento: str
    confianca: float
    probabilidades: dict[str, float]


@lru_cache(maxsize=1)
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Modelo nao treinado. Execute: python -m ml.train")
    # Carregue somente artefatos gerados por voce ou de fonte confiavel (joblib usa pickle).
    return joblib.load(MODEL_PATH)


@app.get("/health")
def health():
    return {"status": "ok", "modelo_disponivel": MODEL_PATH.exists()}


@app.get("/model-info")
def model_info():
    if not METRICS_PATH.exists():
        raise HTTPException(status_code=503, detail="Ainda nao existem metricas de treinamento.")
    return json.loads(METRICS_PATH.read_text(encoding="utf-8"))


@app.post("/predict", response_model=PredictionResponse)
def predict(body: PredictionRequest):
    try:
        model = load_model()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    texto = body.texto.strip()
    if len(texto) < 3:
        raise HTTPException(status_code=422, detail="Informe um texto com pelo menos 3 caracteres.")
    classes = list(model.classes_)
    probabilities = model.predict_proba([texto])[0]
    scores = {str(label): round(float(prob), 4) for label, prob in zip(classes, probabilities)}
    result = str(model.predict([texto])[0])
    return PredictionResponse(
        sentimento=result,
        confianca=round(float(probabilities[classes.index(result)]), 4),
        probabilidades=scores,
    )
