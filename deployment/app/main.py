from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, create_model
import numpy as np

from app.model_loader import load_artifacts

app = FastAPI(
    title="IDS Cloud API",
    description="API de détection d'intrusion (Benign / DDoS / SSH-Bruteforce) basée sur Machine Learning",
    version="1.0.0"
)

# Chargement des artefacts au démarrage de l'API (une seule fois)
model, scaler, label_encoder, feature_names = load_artifacts()

# ============ Construction dynamique du schéma d'entrée ============
# Crée un modèle Pydantic avec un champ float par feature attendue
fields = {name: (float, ...) for name in feature_names}
FlowFeatures = create_model("FlowFeatures", **fields)


class PredictionResponse(BaseModel):
    prediction: str
    prediction_code: int
    probabilities: dict


@app.get("/")
def root():
    return {
        "message": "IDS Cloud API — utilisez POST /predict pour classifier un flux réseau",
        "classes_possibles": list(label_encoder.classes_)
    }


@app.get("/health")
def health_check():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict", response_model=PredictionResponse)
def predict(flow: FlowFeatures):
    try:
        # Reconstituer le vecteur de features dans le bon ordre
        input_dict = flow.dict()
        X = np.array([[input_dict[name] for name in feature_names]])

        # Appliquer la même normalisation que pendant l'entraînement
        X_scaled = scaler.transform(X)

        # Prédiction
        pred_code = int(model.predict(X_scaled)[0])
        pred_label = label_encoder.inverse_transform([pred_code])[0]

        # Probabilités par classe (si le modèle les supporte)
        proba_dict = {}
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_scaled)[0]
            proba_dict = {
                cls: round(float(p), 4)
                for cls, p in zip(label_encoder.classes_, proba)
            }

        return PredictionResponse(
            prediction=pred_label,
            prediction_code=pred_code,
            probabilities=proba_dict
        )

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))