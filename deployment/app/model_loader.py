import json
import os

import joblib

# models/ vit a la racine du depot : deployment/app/model_loader.py -> ../../models
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "models")


def load_artifacts():
    model = joblib.load(os.path.join(MODELS_DIR, "decision_tree_model.pkl"))
    scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
    label_encoder = joblib.load(os.path.join(MODELS_DIR, "label_encoder.pkl"))

    with open(os.path.join(MODELS_DIR, "feature_names.json"), "r") as f:
        feature_names = json.load(f)

    return model, scaler, label_encoder, feature_names
