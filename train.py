"""
train.py
--------
Pipeline d'entrainement pour la detection d'intrusions Cloud sur CSE-CIC-IDS2018.

Reproduit la chaine de traitement decrite dans le rapport :
1. Chargement du dataset pretraite (features selectionnees, deja nettoye)
2. Suppression de la classe FTP-BruteForce (trop peu representee : 53 lignes)
3. Encodage des labels (LabelEncoder)
4. Split train/test stratifie (80/20)
5. Normalisation (StandardScaler, fit sur le train uniquement)
6. Entrainement du modele retenu (Decision Tree, class_weight="balanced")
7. Evaluation (Accuracy, Precision, Recall, F1, AUC-ROC, MCC - moyennes macro)
8. Sauvegarde des artefacts (models/) pour l'API de prediction

Usage :
    python train.py --data data/sample_flows.parquet
    python train.py --data data/df_selected.parquet --model random_forest

Remarque :
    Le depot fournit un echantillon (data/sample_flows.parquet, 18 000 lignes)
    a titre de demonstration. Pour reproduire les resultats du rapport
    (1,57M lignes), utiliser le dataset complet CSE-CIC-IDS2018 (voir data/README.md).
"""

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"

MODEL_REGISTRY = {
    "decision_tree": lambda: DecisionTreeClassifier(
        class_weight="balanced", max_depth=10, random_state=42
    ),
    "random_forest": lambda: RandomForestClassifier(
        n_estimators=100,
        max_depth=8,
        max_features="sqrt",
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    ),
    "logistic_regression": lambda: LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        n_jobs=-1,
        random_state=42,
    ),
    "svm": lambda: LinearSVC(class_weight="balanced", max_iter=2000, random_state=42),
}


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    # FTP-BruteForce est retiree : seulement 53 exemples dans le dataset complet,
    # insuffisant pour un apprentissage fiable (voir rapport, section 3.4)
    if "FTP-BruteForce" in df["Label"].unique():
        df = df[df["Label"] != "FTP-BruteForce"].reset_index(drop=True)
    return df


def evaluer_modele(y_true, y_pred, y_proba, nom_modele, classes):
    """Calcule Accuracy, Precision/Recall/F1 (macro), AUC (OvR macro) et MCC."""
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)

    auc = None
    if y_proba is not None:
        try:
            auc = roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")
        except ValueError:
            auc = None

    print(f"\n===== {nom_modele} =====")
    print(f"Accuracy          : {accuracy:.4f}")
    print(f"Precision (macro) : {precision:.4f}")
    print(f"Recall (macro)    : {recall:.4f}")
    print(f"F1-score (macro)  : {f1:.4f}")
    if auc is not None:
        print(f"AUC-ROC (macro)   : {auc:.4f}")
    print(f"MCC               : {mcc:.4f}")
    print("\nClassification report :")
    print(classification_report(y_true, y_pred, target_names=classes, zero_division=0))

    return {
        "model": nom_modele,
        "accuracy": accuracy,
        "precision_macro": precision,
        "recall_macro": recall,
        "f1_macro": f1,
        "auc_ovr_macro": auc,
        "mcc": mcc,
    }


def main():
    parser = argparse.ArgumentParser(description="Entrainement du modele IDS (CSE-CIC-IDS2018)")
    parser.add_argument(
        "--data",
        type=str,
        default=str(ROOT / "data" / "df_selected.parquet"),
        help="Chemin du dataset pretraite (.parquet, colonne cible 'Label'). "
             "Utiliser data/sample_flows.parquet pour un test rapide.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="decision_tree",
        choices=list(MODEL_REGISTRY.keys()),
        help="Modele a entrainer (decision_tree = modele retenu pour le deploiement)",
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--no-save", action="store_true", help="N'ecrit pas les artefacts dans models/"
    )
    args = parser.parse_args()

    print(f"Chargement du dataset : {args.data}")
    df = load_data(args.data)
    print(f"Shape apres nettoyage : {df.shape}")
    print(df["Label"].value_counts())

    # Encodage des labels
    le = LabelEncoder()
    y = le.fit_transform(df["Label"])
    X = df.drop(columns=["Label"])
    feature_names = X.columns.tolist()

    # Split stratifie
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, stratify=y, random_state=args.random_state
    )
    print(f"\nTrain : {X_train.shape[0]} lignes | Test : {X_test.shape[0]} lignes")

    # Normalisation (fit sur le train uniquement, pas de fuite de donnees)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # Entrainement
    model = MODEL_REGISTRY[args.model]()
    start = time.time()
    model.fit(X_train_s, y_train)
    elapsed = time.time() - start
    print(f"\nTemps d'entrainement ({args.model}) : {elapsed:.2f} sec")

    # Prediction + evaluation
    y_pred = model.predict(X_test_s)
    y_proba = model.predict_proba(X_test_s) if hasattr(model, "predict_proba") else None
    evaluer_modele(y_test, y_pred, y_proba, args.model, le.classes_)

    if not args.no_save:
        MODELS_DIR.mkdir(exist_ok=True)
        joblib.dump(model, MODELS_DIR / "decision_tree_model.pkl")
        joblib.dump(scaler, MODELS_DIR / "scaler.pkl")
        joblib.dump(le, MODELS_DIR / "label_encoder.pkl")
        with open(MODELS_DIR / "feature_names.json", "w") as f:
            json.dump(feature_names, f)
        print(f"\nArtefacts sauvegardes dans : {MODELS_DIR}")


if __name__ == "__main__":
    main()
