"""
evaluate.py

Loads the saved best model and reports real evaluation metrics (accuracy,
precision, recall, F1, confusion matrix) on a held-out test set.
"""

import os

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from data_loader import load_dataset
from preprocess import split_and_preprocess
from utils import load_pickle

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


def evaluate(data_path=None, random_state=42):
    model_path = os.path.join(MODELS_DIR, "best_model.pkl")
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            "No trained model found. Run `python main.py --train` first."
        )

    model = load_pickle(model_path)
    best_name = load_pickle(os.path.join(MODELS_DIR, "best_model_name.pkl"))

    df = load_dataset(data_path)
    _, X_test, _, y_test, _ = split_and_preprocess(df, random_state=random_state)

    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, zero_division=0)
    rec = recall_score(y_test, preds, zero_division=0)
    f1 = f1_score(y_test, preds, zero_division=0)
    cm = confusion_matrix(y_test, preds)

    print(f"Model: {best_name}")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print("Confusion Matrix:")
    print(cm)

    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "confusion_matrix": cm}


if __name__ == "__main__":
    evaluate()
