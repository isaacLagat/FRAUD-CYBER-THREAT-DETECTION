"""
train.py

Trains Logistic Regression, Random Forest, and XGBoost classifiers on the
preprocessed data, compares them by F1-score (a better metric than raw
accuracy for imbalanced fraud/threat data), and saves the best-performing
model plus its preprocessor to disk.
"""

import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from xgboost import XGBClassifier

from data_loader import load_dataset
from preprocess import split_and_preprocess
from utils import save_pickle

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


def get_candidate_models(random_state=42):
    return {
        "logistic_regression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=random_state),
        "random_forest": RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=random_state),
        "xgboost": XGBClassifier(
            n_estimators=200, eval_metric="logloss", random_state=random_state,
            scale_pos_weight=10,
        ),
    }


def train_and_select_best(data_path=None, random_state=42):
    df = load_dataset(data_path)
    X_train, X_test, y_train, y_test, preprocessor = split_and_preprocess(df, random_state=random_state)

    models = get_candidate_models(random_state=random_state)
    results = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        score = f1_score(y_test, preds)
        results[name] = score
        print(f"{name:20s} F1-score: {score:.4f}")

    best_name = max(results, key=results.get)
    best_model = models[best_name]
    print(f"\nBest model: {best_name} (F1-score: {results[best_name]:.4f})")

    save_pickle(best_model, os.path.join(MODELS_DIR, "best_model.pkl"))
    save_pickle(preprocessor, os.path.join(MODELS_DIR, "preprocessor.pkl"))
    save_pickle(best_name, os.path.join(MODELS_DIR, "best_model_name.pkl"))

    return best_model, preprocessor, X_test, y_test, results


if __name__ == "__main__":
    train_and_select_best()
