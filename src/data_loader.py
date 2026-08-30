"""
data_loader.py

Loads transaction/event data for fraud or cyber-threat classification.

If a real dataset (CSV) is provided via --data_path, it is loaded directly.
Otherwise, a synthetic-but-realistic dataset is generated so the full
pipeline can be run and tested end-to-end without requiring an external
dataset. Swap in a real dataset (e.g. Kaggle's Credit Card Fraud dataset)
by pointing --data_path at your CSV — the column names below just need
to match, or you can adjust FEATURE_COLUMNS/LABEL_COLUMN accordingly.
"""

import os
import numpy as np
import pandas as pd

FEATURE_COLUMNS = ["amount", "duration_sec", "transaction_type", "source_ip_reputation"]
LABEL_COLUMN = "label"


def generate_synthetic_dataset(n_samples=5000, fraud_ratio=0.08, random_state=42, save_path=None):
    """
    Generates a synthetic dataset resembling transaction/network-event records,
    with a realistic-but-imperfect signal separating normal (0) from
    fraud/threat (1) records — i.e. classes overlap somewhat, like real data.
    """
    rng = np.random.RandomState(random_state)
    n_fraud = int(n_samples * fraud_ratio)
    n_normal = n_samples - n_fraud

    # Normal transactions: smaller amounts, longer typical duration, better IP reputation.
    # Wide spread + overlap with fraud-like behavior, so classes aren't perfectly separable.
    normal_amount = rng.gamma(shape=2.0, scale=60, size=n_normal)
    normal_duration = rng.normal(loc=100, scale=60, size=n_normal).clip(0.5)
    normal_type = rng.choice(["purchase", "transfer", "withdrawal", "login"], size=n_normal, p=[0.5, 0.2, 0.2, 0.1])
    normal_ip_rep = rng.beta(a=5, b=3, size=n_normal)

    # Fraud/threat: larger or erratic amounts, shorter duration, worse IP reputation —
    # with enough spread that some fraud cases look "normal" and vice versa.
    fraud_amount = rng.gamma(shape=2.0, scale=110, size=n_fraud)
    fraud_duration = rng.normal(loc=40, scale=35, size=n_fraud).clip(0.5)
    fraud_type = rng.choice(["purchase", "transfer", "withdrawal", "login"], size=n_fraud, p=[0.3, 0.4, 0.2, 0.1])
    fraud_ip_rep = rng.beta(a=3, b=5, size=n_fraud)

    df_normal = pd.DataFrame({
        "amount": normal_amount,
        "duration_sec": normal_duration,
        "transaction_type": normal_type,
        "source_ip_reputation": normal_ip_rep,
        LABEL_COLUMN: 0,
    })
    df_fraud = pd.DataFrame({
        "amount": fraud_amount,
        "duration_sec": fraud_duration,
        "transaction_type": fraud_type,
        "source_ip_reputation": fraud_ip_rep,
        LABEL_COLUMN: 1,
    })

    df = pd.concat([df_normal, df_fraud], ignore_index=True)
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df.to_csv(save_path, index=False)
        print(f"Synthetic dataset saved to {save_path} ({len(df)} rows, {df[LABEL_COLUMN].mean():.2%} positive)")

    return df


def load_dataset(data_path=None):
    """
    Loads a dataset from data_path if provided and exists; otherwise generates
    and returns a synthetic dataset (also cached to data/raw/synthetic.csv).
    """
    if data_path and os.path.exists(data_path):
        df = pd.read_csv(data_path)
        print(f"Loaded dataset from {data_path} ({len(df)} rows)")
        return df

    default_path = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "synthetic.csv")
    if os.path.exists(default_path):
        df = pd.read_csv(default_path)
        print(f"Loaded cached synthetic dataset from {default_path} ({len(df)} rows)")
        return df

    print("No dataset found — generating synthetic dataset for demonstration.")
    return generate_synthetic_dataset(save_path=default_path)


if __name__ == "__main__":
    df = load_dataset()
    print(df.head())
    print(df[LABEL_COLUMN].value_counts(normalize=True))
