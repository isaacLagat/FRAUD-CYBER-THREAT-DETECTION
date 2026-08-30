"""
preprocess.py

Handles feature preprocessing: categorical encoding, numerical scaling,
and train/test splitting. Returns processed arrays plus the fitted
transformers so the same preprocessing can be applied at inference time.
"""

from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from data_loader import FEATURE_COLUMNS, LABEL_COLUMN

NUMERICAL_COLUMNS = ["amount", "duration_sec", "source_ip_reputation"]
CATEGORICAL_COLUMNS = ["transaction_type"]


def build_preprocessor():
    """Builds a ColumnTransformer that scales numerical and one-hot encodes categorical features."""
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERICAL_COLUMNS),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLUMNS),
        ]
    )


def split_and_preprocess(df, test_size=0.2, random_state=42):
    """
    Splits the dataframe into train/test sets and fits preprocessing on the
    training set only (to avoid data leakage), then transforms both sets.

    Returns: X_train, X_test, y_train, y_test, fitted_preprocessor
    """
    X = df[FEATURE_COLUMNS]
    y = df[LABEL_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    preprocessor = build_preprocessor()
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    return X_train_processed, X_test_processed, y_train, y_test, preprocessor


if __name__ == "__main__":
    from data_loader import load_dataset

    df = load_dataset()
    X_train, X_test, y_train, y_test, preprocessor = split_and_preprocess(df)
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    print(f"Train positive rate: {y_train.mean():.2%}, Test positive rate: {y_test.mean():.2%}")
