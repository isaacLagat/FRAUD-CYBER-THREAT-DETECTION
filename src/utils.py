"""
utils.py

Shared helper functions for model persistence.
"""

import os
import pickle


def save_pickle(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    print(f"Saved to {path}")


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)
