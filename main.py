"""
main.py

CLI entry point for the fraud/cyber-threat classification pipeline.

Usage:
    python main.py --train
    python main.py --evaluate
    python main.py --train --evaluate --data_path data/raw/my_dataset.csv
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from train import train_and_select_best  # noqa: E402
from evaluate import evaluate  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Fraud & Cyber Threat Detection Pipeline")
    parser.add_argument("--train", action="store_true", help="Train models and save the best one")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate the saved best model")
    parser.add_argument("--data_path", type=str, default=None, help="Path to a CSV dataset (optional)")
    args = parser.parse_args()

    if not args.train and not args.evaluate:
        parser.print_help()
        return

    if args.train:
        train_and_select_best(data_path=args.data_path)

    if args.evaluate:
        evaluate(data_path=args.data_path)


if __name__ == "__main__":
    main()
