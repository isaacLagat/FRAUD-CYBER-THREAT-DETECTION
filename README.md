# Fraud & Cyber Threat Detection — ML Classification Pipeline

An end-to-end machine learning classification pipeline for detecting
fraudulent transactions or cyber threats from structured/tabular data. Built
with a modular structure so preprocessing, training, and evaluation are
cleanly separated and independently testable.

## Features

- Modular Python code structure (`data_loader.py`, `preprocess.py`,
  `train.py`, `evaluate.py`, `utils.py`)
- Automatic preprocessing: numerical scaling + categorical one-hot encoding,
  with a proper train/test split (no data leakage — preprocessing is fit
  on the training set only)
- Trains and compares three models: Logistic Regression, Random Forest, and
  XGBoost — automatically selects the best by F1-score (a better metric
  than raw accuracy for imbalanced fraud data)
- Full evaluation: accuracy, precision, recall, F1, confusion matrix
- Model persistence via pickle

## Project Structure

```
FRAUD-CYBER-THREAT-DETECTION/
├── data/
│   ├── raw/                # source data (synthetic.csv auto-generated here)
│   └── processed/
├── src/
│   ├── data_loader.py      # loads real data or generates a synthetic dataset
│   ├── preprocess.py       # scaling, encoding, train/test split
│   ├── train.py            # trains & compares 3 models, saves the best
│   ├── evaluate.py         # computes real metrics on held-out test data
│   └── utils.py            # pickle save/load helpers
├── models/                 # saved model + preprocessor (generated on train)
├── main.py                 # CLI entry point
├── requirements.txt
└── README.md
```

## Installation

```bash
git clone https://github.com/isaacLagat/FRAUD-CYBER-THREAT-DETECTION.git
cd FRAUD-CYBER-THREAT-DETECTION
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

Train the models (automatically selects and saves the best one):
```bash
python main.py --train
```

Evaluate the saved model on the held-out test set:
```bash
python main.py --evaluate
```

Both at once:
```bash
python main.py --train --evaluate
```

By default, if no real dataset is supplied, the pipeline generates a
synthetic-but-realistic transaction dataset (8% fraud rate, overlapping
feature distributions so the classes aren't trivially separable) so the
whole pipeline can be run and tested immediately. To use a real dataset:

```bash
python main.py --train --evaluate --data_path data/raw/your_dataset.csv
```

Your CSV should have columns: `amount`, `duration_sec`, `transaction_type`,
`source_ip_reputation`, and `label` (0 = normal, 1 = fraud/threat) — or edit
`FEATURE_COLUMNS`/`LABEL_COLUMN` in `src/data_loader.py` to match your schema.

## Example Output

Real output from a run against the built-in synthetic dataset (results vary
slightly run to run):

```
logistic_regression  F1-score: 0.5201
random_forest        F1-score: 0.5414
xgboost              F1-score: 0.5890

Best model: xgboost (F1-score: 0.5890)

Model: xgboost
Accuracy:  0.9330
Precision: 0.5783
Recall:    0.6000
F1 Score:  0.5890
Confusion Matrix:
[[885  35]
 [ 32  48]]
```

Note the accuracy/F1 gap — with only 8% of the data being fraud, a model
that always predicted "normal" would already hit 92% accuracy while being
useless. F1 (which balances precision and recall) is the more honest metric
here, which is why the pipeline selects the best model by F1, not accuracy.

## Status / Roadmap

**Done:**
- [x] Preprocessing pipeline
- [x] Baseline model training + comparison
- [x] Real evaluation metrics + confusion matrix

**Not yet implemented** (planned next):
- [ ] SMOTE / other imbalance-handling techniques
- [ ] Hyperparameter tuning
- [ ] API endpoint (FastAPI) for serving predictions
- [ ] Dockerfile + cloud deployment

This is a training/evaluation pipeline, not a deployed service — there is
currently no live API or inference server.

## License

MIT License
