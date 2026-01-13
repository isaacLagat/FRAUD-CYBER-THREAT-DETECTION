"""
Fraud Detection Using Decision Trees
A machine learning project demonstrating transaction fraud classification
using interpretable decision tree models with real-world data.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    roc_auc_score, 
    roc_curve,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    precision_recall_curve,
    average_precision_score
)
from sklearn.preprocessing import StandardScaler
import warnings
import os
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

class FraudDetectionModel:
    """
    A class to handle fraud detection using Decision Trees.
    Supports both synthetic and real-world datasets.
    """
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.dataset_type = None
        
    def load_real_dataset(self, filepath='creditcard.csv'):
        """
        Load the Kaggle Credit Card Fraud Detection dataset.
        
        Dataset: https://www.kaggle.com/mlg-ulb/creditcardfraud
        
        Parameters:
        - filepath: Path to the CSV file
        
        Returns:
        - DataFrame with the loaded data
        """
        print(f"Loading real fraud dataset from '{filepath}'...")
        
        if not os.path.exists(filepath):
            print(f"\n⚠️  File '{filepath}' not found!")
            print("\nTo use real data:")
            print("1. Download from: https://www.kaggle.com/mlg-ulb/creditcardfraud")
            print("2. Place 'creditcard.csv' in the project directory")
            print("3. Or use generate_synthetic_data() for demonstration\n")
            return None
        
        try:
            df = pd.read_csv(filepath)
            print(f"✓ Loaded {len(df)} transactions")
            print(f"✓ Features: {len(df.columns)-1}")
            print(f"✓ Fraud cases: {df['Class'].sum()} ({df['Class'].mean()*100:.3f}%)")
            print(f"✓ Legitimate cases: {(df['Class']==0).sum()} ({(1-df['Class'].mean())*100:.3f}%)")
            
            self.dataset_type = 'real'
            return df
            
        except Exception as e:
            print(f"Error loading file: {e}")
            return None
    
    def load_sample_real_data(self, n_samples=10000):
        """
        Create a sample dataset that mimics the real Kaggle dataset structure.
        Useful for testing when the full dataset is not available.
        """
        print(f"Creating sample dataset mimicking real fraud data structure...")
        
        # The real dataset has V1-V28 (PCA features), Time, Amount, and Class
        n_fraud = int(n_samples * 0.00172)  # Real dataset fraud rate
        n_legit = n_samples - n_fraud
        
        data = {}
        
        # Time feature (seconds elapsed)
        data['Time'] = np.random.uniform(0, 172792, n_samples)
        
        # V1-V28 (PCA components - generate with different distributions for fraud/legit)
        for i in range(1, 29):
            legit_vals = np.random.normal(0, 1, n_legit)
            fraud_vals = np.random.normal(np.random.uniform(-2, 2), 
                                         np.random.uniform(1, 3), 
                                         n_fraud)
            data[f'V{i}'] = np.concatenate([legit_vals, fraud_vals])
        
        # Amount feature
        legit_amount = np.random.gamma(2, 50, n_legit)
        fraud_amount = np.random.gamma(3, 100, n_fraud)
        data['Amount'] = np.concatenate([legit_amount, fraud_amount])
        
        # Class (target)
        data['Class'] = np.concatenate([np.zeros(n_legit), np.ones(n_fraud)])
        
        df = pd.DataFrame(data)
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        print(f"✓ Created {len(df)} sample transactions")
        print(f"✓ Fraud cases: {df['Class'].sum()} ({df['Class'].mean()*100:.3f}%)")
        
        self.dataset_type = 'sample_real'
        return df
        
    def generate_synthetic_data(self, n_samples=10000, fraud_ratio=0.1):
        """
        Generate synthetic transaction data for demonstration.
        
        Parameters:
        - n_samples: Number of transactions to generate
        - fraud_ratio: Proportion of fraudulent transactions
        """
        print(f"Generating {n_samples} synthetic transactions...")
        
        n_fraud = int(n_samples * fraud_ratio)
        n_legit = n_samples - n_fraud
        
        # Legitimate transactions
        legit_data = {
            'amount': np.random.gamma(shape=2, scale=50, size=n_legit),
            'time_of_day': np.random.normal(12, 4, n_legit) % 24,
            'distance_from_home': np.random.exponential(scale=20, size=n_legit),
            'distance_from_last': np.random.exponential(scale=10, size=n_legit),
            'ratio_to_median': np.random.normal(1.0, 0.3, n_legit),
            'repeat_retailer': np.random.binomial(1, 0.7, n_legit),
            'used_chip': np.random.binomial(1, 0.9, n_legit),
            'used_pin': np.random.binomial(1, 0.85, n_legit),
            'online_order': np.random.binomial(1, 0.3, n_legit),
            'fraud': np.zeros(n_legit)
        }
        
        # Fraudulent transactions (with different patterns)
        fraud_data = {
            'amount': np.random.gamma(shape=5, scale=100, size=n_fraud),
            'time_of_day': np.random.choice([2, 3, 4, 22, 23], n_fraud),
            'distance_from_home': np.random.exponential(scale=100, size=n_fraud),
            'distance_from_last': np.random.exponential(scale=50, size=n_fraud),
            'ratio_to_median': np.random.normal(3.0, 1.5, n_fraud),
            'repeat_retailer': np.random.binomial(1, 0.2, n_fraud),
            'used_chip': np.random.binomial(1, 0.3, n_fraud),
            'used_pin': np.random.binomial(1, 0.2, n_fraud),
            'online_order': np.random.binomial(1, 0.6, n_fraud),
            'fraud': np.ones(n_fraud)
        }
        
        # Combine and shuffle
        df_legit = pd.DataFrame(legit_data)
        df_fraud = pd.DataFrame(fraud_data)
        df = pd.concat([df_legit, df_fraud], ignore_index=True)
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Ensure values are in reasonable ranges
        df['amount'] = df['amount'].clip(lower=0)
        df['time_of_day'] = df['time_of_day'].clip(0, 23)
        df['distance_from_home'] = df['distance_from_home'].clip(lower=0)
        df['distance_from_last'] = df['distance_from_last'].clip(lower=0)
        df['ratio_to_median'] = df['ratio_to_median'].clip(lower=0)
        
        print(f"✓ Generated {len(df)} transactions ({n_fraud} fraudulent, {n_legit} legitimate)")
        
        self.dataset_type = 'synthetic'
        return df
    
    def preprocess_data(self, df):
        """
        Preprocess the dataset based on its type.
        
        Parameters:
        - df: Input DataFrame
        
        Returns:
        - Preprocessed DataFrame
        """
        df_processed = df.copy()
        
        if self.dataset_type == 'real' or self.dataset_type == 'sample_real':
            # For real Kaggle dataset
            # Normalize Amount and Time
            if 'Amount' in df_processed.columns:
                df_processed['Amount_scaled'] = self.scaler.fit_transform(
                    df_processed['Amount'].values.reshape(-1, 1)
                )
            
            if 'Time' in df_processed.columns:
                df_processed['Time_scaled'] = self.scaler.fit_transform(
                    df_processed['Time'].values.reshape(-1, 1)
                )
            
            # Drop original Time and Amount, keep scaled versions
            df_processed = df_processed.drop(['Time', 'Amount'], axis=1, errors='ignore')
            
            print("✓ Preprocessed real dataset (scaled Amount and Time)")
        
        return df_processed
    
    def prepare_data(self, df, test_size=0.2):
        """
        Split data into training and testing sets.
        """
        df_processed = self.preprocess_data(df)
        
        # Determine target column name
        target_col = 'Class' if 'Class' in df_processed.columns else 'fraud'
        
        X = df_processed.drop(target_col, axis=1)
        y = df_processed[target_col]
        
        self.feature_names = X.columns.tolist()
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        print(f"\n{'='*60}")
        print("DATA SPLIT SUMMARY")
        print(f"{'='*60}")
        print(f"Training set: {len(self.X_train)} samples")
        print(f"Testing set: {len(self.X_test)} samples")
        print(f"Fraud rate in training: {self.y_train.mean()*100:.3f}%")
        print(f"Fraud rate in testing: {self.y_test.mean()*100:.3f}%")
        print(f"Number of features: {len(self.feature_names)}")
        print(f"{'='*60}\n")
        
    def train_model(self, max_depth=10, min_samples_split=50, min_samples_leaf=20):
        """
        Train the Decision Tree model with specified hyperparameters.
        """
        print("Training Decision Tree model...")
        
        self.model = DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=42,
            class_weight='balanced'  # Handle imbalanced data
        )
        
        self.model.fit(self.X_train, self.y_train)
        print("✓ Model training complete!")
        
    def optimize_hyperparameters(self, cv=5):
        """
        Use GridSearchCV to find optimal hyperparameters.
        """
        print("\nOptimizing hyperparameters with cross-validation...")
        
        param_grid = {
            'max_depth': [5, 10, 15, 20, None],
            'min_samples_split': [20, 50, 100, 200],
            'min_samples_leaf': [10, 20, 50, 100],
            'criterion': ['gini', 'entropy']
        }
        
        grid_search = GridSearchCV(
            DecisionTreeClassifier(random_state=42, class_weight='balanced'),
            param_grid,
            cv=cv,
            scoring='f1',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(self.X_train, self.y_train)
        
        print(f"\n✓ Best parameters: {grid_search.best_params_}")
        print(f"✓ Best F1 score (CV): {grid_search.best_score_:.4f}")
        
        self.model = grid_search.best_estimator_
        return grid_search.best_params_
    
    def evaluate_model(self):
        """
        Evaluate the model on test data and print comprehensive metrics.
        """
        y_pred = self.model.predict(self.X_test)
        y_pred_proba = self.model.predict_proba(self.X_test)[:, 1]
        
        print("\n" + "="*70)
        print("MODEL EVALUATION RESULTS")
        print("="*70)
        
        print(f"\nAccuracy: {accuracy_score(self.y_test, y_pred):.4f}")
        print(f"Precision: {precision_score(self.y_test, y_pred):.4f}")
        print(f"Recall (Sensitivity): {recall_score(self.y_test, y_pred):.4f}")
        print(f"F1 Score: {f1_score(self.y_test, y_pred):.4f}")
        print(f"ROC-AUC Score: {roc_auc_score(self.y_test, y_pred_proba):.4f}")
        print(f"Average Precision Score: {average_precision_score(self.y_test, y_pred_proba):.4f}")
        
        # Confusion Matrix breakdown
        cm = confusion_matrix(self.y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        print(f"\nConfusion Matrix Breakdown:")
        print(f"  True Negatives (Correctly identified legitimate): {tn}")
        print(f"  False Positives (Legitimate flagged as fraud): {fp}")
        print(f"  False Negatives (Fraud missed): {fn}")
        print(f"  True Positives (Correctly identified fraud): {tp}")
        
        print("\nClassification Report:")
        print(classification_report(self.y_test, y_pred, 
                                   target_names=['Legitimate', 'Fraud']))
        
        return {
            'accuracy': accuracy_score(self.y_test, y_pred),
            'precision': precision_score(self.y_test, y_pred),
            'recall': recall_score(self.y_test, y_pred),
            'f1': f1_score(self.y_test, y_pred),
            'roc_auc': roc_auc_score(self.y_test, y_pred_proba),
            'avg_precision': average_precision_score(self.y_test, y_pred_proba)
        }
    
    def plot_confusion_matrix(self):
        """
        Plot confusion matrix with percentages.
        """
        y_pred = self.model.predict(self.X_test)
        cm = confusion_matrix(self.y_test, y_pred)
        
        # Calculate percentages
        cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Counts
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Legitimate', 'Fraud'],
                   yticklabels=['Legitimate', 'Fraud'],
                   ax=ax1, cbar_kws={'label': 'Count'})
        ax1.set_title('Confusion Matrix (Counts)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Actual')
        ax1.set_xlabel('Predicted')
        
        # Percentages
        sns.heatmap(cm_percent, annot=True, fmt='.2f', cmap='Oranges',
                   xticklabels=['Legitimate', 'Fraud'],
                   yticklabels=['Legitimate', 'Fraud'],
                   ax=ax2, cbar_kws={'label': 'Percentage (%)'})
        ax2.set_title('Confusion Matrix (Percentages)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Actual')
        ax2.set_xlabel('Predicted')
        
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        print("\n✓ Confusion matrix saved as 'confusion_matrix.png'")
        plt.show()
        
    def plot_roc_curve(self):
        """
        Plot ROC curve.
        """
        y_pred_proba = self.model.predict_proba(self.X_test)[:, 1]
        fpr, tpr, _ = roc_curve(self.y_test, y_pred_proba)
        roc_auc = roc_auc_score(self.y_test, y_pred_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=11)
        plt.ylabel('True Positive Rate', fontsize=11)
        plt.title('ROC Curve - Fraud Detection', fontsize=14, fontweight='bold')
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig('roc_curve.png', dpi=300, bbox_inches='tight')
        print("✓ ROC curve saved as 'roc_curve.png'")
        plt.show()
    
    def plot_precision_recall_curve(self):
        """
        Plot Precision-Recall curve (especially useful for imbalanced datasets).
        """
        y_pred_proba = self.model.predict_proba(self.X_test)[:, 1]
        precision, recall, _ = precision_recall_curve(self.y_test, y_pred_proba)
        avg_precision = average_precision_score(self.y_test, y_pred_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='blue', lw=2,
                label=f'PR curve (AP = {avg_precision:.4f})')
        plt.xlabel('Recall', fontsize=11)
        plt.ylabel('Precision', fontsize=11)
        plt.title('Precision-Recall Curve - Fraud Detection', fontsize=14, fontweight='bold')
        plt.legend(loc="upper right")
        plt.grid(alpha=0.3)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.tight_layout()
        plt.savefig('precision_recall_curve.png', dpi=300, bbox_inches='tight')
        print("✓ Precision-Recall curve saved as 'precision_recall_curve.png'")
        plt.show()
        
    def plot_feature_importance(self, top_n=15):
        """
        Plot feature importance from the decision tree.
        """
        importances = self.model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        # Limit to top_n features for readability
        top_indices = indices[:top_n]
        top_importances = importances[top_indices]
        top_features = [self.feature_names[i] for i in top_indices]
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(range(len(top_importances)), top_importances, color='steelblue')
        plt.xticks(range(len(top_importances)), top_features, rotation=45, ha='right')
        plt.xlabel('Features', fontsize=11)
        plt.ylabel('Importance', fontsize=11)
        plt.title(f'Top {top_n} Feature Importance in Fraud Detection', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
        print("✓ Feature importance plot saved as 'feature_importance.png'")
        plt.show()
        
        print(f"\nTop {min(top_n, len(indices))} Most Important Features:")
        for i in range(min(top_n, len(indices))):
            idx = indices[i]
            print(f"{i+1:2d}. {self.feature_names[idx]:20s}: {importances[idx]:.4f}")
    
    def visualize_tree(self, max_depth_display=3):
        """
        Visualize the decision tree (limited depth for readability).
        """
        plt.figure(figsize=(20, 10))
        plot_tree(self.model, 
                 feature_names=self.feature_names,
                 class_names=['Legitimate', 'Fraud'],
                 filled=True,
                 rounded=True,
                 max_depth=max_depth_display,
                 fontsize=9)
        plt.title(f'Decision Tree Visualization (max depth={max_depth_display})', 
                 fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig('decision_tree.png', dpi=300, bbox_inches='tight')
        print(f"✓ Decision tree visualization saved as 'decision_tree.png'")
        plt.show()
    
    def predict_new_transaction(self, transaction_data):
        """
        Predict if a new transaction is fraudulent.
        
        Parameters:
        - transaction_data: dict with feature values
        """
        df_new = pd.DataFrame([transaction_data])
        
        # Ensure all features are present
        missing_features = set(self.feature_names) - set(df_new.columns)
        if missing_features:
            print(f"⚠️  Missing features: {missing_features}")
            return None, None
        
        # Reorder columns to match training data
        df_new = df_new[self.feature_names]
        
        prediction = self.model.predict(df_new)[0]
        probability = self.model.predict_proba(df_new)[0]
        
        print("\n" + "="*70)
        print("NEW TRANSACTION PREDICTION")
        print("="*70)
        print(f"Prediction: {'🚨 FRAUD' if prediction == 1 else '✓ LEGITIMATE'}")
        print(f"Fraud Probability: {probability[1]:.2%}")
        print(f"Legitimate Probability: {probability[0]:.2%}")
        
        if prediction == 1:
            print("\n⚠️  ALERT: This transaction should be flagged for review!")
        
        return prediction, probability


def main():
    """
    Main execution function for the fraud detection pipeline.
    """
    print("="*70)
    print("FRAUD DETECTION USING DECISION TREES")
    print("Real-World Dataset Integration")
    print("="*70)
    
    # Initialize model
    fraud_detector = FraudDetectionModel()
    
    # Try to load real dataset, fallback to sample/synthetic
    print("\n[1] Attempting to load real dataset...")
    df = fraud_detector.load_real_dataset('creditcard.csv')
    
    if df is None:
        print("\n[2] Real dataset not found. Using sample dataset...")
        df = fraud_detector.load_sample_real_data(n_samples=50000)
        
        # Uncomment below to use synthetic data instead
        # print("\n[2] Using synthetic dataset for demonstration...")
        # df = fraud_detector.generate_synthetic_data(n_samples=10000, fraud_ratio=0.1)
    
    # Data exploration
    print(f"\n{'='*70}")
    print("DATASET OVERVIEW")
    print(f"{'='*70}")
    print(df.head())
    print(f"\nDataset shape: {df.shape}")
    print(f"\nData types:\n{df.dtypes}")
    print(f"\nMissing values:\n{df.isnull().sum()}")
    
    # Prepare data
    fraud_detector.prepare_data(df, test_size=0.2)
    
    # Train initial model
    print("\n[3] Training initial model...")
    fraud_detector.train_model(max_depth=15, min_samples_split=100, min_samples_leaf=50)
    
    # Evaluate model
    print("\n[4] Evaluating model...")
    metrics = fraud_detector.evaluate_model()
    
    # Visualizations
    print("\n[5] Generating visualizations...")
    fraud_detector.plot_confusion_matrix()
    fraud_detector.plot_roc_curve()
    fraud_detector.plot_precision_recall_curve()
    fraud_detector.plot_feature_importance(top_n=15)
    fraud_detector.visualize_tree(max_depth_display=3)
    
    # Optional: Optimize hyperparameters (comment out for faster execution)
    print("\n[6] Hyperparameter optimization (optional - uncomment to run)...")
    # best_params = fraud_detector.optimize_hyperparameters(cv=5)
    # metrics = fraud_detector.evaluate_model()
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE!")
    print("="*70)
    print(f"\nFinal Model Performance:")
    for metric, value in metrics.items():
        print(f"  {metric.upper()}: {value:.4f}")
    
    print(f"\nGenerated files:")
    print("  - confusion_matrix.png")
    print("  - roc_curve.png")
    print("  - precision_recall_curve.png")
    print("  - feature_importance.png")
    print("  - decision_tree.png")


if __name__ == "__main__":
    main()