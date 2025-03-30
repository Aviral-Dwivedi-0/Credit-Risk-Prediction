import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc, precision_recall_curve,
    average_precision_score, classification_report
)
from sklearn.calibration import calibration_curve
import joblib
import logging
import os
import glob

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelEvaluator:
    def __init__(self):
        self.metrics = {}
        self.figures_dir = "figures"
        os.makedirs(self.figures_dir, exist_ok=True)
        
    def load_latest_model(self):
        """Load the latest trained model."""
        try:
            model_files = glob.glob("models/*.joblib")
            if not model_files:
                raise FileNotFoundError("No model files found")
            
            latest_model_file = max(model_files, key=os.path.getctime)
            model = joblib.load(latest_model_file)
            logger.info(f"Successfully loaded model from {latest_model_file}")
            return model, latest_model_file
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

    def calculate_metrics(self, y_true, y_pred, y_pred_proba):
        """Calculate comprehensive evaluation metrics."""
        # Basic metrics
        self.metrics['classification_report'] = classification_report(y_true, y_pred)
        
        # ROC curve and AUC
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        self.metrics['auc_roc'] = auc(fpr, tpr)
        self.metrics['roc_curve'] = {'fpr': fpr, 'tpr': tpr}
        
        # Precision-Recall curve and Average Precision
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
        self.metrics['average_precision'] = average_precision_score(y_true, y_pred_proba)
        self.metrics['pr_curve'] = {'precision': precision, 'recall': recall}
        
        # Confusion Matrix
        self.metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred)
        
        logger.info("Calculated all evaluation metrics")
        return self.metrics

    def plot_confusion_matrix(self, cm):
        """Plot confusion matrix."""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(os.path.join(self.figures_dir, 'confusion_matrix.png'))
        plt.close()
        logger.info("Saved confusion matrix plot")

    def plot_roc_curve(self, fpr, tpr, auc_roc):
        """Plot ROC curve."""
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc_roc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(self.figures_dir, 'roc_curve.png'))
        plt.close()
        logger.info("Saved ROC curve plot")

    def plot_precision_recall_curve(self, precision, recall, average_precision):
        """Plot precision-recall curve."""
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='blue', lw=2, 
                label=f'Precision-Recall curve (AP = {average_precision:.2f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="lower left")
        plt.savefig(os.path.join(self.figures_dir, 'precision_recall_curve.png'))
        plt.close()
        logger.info("Saved precision-recall curve plot")

    def plot_calibration_curve(self, y_true, y_pred_proba):
        """Plot calibration curve."""
        plt.figure(figsize=(8, 6))
        prob_true, prob_pred = calibration_curve(y_true, y_pred_proba, n_bins=5)
        plt.plot(prob_pred, prob_true, marker='o', linewidth=2, label='Calibration curve')
        plt.plot([0, 1], [0, 1], linestyle='--', label='Perfect calibration')
        plt.xlabel('Mean predicted probability')
        plt.ylabel('True probability in each bin')
        plt.title('Calibration Curve')
        plt.legend()
        plt.savefig(os.path.join(self.figures_dir, 'calibration_curve.png'))
        plt.close()
        logger.info("Saved calibration curve plot")

    def save_metrics_report(self, metrics, model_file):
        """Save metrics to a text file."""
        report_file = os.path.join(self.figures_dir, 'evaluation_report.txt')
        with open(report_file, 'w') as f:
            f.write("Credit Risk Model Evaluation Report\n")
            f.write("================================\n\n")
            f.write(f"Model: {os.path.basename(model_file)}\n\n")
            f.write("Classification Report:\n")
            f.write(metrics['classification_report'])
            f.write(f"\nROC AUC Score: {metrics['auc_roc']:.4f}\n")
            f.write(f"Average Precision Score: {metrics['average_precision']:.4f}\n")
        
        logger.info(f"Saved evaluation report to {report_file}")

    def evaluate_model(self, X_test, y_test):
        """Main evaluation pipeline."""
        try:
            # Load model
            model, model_file = self.load_latest_model()
            
            # Make predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            metrics = self.calculate_metrics(y_test, y_pred, y_pred_proba)
            
            # Plot confusion matrix
            self.plot_confusion_matrix(metrics['confusion_matrix'])
            
            # Plot ROC curve
            self.plot_roc_curve(
                metrics['roc_curve']['fpr'],
                metrics['roc_curve']['tpr'],
                metrics['auc_roc']
            )
            
            # Plot precision-recall curve
            self.plot_precision_recall_curve(
                metrics['pr_curve']['precision'],
                metrics['pr_curve']['recall'],
                metrics['average_precision']
            )
            
            # Plot calibration curve
            self.plot_calibration_curve(y_test, y_pred_proba)
            
            # Save metrics report
            self.save_metrics_report(metrics, model_file)
            
            logger.info("Model evaluation completed successfully")
            return metrics
            
        except Exception as e:
            logger.error(f"Error in model evaluation: {str(e)}")
            raise

if __name__ == "__main__":
    # Example usage
    evaluator = ModelEvaluator()
    try:
        # Load test data
        X_test = pd.read_csv("data/processed/X_test_engineered.csv")
        y_test = pd.read_csv("data/processed/y_test.csv")['target']
        
        # Evaluate model
        metrics = evaluator.evaluate_model(X_test, y_test)
        
        logger.info("Model evaluation completed successfully")
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}") 