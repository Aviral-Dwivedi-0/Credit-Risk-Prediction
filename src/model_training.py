import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import optuna
from datetime import datetime
import joblib
import logging
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class CreditRiskModel:
    def __init__(self):
        self.numeric_features = [
            'person_age', 'person_income', 'person_emp_length',
            'loan_amnt', 'loan_int_rate', 'loan_percent_income',
            'cb_person_cred_hist_length'
        ]
        
        self.categorical_features = [
            'person_home_ownership', 'loan_intent', 'loan_grade',
            'cb_person_default_on_file'
        ]
        
        self.target = 'loan_status'
        self.model = None
        self.preprocessor = None
        
    def load_and_preprocess_data(self, data_path):
        """Load and preprocess the credit risk dataset"""
        logging.info("Loading data...")
        df = pd.read_csv(data_path)
        logging.info(f"Data loaded with shape: {df.shape}")
        
        # Basic cleaning
        df = df.dropna()
        df = df.drop_duplicates()
        
        return df
    
    def create_preprocessor(self):
        """Create a preprocessing pipeline for numeric and categorical features"""
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
        ])

        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, self.numeric_features),
                ('cat', categorical_transformer, self.categorical_features)
            ])
        
        return self.preprocessor
    
    def objective(self, trial):
        """Optuna objective function for hyperparameter optimization"""
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 20),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'class_weight': trial.suggest_categorical('class_weight', ['balanced', 'balanced_subsample']),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2'])
        }
        
        clf = RandomForestClassifier(**params, random_state=42)
        pipeline = Pipeline([
            ('preprocessor', self.preprocessor),
            ('classifier', clf)
        ])
        
        # Use simple cross-validation without parallel processing
        scores = cross_val_score(
            pipeline, self.X_train, self.y_train, 
            scoring='roc_auc', cv=5,
            n_jobs=1  # Disable parallel processing
        )
        
        return scores.mean()
    
    def train(self, data_path, n_trials=50):
        """Train the model with hyperparameter optimization"""
        # Load and split data
        df = self.load_and_preprocess_data(data_path)
        
        X = df.drop(columns=[self.target])
        y = df[self.target]
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        logging.info(f"Training set shape: {self.X_train.shape}")
        logging.info(f"Test set shape: {self.X_test.shape}")
        
        # Create preprocessor
        self.create_preprocessor()
        
        # Optimize hyperparameters
        logging.info("Starting hyperparameter optimization...")
        study = optuna.create_study(direction='maximize')
        study.optimize(self.objective, n_trials=n_trials)
        
        best_params = study.best_params
        logging.info(f"Best parameters: {best_params}")
        
        # Train final model with best parameters
        final_clf = RandomForestClassifier(
            **best_params,
            random_state=42,
            n_jobs=-1  # Use parallel processing for final training
        )
        
        self.model = Pipeline([
            ('preprocessor', self.preprocessor),
            ('classifier', final_clf)
        ])
        
        logging.info("Training final model...")
        self.model.fit(self.X_train, self.y_train)
        
        # Evaluate model
        self.evaluate_model()
        
        # Save model
        self.save_model()
    
    def evaluate_model(self):
        """Evaluate the model's performance"""
        logging.info("Evaluating model performance...")
        
        # Make predictions
        y_pred = self.model.predict(self.X_test)
        y_pred_proba = self.model.predict_proba(self.X_test)[:, 1]
        
        # Calculate metrics
        roc_auc = roc_auc_score(self.y_test, y_pred_proba)
        
        # Generate classification report
        report = classification_report(self.y_test, y_pred)
        
        # Calculate confusion matrix
        cm = confusion_matrix(self.y_test, y_pred)
        
        # Log results
        logging.info(f"\nROC AUC Score: {roc_auc:.4f}")
        logging.info(f"\nClassification Report:\n{report}")
        logging.info(f"\nConfusion Matrix:\n{cm}")
        
        # Feature importance
        feature_names = (
            self.numeric_features + 
            [f"{feat}_{val}" for feat, vals in 
             zip(self.categorical_features, 
                 self.model.named_steps['preprocessor']
                 .named_transformers_['cat']
                 .named_steps['onehot'].categories_) 
             for val in vals[1:]]
        )
        
        importances = self.model.named_steps['classifier'].feature_importances_
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        logging.info("\nTop 10 Most Important Features:")
        logging.info(feature_importance.head(10))
        
        # Save evaluation results
        Path('evaluation').mkdir(exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        with open(f'evaluation/model_evaluation_{timestamp}.txt', 'w') as f:
            f.write(f"Model Evaluation Results\n")
            f.write(f"========================\n\n")
            f.write(f"ROC AUC Score: {roc_auc:.4f}\n\n")
            f.write(f"Classification Report:\n{report}\n")
            f.write(f"\nConfusion Matrix:\n{cm}\n")
            f.write(f"\nTop 10 Most Important Features:\n{feature_importance.head(10)}")
    
    def save_model(self):
        """Save the trained model"""
        Path('models').mkdir(exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_path = f'models/credit_risk_model_{timestamp}.joblib'
        
        model_info = {
            'model': self.model,
            'numeric_features': self.numeric_features,
            'categorical_features': self.categorical_features,
            'feature_names': (
                self.numeric_features + 
                [f"{feat}_{val}" for feat, vals in 
                 zip(self.categorical_features, 
                     self.model.named_steps['preprocessor']
                     .named_transformers_['cat']
                     .named_steps['onehot'].categories_) 
                 for val in vals[1:]]
            )
        }
        
        joblib.dump(model_info, model_path)
        logging.info(f"Model saved to {model_path}")

if __name__ == "__main__":
    # Train model
    model = CreditRiskModel()
    model.train("data/credit_risk_dataset.csv", n_trials=20)  # Reduced trials for faster training 