import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA
import logging
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeatureEngineer:
    def __init__(self):
        self.feature_selector = None
        self.pca = None
        self.selected_features = None
        
    def create_interaction_features(self, df):
        """Create interaction features between numerical columns."""
        numerical_cols = ['age', 'income', 'employment_length', 'credit_score', 
                         'debt_ratio', 'loan_amount', 'loan_term']
        interaction_features = pd.DataFrame(index=df.index)
        
        for i in range(len(numerical_cols)):
            for j in range(i+1, len(numerical_cols)):
                col1, col2 = numerical_cols[i], numerical_cols[j]
                if col1 in df.columns and col2 in df.columns:
                    interaction_features[f"{col1}_{col2}_interaction"] = df[col1] * df[col2]
                    # For ratio features, we'll add a small constant to avoid division by zero
                    interaction_features[f"{col1}_{col2}_ratio"] = df[col1] / (df[col2] + 1e-8)
        
        return pd.concat([df, interaction_features], axis=1)

    def create_polynomial_features(self, df, degree=2):
        """Create polynomial features for numerical columns."""
        numerical_cols = ['age', 'income', 'employment_length', 'credit_score', 
                         'debt_ratio', 'loan_amount', 'loan_term']
        polynomial_features = pd.DataFrame(index=df.index)
        
        for col in numerical_cols:
            if col in df.columns:
                for d in range(2, degree + 1):
                    polynomial_features[f"{col}_power_{d}"] = df[col] ** d
        
        return pd.concat([df, polynomial_features], axis=1)

    def select_features(self, X, y, k=None):
        """Select top k features using SelectKBest with f_classif."""
        if k is None:
            k = min(10, X.shape[1])
        
        try:
            self.feature_selector = SelectKBest(score_func=f_classif, k=k)
            X_selected = self.feature_selector.fit_transform(X, y)
            self.selected_features = X.columns[self.feature_selector.get_support()].tolist()
            
            logger.info(f"Selected {len(self.selected_features)} features")
            return pd.DataFrame(X_selected, columns=self.selected_features, index=X.index)
        except Exception as e:
            logger.error(f"Error in feature selection: {str(e)}")
            # If feature selection fails, return original features
            logger.info("Returning original features without selection")
            self.selected_features = X.columns.tolist()
            return X

    def apply_pca(self, X, n_components=0.95):
        """Apply PCA for dimensionality reduction."""
        try:
            self.pca = PCA(n_components=n_components)
            X_pca = self.pca.fit_transform(X)
            
            logger.info(f"PCA reduced dimensions from {X.shape[1]} to {X_pca.shape[1]}")
            return pd.DataFrame(X_pca, 
                              columns=[f"PC{i+1}" for i in range(X_pca.shape[1])],
                              index=X.index)
        except Exception as e:
            logger.error(f"Error in PCA: {str(e)}")
            # If PCA fails, return original features
            logger.info("Returning original features without PCA")
            return X

    def save_engineered_data(self, X_train_eng, X_test_eng, y_train, y_test):
        """Save engineered data to files."""
        processed_dir = "data/processed"
        os.makedirs(processed_dir, exist_ok=True)
        
        X_train_eng.to_csv(f"{processed_dir}/X_train_engineered.csv", index=False)
        X_test_eng.to_csv(f"{processed_dir}/X_test_engineered.csv", index=False)
        
        logger.info("Saved engineered data to files")

    def engineer_features(self, X_train, X_test, y_train, y_test):
        """Main feature engineering pipeline."""
        try:
            logger.info(f"Initial shapes - X_train: {X_train.shape}, X_test: {X_test.shape}")
            
            # Create interaction features
            X_train = self.create_interaction_features(X_train)
            X_test = self.create_interaction_features(X_test)
            logger.info(f"After interactions - X_train: {X_train.shape}, X_test: {X_test.shape}")
            
            # Create polynomial features
            X_train = self.create_polynomial_features(X_train)
            X_test = self.create_polynomial_features(X_test)
            logger.info(f"After polynomials - X_train: {X_train.shape}, X_test: {X_test.shape}")
            
            # Select features
            X_train_selected = self.select_features(X_train, y_train)
            X_test_selected = X_test[self.selected_features]
            logger.info(f"After selection - X_train: {X_train_selected.shape}, X_test: {X_test_selected.shape}")
            
            # Apply PCA
            X_train_pca = self.apply_pca(X_train_selected)
            if self.pca is not None:
                X_test_pca = pd.DataFrame(
                    self.pca.transform(X_test_selected),
                    columns=[f"PC{i+1}" for i in range(X_train_pca.shape[1])],
                    index=X_test_selected.index
                )
            else:
                X_test_pca = X_test_selected
            
            logger.info(f"Final shapes - X_train: {X_train_pca.shape}, X_test: {X_test_pca.shape}")
            
            # Save engineered data
            self.save_engineered_data(X_train_pca, X_test_pca, y_train, y_test)
            
            return X_train_pca, X_test_pca, y_train, y_test
            
        except Exception as e:
            logger.error(f"Error in feature engineering: {str(e)}")
            raise

if __name__ == "__main__":
    # Example usage
    feature_engineer = FeatureEngineer()
    try:
        # Load processed data
        X_train = pd.read_csv("data/processed/X_train.csv")
        X_test = pd.read_csv("data/processed/X_test.csv")
        y_train = pd.read_csv("data/processed/y_train.csv")['target']
        y_test = pd.read_csv("data/processed/y_test.csv")['target']
        
        # Engineer features
        X_train_eng, X_test_eng, y_train, y_test = feature_engineer.engineer_features(
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test
        )
        
        logger.info("Feature engineering completed successfully")
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}") 