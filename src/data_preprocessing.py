import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import logging
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CreditDataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.imputer = SimpleImputer(strategy='mean')
        self.categorical_columns = None
        self.numerical_columns = None
        
    def load_data(self, file_path):
        """Load the credit risk dataset."""
        try:
            df = pd.read_csv(file_path)
            logger.info(f"Successfully loaded data with shape: {df.shape}")
            return df
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise

    def identify_columns(self, df):
        """Identify categorical and numerical columns."""
        self.numerical_columns = ['age', 'income', 'employment_length', 'credit_score', 
                                'debt_ratio', 'loan_amount', 'loan_term']
        self.categorical_columns = ['payment_history']
        logger.info(f"Categorical columns: {self.categorical_columns}")
        logger.info(f"Numerical columns: {self.numerical_columns}")

    def handle_missing_values(self, df):
        """Handle missing values in the dataset."""
        # For numerical columns
        df[self.numerical_columns] = self.imputer.fit_transform(df[self.numerical_columns])
        
        # For categorical columns
        df[self.categorical_columns] = df[self.categorical_columns].fillna('missing')
        
        logger.info("Handled missing values")
        return df

    def encode_categorical_variables(self, df):
        """Encode categorical variables using Label Encoding."""
        for col in self.categorical_columns:
            df[col] = self.label_encoder.fit_transform(df[col])
        logger.info("Encoded categorical variables")
        return df

    def scale_numerical_features(self, df):
        """Scale numerical features using StandardScaler."""
        df[self.numerical_columns] = self.scaler.fit_transform(df[self.numerical_columns])
        logger.info("Scaled numerical features")
        return df

    def split_data(self, df, target_column, test_size=0.2, random_state=42):
        """Split data into training and testing sets."""
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        logger.info(f"Split data into training set: {X_train.shape} and test set: {X_test.shape}")
        return X_train, X_test, y_train, y_test

    def save_processed_data(self, X_train, X_test, y_train, y_test):
        """Save processed data to files."""
        processed_dir = "data/processed"
        os.makedirs(processed_dir, exist_ok=True)
        
        # Save features
        X_train.to_csv(f"{processed_dir}/X_train.csv", index=False)
        X_test.to_csv(f"{processed_dir}/X_test.csv", index=False)
        
        # Save target variables as Series to preserve the values
        pd.Series(y_train, name='target').to_csv(f"{processed_dir}/y_train.csv", index=False)
        pd.Series(y_test, name='target').to_csv(f"{processed_dir}/y_test.csv", index=False)
        
        logger.info("Saved processed data to files")

    def preprocess_data(self, file_path, target_column):
        """Main preprocessing pipeline."""
        try:
            # Load data
            df = self.load_data(file_path)
            
            # Identify column types
            self.identify_columns(df)
            
            # Handle missing values
            df = self.handle_missing_values(df)
            
            # Encode categorical variables
            df = self.encode_categorical_variables(df)
            
            # Scale numerical features
            df = self.scale_numerical_features(df)
            
            # Split data
            X_train, X_test, y_train, y_test = self.split_data(df, target_column)
            
            # Save processed data
            self.save_processed_data(X_train, X_test, y_train, y_test)
            
            return X_train, X_test, y_train, y_test
            
        except Exception as e:
            logger.error(f"Error in preprocessing: {str(e)}")
            raise

if __name__ == "__main__":
    # Example usage
    preprocessor = CreditDataPreprocessor()
    try:
        X_train, X_test, y_train, y_test = preprocessor.preprocess_data(
            file_path="data/credit_data.csv",
            target_column="default"
        )
        logger.info("Data preprocessing completed successfully")
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}") 