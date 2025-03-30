import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import logging
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataVisualizer:
    def __init__(self):
        self.figures_dir = "figures"
        os.makedirs(self.figures_dir, exist_ok=True)
        
    def plot_feature_distributions(self, df, target_col):
        """Plot distributions of features by target class."""
        numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
        n_cols = 3
        n_rows = (len(numerical_cols) + n_cols - 1) // n_cols
        
        plt.figure(figsize=(15, 5*n_rows))
        for i, col in enumerate(numerical_cols, 1):
            plt.subplot(n_rows, n_cols, i)
            sns.boxplot(x=target_col, y=col, data=df)
            plt.title(f'{col} Distribution by Target')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_dir, 'feature_distributions.png'))
        plt.close()
        logger.info("Saved feature distributions plot")

    def plot_correlation_matrix(self, df):
        """Plot correlation matrix of numerical features."""
        numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
        corr_matrix = df[numerical_cols].corr()
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title('Correlation Matrix')
        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_dir, 'correlation_matrix.png'))
        plt.close()
        logger.info("Saved correlation matrix plot")

    def plot_target_distribution(self, df, target_col):
        """Plot target variable distribution."""
        plt.figure(figsize=(8, 6))
        sns.countplot(x=target_col, data=df)
        plt.title('Target Variable Distribution')
        plt.savefig(os.path.join(self.figures_dir, 'target_distribution.png'))
        plt.close()
        logger.info("Saved target distribution plot")

    def plot_feature_importance_boxplot(self, df, target_col):
        """Plot feature importance using statistical tests."""
        numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
        importance_scores = []
        
        for col in numerical_cols:
            if col != target_col:
                # Calculate t-statistic
                t_stat, p_value = stats.ttest_ind(
                    df[df[target_col] == 0][col],
                    df[df[target_col] == 1][col]
                )
                importance_scores.append({
                    'feature': col,
                    'importance': abs(t_stat)
                })
        
        importance_df = pd.DataFrame(importance_scores)
        importance_df = importance_df.sort_values('importance', ascending=False)
        
        plt.figure(figsize=(12, 6))
        sns.barplot(x='importance', y='feature', data=importance_df)
        plt.title('Feature Importance (Based on t-statistics)')
        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_dir, 'feature_importance_boxplot.png'))
        plt.close()
        logger.info("Saved feature importance boxplot")

    def plot_risk_score_distribution(self, df, target_col, risk_score_col):
        """Plot risk score distribution by target class."""
        plt.figure(figsize=(10, 6))
        sns.histplot(data=df, x=risk_score_col, hue=target_col, bins=50)
        plt.title('Risk Score Distribution by Target Class')
        plt.savefig(os.path.join(self.figures_dir, 'risk_score_distribution.png'))
        plt.close()
        logger.info("Saved risk score distribution plot")

    def plot_feature_vs_risk(self, df, target_col, risk_score_col):
        """Plot feature values against risk scores."""
        numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
        numerical_cols = [col for col in numerical_cols if col not in [target_col, risk_score_col]]
        
        n_cols = 3
        n_rows = (len(numerical_cols) + n_cols - 1) // n_cols
        
        plt.figure(figsize=(15, 5*n_rows))
        for i, col in enumerate(numerical_cols, 1):
            plt.subplot(n_rows, n_cols, i)
            sns.scatterplot(data=df, x=col, y=risk_score_col, hue=target_col, alpha=0.5)
            plt.title(f'{col} vs Risk Score')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_dir, 'feature_vs_risk.png'))
        plt.close()
        logger.info("Saved feature vs risk plots")

    def create_visualizations(self, df, target_col, risk_score_col=None):
        """Main visualization pipeline."""
        try:
            # Plot target distribution
            self.plot_target_distribution(df, target_col)
            
            # Plot feature distributions
            self.plot_feature_distributions(df, target_col)
            
            # Plot correlation matrix
            self.plot_correlation_matrix(df)
            
            # Plot feature importance boxplot
            self.plot_feature_importance_boxplot(df, target_col)
            
            # Plot risk score distribution if available
            if risk_score_col:
                self.plot_risk_score_distribution(df, target_col, risk_score_col)
                self.plot_feature_vs_risk(df, target_col, risk_score_col)
            
            logger.info("All visualizations created successfully")
            
        except Exception as e:
            logger.error(f"Error in visualization: {str(e)}")
            raise

if __name__ == "__main__":
    # Example usage
    visualizer = DataVisualizer()
    try:
        # Load data
        df = pd.read_csv("data/credit_data.csv")
        
        # Create visualizations
        visualizer.create_visualizations(
            df=df,
            target_col='default',
            risk_score_col='risk_score'  # if available
        )
        
        logger.info("Visualization process completed successfully")
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}") 