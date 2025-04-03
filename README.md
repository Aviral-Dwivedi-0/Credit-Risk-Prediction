# Credit Risk Prediction System

An end-to-end machine learning application for predicting credit risk using Random Forest Classifier.

## Features

- **Data Analysis Dashboard**: Visualize and analyze credit risk data
- **Risk Prediction**: Real-time credit risk assessment
- **Feature Importance**: Understand factors affecting credit risk
- **Interactive UI**: User-friendly Streamlit interface

## Tech Stack

- Python 3.x
- Scikit-learn
- Pandas
- Streamlit
- Plotly
- Optuna (for hyperparameter optimization)

## Project Structure

```
Credit-Risk-Prediction/
├── data/
│   └── credit_risk_dataset.csv
├── models/
│   └── credit_risk_model_*.joblib
├── src/
│   ├── app.py
│   └── model_training.py
├── evaluation/
│   └── model_evaluation_*.txt
├── requirements.txt
└── README.md
```

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/Aviral-Dwivedi-0/Credit-Risk-Prediction
   cd Credit-Risk-Prediction
   ```

2. Create and activate virtual environment:

   ```bash
   python -m venv venv
   venv\Scripts\activate  # Windows
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Train the model:

   ```bash
   python src/model_training.py --quick  # Quick training (20 trials)
   python src/model_training.py --medium # Medium training (50 trials)
   python src/model_training.py --full   # Full training (100 trials)
   ```

2. Run the Streamlit app:
   ```bash
   python -m streamlit run src/app.py
   ```

## Model Details

- Algorithm: Random Forest Classifier
- Features: Both numerical and categorical inputs
- Preprocessing: Standard scaling for numerical features, one-hot encoding for categorical features
- Optimization: Hyperparameter tuning using Optuna
- Evaluation Metrics: ROC-AUC, Precision, Recall, F1-Score

## Contributing

Feel free to open issues and pull requests!
