import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import joblib
import sys
import os
import kagglehub

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set page config
st.set_page_config(
    page_title="Credit Risk Prediction",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        margin-top: 1rem;
    }
    .css-1d391kg {
        padding-top: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_data
def load_model():
    """Load the latest trained model and preprocessing information"""
    try:
        model_dir = Path("models")
        model_files = list(model_dir.glob("credit_risk_model_*.joblib"))
        if not model_files:
            st.error("No trained model found. Please train the model first.")
            return None
        
        latest_model = max(model_files, key=lambda x: x.stat().st_mtime)
        model_info = joblib.load(latest_model)
        
        return model_info
        
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def prepare_features(data_dict, model_info):
    """Prepare features in the correct format for model prediction"""
    # Create a DataFrame with all input features
    df = pd.DataFrame([data_dict])
    return df

@st.cache_data
def load_data():
    """Load the credit risk dataset from data directory"""
    try:
        # Load the dataset from data directory
        dataset_path = "data/credit_risk_dataset.csv"
        
        # Try to read the CSV file
        try:
            df = pd.read_csv(dataset_path)
        except Exception as e:
            st.error(f"Error reading dataset: {str(e)}")
            return None
        
        # Display raw data info for debugging
        st.sidebar.write("Raw Data Info:")
        st.sidebar.write("Available columns:", df.columns.tolist())
        st.sidebar.write(f"Initial shape: {df.shape}")
        
        # Clean the data
        # Remove any rows with missing values
        df = df.dropna()
        
        # Remove any duplicate rows
        df = df.drop_duplicates()
        
        # Convert numeric columns to appropriate types
        numeric_columns = ['person_age', 'person_income', 'person_emp_length', 
                         'loan_amnt', 'loan_int_rate', 'loan_percent_income',
                         'cb_person_cred_hist_length']
        
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Drop rows with any remaining NaN values after conversion
        df = df.dropna()
        
        # Display final dataset info in sidebar
        st.sidebar.write("Clean Dataset Info:")
        st.sidebar.write(f"- Final Rows: {len(df):,}")
        st.sidebar.write(f"- Final Columns: {len(df.columns)}")
        st.sidebar.write(f"- Memory Usage: {df.memory_usage().sum() / 1024**2:.1f} MB")
        
        return df
        
    except Exception as e:
        st.error(f"Error loading the dataset: {str(e)}")
        st.write("Error details:", str(e))
        return None

def main():
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Home", "Data Analysis", "Model Prediction", "About"])

    if page == "Home":
        st.title("🏦 Credit Risk Prediction System")
        st.markdown("""
            ### Welcome to the Credit Risk Prediction System
            
            This application helps financial institutions assess the credit risk of loan applicants
            using machine learning models. The system analyzes various factors including:
            
            - Personal Information
                - Age
                - Income
                - Employment Length
                - Home Ownership
            
            - Loan Information
                - Amount
                - Term
                - Interest Rate
                - Purpose/Intent
                - Grade
            
            - Credit History
                - Default History
                - Credit History Length
            
            Use the sidebar to navigate through different sections:
            
            1. **Data Analysis**: Explore the dataset and view insights
            2. **Model Prediction**: Make predictions for new loan applications
            3. **About**: Learn more about the project
        """)
        
        # Load data for metrics
        data = load_data()
        if data is not None:
            # Display key metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(label="Total Applications", value=f"{len(data):,}")
            with col2:
                st.metric(label="Default Rate", 
                         value=f"{(data['loan_status'] == 1).mean():.1%}")
            with col3:
                avg_loan = data['loan_amnt'].mean()
                st.metric(label="Average Loan Amount", 
                         value=f"${avg_loan:,.0f}")

    elif page == "Data Analysis":
        st.title("📊 Data Analysis")
        
        data = load_data()
        if data is not None:
            # Data Overview
            st.subheader("Dataset Overview")
            st.write("First few rows of the dataset:")
            st.dataframe(data.head())
            
            # Feature Distribution
            st.subheader("Feature Distribution")
            
            # Separate numerical and categorical features
            numerical_features = ['person_age', 'person_income', 'person_emp_length', 
                                'loan_amnt', 'loan_int_rate', 'loan_percent_income',
                                'cb_person_cred_hist_length']
            
            categorical_features = ['person_home_ownership', 'loan_intent', 'loan_grade',
                                  'cb_person_default_on_file', 'loan_status']
            
            # Let user choose feature type
            feature_type = st.radio("Select Feature Type", ["Numerical", "Categorical"])
            
            if feature_type == "Numerical":
                feature = st.selectbox("Select Feature", numerical_features)
                
                # Calculate and display statistics first
                st.subheader(f"Statistics for {feature}")
                stats_df = pd.DataFrame({
                    'Statistic': ['Count', 'Mean', 'Std', 'Min', '25%', '50%', '75%', 'Max'],
                    'Value': [
                        f"{data[feature].count():,.0f}",
                        f"{data[feature].mean():,.2f}",
                        f"{data[feature].std():,.2f}",
                        f"{data[feature].min():,.2f}",
                        f"{data[feature].quantile(0.25):,.2f}",
                        f"{data[feature].quantile(0.50):,.2f}",
                        f"{data[feature].quantile(0.75):,.2f}",
                        f"{data[feature].max():,.2f}"
                    ]
                })
                st.table(stats_df)
                
                # Display histogram
                st.subheader(f"Distribution of {feature}")
                fig = px.histogram(
                    data, 
                    x=feature, 
                    title=f"Distribution of {feature}",
                    nbins=50,
                    marginal="box"
                )
                fig.update_layout(
                    xaxis_title=feature,
                    yaxis_title="Count",
                    showlegend=False
                )
                st.plotly_chart(fig, use_container_width=True)
                
            else:
                feature = st.selectbox("Select Feature", categorical_features)
                
                # Calculate and display value counts
                st.subheader(f"Value Counts for {feature}")
                value_counts = data[feature].value_counts()
                counts_df = pd.DataFrame({
                    'Category': value_counts.index,
                    'Count': value_counts.values,
                    'Percentage': (value_counts.values / len(data) * 100).round(2)
                })
                counts_df['Percentage'] = counts_df['Percentage'].apply(lambda x: f"{x:.2f}%")
                st.table(counts_df)
                
                # Display bar chart
                st.subheader(f"Distribution of {feature}")
                fig = px.bar(
                    counts_df,
                    x='Category',
                    y='Count',
                    title=f"Distribution of {feature}",
                    text='Percentage'
                )
                fig.update_traces(textposition='outside')
                fig.update_layout(
                    xaxis_title=feature,
                    yaxis_title="Count",
                    showlegend=False
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Correlation Matrix for numerical features
            st.subheader("Correlation Matrix")
            numeric_data = data[numerical_features]
            corr_matrix = numeric_data.corr().round(2)
            
            fig = px.imshow(
                corr_matrix,
                title="Correlation Matrix",
                color_continuous_scale="RdBu",
                aspect="auto",
                labels=dict(color="Correlation")
            )
            fig.update_traces(text=corr_matrix.values, texttemplate="%{text}")
            st.plotly_chart(fig, use_container_width=True)

    elif page == "Model Prediction":
        st.title("🔮 Credit Risk Prediction")
        
        model_info = load_model()
        if model_info is not None:
            model = model_info['model']
            st.markdown("""
                ### Enter Applicant Information
                Please fill in the details below to predict credit risk.
            """)
            
            # Create input fields
            col1, col2 = st.columns(2)
            
            # Collect all inputs in a dictionary
            inputs = {}
            
            with col1:
                inputs['person_age'] = st.number_input("Age", min_value=18, max_value=100, value=30)
                inputs['person_income'] = st.number_input("Annual Income ($)", min_value=0, value=50000)
                inputs['person_emp_length'] = st.number_input("Employment Length (years)", min_value=0, value=5)
                inputs['person_home_ownership'] = st.selectbox("Home Ownership", 
                                                             ["RENT", "OWN", "MORTGAGE", "OTHER"])
                inputs['loan_intent'] = st.selectbox("Loan Intent",
                                                   ["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE",
                                                    "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"])
            
            with col2:
                inputs['loan_amnt'] = st.number_input("Loan Amount ($)", min_value=0, value=10000)
                inputs['loan_grade'] = st.selectbox("Loan Grade", ["A", "B", "C", "D", "E", "F"])
                inputs['loan_int_rate'] = st.number_input("Interest Rate (%)", min_value=0.0, max_value=30.0, value=10.0)
                inputs['loan_percent_income'] = st.number_input("Loan Percent Income", min_value=0.0, max_value=1.0, value=0.3)
                inputs['cb_person_cred_hist_length'] = st.number_input("Credit History Length (years)", min_value=0, value=5)
                inputs['cb_person_default_on_file'] = st.selectbox("Previous Defaults", ["N", "Y"])
            
            if st.button("Predict Credit Risk"):
                # Prepare features
                features = prepare_features(inputs, model_info)
                
                # Make prediction using the pipeline
                prediction = model.predict(features)[0]
                probability = model.predict_proba(features)[0]
                
                # Display results
                st.subheader("Prediction Results")
                
                col1, col2 = st.columns(2)
                with col1:
                    risk_level = "High Risk" if prediction == 1 else "Low Risk"
                    risk_color = "red" if prediction == 1 else "green"
                    st.markdown(f"### Risk Level: <span style='color:{risk_color}'>{risk_level}</span>", 
                              unsafe_allow_html=True)
                    st.write(f"Probability of Default: {probability[1]:.2%}")
                    
                    # Display feature importance if available
                    if 'feature_names' in model_info:
                        st.write("\n### Top Contributing Factors")
                        feature_importance = pd.DataFrame({
                            'Feature': model_info['feature_names'],
                            'Importance': model.named_steps['classifier'].feature_importances_
                        }).sort_values('Importance', ascending=False).head(5)
                        
                        for _, row in feature_importance.iterrows():
                            st.write(f"- {row['Feature']}: {row['Importance']:.2%}")
                
                with col2:
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=probability[1] * 100,
                        domain={'x': [0, 1], 'y': [0, 1]},
                        title={'text': "Risk Probability"},
                        gauge={'axis': {'range': [0, 100]},
                               'bar': {'color': risk_color},
                               'steps': [
                                   {'range': [0, 30], 'color': "lightgreen"},
                                   {'range': [30, 70], 'color': "yellow"},
                                   {'range': [70, 100], 'color': "lightcoral"}
                               ]}))
                    st.plotly_chart(fig, use_container_width=True)

    else:  # About page
        st.title("ℹ️ About")
        st.markdown("""
            ### Credit Risk Prediction System
            
            This project is designed to help financial institutions make data-driven decisions
            about loan applications. It uses machine learning to predict the likelihood of
            loan default based on various applicant characteristics.
            
            #### Dataset Features
            
            The model analyzes the following features:
            
            **Personal Information**
            - Age
            - Income
            - Employment Length
            - Home Ownership (RENT/OWN/MORTGAGE/OTHER)
            
            **Loan Information**
            - Amount
            - Interest Rate
            - Percent of Income
            - Intent/Purpose
            - Grade (A-F)
            
            **Credit History**
            - Default History
            - Credit History Length
            
            #### Technical Details
            
            - Built with Python and Streamlit
            - Uses scikit-learn for machine learning
            - Features data preprocessing and engineering
            - Interactive visualizations with Plotly
            
            #### Model Performance
            
            The model's performance metrics are calculated on a test set:
            - Accuracy
            - ROC AUC
            - Precision
            - Recall
            - F1 Score
            
            Note: The actual performance metrics will vary based on the model training results.
        """)

if __name__ == "__main__":
    main() 