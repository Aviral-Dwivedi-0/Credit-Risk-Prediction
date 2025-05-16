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
    initial_sidebar_state="collapsed"
)

# Custom CSS
st.markdown("""
    <style>
    /* General body styling */
    body {
        background-color: #ffffff;
        color: #333333;
        font-family: 'Arial', sans-serif;
    }

    /* Header styling */
    .header {
        background-color: #1E88E5;
        padding: 1rem 2rem;
        margin-bottom: 2rem;
        border-radius: 0 0 10px 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }

    .header h1 {
        color: white;
        margin: 0;
        font-size: 2rem;
    }

    /* Navigation styling */
    .nav {
        display: flex;
        gap: 1rem;
        margin-top: 1rem;
    }

    .nav-link {
        color: white;
        text-decoration: none;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        transition: background-color 0.3s;
    }

    .nav-link:hover {
        background-color: rgba(255,255,255,0.1);
    }

    /* Main content area */
    .main {
        padding: 2rem;
        max-width: 1200px;
        margin: 0 auto;
    }

    /* Cards */
    .card {
        background: linear-gradient(145deg, #ffffff, #f5f7fa);
        border-radius: 15px;
        padding: 2rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        margin-bottom: 2rem;
        border: 1px solid rgba(0,0,0,0.05);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }

    .card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.08);
    }

    .card h2 {
        color: #2c3e50;
        font-size: 1.8rem;
        margin-bottom: 1rem;
        font-weight: 600;
    }

    .card h3 {
        color: #34495e;
        font-size: 1.4rem;
        margin: 1.5rem 0 1rem 0;
        font-weight: 500;
    }

    .card h4 {
        color: #2c3e50;
        font-size: 1.2rem;
        margin: 1.2rem 0 0.8rem 0;
        font-weight: 500;
    }

    .card p {
        color: #4a5568;
        font-size: 1.1rem;
        line-height: 1.6;
        margin-bottom: 1rem;
    }

    .card ul {
        color: #4a5568;
        font-size: 1.1rem;
        line-height: 1.8;
        padding-left: 1.5rem;
    }

    .card li {
        margin-bottom: 0.5rem;
    }

    /* Input fields within cards */
    .card .stNumberInput>div>div>input,
    .card .stSelectbox>div>div>select {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 0.75rem;
        color: #2d3748;
        font-size: 1rem;
        transition: border-color 0.2s ease, box-shadow 0.2s ease;
    }

    .card .stNumberInput>div>div>input:focus,
    .card .stSelectbox>div>div>select:focus {
        border-color: #4299e1;
        box-shadow: 0 0 0 3px rgba(66, 153, 225, 0.1);
        outline: none;
    }

    /* Results section in cards */
    .card .stMarkdown {
        color: #4a5568;
    }

    .card .stMarkdown h3 {
        color: #2c3e50;
        font-size: 1.4rem;
        margin: 1.5rem 0 1rem 0;
    }

    /* Prediction results specific styling */
    .prediction-result {
        background: linear-gradient(145deg, #ffffff, #f8fafc);
        border-radius: 12px;
        padding: 1.5rem;
        margin-top: 1rem;
        border: 1px solid rgba(0,0,0,0.05);
    }

    .prediction-result h3 {
        color: #2c3e50;
        font-size: 1.4rem;
        margin-bottom: 1rem;
    }

    .prediction-result p {
        color: #4a5568;
        font-size: 1.1rem;
        line-height: 1.6;
    }

    /* Input fields */
    .stTextInput>div>div>input, .stNumberInput>div>div>input, .stSelectbox>div>div>select {
        border-radius: 5px;
        border: 1px solid #ddd;
        padding: 0.5rem;
    }

    /* Buttons */
    .stButton>button {
        background-color: #1E88E5;
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        cursor: pointer;
        transition: background-color 0.3s;
    }

    .stButton>button:hover {
        background-color: #1976D2;
    }

    /* Metrics */
    .metric-card {
        background: white;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }

    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #1E88E5;
    }

    .metric-label {
        color: #666;
        font-size: 0.9rem;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state for page navigation
if 'page' not in st.session_state:
    st.session_state.page = 'Home'

def create_header():
    st.markdown("""
        <div class="header">
            <h1>🏦 Credit Risk Prediction System</h1>
            <div class="nav">
                <a href="#" class="nav-link" onclick="document.querySelector('[data-testid=stButton]').click()">Home</a>
                <a href="#" class="nav-link" onclick="document.querySelector('[data-testid=stButton]').click()">About</a>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("Home", key="home_btn"):
            st.session_state.page = 'Home'
    with col2:
        if st.button("About", key="about_btn"):
            st.session_state.page = 'About'

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
    df = pd.DataFrame([data_dict])
    return df

@st.cache_data
def load_data():
    """Load the credit risk dataset from data directory"""
    try:
        dataset_path = "data/credit_risk_dataset.csv"
        df = pd.read_csv(dataset_path)
        df = df.dropna()
        df = df.drop_duplicates()
        
        numeric_columns = ['person_age', 'person_income', 'person_emp_length', 
                         'loan_amnt', 'loan_int_rate', 'loan_percent_income',
                         'cb_person_cred_hist_length']
        
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df = df.dropna()
        return df
        
    except Exception as e:
        st.error(f"Error loading the dataset: {str(e)}")
        return None

def show_prediction_page():
    st.markdown("""
        <div class="card">
            <h2>🔮 Credit Risk Prediction</h2>
            <p>Enter applicant information to predict credit risk.</p>
        </div>
    """, unsafe_allow_html=True)
    
    model_info = load_model()
    if model_info is not None:
        model = model_info['model']
        
        # Create input fields
        col1, col2 = st.columns(2)
        
        # Collect all inputs in a dictionary
        inputs = {}
        
        with col1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            inputs['person_age'] = st.number_input("Age", min_value=18, max_value=100, value=30)
            inputs['person_income'] = st.number_input("Annual Income ($)", min_value=0, value=50000)
            inputs['person_emp_length'] = st.number_input("Employment Length (years)", min_value=0, value=5)
            inputs['person_home_ownership'] = st.selectbox("Home Ownership", 
                                                         ["RENT", "OWN", "MORTGAGE", "OTHER"])
            inputs['loan_intent'] = st.selectbox("Loan Intent",
                                               ["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE",
                                                "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"])
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            inputs['loan_amnt'] = st.number_input("Loan Amount ($)", min_value=0, value=10000)
            inputs['loan_grade'] = st.selectbox("Loan Grade", ["A", "B", "C", "D", "E", "F"])
            inputs['loan_int_rate'] = st.number_input("Interest Rate (%)", min_value=0.0, max_value=30.0, value=10.0)
            inputs['loan_percent_income'] = st.number_input("Loan Percent Income", min_value=0.0, max_value=1.0, value=0.3)
            inputs['cb_person_cred_hist_length'] = st.number_input("Credit History Length (years)", min_value=0, value=5)
            inputs['cb_person_default_on_file'] = st.selectbox("Previous Defaults", ["N", "Y"])
            st.markdown('</div>', unsafe_allow_html=True)
        
        if st.button("Predict Credit Risk", key="predict_btn"):
            # Prepare features
            features = prepare_features(inputs, model_info)
            
            # Make prediction
            prediction = model.predict(features)[0]
            probability = model.predict_proba(features)[0]
            
            # Display results
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("Prediction Results")
            
            col1, col2 = st.columns(2)
            with col1:
                risk_level = "High Risk" if prediction == 1 else "Low Risk"
                risk_color = "red" if prediction == 1 else "green"
                st.markdown(f"### Risk Level: <span style='color:{risk_color};'>{risk_level}</span>",
                          unsafe_allow_html=True)
                st.write(f"Probability of Default: {probability[1]:.2%}")
                
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
            st.markdown('</div>', unsafe_allow_html=True)

def show_about_page():
    st.markdown("""
        <div class="card">
            <h2>ℹ️ About the Credit Risk Prediction System</h2>
            <p>This project helps financial institutions make data-driven decisions about loan applications using machine learning.</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
        <div class="card">
            <h3>📊 Dataset Features</h3>
            <h4>Personal Information</h4>
            <ul>
                <li>Age</li>
                <li>Income</li>
                <li>Employment Length</li>
                <li>Home Ownership (RENT/OWN/MORTGAGE/OTHER)</li>
            </ul>
            
            <h4>Loan Information</h4>
            <ul>
                <li>Amount</li>
                <li>Interest Rate</li>
                <li>Percent of Income</li>
                <li>Intent/Purpose</li>
                <li>Grade (A-F)</li>
            </ul>
            
            <h4>Credit History</h4>
            <ul>
                <li>Default History</li>
                <li>Credit History Length</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
        <div class="card">
            <h3>🛠️ Technical Details</h3>
            <ul>
                <li>Built with Python and Streamlit</li>
                <li>Uses scikit-learn for machine learning</li>
                <li>Features data preprocessing and engineering</li>
                <li>Interactive visualizations with Plotly</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)

def main():
    create_header()
    
    if st.session_state.page == 'Home':
        show_prediction_page()
    else:
        show_about_page()

if __name__ == "__main__":
    main() 