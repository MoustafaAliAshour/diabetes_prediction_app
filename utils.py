import joblib
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

def load_model_and_preprocessor():
    """Load the saved XGBoost model and preprocessor."""
    try:
        model = joblib.load('models/xgboost_diabetes_model.pkl')
        preprocessor = joblib.load('models/preprocessor_pipeline.pkl')
        return model, preprocessor
    except FileNotFoundError as e:
        st.error(f"Error loading model files: {e}")
        st.stop()

def recategorize_smoking(smoking_status):
    """Recategorize smoking status to match the training data."""
    if smoking_status in ['never', 'No Info']:
        return 'non-smoker'
    elif smoking_status == 'current':
        return 'current'
    elif smoking_status in ['ever', 'former', 'not current']:
        return 'past_smoker'
    return smoking_status

def preprocess_input(data, preprocessor):
    """Preprocess user input data using the saved preprocessor."""
    # Convert to DataFrame for preprocessing
    input_df = pd.DataFrame([data])
    
    # Apply preprocessing
    processed_data = preprocessor.transform(input_df)
    return processed_data

def predict_diabetes(model, processed_data):
    """Make a prediction using the preprocessed data."""
    prediction = model.predict(processed_data)
    prediction_proba = model.predict_proba(processed_data)
    return prediction[0], prediction_proba[0]

def create_gauge_chart(probability):
    """Create a gauge chart to visualize the prediction probability."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Risk of Diabetes", 'font': {'size': 24}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 30], 'color': 'green'},
                {'range': [30, 70], 'color': 'yellow'},
                {'range': [70, 100], 'color': 'red'}],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': probability * 100}}))
    
    fig.update_layout(height=300)
    return fig

def display_feature_importance(model):
    """Display feature importance from the model."""
    # Get feature importances and display
    # Note: You would need to have the feature names in the correct order
    numeric_features = ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level', 'hypertension', 'heart_disease']
    onehot_columns = ['gender_Female', 'gender_Male', 
                      'smoking_history_current', 'smoking_history_non-smoker', 'smoking_history_past_smoker']
    feature_names = numeric_features + onehot_columns
    
    importances = model.feature_importances_
    
    # Create DataFrame for plotting
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)
    
    # Create bar chart
    fig = px.bar(importance_df, x='Importance', y='Feature', orientation='h',
                 title='Feature Importance', color='Importance',
                 color_continuous_scale='Blues')
    
    return fig

def display_risk_factors(user_data):
    """Display risk factors based on user input."""
    risk_factors = []
    
    # Check BMI
    if user_data['bmi'] >= 30:
        risk_factors.append("High BMI (≥30)")
    
    # Check blood glucose
    if user_data['blood_glucose_level'] > 140:
        risk_factors.append("Elevated blood glucose (>140 mg/dL)")
    
    # Check HbA1c
    if user_data['HbA1c_level'] >= 6.5:
        risk_factors.append("Elevated HbA1c (≥6.5%)")
    
    # Check age
    if user_data['age'] > 45:
        risk_factors.append("Age over 45")
    
    # Check hypertension and heart disease
    if user_data['hypertension'] == 1:
        risk_factors.append("History of hypertension")
    
    if user_data['heart_disease'] == 1:
        risk_factors.append("History of heart disease")
    
    return risk_factors

def generate_health_recommendations(prediction, risk_factors):
    """Generate health recommendations based on prediction and risk factors."""
    recommendations = []
    
    if prediction == 1:
        recommendations.append("Based on your results, we recommend consulting with a healthcare provider for further evaluation.")
        recommendations.append("Regular monitoring of blood glucose levels may be beneficial.")
    
    # General recommendations
    recommendations.append("Maintain a balanced diet rich in vegetables, fruits, and whole grains.")
    recommendations.append("Engage in regular physical activity (at least 150 minutes per week).")
    recommendations.append("Maintain a healthy weight or work towards achieving it.")
    
    # Specific recommendations based on risk factors
    if "High BMI (≥30)" in risk_factors:
        recommendations.append("Consider working with a nutritionist to develop a weight management plan.")
    
    if "Elevated blood glucose (>140 mg/dL)" in risk_factors or "Elevated HbA1c (≥6.5%)" in risk_factors:
        recommendations.append("Limit intake of refined carbohydrates and added sugars.")
        recommendations.append("Consider the glycemic index when planning meals.")
    
    if "History of hypertension" in risk_factors:
        recommendations.append("Limit sodium intake and consider the DASH diet approach.")
        recommendations.append("Monitor blood pressure regularly.")
    
    if "History of heart disease" in risk_factors:
        recommendations.append("Focus on heart-healthy foods and regular cardio exercises.")
        recommendations.append("Consider discussing medication options with your doctor.")
    
    return recommendations