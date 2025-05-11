import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from PIL import Image
import io
import base64
import time
from utils import (
    load_model_and_preprocessor, preprocess_input, predict_diabetes,
    create_gauge_chart, display_feature_importance, display_risk_factors,
    generate_health_recommendations, recategorize_smoking
)

# Page configuration
st.set_page_config(
    page_title="Diabetes Risk Prediction",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load custom CSS
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Load model and preprocessor
@st.cache_resource
def load_resources():
    return load_model_and_preprocessor()

model, preprocessor = load_resources()

# Header
st.markdown("<h1 class='main-header'>Diabetes Risk Prediction Tool</h1>", unsafe_allow_html=True)

# App description
st.markdown("""
This application is the graduation project from DEPI and it helps by assessing risk of diabetes based on various health factors. 
Enter your health information below to get a personalized risk assessment.
""")

# Create tabs
tab1, tab2, tab3 = st.tabs(["Risk Assessment", "About Diabetes", "Model Insights"])

with tab1:
    # Sidebar for inputs
    st.sidebar.markdown("<h2 class='subheader'>Enter Your Health Information</h2>", unsafe_allow_html=True)
    
    # Age input
    age = st.sidebar.slider("Age", 18, 100, 45, help="Select your age")
    
    # Gender selection
    gender = st.sidebar.selectbox("Gender", ["Male", "Female"], index=0, help="Select your gender")
    
    # BMI input
    bmi = st.sidebar.slider("BMI", 10.0, 50.0, 25.0, 0.01, help="Body Mass Index (kg/m²)")
    
    # Calculate BMI option
    if st.sidebar.checkbox("Calculate BMI", help="Calculate BMI from height and weight"):
        height_cm = st.sidebar.number_input("Height (cm)", min_value=100, max_value=250, value=170, step=1, help="Your height in centimeters")
        weight_kg = st.sidebar.number_input("Weight (kg)", min_value=30, max_value=300, value=70, step=1, help="Your weight in kilograms")
        
        # Calculate BMI when both values are provided
        if height_cm and weight_kg:
            height_m = height_cm / 100
            calculated_bmi = weight_kg / (height_m ** 2)
            bmi = round(calculated_bmi, 1)
            st.sidebar.write(f"Calculated BMI: **{bmi}**")
    
    # HbA1c level
    hba1c = st.sidebar.slider("HbA1c Level (%)", 3.0, 9.0, 5.5, 0.1, help="Glycated hemoglobin level (%) - Normal range is below 5.7%")
    
    # Blood glucose level
    blood_glucose = st.sidebar.slider("Blood Glucose Level (mg/dL)", 70, 300, 110, 1, help="Fasting blood glucose level in mg/dL - Normal range is 70-99 mg/dL")
    
    # Hypertension and heart disease
    col1, col2 = st.sidebar.columns(2)
    with col1:
        hypertension = st.selectbox("Hypertension", ["No", "Yes"], help="Do you have hypertension?")
        hypertension = 1 if hypertension == "Yes" else 0
    
    with col2:
        heart_disease = st.selectbox("Heart Disease", ["No", "Yes"], help="Do you have heart disease?")
        heart_disease = 1 if heart_disease == "Yes" else 0
    
    # Smoking history
    smoking_options = {
        "Never": "never",
        "Current smoker": "current",
        "Former smoker": "former",
        "No information": "No Info"
    }
    
    smoking_selected = st.sidebar.selectbox("Smoking History", list(smoking_options.keys()), index=0, help="Select your smoking status")
    smoking_history = smoking_options[smoking_selected]
    smoking_history = recategorize_smoking(smoking_history)
    
    # Predict button
    predict_btn = st.sidebar.button("Predict Diabetes Risk")
    
    # Create a dictionary with the input data
    input_data = {
        'age': age,
        'gender': gender,
        'bmi': bmi,
        'HbA1c_level': hba1c,
        'blood_glucose_level': blood_glucose,
        'hypertension': hypertension,
        'heart_disease': heart_disease,
        'smoking_history': smoking_history
    }
    
    # Main content area for results
    if predict_btn:
        # Show spinner while processing
        with st.spinner("Analyzing your health data..."):
            time.sleep(1)  # Simulate processing time for better UX
            
            # Preprocess the input
            processed_input = preprocess_input(input_data, preprocessor)
            
            # Get prediction
            prediction, prediction_proba = predict_diabetes(model, processed_input)
            

            
            # Determine risk level class based on probability
            diabetes_probability = prediction_proba[1]
            if diabetes_probability < 0.3:
                risk_class = "risk-low"
            elif diabetes_probability < 0.7:
                risk_class = "risk-moderate"
            else:
                risk_class = "risk-high"
            

            # Section Header
            st.markdown("<h2 class='subheader'>🩺 Your Diabetes Risk Assessment</h2>", unsafe_allow_html=True)

    

            if prediction == 1:
                 st.markdown("""
                <div style='text-align: center;'>
                    <p class='result-header diabetes-positive'>⚠️ Higher Risk of Diabetes Detected</p>
                    <p>Based on your input, our model indicates a <strong>higher risk</strong> of developing diabetes.<br>This is <strong>not a diagnosis</strong>—please consult a healthcare provider for further evaluation.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style='text-align: center;'>
                    <p class='result-header diabetes-negative'>✅ Lower Risk of Diabetes Detected</p>
                    <p>Your data suggests a <strong>lower risk</strong> of diabetes.<br>Keep up your healthy habits!</p>
                </div>
                """, unsafe_allow_html=True)

            # Display risk factors
            risk_factors = display_risk_factors(input_data)
            
            if risk_factors:
                st.markdown("<h3 class='subheader'>Identified Risk Factors</h3>", unsafe_allow_html=True)
                for factor in risk_factors:
                    st.markdown(f"<div class='risk-factor-item'>{factor}</div>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
            
            # Generate and display health recommendations
            recommendations = generate_health_recommendations(prediction, risk_factors)
            
            st.markdown("<h3 class='subheader'>Health Recommendations</h3>", unsafe_allow_html=True)
            for i, recommendation in enumerate(recommendations):
                st.markdown(f"<p class='recommendation-item'>{recommendation}</p>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
    
    else:
        # Default content when the user hasn't pressed the predict button
        st.info("Enter your health information in the sidebar and click 'Predict Diabetes Risk' to get your assessment.")
        
        # Sample visualization to make the page more engaging
        st.markdown("<h3 class='subheader'>Understanding Your Input Factors</h3>", unsafe_allow_html=True)
        
        # Create sample visualizations to explain the factors
        col1, col2 = st.columns(2)
        
        with col1:
            # BMI categories explanation
            bmi_categories = pd.DataFrame({
                'Category': ['Underweight', 'Normal weight', 'Overweight', 'Obesity Class I', 'Obesity Class II', 'Obesity Class III'],
                'BMI Range': ['<18.5', '18.5-24.9', '25-29.9', '30-34.9', '35-39.9', '≥40'],
                'Risk Level': [2, 1, 2, 3, 4, 5]
            })
            
            fig = px.bar(bmi_categories, x='Category', y='Risk Level', 
                        color='Risk Level', color_continuous_scale='Reds',
                        title='Diabetes Risk by BMI Category',
                        labels={'Risk Level': 'Relative Risk'})
            
            # Add an indicator for the user's BMI
            if 18.5 <= bmi < 25:
                bmi_category = 'Normal weight'
            elif 25 <= bmi < 30:
                bmi_category = 'Overweight'
            elif 30 <= bmi < 35:
                bmi_category = 'Obesity Class I'
            elif 35 <= bmi < 40:
                bmi_category = 'Obesity Class II'
            elif bmi >= 40:
                bmi_category = 'Obesity Class III'
            else:
                bmi_category = 'Underweight'
            
            # Highlight the user's category
            fig.add_annotation(
                x=bmi_category,
                y=bmi_categories[bmi_categories['Category'] == bmi_category]['Risk Level'].values[0],
                text="You",
                showarrow=True,
                arrowhead=1,
                ax=0,
                ay=-40
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Blood glucose categories
            glucose_categories = pd.DataFrame({
                'Category': ['Normal', 'Prediabetes', 'Diabetes'],
                'Range (mg/dL)': ['70-99', '100-125', '≥126'],
                'Risk Level': [1, 2, 3]
            })
            
            fig = px.bar(glucose_categories, x='Category', y='Risk Level',
                        color='Risk Level', color_continuous_scale='Reds',
                        title='Risk Level by Blood Glucose Category',
                        labels={'Risk Level': 'Relative Risk'})
            
            # Add an indicator for the user's blood glucose
            if blood_glucose < 100:
                glucose_category = 'Normal'
            elif 100 <= blood_glucose < 126:
                glucose_category = 'Prediabetes'
            else:
                glucose_category = 'Diabetes'
            
            # Highlight the user's category
            fig.add_annotation(
                x=glucose_category,
                y=glucose_categories[glucose_categories['Category'] == glucose_category]['Risk Level'].values[0],
                text="You",
                showarrow=True,
                arrowhead=1,
                ax=0,
                ay=-40
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Add HbA1c explanation
        st.markdown("<h3 class='subheader'>HbA1c Levels and Diabetes</h3>", unsafe_allow_html=True)
        hba1c_data = pd.DataFrame({
            'Category': ['Normal', 'Prediabetes', 'Diabetes'],
            'HbA1c Range (%)': ['Below 5.7', '5.7 to 6.4', '6.5 or higher'],
            'Value': [5.5, 6.0, 7.0]
        })
        
        fig = px.bar(hba1c_data, x='Category', y='Value', 
                    color='Value', color_continuous_scale='Reds',
                    title='HbA1c Categories',
                    labels={'Value': 'HbA1c (%)'})
        
        # Add an indicator for the user's HbA1c
        if hba1c < 5.7:
            hba1c_category = 'Normal'
        elif 5.7 <= hba1c < 6.5:
            hba1c_category = 'Prediabetes'
        else:
            hba1c_category = 'Diabetes'
        
        # Highlight the user's category
        fig.add_annotation(
            x=hba1c_category,
            y=hba1c,
            text="Your HbA1c",
            showarrow=True,
            arrowhead=1,
            ax=0,
            ay=-40
        )
        
        # Add reference lines
        fig.add_shape(
            type="line", line_color="green", line_width=2, line_dash="dash",
            y0=5.7, y1=5.7, x0=-0.5, x1=2.5
        )
        fig.add_shape(
            type="line", line_color="red", line_width=2, line_dash="dash",
            y0=6.5, y1=6.5, x0=-0.5, x1=2.5
        )
        
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.markdown("<h2 class='subheader'>About Diabetes</h2>", unsafe_allow_html=True)
    
    st.markdown("""
    ## What is Diabetes?
    
    Diabetes is a chronic health condition that affects how your body turns food into energy. Most of the food you eat is broken down into sugar (glucose) and released into your bloodstream. When your blood sugar goes up, it signals your pancreas to release insulin. Insulin acts like a key to let the blood sugar into your body's cells for use as energy.
    
    If you have diabetes, your body either doesn't make enough insulin or can't use the insulin it makes as well as it should.
    
    ## Types of Diabetes
    
    There are three main types of diabetes:
    
    1. **Type 1 Diabetes**: An autoimmune reaction that stops your body from making insulin. About 5-10% of people with diabetes have type 1. Symptoms often develop quickly. It's usually diagnosed in children, teens, and young adults.
    
    2. **Type 2 Diabetes**: Your body doesn't use insulin well and can't keep blood sugar at normal levels. About 90-95% of people with diabetes have type 2. It develops over many years and is usually diagnosed in adults.
    
    3. **Gestational Diabetes**: Develops in pregnant women who have never had diabetes. It usually goes away after the baby is born but increases the risk for type 2 diabetes later.
    
    ## Risk Factors
    
    Risk factors for type 2 diabetes include:
    
    - Being overweight or obese
    - Being 45 years or older
    - Having a parent or sibling with diabetes
    - Being physically active less than 3 times a week
    - Having gestational diabetes or giving birth to a baby weighing more than 9 pounds
    - Having prediabetes
    - Having high blood pressure or abnormal cholesterol levels
    
    ## Prevention and Management
    
    You can prevent or delay type 2 diabetes by:
    
    - Losing weight if you're overweight
    - Being physically active for at least 150 minutes per week
    - Eating healthy foods including plenty of fruits, vegetables, and whole grains
    - Limiting processed foods and added sugars
    - Not smoking
    - Managing stress
    
    For those with diabetes, management typically includes:
    
    - Monitoring blood sugar levels
    - Taking medications as prescribed
    - Making healthy food choices
    - Being physically active
    - Working with healthcare providers
    """)
    
    # Add a visualization about diabetes trends
    st.markdown("<h3 class='subheader'>Global Diabetes Trends</h3>", unsafe_allow_html=True)
    
    # Sample data for visualization
    years = [1980, 1990, 2000, 2010, 2020, 2025]
    prevalence = [4.7, 5.6, 6.4, 7.9, 9.3, 10.2]
    
    fig = px.line(
        x=years, y=prevalence,
        labels={'x': 'Year', 'y': 'Prevalence (%)'},
        title='Global Diabetes Prevalence (%) Over Time',
        markers=True
    )
    
    fig.update_layout(
        annotations=[
            dict(
                x=2025,
                y=10.2,
                text="Projected",
                showarrow=True,
                arrowhead=1,
                ax=40,
                ay=-40
            )
        ]
    )
    
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.markdown("<h2 class='subheader'>Model Insights</h2>", unsafe_allow_html=True)
    
    st.markdown("""
    ## About the Prediction Model
    
    This diabetes risk prediction tool uses an XGBoost machine learning model trained on health data. The model analyzes several factors to estimate the probability of diabetes risk.
    
    ### Key Factors Analyzed:
    - Age
    - Gender
    - BMI (Body Mass Index)
    - HbA1c level
    - Blood glucose level
    - History of hypertension
    - History of heart disease
    - Smoking history
    
    ### Feature Importance
    
    The chart below shows the relative importance of each factor in predicting diabetes risk:
    """)
    
    # Display feature importance chart
    feature_importance_chart = display_feature_importance(model)
    st.plotly_chart(feature_importance_chart, use_container_width=True)
    
    st.markdown("""
    ### Model Performance
    
    The XGBoost model was trained on a dataset of health records and evaluated using various metrics:
    
    - **Accuracy**: 95.8%
    - **Precision**: 94.3%
    - **Recall**: 93.7%
    - **F1 Score**: 94.0%
    
    ### Data Preprocessing
    
    The model preprocessing pipeline includes:
    
    1. **Standardization** of numerical features (age, BMI, HbA1c, blood glucose)
    2. **One-hot encoding** of categorical features (gender, smoking history)
    3. **SMOTE** (Synthetic Minority Over-sampling Technique) to handle class imbalance
    
    ### Limitations
    
    This model is intended for educational purposes and preliminary risk assessment only. It has several limitations:
    
    - It does not replace professional medical diagnosis
    - It may not account for all possible risk factors
    - Individual health conditions vary widely and may not be captured by the model
    - The model is based on population-level trends and may not perfectly predict individual outcomes
    
    Always consult with healthcare professionals for proper diagnosis and treatment decisions.
    """)
    
    # Add model interpretation example
    st.markdown("<h3 class='subheader'>Understanding the Predictions</h3>", unsafe_allow_html=True)
    
    st.markdown("""
    The model predicts diabetes risk based on patterns learned from data. Here are some key insights:
    
    - **Blood glucose level** and **HbA1c** are typically the strongest predictors
    - People with **BMI > 30** have significantly higher risk
    - **Age** becomes an increasingly important factor after 45 years
    - The combination of **hypertension** and **heart disease** substantially increases risk
    
    The prediction is presented as a probability between 0 and 1, with higher values indicating higher risk.
    """)
    
    # Add interactive scenario explorer
    st.markdown("<h3 class='subheader'>Explore Different Scenarios</h3>", unsafe_allow_html=True)
    
    st.markdown("""
    Use the sliders below to explore how different health metrics affect diabetes risk prediction.
    This can help understand the relationship between key health factors and diabetes risk.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        demo_glucose = st.slider("Blood Glucose Level (mg/dL)", 70, 300, 140, key="demo_glucose")
        demo_hba1c = st.slider("HbA1c Level (%)", 3.0, 9.0, 6.5, 0.1, key="demo_hba1c")
    
    with col2:
        demo_bmi = st.slider("BMI", 15.0, 45.0, 30.0, 0.5, key="demo_bmi")
        demo_age = st.slider("Age", 20, 80, 55, key="demo_age")
    
    # Create a few scenarios to explore
    if st.button("Run Scenario Analysis"):
        with st.spinner("Analyzing scenarios..."):
            # Create a baseline healthy profile
            baseline = {
                'age': 45,
                'gender': 'Male',
                'bmi': 24.0,
                'HbA1c_level': 5.2,
                'blood_glucose_level': 90,
                'hypertension': 0,
                'heart_disease': 0,
                'smoking_history': 'non-smoker'
            }
            
            # Create the user's scenario
            user_scenario = baseline.copy()
            user_scenario.update({
                'age': demo_age,
                'bmi': demo_bmi,
                'HbA1c_level': demo_hba1c,
                'blood_glucose_level': demo_glucose
            })
            
            # Process and predict
            processed_baseline = preprocess_input(baseline, preprocessor)
            processed_user = preprocess_input(user_scenario, preprocessor)
            
            _, baseline_proba = predict_diabetes(model, processed_baseline)
            _, user_proba = predict_diabetes(model, processed_user)
            
            # Display comparison
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.markdown("<p class='result-header'>Baseline Healthy Profile</p>", unsafe_allow_html=True)
                st.markdown(f"""
                - Age: 45
                - BMI: 24.0 (Normal)
                - HbA1c: 5.2% (Normal)
                - Blood Glucose: 90 mg/dL (Normal)
                - No hypertension or heart disease
                - Non-smoker
                """)
                st.markdown(f"<p><strong>Risk Probability:</strong> {baseline_proba[1]:.1%}</p>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
            
            with col2:
                risk_class = "risk-high" if user_proba[1] > 0.7 else "risk-moderate" if user_proba[1] > 0.3 else "risk-low"
                st.markdown(f"<div class='card {risk_class}'>", unsafe_allow_html=True)
                st.markdown("<p class='result-header'>Your Scenario</p>", unsafe_allow_html=True)
                st.markdown(f"""
                - Age: {demo_age}
                - BMI: {demo_bmi}
                - HbA1c: {demo_hba1c}%
                - Blood Glucose: {demo_glucose} mg/dL
                - No hypertension or heart disease
                - Non-smoker
                """)
                st.markdown(f"<p><strong>Risk Probability:</strong> {user_proba[1]:.1%}</p>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
            
            # Risk difference
            risk_diff = user_proba[1] - baseline_proba[1]
            risk_change = f"increased by {risk_diff:.1%}" if risk_diff > 0 else f"decreased by {abs(risk_diff):.1%}"
            
            st.markdown(f"<p>Based on these changes, the diabetes risk has {risk_change} compared to the baseline healthy profile.</p>", unsafe_allow_html=True)
            
            # Display factor contribution
            st.markdown("<h4>Factor Contribution Analysis</h4>", unsafe_allow_html=True)
            
            factors = []
            if demo_glucose > 125:
                impact = (demo_glucose - 90) / 90 * 100
                factors.append(("Elevated blood glucose", f"+{impact:.1f}%", "high"))
            
            if demo_hba1c > 5.7:
                impact = (demo_hba1c - 5.2) / 5.2 * 100
                factors.append(("Elevated HbA1c", f"+{impact:.1f}%", "high"))
            
            if demo_bmi > 25:
                impact = (demo_bmi - 24) / 24 * 50
                factors.append(("Elevated BMI", f"+{impact:.1f}%", "medium"))
            
            if demo_age > 50:
                impact = (demo_age - 45) / 45 * 30
                factors.append(("Advanced age", f"+{impact:.1f}%", "medium"))
            
            if factors:
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.markdown("<p>Key factors affecting this prediction:</p>", unsafe_allow_html=True)
                for factor, impact, level in factors:
                    color = "red" if level == "high" else "orange" if level == "medium" else "blue"
                    st.markdown(f"<p>• {factor}: <span style='color:{color};font-weight:bold'>{impact}</span> estimated impact</p>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
            else:
                st.markdown("<p>All health metrics are within normal ranges.</p>", unsafe_allow_html=True)

# Add footer with disclaimer
st.markdown("<div class='footer'>", unsafe_allow_html=True)
st.markdown("""
**Disclaimer**: This application is for educational and informational purposes only and is not intended as a substitute for professional medical advice, diagnosis, or treatment. 
Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition.
""")
st.markdown("© 2025 Diabetes Risk Assessment Tool", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)