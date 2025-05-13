import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from PIL import Image
import io
import base64
import time
from utils import (
load_model_and_preprocessor, preprocess_input, predict_diabetes, display_feature_importance, 
display_risk_factors, generate_health_recommendations, recategorize_smoking
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
tab1, tab2, tab3, tab4 = st.tabs(["Risk Assessment", "About Diabetes", "Model Insights", "Exploratory Data Analysis"])

with tab1: 
    # Sidebar for inputs
    st.sidebar.markdown("<h2 class='subheader'>Enter Your Health Information</h2>", unsafe_allow_html=True)

    age = st.sidebar.slider("Age", 0, 100, 45)
    gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
    bmi = st.sidebar.slider("BMI", 10.0, 50.0, 25.0, 0.01)

    if st.sidebar.checkbox("Calculate BMI"):
        height_cm = st.sidebar.number_input("Height (cm)", min_value=100, max_value=250, value=170)
        weight_kg = st.sidebar.number_input("Weight (kg)", min_value=30, max_value=300, value=70)
        if height_cm and weight_kg:
            height_m = height_cm / 100
            calculated_bmi = weight_kg / (height_m ** 2)
            bmi = round(calculated_bmi, 1)
            st.sidebar.write(f"Calculated BMI: **{bmi}**")

    hba1c = st.sidebar.slider("HbA1c Level (%)", 3.0, 9.0, 5.5, 0.1)
    blood_glucose = st.sidebar.slider("Blood Glucose Level (mg/dL)", 70, 300, 110, 1)

    col1, col2 = st.sidebar.columns(2)
    with col1:
        hypertension = st.selectbox("Hypertension", ["No", "Yes"])
        hypertension = 1 if hypertension == "Yes" else 0

    with col2:
        heart_disease = st.selectbox("Heart Disease", ["No", "Yes"])
        heart_disease = 1 if heart_disease == "Yes" else 0

    smoking_options = {
        "Never": "never",
        "Current smoker": "current",
        "Former smoker": "former",
        "No information": "No Info"
    }

    smoking_selected = st.sidebar.selectbox("Smoking History", list(smoking_options.keys()))
    smoking_history = recategorize_smoking(smoking_options[smoking_selected])

    predict_btn = st.sidebar.button("Predict Diabetes Risk")

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

    if predict_btn:
        with st.spinner("Analyzing your health data..."):
            time.sleep(1)

            processed_input = preprocess_input(input_data, preprocessor)
            prediction = predict_diabetes(model, processed_input)


            st.markdown("<h2 class='subheader'>🩺 Your Diabetes Risk Assessment</h2>", unsafe_allow_html=True)

            if prediction == 1:
                st.markdown("""
                <div style='text-align: center;'>
                    <p style="font-size: 4.5em;" class='result-header diabetes-positive'>⚠️ Higher Risk of Diabetes Detected</p>
                    <p>Based on your input, our model indicates a <strong>higher risk</strong> of developing diabetes.<br>This is <strong>not a diagnosis</strong>—please consult a healthcare provider for further evaluation.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style='text-align: center;'>
                    <p style="font-size: 4.5em;" class='result-header diabetes-negative'>✅ Lower Risk of Diabetes Detected</p>
                    <p>Your data suggests a <strong>lower risk</strong> of diabetes.<br>Keep up your healthy habits!</p>
                </div>
                """, unsafe_allow_html=True)

            risk_factors = display_risk_factors(input_data)
            
           # Display identified risk factors without background color or cards
            if risk_factors:
                st.markdown("<h3 class='subheader'>Identified Risk Factors</h3>", unsafe_allow_html=True)
                for factor in risk_factors:
                    st.markdown(f"<p class='risk-factor-item'>{factor}</p>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

 
            recommendations = generate_health_recommendations(prediction, risk_factors)

            st.markdown("<h3 class='subheader'>Health Recommendations</h3>", unsafe_allow_html=True)
            for i, recommendation in enumerate(recommendations):
                st.markdown(f"<p class='recommendation-item'>{recommendation}</p>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

    
    else:
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
    
    - **Accuracy**: 97.61%
    - **Precision**: 96%
    - **Recall**: 96%
    - **F1 Score**: 98%
    
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
    
    st.markdown("<h3 class='subheader'>Understanding the Predictions</h3>", unsafe_allow_html=True)
    
    st.markdown("""
    The model predicts diabetes risk based on patterns learned from data. Here are some key insights:
    
    - **Blood glucose level** and **HbA1c** are typically the strongest predictors
    - People with **BMI > 30** have significantly higher risk
    - **Age** becomes an increasingly important factor after 45 years
    - The combination of **hypertension** and **heart disease** substantially increases risk
    
    The prediction is presented as a probability between 0 and 1, with higher values indicating higher risk.
    """)
    

with tab4:
    st.markdown("<h2 class='subheader'>Exploratory Data Analysis</h2>", unsafe_allow_html=True)
    
    # Dataset Statistics Section
    st.markdown("""
    ## Dataset Overview
    
    Here are the key statistics about the dataset used for training the diabetes prediction model:
    """)
    
    # Display the statistics table
    st.markdown("### Basic Statistics")
    stats_df = pd.DataFrame({
        'Metric': ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max'],
        'age': [96128.00, 41.80, 22.46, 0.08, 24.00, 43.00, 59.00, 80.00],
        'hypertension': [96128.00, 0.08, 0.27, 0.00, 0.00, 0.00, 0.00, 1.00],
        'heart_disease': [96128.00, 0.04, 0.20, 0.00, 0.00, 0.00, 0.00, 1.00],
        'bmi': [96128.00, 27.32, 6.77, 10.01, 23.40, 27.32, 29.86, 95.69],
        'HbA1c_level': [96128.00, 5.53, 1.07, 3.50, 4.80, 5.80, 6.20, 9.00],
        'blood_glucose_level': [96128.00, 138.22, 40.91, 80.00, 100.00, 140.00, 159.00, 300.00],
        'diabetes': [96128.00, 0.09, 0.28, 0.00, 0.00, 0.00, 0.00, 1.00]
    })
    st.dataframe(stats_df.set_index('Metric'), use_container_width=True)
    
    # Data Quality Information
    st.markdown("""
    ## Data Quality Analysis
    
    ### Duplicate Rows
    - Number of duplicate rows: 3,854 (4% of dataset)
    
    ### Distinct Values
    - Gender: 3 distinct values (Male, Female, Other - 'Other' was removed as it was only 0.00195% of data)
    - Age: 102 distinct values
    - Hypertension: 2 distinct values (0, 1)
    - Heart Disease: 2 distinct values (0, 1)
    - Smoking History: 6 distinct values
    - BMI: 4,247 distinct values
    - HbA1c Level: 18 distinct values
    - Blood Glucose Level: 18 distinct values
    - Diabetes: 2 distinct values (0, 1)
    
    ### Missing Values
    - No missing values in any columns
    """)
    
    st.markdown("""
    ## Dataset Insights
    
    This section provides visual insights from the dataset used to train the diabetes prediction model.
    The visualizations help understand the relationships between different health factors and diabetes risk.
    """)
    
    # Create a dropdown to select different visualizations
    eda_options = {
        "Pairplot of Features": "pairplot.png",
        "BMI Distribution": "BMI Distribution.png",
        "Gender Distribution": "Gender Distribution.png",
        "Diabetes Distribution": "Diabetes Distribution.png",
        "Smoking History Distribution": "Smoking History Distribution.png",
        "Age Distribution": "Age Distribution.png",
        "Age vs Diabetes": "Age vs Diabetes.png",
        "BMI vs Diabetes": "BMI vs Diabetes.png",
        "Blood Glucose vs Diabetes": "Blood Glucose Level vs Diabetes.png",
        "HbA1c Level vs Diabetes": "HbA1c level vs Diabetes.png",
        "Gender vs Diabetes": "Gender vs Diabetes.png",
        "Age Distribution by Diabetes Status and Gender": "Age Distribution by Diabetes Status and Gender.png",
        "BMI Distribution by Diabetes Status and Gender": "BMI Distribution by Diabetes Status and Gender.png",
        "BMI vs Diabetes split by Gender": "BMI vs Diabetes split by Gender.png",
        "Age vs BMI": "Age vs BMI.png",
        "Correlation Matrix Heatmap": "Correlation Matrix Heatmap.png",
        "Correlation with Diabetes": "Correlation with Diabetes.png",
        "Feature Importance": "xgb_feature_importances.png",
        "Confusion Matrix": "confusion_matrix.png"
    }
    
    selected_eda = st.selectbox("Select Visualization", list(eda_options.keys()))
    
    # Display the selected visualization
    try:
        image_path = f"EDA/{eda_options[selected_eda]}"
        st.image(image_path, use_container_width=True)  # Updated parameter here
        
        # Add descriptions for each visualization
        if selected_eda == "Age Distribution":
            st.markdown("""
            **Insight**: Shows the distribution of ages in the dataset. Diabetes risk typically increases with age.
            """)
        elif selected_eda == "BMI Distribution":
            st.markdown("""
            **Insight**: Displays the distribution of BMI values. Higher BMI is associated with increased diabetes risk.
            """)
        elif selected_eda == "Gender Distribution":
            st.markdown("""
            **Insight**: Shows the gender distribution in the dataset. Some studies suggest gender differences in diabetes prevalence.
            """)
        elif selected_eda == "Diabetes Distribution":
            st.markdown("""
            **Insight**: Illustrates the proportion of diabetes cases in the dataset. This helps understand the class balance.
            """)
        elif selected_eda == "Age vs Diabetes":
            st.markdown("""
            **Insight**: Visualizes the relationship between age and diabetes status. Older individuals tend to have higher diabetes prevalence.
            """)
        elif selected_eda == "BMI vs Diabetes":
            st.markdown("""
            **Insight**: Shows how BMI relates to diabetes status. Higher BMI values correlate with increased diabetes risk.
            """)
        elif selected_eda == "Blood Glucose vs Diabetes":
            st.markdown("""
            **Insight**: Demonstrates the strong relationship between blood glucose levels and diabetes status.
            """)
        elif selected_eda == "HbA1c Level vs Diabetes":
            st.markdown("""
            **Insight**: HbA1c is a key indicator of long-term blood glucose control and diabetes status.
            """)
        elif selected_eda == "Feature Importance":
            st.markdown("""
            **Insight**: Shows which features were most important in the XGBoost model's predictions.
            """)
        elif selected_eda == "Confusion Matrix":
            st.markdown("""
            **Insight**: Illustrates the model's performance in terms of true positives, false positives, etc.
            """)
        elif selected_eda == "Pairplot of Features":
            st.markdown("""
            **Insight**: Displays pairwise relationships between multiple features in the dataset.
            """)
            
    except Exception as e:
        st.error(f"Error loading visualization: {str(e)}")
    
    # Add a section with key findings
    st.markdown("""
    ## Key Findings from EDA
    
    1. **Age and Diabetes Risk**: The data shows a clear increase in diabetes prevalence with age, particularly after 45 years.
    
    2. **BMI Impact**: Individuals with BMI > 30 have significantly higher diabetes rates compared to those with normal BMI.
    
    3. **Blood Glucose Levels**: As expected, higher blood glucose levels are strongly associated with diabetes.
    
    4. **HbA1c Importance**: HbA1c levels show a clear threshold effect around 5.7% (prediabetes) and 6.5% (diabetes).
    
    5. **Gender Differences**: The dataset shows some gender-based differences in diabetes prevalence and risk factor distributions.
    
    6. **Comorbid Conditions**: Presence of hypertension or heart disease is associated with higher diabetes rates.
    """)
    
    # Add interactive correlation explorer
    st.markdown("<h3 class='subheader'>Feature Correlation Explorer</h3>", unsafe_allow_html=True)
    
    st.markdown("""
    The heatmap below shows correlations between different features in the dataset.
    Stronger correlations (positive or negative) are indicated by darker colors.
    """)
    
    # Display the correlation visualization
    try:
        st.image("EDA/Correlation with Diabetes.png", use_container_width=True)  # Updated parameter here
    except:
        st.warning("Correlation visualization not available")
    
    st.markdown("""
    ### Interpretation Guide:
    
    - **+1.0**: Perfect positive correlation (as one increases, the other increases)
    - **0.0**: No correlation
    - **-1.0**: Perfect negative correlation (as one increases, the other decreases)
    
    In diabetes risk assessment:
    - Blood glucose and HbA1c show strong positive correlation with diabetes
    - Age and BMI show moderate positive correlation
    - Some features show little correlation with diabetes but may be important in combination with others
    """)
# Add footer 
st.markdown("<div class='footer'>", unsafe_allow_html=True)
st.markdown("""
This project, developed under the Digital Egypt Pioneers Initiative (DEPI), is centered around predictive modeling in healthcare—specifically, 
            the classification of diabetes risk for individuals. Diabetes remains one of the most prevalent and debilitating diseases worldwide, 
            and early risk detection, combined with personalized health recommendations, can significantly improve patient management and outcomes. 
            The aim of this project is to harness the power of data science to support healthcare professionals in making more informed decisions through 
            data-driven insights, ultimately helping individuals at risk of diabetes to better manage their health.
""")
