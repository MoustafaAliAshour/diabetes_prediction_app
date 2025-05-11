# Diabetes Risk Prediction App

An interactive Streamlit application for diabetes risk assessment based on health metrics and XGBoost machine learning model.

![Diabetes Risk Prediction App](assets/diabetes_header.png)

## Features

- Interactive health data input with intuitive UI
- Real-time diabetes risk prediction
- Personalized risk assessment with visual indicators
- Health recommendations based on identified risk factors
- Educational content about diabetes
- Model insights and feature importance visualization
- Scenario explorer to understand risk factors

## Getting Started

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

### Installation

1. Clone this repository:
   ```
   git clone https://github.com/rezk1834/diabetes-prediction-app.git
   cd diabetes-prediction-app
   ```

2. Run the setup script:
   ```
   bash run.sh
   ```

   This script will:
   - Create a virtual environment
   - Install the required dependencies
   - Start the Streamlit application

   Alternatively, you can set up manually:

   ```
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   streamlit run app.py
   ```

3. The application will open in your default web browser at `http://localhost:8501`

### Model Files

The application requires two model files in the `models/` directory:
- `xgboost_diabetes_model.pkl` - The trained XGBoost model
- `preprocessor_pipeline.pkl` - The preprocessing pipeline

These files are generated from the Jupyter notebook analysis and should be placed in the models directory before running the application.

## Project Structure

```
diabetes_prediction_app/
│
├── app.py                  # Main Streamlit application
├── utils.py                # Helper functions
├── style.css               # Custom CSS for styling
├── requirements.txt        # Project dependencies
├── run.sh                  # Setup and run script
├── README.md               # Project documentation
├── models/                 # Saved model files
│   ├── xgboost_diabetes_model.pkl
│   └── preprocessor_pipeline.pkl
└── assets/                 # Images and other static assets
    └── diabetes_header.png
```

## Usage

1. Enter your health information in the sidebar:
   - Age
   - Gender
   - BMI (or calculate from height and weight)
   - HbA1c level
   - Blood glucose level
   - Hypertension status
   - Heart disease status
   - Smoking history

2. Click the "Predict Diabetes Risk" button to get your assessment

3. View your results, including:
   - Risk prediction with probability
   - Identified risk factors
   - Personalized health recommendations

4. Explore the other tabs for:
   - Educational content about diabetes
   - Model insights and feature importance
   - Interactive scenario exploration

## Customization

You can customize the application by:

1. Modifying the `style.css` file to change the appearance
2. Adding new visualizations or content in the `app.py` file
3. Updating the health recommendations logic in `utils.py`

## Deployment

The application can be deployed to Streamlit Cloud, Heroku, or any other platform that supports Python applications.

For Streamlit Cloud:
1. Push your code to a GitHub repository
2. Log in to [Streamlit Cloud](https://streamlit.io/cloud)
3. Create a new app and connect it to your repository

## Important Note

This application is for educational purposes only and should not replace professional medical advice. The predictions are based on a machine learning model and should be considered as general guidance rather than a definitive diagnosis.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Data source: [Kaggle Diabetes Prediction Dataset](https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset)
- Built with [Streamlit](https://streamlit.io/)"# -diabetes_prediction_app" 
