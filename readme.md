## 🧠 Diabetes Risk Prediction Using Machine Learning

![Header](./assets/diabetes_header.png)

* **Initiative**: Digital Egypt Pioneers Initiative (DEPI)
* **Track**: Data Science
* **Group Code**: ALX2 AIS4 S10 Data Scientist
* **Project Supervisor**: Mahmoud Khorshed
* **Organization**: Ministry of Communications and Information Technology (MCIT)

### Team Members

* **Moustafa Rezk**
* **Moustafa Ashour**
* **Mahmoud Elsanhouri**
* **Seif Mountaser**
* **Abdelrahman Ata**
* **Mohammed Elzayat**
  
[🔗 Streamlit App](https://diabetespredictionapp-qhgxj9apfkxkxvxzjvvye8.streamlit.app)

[📂 Project Repo](https://github.com/rezk1834/diabetes_prediction_app)

---

### 📌 Overview

This project leverages machine learning to assess diabetes risk based on individual health indicators. By deploying an XGBoost classifier within a full data science pipeline, the model achieves near-clinical accuracy while remaining user-friendly through a Streamlit interface.

---

### 🚀 Objective

To build a predictive tool for diabetes diagnosis using a machine learning pipeline trained on patient health records. This tool supports early detection and informed clinical decision-making.

---

### 🧠 Technologies & Libraries

- **Python 3**
- **Pandas, NumPy** – Data manipulation
- **Scikit-learn, XGBoost** – ML models & evaluation
- **Matplotlib, Seaborn** – Visualizations
- **Streamlit** – Web application deployment

---


### 📊 Dataset Overview

* **Source:** [Kaggle - Diabetes Prediction Dataset](https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset)
* **Size:** 96,128 patient records
* **Target Variable:** Diabetes status (0 = No, 1 = Yes)

#### Key Features:

| Feature             | Type        | Description                       |
| ------------------- | ----------- | --------------------------------- |
| Age                 | Numerical   | Age in years (0.08–80)            |
| Gender              | Categorical | Male, Female                      |
| BMI                 | Numerical   | Body Mass Index (10–95.69)        |
| HbA1c Level         | Numerical   | Glycated hemoglobin % (3.5–9.0)   |
| Blood Glucose Level | Numerical   | Fasting glucose in mg/dL (80–300) |
| Hypertension        | Binary      | 0 = No, 1 = Yes                   |
| Heart Disease       | Binary      | 0 = No, 1 = Yes                   |
| Smoking History     | Categorical | never, current, former, No Info   |

---

### 🧹 Data Preprocessing

* **Duplicates Removed:** 3,854 rows (\~4%)

* **Missing Values:** None

* **Outlier Handling:** All values within plausible medical ranges

* **Categorical Recoding:**

  * Consolidated smoking history into 3 buckets (never, current/former, unknown)
  * Removed rare category “Other” from gender

* **Encoding:** One-Hot Encoding applied to categorical variables

* **Scaling:** StandardScaler applied to numerical features

* **Imbalance Treatment:** SMOTE used to oversample minority class

---

### 📈 Exploratory Data Analysis

EDA helped uncover critical patterns:

* **Age, BMI, and Blood Glucose** show strong right-skewed distributions
* **Diabetes prevalence** is skewed, necessitating class balance treatment
* **HbA1c and Glucose levels** were most predictive of diabetes status
* **Smoking history** showed nuanced patterns—"current/former" had higher prevalence than "never" or "unknown"

📷 *Examples of visualizations in the `/EDA/` folder include:*

* Age and BMI distributions
* Diabetes by glucose level
* Gender and smoking distributions
* Feature correlations

---

### 🛠️ Feature Engineering

* Created interaction terms between BMI, HbA1c, and glucose
* Assessed feature correlation and variance
* XGBoost feature importances confirmed:
  `Glucose > HbA1c > Age > BMI > Hypertension > Smoking > Heart Disease`

---

### 🤖 Model Development

Two models were tested:

* **Random Forest Classifier**
* **XGBoost Classifier** (selected)

**Why XGBoost?**

* Higher precision-recall balance
* Handles imbalanced data effectively
* Native support for feature importance

**Tuning:**

* GridSearchCV with cross-validation to tune:

  * `max_depth`, `learning_rate`, `n_estimators`, `subsample`

---

### 📊 Performance Metrics

| Metric    | Score  |
| --------- | ------ |
| Accuracy  | 97.61% |
| Precision | 96%    |
| Recall    | 96%    |
| F1 Score  | 98%    |

✅ Confusion matrix shows very low false positives/negatives, critical for healthcare models.

📌 Note: SMOTE was crucial for avoiding majority-class bias.

---
## 🌐 Deployment

The model is deployed via **Streamlit**, providing an interactive interface for users to input health parameters and receive real-time diabetes risk predictions.

### How to Run:
```bash
git clone https://github.com/rezk1834/diabetes_prediction_app
cd diabetes_prediction_app
pip install -r requirements.txt
streamlit run main_app.py
````

---

### 📁 Project Structure

```
diabetes_prediction_app/
├── .ipynb_checkpoints/
├── assets/
│   └── diabetes_header.png
├── EDA/
│   ├── [EDA Visualizations - 20+ PNGs]
├── models/
│   └── [Trained ML models]
├── Diabetes_DEPI.ipynb         ← Full notebook: EDA + Modeling
├── main_app.py                 ← Streamlit web app
├── utils.py                    ← Helper functions
├── style.css                   ← Custom UI styling
├── run_script.sh               ← Shell script for deployment
├── requirements.txt
└── readme.md
```


Key Features:

* Responsive layout for mobile and desktop
* Error handling for edge cases
* Lightweight backend with preloaded XGBoost model
* Deployed on Streamlit Cloud for public access

---

### 🔮 Future Directions

* Incorporate **diet, exercise, sleep** data from wearables
* Add time-series modeling for tracking patient risk progression
* Explore **Explainable AI (XAI)** frameworks like SHAP, LIME
* Integrate into **EHR systems** via APIs
* Mobile app version for larger-scale public access

---

### 📚 References

1. [Project Repository](https://github.com/rezk1834/diabetes_prediction_app)
2. [Deployed Streamlit App](https://diabetespredictionapp-qhgxj9apfkxkxvxzjvvye8.streamlit.app/)
3. [Kaggle Dataset](https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset)

---

### 📫 Contact

GitHub: [rezk1834](https://github.com/rezk1834)


