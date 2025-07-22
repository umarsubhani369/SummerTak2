
# 🔮 Telco Customer Churn Predictor (Flask App)

A machine learning web app built with **Flask** to predict whether a Telco customer will churn based on contract type, tech support, internet service, and other customer details.

---

## 📂 Foldeer Structure
📁 End-to-End ML Pipeline with Scikit-learn Pipeline API/
├── app.py # Flask application backend

├── churn_model_pipeline.joblib # Trained ML pipeline

├── templates/

│ └── index.html # Web form for input (Jinja2)

├── WA_Fn-UseC_-Telco-Customer-Churn.csv # Original dataset

└── README.md # You're here


## 🧠 Model Overview

- **Algorithm**: Logistic Regression (tuned via GridSearchCV)
- **Preprocessing**: Scaling + Encoding using `ColumnTransformer`
- **Exported Using**: `joblib`
- **Accuracy**: ~81%
- **Target Variable**: `Churn` (Yes/No)

---

## 🚀 How to Run the App

### 1. Install Dependencies
At bash
pip install -r requirements.txt
## Launch the Flask App
bash
python app.py
Then open in your browser:
📍 http://127.0.0.1:5000/
## Features
Clean two-column input layout (no sidebar)
Predicts churn in real-time
Shows confidence score
Fully browser-based — no setup beyond Python

## Prediction Output
If prediction = 1 → Customer is likely to churn
If prediction = 0 → Customer is likely to stay
Displays a confidence percentage based on model probability

## Developed By
Engr.Umer Hayat
