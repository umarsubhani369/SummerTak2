# SummerTak2 - ML Project Collection
This repository contains three end-to-end machine learning projects which are covering classification, regression, and NLP use cases. Each folder is self-contained and
deployable with either Flask or Streamlit.

## 1. Telco Customer Churn Predictor (Flask App)
Predicts whether a customer will churn based on contract type, services used, and account details.

### Model:
Logistic Regression with preprocessing pipeline
### UI:
Flask + Jinja2
### Input:
Customer features (contract, tech support, etc.)
### Output:
Churn or Not Churn with confidence %

---

## 2. Multimodal House Price Prediction
Predicts housing prices using both tabular data (bed, bath, area, zip) and images (bathroom,
bedroom, frontal, kitchen).
### Model:
EfficientNetB0 (images) + Dense layers (tabular) merged
### UI:
Flask
### Input:
Structured data + 4 images
### Output:
Predicted house price

---

## 3. Support Ticket Auto-Tagger (Streamlit App)
Tags support tickets using semantic similarity via a sentence transformer (all-MiniLM-L6-v2).
### Model:
SentenceTransformer + Cosine Similarity
### UI:
Streamlit
### Input:
Free-text ticket
### Output:
Top-3 most relevant tags

---
## Quick Start:
<pre> git clone https://github.com/umarsubhani369/SummerTak2.git
cd "SummerTak2" </pre>
### Choose the project folder you want to run:
Support Ticket Auto-Tagger/
Multimodal House Price Prediction/
End-to-End ML Pipeline with Scikit-learn Pipeline API/
cd "Your_Selected_Project"
pip install -r requirements.txt

---
## To run the app:
### For Streamlit apps
streamlit run app.py
### For Flask apps
python app.py

---
### Developed By:
Umer Hayat
GitHub: @umarsubhani369
Passionate about applied ML, model deployment, and building real-world AI systems.
