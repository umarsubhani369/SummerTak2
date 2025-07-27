# SummerTak2 - ML Project Collection
This repository contains three end-to-end machine learning projects which are covering classification, regression, and NLP use cases. Each folder is self-contained and
deployable with either Flask or Streamlit.

## 1. Telco Customer Churn Predictor (Flask App)
Predicts whether a customer will churn based on contract type, services used, and account details.
Model: Logistic Regression with preprocessing pipeline
UI: Flask + Jinja2
Input: Customer features (contract, tech support, etc.)
Output: Churn or Not Churn with confidence %
Quick Start:
cd "End-to-End ML Pipeline with Scikit-learn Pipeline API"
pip install -r requirements.txt
python app.py
Open http://127.0.0.1:5000/
## 2. Multimodal House Price Prediction
Predicts housing prices using both tabular data (bed, bath, area, zip) and images (bathroom,
bedroom, frontal, kitchen).
Model: EfficientNetB0 (images) + Dense layers (tabular) merged
UI: Flask
Input: Structured data + 4 images
Output: Predicted house price
Quick Start:
cd "Multimodal House Price Prediction"
pip install -r requirements.txt
python model.py # Train the model
python app.py # Run Flask app
Open http://127.0.0.1:5000/
## 3. Support Ticket Auto-Tagger (Streamlit App)
Tags support tickets using semantic similarity via a sentence transformer (all-MiniLM-L6-v2).
Model: SentenceTransformer + Cosine Similarity
UI: Streamlit
Input: Free-text ticket
Output: Top-3 most relevant tags
Quick Start:
cd "Support Ticket Auto-Tagger"
pip install -r requirements.txt
streamlit run app.py
Requirements:
Each project contains its own requirements.txt file. Install dependencies per folder.
### Developed By:
Umer Hayat
GitHub: @umarsubhani369
Passionate about applied ML, model deployment, and building real-world AI systems.
