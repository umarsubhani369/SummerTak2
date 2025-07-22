import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Load dataset
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer.csv")

# Columns to use
columns_to_use = [
    'gender', 'SeniorCitizen', 'tenure', 'InternetService', 'TechSupport',
    'Contract', 'PaymentMethod', 'MonthlyCharges', 'Churn'
]
df = df[columns_to_use].copy()

# Convert Churn to binary
df['Churn'] = df['Churn'].map({'No': 0, 'Yes': 1})

# Feature/target split
X = df.drop(columns='Churn')
y = df['Churn']

# Column types
categorical_cols = ['gender', 'InternetService', 'TechSupport', 'Contract', 'PaymentMethod']
numerical_cols = ['SeniorCitizen', 'tenure', 'MonthlyCharges']

# Preprocessing
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

numerical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler())
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numerical_transformer, numerical_cols),
    ("cat", categorical_transformer, categorical_cols)
])

# Pipeline with Random Forest + class_weight
pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(class_weight='balanced', random_state=42))
])

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Hyperparameter tuning (small grid)
param_grid = {
    "classifier__n_estimators": [100, 200],
    "classifier__max_depth": [5, 10, None]
}

grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Evaluation
y_pred = grid_search.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n✅ Model trained with RandomForestClassifier + class_weight.")
print(f"🔍 Accuracy on test set: {accuracy * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Export model
joblib.dump(grid_search.best_estimator_, "churn_model_pipeline.joblib")
print("\n💾 Model saved as: churn_model_pipeline.joblib")
