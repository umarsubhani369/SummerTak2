from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)
model = joblib.load("churn_model_pipeline.joblib")

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        form = request.form
        data = pd.DataFrame([{
            "gender": form["gender"],
            "SeniorCitizen": 1 if form["senior"] == "Yes" else 0,
            "tenure": int(form["tenure"]),
            "InternetService": form["internet"],
            "TechSupport": form["tech_support"],
            "Contract": form["contract"],
            "PaymentMethod": form["payment"],
            "MonthlyCharges": float(form["monthly_charges"])
        }])
        prediction = model.predict(data)[0]
        proba = model.predict_proba(data)[0][prediction] * 100
        result = "Customer is likely to churn." if prediction == 1 else "Customer is likely to stay."
        confidence = f"Confidence: {proba:.2f}%"
        return render_template("index.html", prediction=result, confidence=confidence)

    return render_template("index.html")
if __name__ == "__main__":
    app.run(debug=True)