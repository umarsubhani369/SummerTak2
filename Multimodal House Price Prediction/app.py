import os
import numpy as np
from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.efficientnet import preprocess_input
from sklearn.preprocessing import StandardScaler
import joblib

app = Flask(__name__)
STATIC_DIR = "static"
os.makedirs(STATIC_DIR, exist_ok=True)

model = load_model("multimodal_model.keras")
scaler = joblib.load("scaler.save")

IMAGE_SIZE = (224, 224)
TABULAR_COLUMNS = ['bed', 'bath', 'area', 'zip']

def process_image(path):
    img = load_img(path, target_size=IMAGE_SIZE)
    img_array = img_to_array(img)
    return preprocess_input(img_array)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Tabular input
        try:
            bed = float(request.form["bed"])
            bath = float(request.form["bath"])
            area = float(request.form["area"])
            zip_code = float(request.form["zip"])
        except:
            return "Invalid input"

        tabular = np.array([[bed, bath, area, zip_code]])
        tabular_scaled = scaler.transform(tabular).reshape(1, -1)

        # Correct order: bathroom, bedroom, frontal, kitchen
        image_order = ["bathroom", "bedroom", "frontal", "kitchen"]
        images = []

        for field in image_order:
            file = request.files[field]
            path = os.path.join(STATIC_DIR, f"{field}.jpg")
            file.save(path)
            img_array = process_image(path)
            images.append(np.expand_dims(img_array, axis=0))

        prediction = model.predict(images + [tabular_scaled])[0][0]
        predicted_price = round(prediction, 2)

        return render_template("index.html", prediction=predicted_price)

    return render_template("index.html", prediction=None)

if __name__ == "__main__":
    app.run(debug=True)
