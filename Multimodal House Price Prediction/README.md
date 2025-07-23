
# 🏠 Multimodal House Price Prediction

Predict housing prices using a combination of **images** (bedroom, bathroom, frontal, kitchen) and **structured data** (bedrooms, bathrooms, area, zip code). This deep learning model uses both CNN and dense layers to make realistic price predictions.

---

## 🚀 Overview

Unlike traditional models that rely solely on numbers, this project integrates image data—similar to how a human evaluates a property. The model takes:

- 📊 Tabular data:  
  - `bed` (number of bedrooms)  
  - `bath` (number of bathrooms)  
  - `area` (square footage)  
  - `zip` (location indicator)

- 🖼️ Image data:  
  - `bathroom_image`  
  - `bedroom_image`  
  - `frontal_image`  
  - `kitchen_image`

These inputs are passed into a hybrid model that combines image and tabular features.

---

## 🧠 Model Summary

- **Image backbone**: EfficientNetB0 (pretrained, frozen)
- **Tabular layers**: Dense → BatchNorm → Dropout
- **Fusion**: All inputs concatenated and passed through dense layers
- **Loss**: Mean Absolute Error (MAE)
- **Output**: Final house price prediction

## Tech Stack

Python
TensorFlow / Keras
Flask
Scikit-learn
EfficientNetB0 (pretrained)

## Model Design
4 image branches using EfficientNetB0
1 tabular branch with dense layers
All merged and passed through fully connected layers
Output: House price

---

## 🔧 How to Use

### ✅ Training the Model

1. Install dependencies:
   ```bash
   pip install -r requirements.txt


## Run training:

python model.py
This will create:
multimodal_model.keras
scaler.save


## Launch Flask app
python app.py
Then open your browser to:
http://127.0.0.1:5000

## Notes
Don’t pass blank or negative values.
Images must exist and match filenames in CSV.
Keep image sizes moderate for faster predictions.
Use Chrome for best experience.


