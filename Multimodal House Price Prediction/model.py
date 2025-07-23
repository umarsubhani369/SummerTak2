import pandas as pd
import numpy as np
import joblib
import os
import tensorflow as tf
from tensorflow.keras import layers, Model, Input
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt

CSV_PATH = "houses.csv"
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20

df = pd.read_csv(CSV_PATH)

# Tabular features
tabular_features = df[['bed', 'bath', 'area', 'zip']].values
target = df['price'].values

scaler = StandardScaler()
tabular_features = scaler.fit_transform(tabular_features)

def load_and_preprocess_image(path):
    img = load_img(path, target_size=IMAGE_SIZE)
    img_array = img_to_array(img)
    return preprocess_input(img_array)

def extract_image_features(df, column):
    return np.array([load_and_preprocess_image(path) for path in df[column]])

# Order: bathroom, bedroom, frontal, kitchen
bathroom_imgs = extract_image_features(df, 'bathroom_image')
bedroom_imgs = extract_image_features(df, 'bedroom_image')
frontal_imgs  = extract_image_features(df, 'frontal_image')
kitchen_imgs  = extract_image_features(df, 'kitchen_image')

X_tab_train, X_tab_test, bath_train, bath_test, bed_train, bed_test, front_train, front_test, kit_train, kit_test, y_train, y_test = train_test_split(
    tabular_features, bathroom_imgs, bedroom_imgs, frontal_imgs, kitchen_imgs, target, test_size=0.2, random_state=42
)

def build_image_branch():
    base = EfficientNetB0(include_top=False, input_shape=(224, 224, 3), pooling='avg', weights='imagenet')
    base.trainable = False
    inp = Input(shape=(224, 224, 3))
    x = base(inp)
    x = layers.Dense(64, activation='relu')(x)
    return Model(inputs=inp, outputs=x)

# Build CNNs
bath_model = build_image_branch()
bed_model  = build_image_branch()
front_model = build_image_branch()
kit_model = build_image_branch()

# Inputs
bath_input = Input(shape=(224, 224, 3))
bed_input  = Input(shape=(224, 224, 3))
front_input = Input(shape=(224, 224, 3))
kit_input = Input(shape=(224, 224, 3))
tabular_input = Input(shape=(X_tab_train.shape[1],))

# Process each input
bath_feat = bath_model(bath_input)
bed_feat  = bed_model(bed_input)
front_feat = front_model(front_input)
kit_feat   = kit_model(kit_input)

x_tab = layers.Dense(64, activation='relu')(tabular_input)
x_tab = layers.BatchNormalization()(x_tab)
x_tab = layers.Dropout(0.3)(x_tab)
x_tab = layers.Dense(32, activation='relu')(x_tab)

# Merge everything
combined = layers.concatenate([bath_feat, bed_feat, front_feat, kit_feat, x_tab])
x = layers.Dense(128, activation='relu')(combined)
x = layers.Dropout(0.3)(x)
x = layers.Dense(64, activation='relu')(x)
x = layers.Dense(32, activation='relu')(x)
output = layers.Dense(1)(x)

model = Model(inputs=[bath_input, bed_input, front_input, kit_input, tabular_input], outputs=output)
model.compile(optimizer='adam', loss='mae', metrics=['mae'])

# Train
model.fit(
    [bath_train, bed_train, front_train, kit_train, X_tab_train],
    y_train,
    validation_split=0.1,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE
)

# Evaluate
predictions = model.predict([bath_test, bed_test, front_test, kit_test, X_tab_test])
print("MAE:", mean_absolute_error(y_test, predictions))
print("RMSE:", np.sqrt(mean_squared_error(y_test, predictions)))

# Save
model.save("multimodal_model.keras")
joblib.dump(scaler, "scaler.save")
