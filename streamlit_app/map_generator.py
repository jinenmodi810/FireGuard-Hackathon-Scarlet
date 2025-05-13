import pandas as pd
import joblib
import folium
import json
import os

# Paths (adjust if needed)
DATA_PATH = os.path.join("data", "illinois_fire_risk_dataset.json")
MODEL_PATH = os.path.join("models", "fire_risk_model_xgboost.joblib")
VEG_ENCODER_PATH = os.path.join("models", "vegetation_encoder.joblib")
RISK_ENCODER_PATH = os.path.join("models", "risk_encoder.joblib")
OUTPUT_MAP = os.path.join("outputs", "fire_risk_map.html")

# Load dataset
with open(DATA_PATH, "r") as f:
    data = json.load(f)
df = pd.DataFrame(data)

# Drop rows with missing values required for prediction
df = df.dropna(subset=["temperature_c", "humidity", "wind_kph", "vegetation_level"])

# Load model and encoders
model = joblib.load(MODEL_PATH)
veg_encoder = joblib.load(VEG_ENCODER_PATH)
risk_encoder = joblib.load(RISK_ENCODER_PATH)

# Encode vegetation_level
df["vegetation_level_encoded"] = veg_encoder.transform(df["vegetation_level"])

# Predict risk level
X = df[["temperature_c", "humidity", "wind_kph", "vegetation_level_encoded"]]
predictions = model.predict(X)
df["predicted_risk"] = risk_encoder.inverse_transform(predictions)

# Initialize folium map centered on Illinois
m = folium.Map(location=[40.0, -89.0], zoom_start=6, tiles="CartoDB positron")

# Define color mapping
color_map = {
    "Low ✅": "green",
    "Moderate ⚠️": "orange",
    "High 🚨": "red",
    "Extreme 🔥": "darkred"
}

# Plot each ZIP code on the map
for _, row in df.iterrows():
    folium.CircleMarker(
        location=(row["latitude"], row["longitude"]),
        radius=6,
        popup=f"ZIP: {row['zip']}<br>Risk: {row['predicted_risk']}",
        color=color_map.get(row["predicted_risk"], "blue"),
        fill=True,
        fill_opacity=0.8
    ).add_to(m)

# Make sure output folder exists
os.makedirs("outputs", exist_ok=True)

# Save the map
m.save(OUTPUT_MAP)
print(f"✅ Map generated and saved to {OUTPUT_MAP}")
