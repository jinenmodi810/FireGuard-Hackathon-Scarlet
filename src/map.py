#!/usr/bin/env python
import pandas as pd
import folium
from folium.plugins import HeatMap, MarkerCluster
import joblib
import os
import re
import branca.colormap as cm

def main():
    # File paths
    data_path = "data/cleaned_state_data.csv"
    model_path = "models/fire_risk_model_xgboost.joblib"  # Your trained model filename
    output_map_path = "src/static/fire_risk_map.html"

    # Load data and model
    df = pd.read_csv(data_path)
    model = joblib.load(model_path)

    # Preprocess features:
    # Clean Temperature: remove "°F" and convert to float if needed
    if df["Temperature"].dtype == object:
        df["Temperature"] = df["Temperature"].str.replace("°F", "").astype(float)
    # Clean Humidity: remove "%" and convert to float
    if df["Humidity"].dtype == object:
        df["Humidity"] = df["Humidity"].str.replace("%", "").astype(float)
    # Clean Wind Speed: extract the numeric part and convert to float
    if df["Wind Speed"].dtype == object:
        df["Wind Speed"] = df["Wind Speed"].apply(
            lambda x: float(re.findall(r"\d+", str(x))[0]) if pd.notnull(x) and re.search(r"\d+", str(x)) else 0.0
        )
    # Ensure Dryness Score is numeric
    df["Dryness Score"] = df["Dryness Score"].astype(float)

    # Use original column names for features.
    # One-hot encode "Vegetation Estimate" if not already encoded
    if "Vegetation Estimate" in df.columns and not any(col.startswith("Vegetation Estimate_") for col in df.columns):
        df = pd.get_dummies(df, columns=["Vegetation Estimate"], prefix="Vegetation Estimate")
    
    # Set the required features exactly as used during training.
    required_features = [
        "Temperature", "Humidity", "Wind Speed", "Barometer", "Dewpoint",
        "Visibility", "Wind Chill",
        "Vegetation Estimate_Forested/Parkland",
        "Vegetation Estimate_Mixed/Unknown",
        "Vegetation Estimate_Sparse/Suburban",
        "Vegetation Estimate_Urban/Concrete"
    ]
    
    # Ensure all required features exist; if missing, create them with default value 0.0.
    for feature in required_features:
        if feature not in df.columns:
            print(f"[!] Warning: Feature '{feature}' not found in the data. Creating it with default value 0.0.")
            df[feature] = 0.0
    df[required_features] = df[required_features].fillna(df[required_features].mean())

    # Predict continuous risk using the trained model.
    df["Prediction"] = model.predict(df[required_features])
    
    # Create a linear colormap for the heatmap overlay.
    # We assume the prediction scale is roughly 0 (low) to 3 (extreme).
    colormap = cm.LinearColormap(colors=["green", "yellow", "red"], vmin=0, vmax=3)
    
    # Optionally, generate a discrete risk label for markers.
    def risk_label(score):
        if score < 1:
            return "Normal"
        elif score < 2:
            return "Dry"
        elif score < 2.5:
            return "High Risk"
        else:
            return "Extreme"
    df["RiskLabel"] = df["Prediction"].apply(risk_label)

    # Drop rows without geographic coordinates
    df = df.dropna(subset=["Latitude", "Longitude"])

    # Create a Folium map centered on Chicago
    chicago_map = folium.Map(location=[41.8781, -87.6298], zoom_start=10)

    # --- HeatMap Layer (Transparent Gradient) ---
    # Prepare heatmap data: each entry is [lat, lon, intensity]
    heat_data = [
        [row["Latitude"], row["Longitude"], row["Prediction"]]
        for _, row in df.iterrows()
    ]
    # Convert gradient keys to strings to avoid the 'split' error
    gradient = {str(0.0): 'green', str(0.5): 'yellow', str(1.0): 'red'}
    HeatMap(
        heat_data,
        min_opacity=0.3,  # 30% opacity for a high transparency effect
        max_val=3,        # Prediction scale maximum value
        radius=25,        # Increase radius for a smoother heat
        blur=20,          # Increase blur for softer edges
        gradient=gradient
    ).add_to(chicago_map)
    
    # --- Marker Cluster Layer (Individual Dots) ---
    marker_cluster = MarkerCluster().add_to(chicago_map)
    for _, row in df.iterrows():
        marker_color = colormap(row["Prediction"])
        popup_content = (
            f"ZIP: {row['ZIP Code']}<br>"
            f"Risk: {row['RiskLabel']}<br>"
            f"Score: {row['Prediction']:.2f}"
        )
        folium.CircleMarker(
            location=(row["Latitude"], row["Longitude"]),
            radius=6,
            popup=popup_content,
            color=marker_color,
            fill=True,
            fill_color=marker_color,
            fill_opacity=0.8,
            opacity=0.8
        ).add_to(marker_cluster)
    
    # Add colormap legend to the map
    colormap.caption = "Fire Risk Score (0 = Low, 3 = Extreme)"
    chicago_map.add_child(colormap)

    # Ensure output directory exists and save the map
    os.makedirs(os.path.dirname(output_map_path), exist_ok=True)
    chicago_map.save(output_map_path)
    print(f"✅ Map saved to: {output_map_path}")

if __name__ == "__main__":
    main()