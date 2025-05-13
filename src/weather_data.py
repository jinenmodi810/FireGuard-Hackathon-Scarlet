import pandas as pd
import os
import requests
import json
from dotenv import load_dotenv

# Load your .env containing WEATHER_API_KEY
load_dotenv(dotenv_path=".env")
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")

# Step 1: Load ZIP CSV
df = pd.read_csv("zip_code_database.csv")
il_df = df[df['state'] == 'IL'][['zip', 'primary_city', 'latitude', 'longitude', 'country']].dropna().reset_index(drop=True)

# Optional: Reduce the sample size for testing
#il_df = il_df.head(10)  # You can increase this once you're ready

# Step 2: API helper functions
def get_weather_data(zipcode):
    url = f"http://api.weatherapi.com/v1/current.json?key={WEATHER_API_KEY}&q={zipcode}"
    try:
        response = requests.get(url)
        data = response.json()
        if "error" in data:
            return None
        return {
            "temp_c": data["current"]["temp_c"],
            "humidity": data["current"]["humidity"],
            "wind_kph": data["current"]["wind_kph"],
            "condition": data["current"]["condition"]["text"],
            "location": data["location"]["name"]
        }
    except Exception as e:
        print(f"Weather error for ZIP {zipcode}: {e}")
        return None

def get_ndvi_data(lat, lon):
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=ndvi,soil_moisture"
    try:
        res = requests.get(url)
        data = res.json()
        ndvi = data["current"].get("ndvi")
        soil = data["current"].get("soil_moisture")
        return ndvi, soil
    except Exception as e:
        print(f"NDVI error for {lat}, {lon}: {e}")
        return None, None

def classify_vegetation(ndvi):
    if ndvi is None:
        return "Unknown"
    elif ndvi < 0.2:
        return "Low"
    elif ndvi < 0.4:
        return "High"
    else:
        return "Medium"

def fire_risk_score(temp, humidity, wind, vegetation_level):
    score = 0
    if temp > 30: score += 2
    if humidity < 30: score += 2
    if wind > 20: score += 2
    veg_score = {"Low": 1, "Medium": 2, "High": 3, "Unknown": 0}.get(vegetation_level, 0)
    score += veg_score

    if score >= 7:
        return "Extreme 🔥"
    elif score >= 5:
        return "High 🚨"
    elif score >= 3:
        return "Moderate ⚠️"
    else:
        return "Low ✅"

# Step 3: Build the dataset
dataset = []

for _, row in il_df.iterrows():
    zipcode = row['zip']
    lat, lon = row['latitude'], row['longitude']
    weather = get_weather_data(zipcode)
    ndvi, soil = get_ndvi_data(lat, lon)
    vegetation = classify_vegetation(ndvi)

    if weather:
        risk = fire_risk_score(weather["temp_c"], weather["humidity"], weather["wind_kph"], vegetation)
        dataset.append({
            "zip": zipcode,
            "city": row["primary_city"],
            "latitude": lat,
            "longitude": lon,
            "country": row["country"],
            "temperature_c": weather["temp_c"],
            "humidity": weather["humidity"],
            "wind_kph": weather["wind_kph"],
            "condition": weather["condition"],
            "ndvi": ndvi,
            "soil_moisture": soil,
            "vegetation_level": vegetation,
            "risk_label": risk
        })

# Step 4: Save to JSON
with open("illinois_fire_risk_dataset.json", "w") as f:
    json.dump(dataset, f, indent=4)

print("✅ Dataset saved to: illinois_fire_risk_dataset.json")
