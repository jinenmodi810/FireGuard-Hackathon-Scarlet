from flask import Blueprint, render_template, request, jsonify
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
import requests
import os
from openai import OpenAI
from dotenv import load_dotenv

# Load environment
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

# 🔹 Single Blueprint for all routes
main = Blueprint("main", __name__, template_folder="templates")

# Load ML model and dataset
model = joblib.load("models/chicago_fire_risk_model.joblib")
df = pd.read_csv("data/cleaned_chicago_data.csv")

# ---------- UTILITIES ----------
def get_weather_data(query):
    WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")
    url = f"http://api.weatherapi.com/v1/current.json?key={WEATHER_API_KEY}&q={query}"
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
        print(f"Error fetching weather: {e}")
        return None

def get_ndvi_data(lat, lon):
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=ndvi,soil_moisture"
    try:
        res = requests.get(url)
        data = res.json()
        current = data.get("current", {})
        ndvi = current.get("ndvi")
        soil = current.get("soil_moisture")
        return ndvi, soil
    except Exception as e:
        print(f"Error fetching NDVI: {e}")
        return None, None

def classify_vegetation(ndvi):
    if ndvi is None:
        return "Unknown"
    elif ndvi < 0.2:
        return "Low"
    elif ndvi < 0.4:
        return "Medium"
    else:
        return "High"

def fire_risk_score(temp, humidity, wind, vegetation_level):
    score = 0
    if temp > 30: score += 2
    if humidity < 30: score += 2
    if wind > 20: score += 2
    veg_score = {"Low": 1, "Medium": 2, "High": 3, "Unknown": 0}.get(vegetation_level, 0)
    score += veg_score

    if score >= 7: return "Extreme 🔥"
    elif score >= 5: return "High 🚨"
    elif score >= 3: return "Moderate ⚠️"
    else: return "Low ✅"

# ---------- ROUTES ----------

@main.route("/")
def home():
    return render_template("index.html")

@main.route("/map")
def fire_map():
    return render_template("fire_risk_map.html")

@main.route("/predict", methods=["POST"])
def predict():
    zipcode = request.form.get("zipcode")
    row = df[df["ZIP Code"] == int(zipcode)].copy()

    if row.empty:
        return render_template("index.html", risk_level="❌ ZIP code not found.")

    row["Temperature"] = row["Temperature"].str.replace("°F", "").astype(float)
    row["Humidity"] = row["Humidity"].str.replace("%", "").astype(float)
    row["Wind Speed"] = row["Wind Speed"].str.extract(r"(\d+)").astype(float)
    row["Dryness Score"] = row["Dryness Score"].astype(float)

    row = row.rename(columns={
        "Temperature": "Temp_F",
        "Humidity": "Humidity",
        "Wind Speed": "Wind_kph",
        "Dryness Score": "SoilMoisture",
        "Vegetation Estimate": "NDVI"
    })

    if row["NDVI"].dtype == object:
        row["NDVI"] = LabelEncoder().fit_transform(row["NDVI"])

    features = ["Temp_F", "Humidity", "Wind_kph", "NDVI", "SoilMoisture"]
    prediction = model.predict(row[features])[0]

    label_map = {0: "Normal"}  # Extend if needed
    risk_label = label_map.get(prediction, "Unknown")

    return render_template("index.html", risk_level=f"🔥 Risk Level: {risk_label}")


@main.route("/firebot", methods=["POST"])
def firebot():
    data = request.json
    prompt = data.get("prompt", "")

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful fire safety assistant named FireBot. You help users with wildfire risk, safety tips, and reporting incidents."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=300
        )

        reply = response.choices[0].message.content.strip()
        return jsonify({"response": reply})

    except Exception as e:
        return jsonify({"response": "I'm currently unavailable. Please try again later.", "error": str(e)}), 500
    


@main.route("/predict", methods=["GET"])
def predict_from_location():
    lat = request.args.get("lat", type=float)
    lon = request.args.get("lon", type=float)
    if lat is None or lon is None:
        return jsonify({"error": "Missing latitude or longitude"}), 400

    # 1. 🌍 Get ZIP code using reverse geocoding (e.g., OpenCage or Nominatim)
    try:
        zip_code = None
        geocode_res = requests.get(
            f"https://nominatim.openstreetmap.org/reverse?lat={lat}&lon={lon}&format=json"
        )
        if geocode_res.status_code == 200:
            geo_data = geocode_res.json()
            zip_code = geo_data.get("address", {}).get("postcode")
    except Exception as e:
        print("Error in reverse geocoding:", e)
        zip_code = None

    # 2. Get weather data (already working function)
    weather = get_weather_data(f"{lat},{lon}")
    if not weather:
        return jsonify({"error": "Weather data not available"}), 500

    # 3. Get NDVI and Soil moisture
    ndvi, soil = get_ndvi_data(lat, lon)

    # 4. Classify vegetation
    vegetation = classify_vegetation(ndvi)

    # 5. Predict fire risk
    risk = fire_risk_score(
        temp=weather["temp_c"],
        humidity=weather["humidity"],
        wind=weather["wind_kph"],
        vegetation_level=vegetation
    )

    return jsonify({
        "zip_code": zip_code,
        "latitude": lat,
        "longitude": lon,
        "temperature": weather["temp_c"],
        "humidity": weather["humidity"],
        "wind_kph": weather["wind_kph"],
        "condition": weather["condition"],
        "ndvi": ndvi,
        "soil_moisture": soil,
        "vegetation_level": vegetation,
        "risk_label": risk
    })