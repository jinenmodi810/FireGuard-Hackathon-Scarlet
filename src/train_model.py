import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import joblib
import os


def train_model():
    data_path = "data/cleaned_chicago.csv"
    output_model_path = "models/chicago_fire_risk_model.joblib"

    # Load data
    df = pd.read_csv(data_path)

    # Preprocess features
    if df["Temperature"].dtype != "object":
        df["Temperature"] = df["Temperature"].astype(str)  # Convert to string if not already
    df["Temperature"] = df["Temperature"].str.replace("°F", "", regex=False)

    # Handle missing or invalid values
    df["Temperature"] = pd.to_numeric(df["Temperature"], errors="coerce")  # Convert to float, set invalid values to NaN
    df["Temperature"].fillna(df["Temperature"].mean(), inplace=True)  # Replace NaN with the column mean

    df["Humidity"] = df["Humidity"].astype(float)
    df["Wind Speed"] = df["Wind Speed"].astype(float)
    df["Dryness Score"] = df["Dryness Score"].astype(float)

    # Rename for consistency
    df = df.rename(columns={
        "Temperature": "Temp_F",
        "Humidity": "Humidity",
        "Wind Speed": "Wind_kph",
        "Dryness Score": "SoilMoisture",
        "Vegetation Estimate": "NDVI"
    })

    # Encode vegetation labels (if categorical)
    if df["NDVI"].dtype == object:
        df["NDVI"] = LabelEncoder().fit_transform(df["NDVI"])

    # Features and target
    features = ["Temp_F", "Humidity", "Wind_kph", "NDVI", "SoilMoisture"]
    target = "Dryness Level"

    X = df[features]
    y = df[target]

    # Encode target labels
    y_encoded = LabelEncoder().fit_transform(y)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    print("\n✅ Model Evaluation Report:")
    print(classification_report(y_test, y_pred))

    # Save model
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, output_model_path)
    print(f"\n📦 Model saved to: {output_model_path}")


if __name__ == "__main__":
    train_model()