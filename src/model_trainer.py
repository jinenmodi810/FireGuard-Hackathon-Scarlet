import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import OneHotEncoder
import joblib
import matplotlib.pyplot as plt

def load_and_preprocess_data(filepath="data/cleaned_state_data.csv"):
    # Load the combined cleaned data from CSV
    df = pd.read_csv(filepath)
    
    # Drop rows with missing target variable
    df = df.dropna(subset=["Dryness Score"])
    
    # Identify potential numeric features (if they exist)
    potential_numeric = ["Temperature", "Humidity", "Wind Speed", "Barometer", "Dewpoint", "Visibility", "Wind Chill"]
    numeric_features = [col for col in potential_numeric if col in df.columns]
    
    # Fill missing values for numeric features with the mean
    X_numeric = df[numeric_features].fillna(df[numeric_features].mean())
    
    # If the dataset contains a categorical column (e.g., Vegetation Estimate), one-hot encode it
    if "Vegetation Estimate" in df.columns:
        X_cat = df[["Vegetation Estimate"]].fillna("Unknown")
        encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        X_cat_encoded = encoder.fit_transform(X_cat)
        cat_feature_names = encoder.get_feature_names_out(["Vegetation Estimate"])
        X_cat_df = pd.DataFrame(X_cat_encoded, columns=cat_feature_names, index=X_numeric.index)
        X = pd.concat([X_numeric, X_cat_df], axis=1)
    else:
        X = X_numeric.copy()
    
    # The target variable is "Dryness Score"
    y = df["Dryness Score"]
    
    # Quick data inspection for leakage
    print("Sample Features (X):\n", X.head())
    print("Sample Target (y):\n", y.head())
    
    return X, y

def train_model(X, y, algorithm="xgboost"):
    # Split data into training and testing sets (80/20 split)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Select the algorithm based on user input
    if algorithm == "rf":
        model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    elif algorithm == "xgboost":
        model = XGBRegressor(n_estimators=150, max_depth=5, learning_rate=0.1, random_state=42, n_jobs=-1, objective="reg:squarederror")
    else:
        raise ValueError("Unsupported algorithm. Choose 'rf' or 'xgboost'.")
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Evaluate on training set (to check overfitting)
    y_train_pred = model.predict(X_train)
    train_r2 = r2_score(y_train, y_train_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    
    # Evaluate on test set
    y_test_pred = model.predict(X_test)
    test_r2 = r2_score(y_test, y_test_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    
    # Cross-validation for robustness
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
    
    # Print evaluation
    print("Model Evaluation:")
    print(f"  Training R²: {train_r2:.4f}")
    print(f"  Test R²: {test_r2:.4f}")
    print(f"  Test MAE: {test_mae:.4f}")
    print(f"  Test RMSE: {test_rmse:.4f}")
    print(f"  5-Fold CV R²: Mean = {cv_scores.mean():.4f}, Std = {cv_scores.std():.4f}")
    
    return model

def plot_feature_importances(model, X):
    # Get feature importances from the model and sort them
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    feature_names = X.columns[indices]
    
    plt.figure(figsize=(10, 6))
    plt.title("Feature Importances")
    plt.bar(range(len(importances)), importances[indices], align="center")
    plt.xticks(range(len(importances)), feature_names, rotation=90)
    plt.tight_layout()
    plt.show()

def main():
    # Load and preprocess the cleaned dataset
    X, y = load_and_preprocess_data("data/cleaned_state_data.csv")
    
    # Choose algorithm: "rf" for Random Forest or "xgboost" for XGBoost
    algorithm = "xgboost"  # Change this to "rf" if desired
    model = train_model(X, y, algorithm=algorithm)
    
    # Visualize feature importances (uncomment to use)
    plot_feature_importances(model, X)
    
    # Save the trained model
    model_filename = f"models/fire_risk_model_{algorithm}.joblib"
    joblib.dump(model, model_filename)
    print(f"Trained model saved to '{model_filename}'")

if __name__ == "__main__":
    main()