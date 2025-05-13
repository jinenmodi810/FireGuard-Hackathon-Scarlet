@main.route("/predict", methods=["GET"])
def predict():
    zip_code = request.args.get("zip")
    lat = request.args.get("lat", type=float)
    lon = request.args.get("lon", type=float)

    if zip_code:
        query = zip_code
    elif lat and lon:
        query = f"{lat},{lon}"
    else:
        return jsonify({"error": "Missing ZIP code or latitude/longitude"}), 400

    weather = get_weather_data(query)
    if not weather:
        return jsonify({"error": "Weather data not available"}), 500

    # NDVI/soil logic (only works for lat/lon)
    if lat and lon:
        ndvi, soil = get_ndvi_data(lat, lon)
    else:
        ndvi, soil = None, None

    vegetation = classify_vegetation(ndvi)

    risk = fire_risk_score(
        temp=weather["temp_c"],
        humidity=weather["humidity"],
        wind=weather["wind_kph"],
        vegetation_level=vegetation
    )

    return jsonify({
        "risk_label": risk,
        "vegetation_level": vegetation,
        "temperature": weather["temp_c"],
        "humidity": weather["humidity"],
        "wind_kph": weather["wind_kph"],
        "ndvi": ndvi,
        "soil_moisture": soil,
        "condition": weather["condition"]
    })
