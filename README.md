# FireGuard-Hackthon-Scarlet

Wildfires have become increasingly frequent and destructive, yet public tools often react after the damage is already done. Most weather APIs are either limited or prohibitively expensive, and few offer actionable insights at the community level. We set out to create FireGuard—a real-time wildfire risk prediction and alert system that is free, scalable, and AI-driven.

What It Does

FireGuard dynamically collects live environmental data (temperature, humidity, wind speed, barometric pressure, dew point, visibility, wind chill, etc.) at the ZIP-code level across the United States. What makes FireGuard different is that we do not rely on paid APIs—we scrape and clean the data directly from trusted public sources like NOAA, making this solution highly scalable and free to use.

We then apply a custom AI-driven dryness score model that categorizes wildfire risk into levels such as Normal, Dry, and Very Dry. This scoring is derived from multiple cleaned data inputs, and overcoming inconsistencies in format and structure across sources was a key data engineering challenge.

How We Built It

1.Data Collection & Cleaning: We scraped live weather data from NOAA and extracted a comprehensive set of parameters. Most APIs only give temperature or humidity, but we engineered access to richer metrics like wind chill, visibility, dew point, and barometric pressure. Vegetation estimates were inferred via heuristics and geospatial lookups. One of the biggest challenges was cleaning and standardizing this data across 40,000+ ZIP codes without APIs, which we handled using custom parsers and validation layers.

2.ML Model Development: We developed a machine learning model trained on historical wildfire data and engineered features like the dryness score. We serialized the trained model using joblib and deployed it for real-time inference against live ZIP-code data. Challenges here included aligning data features in inference and training phases, and ensuring model performance under real-time constraints.

3.Dynamic Risk Mapping & Visualization: The output of the model was used to create a dynamic fire-risk dashboard, showing color-coded risk zones on a map with real-time environmental indicators. The backend feeds updated risk scores to the frontend, which visualizes which areas are in danger using Streamlit and geospatial overlays.

4.LLM-Powered Community Alerting: We built an intelligent assistant using OpenAI’s LLM to answer user questions such as: “Is it safe in my area?” “What precautions should I take?” “Should I evacuate?” The assistant responds based on live inputs and provides context-aware safety guidance.

5.Social + Human-Centric Output: To make this system usable for communities, we export structured Excel reports with summaries and feed those into the LLM to generate human-readable alerts. These can then be delivered via SMS, email, or shared on social media for rapid awareness.

Challenges We Faced (1) Collecting and cleaning live weather data without API limits (2)Engineering a reliable dryness model using non-standardized parameters (3)Integrating an ML model with real-time inferencing and data pipelines (4)Dynamically mapping changing wildfire risks across thousands of ZIP codes (5)Building a usable, informative frontend UI to communicate real-time risk clearly

What We Learned

We learned how to build a real-time AI system from the ground up by integrating: (1)Scalable data pipelines (2)Live environmental monitoring (3)ML-powered prediction (4)NLP-based human safety systems

We also learned how to work as a team across data, backend, frontend, and AI model integration under strict time constraints.


Impact FireGuard is not just a wildfire risk predictor—it’s a life-saving system designed for proactive prevention and community safety. It combines deep technical work with real social utility and accessibility.

This is a full-fledged end-to-end system, with real-time data, AI/ML models, and a user-first design—built with purpose.
![img1](https://github.com/user-attachments/assets/46b3853c-bc41-4d76-9f04-0d298efe4532)

![img2](https://github.com/user-attachments/assets/b3442719-129c-4f4f-8c08-d0d846d075c9)

![img3](https://github.com/user-attachments/assets/c3792332-25a8-4e37-8b0a-7f0fd65b23ee)

![img4](https://github.com/user-attachments/assets/13c7bcd5-7fa6-4288-bcf7-800a5b3ab2fe)





