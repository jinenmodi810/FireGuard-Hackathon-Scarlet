import streamlit as st
import os

st.set_page_config(layout="wide")
st.title("🔥 FireGuard: Wildfire Risk Predictor")

# Description
st.markdown("""
Welcome to FireGuard – a wildfire risk detection system powered by real-time data and machine learning.  
This map shows predicted fire risk levels across Illinois ZIP codes based on temperature, humidity, wind, and vegetation.
""")

# Embed the folium map
map_path = os.path.join("..", "outputs", "fire_risk_map.html")
if os.path.exists(map_path):
    st.components.v1.iframe(src=map_path, width=1000, height=600)
else:
    st.error("🚫 Map file not found. Please generate it using map_generator.py.")
