import streamlit as st
import os

st.set_page_config(layout="wide")
st.title("🔥 FireGuard — Stop Fires Before They Start")

# Define paths
base_dir = os.getcwd()
site_html_path = os.path.join(base_dir, "fireguard_site.html")
map_html_path = os.path.join(base_dir, "fire_risk_map.html")

# Load and inject HTML content
if os.path.exists(site_html_path):
    with open(site_html_path, "r", encoding="utf-8") as f:
        site_html = f.read()

    if os.path.exists(map_html_path):
        with open(map_html_path, "r", encoding="utf-8") as mf:
            map_html = mf.read()

        # Inject full map HTML in place of iframe tag
        site_html = site_html.replace(
            '<iframe src="fire_risk_map.html" width="100%" height="300" style="border:none; margin-top:1rem; border-radius: 8px;" loading="lazy"></iframe>',
            f'<div style="margin-top: 1rem; border-radius: 8px; overflow: hidden;">{map_html}</div>'
        )
    else:
        st.warning("fire_risk_map.html not found. Displaying homepage without embedded map.")

    st.components.v1.html(site_html, height=2000, scrolling=True)

else:
    st.error("Homepage file fireguard_site.html not found.")
