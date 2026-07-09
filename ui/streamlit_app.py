"""
Streamlit frontend for the Crop Disease Detector.

This app does NOT load the model itself — it calls the already-deployed
FastAPI backend (on Render) over HTTP, exactly like a real production
frontend/backend split. This separation matters: the UI and the model-serving
API can be deployed, scaled, and updated independently of each other.

Run locally with:
    streamlit run ui/streamlit_app.py
"""

import requests
import streamlit as st

st.set_page_config(page_title="Crop Disease Detector", page_icon="🌿", layout="centered")

# Default points at the live Render deployment — change this in the sidebar
# if testing against a local API instead (e.g. http://localhost:8000).
DEFAULT_API_URL = "https://crop-disease-detector-8fao.onrender.com"

if "api_url" not in st.session_state:
    st.session_state.api_url = DEFAULT_API_URL

st.title("🌿 Crop Disease Detector")
st.write("Upload a photo of a crop leaf to get an instant disease diagnosis.")

with st.sidebar:
    st.header("Settings")
    st.session_state.api_url = st.text_input("API URL", value=st.session_state.api_url)

    if st.button("Check API health"):
        try:
            response = requests.get(f"{st.session_state.api_url}/health", timeout=60)
            if response.status_code == 200:
                data = response.json()
                st.success(f"API is up — {data['num_classes']} classes loaded")
            else:
                st.error(f"API returned status {response.status_code}")
        except requests.exceptions.RequestException as e:
            st.error(f"Could not reach API: {e}")

    st.caption(
        "Note: the free-tier API sleeps after inactivity. The first request "
        "after a while may take 30-60 seconds to wake it up — this is "
        "normal, not a bug."
    )

tab_diagnose, tab_metrics = st.tabs(["🔍 Diagnose", "📊 Live Metrics"])

with tab_diagnose:
    uploaded_file = st.file_uploader("Choose a leaf image", type=["jpg", "jpeg", "png", "webp"])

    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded image", use_container_width=True)

        if st.button("Diagnose", type="primary"):
            with st.spinner("Analyzing... (may take up to a minute if the API is waking up)"):
                try:
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                    response = requests.post(
                        f"{st.session_state.api_url}/predict", files=files, timeout=90
                    )

                    if response.status_code == 200:
                        result = response.json()

                        predicted_class = result["predicted_class"].replace("___", " — ").replace("_", " ")
                        confidence_pct = result["confidence"] * 100

                        st.success(f"**Diagnosis:** {predicted_class}")
                        st.metric("Confidence", f"{confidence_pct:.1f}%")

                        st.write("**Top predictions:**")
                        for pred in result["top_predictions"]:
                            label = pred["class_name"].replace("___", " — ").replace("_", " ")
                            st.progress(pred["confidence"], text=f"{label} — {pred['confidence'] * 100:.1f}%")
                    else:
                        st.error(f"API error ({response.status_code}): {response.json().get('detail', 'Unknown error')}")

                except requests.exceptions.Timeout:
                    st.error("Request timed out. The free-tier API may still be waking up — try again in a moment.")
                except requests.exceptions.RequestException as e:
                    st.error(f"Could not reach API: {e}")

with tab_metrics:
    st.write(
        "Live stats pulled directly from the API's `/metrics` endpoint. "
        "Resets whenever the free-tier API restarts/cold-starts — this is "
        "an in-memory demo dashboard, not a persistent monitoring system."
    )

    if st.button("Refresh metrics"):
        st.rerun()

    try:
        response = requests.get(f"{st.session_state.api_url}/metrics", timeout=60)
        if response.status_code == 200:
            data = response.json()

            col1, col2, col3 = st.columns(3)
            col1.metric("Total requests", data["total_requests_since_start"])
            col2.metric("Avg latency", f"{data['average_latency_ms']:.0f} ms")
            col3.metric("Avg confidence", f"{data['average_confidence'] * 100:.1f}%")

            if data["class_distribution"]:
                st.write("**Prediction distribution by class:**")
                readable_distribution = {
                    k.replace("___", " — ").replace("_", " "): v
                    for k, v in data["class_distribution"].items()
                }
                st.bar_chart(readable_distribution)
            else:
                st.info("No predictions recorded yet — try the Diagnose tab first.")
        else:
            st.error(f"Could not fetch metrics (status {response.status_code})")
    except requests.exceptions.RequestException as e:
        st.error(f"Could not reach API: {e}")