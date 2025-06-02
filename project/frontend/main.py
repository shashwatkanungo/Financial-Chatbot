import streamlit as st
import requests
import pandas as pd
from PIL import Image
import json
import os

API_URL = "http://localhost:8000/api"  # adjust if running on a different host/port

st.set_page_config(page_title="Invoice AI System", layout="wide")
st.title("📊 Invoice Intelligence Dashboard")

# Sidebar navigation
page = st.sidebar.selectbox("Navigate", ["Upload", "Analyze", "Predict", "Ask"])

# --- Upload Page ---
if page == "Upload":
    st.header("📁 Upload Data File")
    file = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx"])
    if file:
        files = {"file": (file.name, file.getvalue())}
        response = requests.post(f"{API_URL}/upload", files=files)
        if response.status_code == 200:
            st.success("File uploaded successfully.")
        else:
            st.error(f"Upload failed: {response.json()['detail']}")

# --- Analyze Page ---
elif page == "Analyze":
    st.header("🔍 Run EDA Analysis")

    if st.button("Run Analysis"):
        response = requests.post(f"{API_URL}/analyze")
        if response.status_code == 200:
            result = response.json()
            st.success("EDA Completed.")

            summary = result.get("insights", "No insights found.")
            st.text_area("📊 EDA Summary", summary, height=500)
        else:
            st.error(f"Error: {response.json()['detail']}")

    st.subheader("📂 EDA Visualizations")

    base_dir = os.path.dirname(os.path.abspath(__file__))  # frontend/
    image_dir = os.path.abspath(os.path.join(base_dir, "..", "outputs", "eda"))


    if os.path.exists(image_dir):
        image_files = [f for f in os.listdir(image_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
        if image_files:
            for img_file in image_files:
                image_path = os.path.join(image_dir, img_file)
                st.image(Image.open(image_path), caption=img_file, use_column_width=True)
        else:
            st.info("No image files found in the EDA folder.")
    else:
        st.warning("EDA outputs folder not found.")

# --- Predict Page ---
elif page == "Predict":
    st.header("🤖 Make Predictions")

    st.subheader("📥 Enter Invoice Details")
    
    amount = st.number_input("Amount", min_value=0.0, value=1000.0)
    days_until_due = st.number_input("Days Until Due", min_value=0, value=30)
    invoice_day = st.number_input("Invoice Day", min_value=1, max_value=31, value=15)
    invoice_month = st.number_input("Invoice Month", min_value=1, max_value=12, value=5)
    invoice_year = st.number_input("Invoice Year", min_value=2000, max_value=2100, value=2024)

    if st.button("Run Prediction"):
        payload = {
            "Amount": amount,
            "Days_Until_Due": days_until_due,
            "Invoice_day": invoice_day,
            "Invoice_month": invoice_month,
            "Invoice_year": invoice_year,
        }

        response = requests.post(f"{API_URL}/predict", json=payload)
        if response.status_code == 200:
            result = response.json()
            st.success("✅ Prediction Completed")
            st.markdown(
                f"**Predicted Days to Pay:** {result['Predicted Days to Pay']} days  \n"
                f"**Expected Payment Date:** {result['Expected Payment Date']}"
            )
        else:
            st.error(f"❌ Error: {response.json().get('detail', 'Unknown error')}")

# --- Ask Page ---
elif page == "Ask":
    st.header("💬 Ask Questions About the Data")
    user_question = st.text_input("Enter your question:")
    if st.button("Ask"):
        payload = {"question": user_question}
        response = requests.post(f"{API_URL}/ask", json=payload)
        if response.status_code == 200:
            result = response.json()
            st.success("Here's the answer:")
            st.markdown(result["answer"])
        else:
            st.error(f"Error: {response.json()['detail']}")
