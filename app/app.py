import os
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
import time
from io import BytesIO
from fpdf import FPDF
import json

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "model", "tomato_disease_model.h5")
model = tf.keras.models.load_model(MODEL_PATH)

CLASS_NAMES = [
    "Tomato_Early_blight",
    "Tomato_Late_blight",
    "Tomato_Spider_mites_Two_spotted_spider_mite"
]

SPRAY_RECOMMENDATIONS = {
    "Tomato_Early_blight": {
        "Spray": "Mancozeb 75% WP @ 2.5g/L",
        "Interval": "7–10 days",
        "Note": "Avoid overhead irrigation; spray preventively."
    },
    "Tomato_Late_blight": {
        "Spray": "Metalaxyl + Mancozeb @ 2g/L",
        "Interval": "5–7 days",
        "Note": "Spray before rains; maintain field sanitation."
    },
    "Tomato_Spider_mites_Two_spotted_spider_mite": {
        "Spray": "Abamectin 1.8% EC @ 0.5ml/L",
        "Interval": "10 days",
        "Note": "Spray early morning; avoid repeated same chemical."
    }
}

def display_spray_info(predicted_class):
    rec = SPRAY_RECOMMENDATIONS.get(predicted_class, None)
    if rec is None:
        st.info("No spray recommendation available.")
        return
    st.markdown(f"""
    <div style="
        background: linear-gradient(90deg,#0b6b2b,#0f66b8);
        color: white;
        padding:16px;
        border-radius:12px;
        box-shadow:0 6px 18px rgba(0,0,0,0.12);
        font-size:15px;
        line-height:1.6;
    ">
      <b style="font-size:16px;"> Spray:</b> {rec['Spray']}<br>
      <b>Interval:</b> {rec['Interval']}<br>
      <b>Note:</b> {rec['Note']}
    </div>
    """, unsafe_allow_html=True)

def render_gradient_bar(percent: int, steps: int = 10):
    """
    Renders a stepwise horizontal bar with green→blue gradient fill.
    percent: 0-100
    steps: number of visual segments (default 10)
    """
    filled_segments = int((percent / 100) * steps)
    segment_html = ""
    seg_width = max(6, int(100/steps) - 1)  
    for i in range(steps):
        if i < filled_segments:
            segment_html += f"""
            <div style="
                display:inline-block;
                width:{seg_width}%;
                height:22px;
                margin-right:1%;
                border-radius:6px;
                background: linear-gradient(90deg, #28a745, #1E90FF);
                box-shadow: 0 1px 3px rgba(0,0,0,0.12);
            "></div>
            """
        else:
            segment_html += f"""
            <div style="
                display:inline-block;
                width:{seg_width}%;
                height:22px;
                margin-right:1%;
                border-radius:6px;
                background: rgba(230,230,230,0.9);
                border: 1px solid rgba(200,200,200,0.6);
            "></div>
            """

    html = f"""
    <div style="width:100%; padding:6px 0;">
      <div style="font-weight:600; color:#0b1320; margin-bottom:8px;">Spray Progress: {percent}%</div>
      <div style="width:100%; display:flex; align-items:center;">{segment_html}</div>
    </div>
    """
    return html

USER_CSV = os.path.join(BASE_DIR, "users.csv")
REPORT_CSV = os.path.join(BASE_DIR, "user_reports.csv")

if not os.path.exists(USER_CSV):
    pd.DataFrame(columns=["username", "password"]).to_csv(USER_CSV, index=False)
if not os.path.exists(REPORT_CSV):
    pd.DataFrame(columns=["username", "image_name", "disease", "confidence"]).to_csv(REPORT_CSV, index=False)

st.set_page_config(page_title="Tomato Disease Detection 🍅", layout="wide")
st.title("🍅 Tomato Crop Disease Detection System")
st.markdown("<h5 style='color:gray;'>AI-powered detection, spray simulation & smart tips for farmers 🌱</h5>", unsafe_allow_html=True)

def predict_disease(image: Image.Image):
    img = image.convert("RGB").resize((224, 224))
    img_array = np.array(img)/255.0
    img_array = np.expand_dims(img_array, axis=0)
    preds = model.predict(img_array)
    class_idx = int(np.argmax(preds[0]))
    confidence = float(preds[0][class_idx])
    return CLASS_NAMES[class_idx], confidence, preds[0]

def save_report(username, image_name, disease, confidence):
    df = pd.read_csv(REPORT_CSV)
    df = pd.concat([df, pd.DataFrame([[username, image_name, disease, confidence]], columns=df.columns)], ignore_index=True)
    df.to_csv(REPORT_CSV, index=False)

def generate_pdf(username):
    df = pd.read_csv(REPORT_CSV)
    user_df = df[df['username'] == username]

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, f"Crop Disease Report for {username}", ln=True, align="C")
    pdf.ln(10)
    pdf.set_font("Arial", "", 12)

    if user_df.empty:
        pdf.cell(0, 8, "No records found.", ln=True)
    else:
        for idx, row in user_df.iterrows():
            pdf.cell(0, 8, f"Image: {row['image_name']}, Disease: {row['disease']}, Confidence: {row['confidence']*100:.2f}%", ln=True)

    pdf_bytes = pdf.output(dest='S').encode('latin1')
    pdf_file = BytesIO(pdf_bytes)
    pdf_file.seek(0)
    return pdf_file

tab_login, tab_detection, tab_dashboard, tab_spray, tab_help = st.tabs(
    [" Login/Register", " Detection", " Dashboard", " Spray Control", " Help & Tips"]
)

with tab_login:
    st.subheader("Create Account / Login")
    col1, col2 = st.columns(2)
    with col1:
        new_user = st.text_input("Enter Name", key="new_user")
        new_pass = st.text_input("Enter Password", type="password", key="new_pass")
        if st.button("Create Account"):
            if new_user.strip() == "" or new_pass.strip() == "":
                st.error("Please enter both name and password.")
            else:
                df_users = pd.read_csv(USER_CSV)
                if new_user in df_users['username'].values:
                    st.error("Username already exists!")
                else:
                    df_users = pd.concat([df_users, pd.DataFrame([[new_user,new_pass]], columns=df_users.columns)], ignore_index=True)
                    df_users.to_csv(USER_CSV, index=False)
                    st.success("Account created successfully!")

    with col2:
        login_user = st.text_input("Username", key="login_user")
        login_pass = st.text_input("Password", type="password", key="login_pass")
        if st.button("Login"):
            df_users = pd.read_csv(USER_CSV)
            if login_user in df_users['username'].values:
                saved_pass = df_users[df_users['username']==login_user]['password'].values[0]
                if login_pass == saved_pass:
                    st.success("Login successful!")
                    st.session_state['username'] = login_user
                else:
                    st.error("Incorrect password")
            else:
                st.error("User not found")

if 'username' in st.session_state:
    username = st.session_state['username']

    with tab_detection:
        st.subheader("Upload Leaf Image for Detection")
        uploaded_file = st.file_uploader("Upload Tomato Leaf Image", type=["jpg","jpeg","png"])
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, use_container_width=True)

            confidence_threshold = st.slider("Confidence Threshold for Auto Spray", 0.0, 1.0, 0.7, key="confidence_threshold")

            with st.spinner("Analyzing Image..."):
                time.sleep(1)
                predicted_class, confidence, all_probs = predict_disease(image)

            st.session_state['predicted_class'] = predicted_class
            st.session_state['confidence'] = confidence

            st.success(f"Disease Detected: **{predicted_class}** ({confidence*100:.2f}%)")

            display_spray_info(predicted_class)

            fig, ax = plt.subplots(figsize=(6,3))
            bars = ax.bar(CLASS_NAMES, all_probs, color=["#32CD32","#1E90FF","#FFD700"])
            ax.set_ylabel("Prediction Probability")
            ax.set_ylim(0, 1)
            ax.set_xticks(range(len(CLASS_NAMES)))
            ax.set_xticklabels(CLASS_NAMES, rotation=20, ha='right')
            for bar, prob in zip(bars, all_probs):
                ax.text(bar.get_x() + bar.get_width()/2, prob + 0.02, f"{prob:.2f}", ha='center', fontsize=9)
            st.pyplot(fig, use_container_width=True)

            save_report(username, uploaded_file.name, predicted_class, confidence)

    with tab_dashboard:
        st.subheader("Your Reports Dashboard")
        df_reports = pd.read_csv(REPORT_CSV)
        user_reports = df_reports[df_reports['username']==username]
        st.dataframe(user_reports)
        pdf_file = generate_pdf(username)
        st.download_button("Download PDF Report", pdf_file, file_name=f"{username}_report.pdf")

    with tab_spray:
        st.markdown("<h3 style='color:#32CD32;'>Spray Simulation Panel (Stepwise Gradient Fill)</h3>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)

    with col1:
        st.markdown("<h4 style='color:#00BFFF;'>Manual Spray</h4>", unsafe_allow_html=True)
        if st.button("Start Manual Spray"):
            st.info("Starting manual spray process...")
            steps = 12
            delay = 0.08
            placeholder = st.empty()

            for s in range(steps + 1):
                percent = int((s / steps) * 100)
                bar_html = f"""
                <div style='width:100%; background-color:rgba(255,255,255,0.1); border-radius:12px; height:25px;'>
                    <div style='width:{percent}%; height:25px; background:linear-gradient(90deg, #28a745, #1E90FF);
                    border-radius:12px; box-shadow:0 0 8px rgba(0,0,0,0.3); transition:width 0.2s ease;'></div>
                </div>
                <p style='color:white; font-weight:bold; margin-top:6px;'>Progress: {percent}%</p>
                """
                placeholder.markdown(bar_html, unsafe_allow_html=True)
                time.sleep(delay)

            st.success("Manual Spray Completed Successfully!")
            st.balloons() 

    with col2:
        st.markdown("<h4 style='color:#1E90FF;'> Auto Spray Mode</h4>", unsafe_allow_html=True)
        auto_mode = st.toggle("Enable Auto Spray")

        if auto_mode:
            if 'predicted_class' in locals() and 'confidence' in locals() and confidence > confidence_threshold:
                st.info("High-confidence disease detected. Auto spray will start now...")
                steps = 12
                delay = 0.06
                placeholder2 = st.empty()

                for s in range(steps + 1):
                    percent = int((s / steps) * 100)
                    bar_html = f"""
                    <div style='width:100%; background-color:rgba(255,255,255,0.1); border-radius:12px; height:25px;'>
                        <div style='width:{percent}%; height:25px; background:linear-gradient(90deg, #28a745, #1E90FF);
                        border-radius:12px; box-shadow:0 0 8px rgba(0,0,0,0.3); transition:width 0.2s ease;'></div>
                    </div>
                    <p style='color:white; font-weight:bold; margin-top:6px;'>Progress: {percent}%</p>
                    """
                    placeholder2.markdown(bar_html, unsafe_allow_html=True)
                    time.sleep(delay)

                st.success("Automatic Spray Completed Successfully!")
                st.snow()  
            else:
                st.warning("Auto mode is ON, but no high-confidence disease detected yet.")

    with tab_help:
        st.subheader("Helpful Farming Tips")
        st.markdown("""
        <div style="background-color:#28a745;color:white;padding:18px;border-radius:12px;">
        <b>Sunlight:</b> Provide at least 6 hours daily.<br>
        <b>Monitoring:</b> Check plants every morning.<br>
        <b>Overwatering:</b> Avoid standing water.<br>
        <b>Spray Schedule:</b> Follow recommended intervals.<br>
        <b>Hygiene:</b> Keep field tools and soil clean.<br>
        <b>Rotation:</b> Change chemicals to prevent resistance.<br>
        </div>
        """, unsafe_allow_html=True)

else:
    st.info("Please create an account or login (Login/Register tab) to use the app.")
