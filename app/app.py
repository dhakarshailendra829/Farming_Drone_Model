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
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

st.set_page_config(page_title="AI Tomato Health Assistant", layout="wide")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "model", "tomato_disease_model.h5")

try:
    model = tf.keras.models.load_model(MODEL_PATH)
except Exception as e:
    st.warning("Model could not be loaded. Detection disabled until model file is present.")
    model = None

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
        "Note": "Spray before rains, maintain field sanitation."
    },
    "Tomato_Spider_mites_Two_spotted_spider_mite": {
        "Spray": "Abamectin 1.8% EC @ 0.5ml/L",
        "Interval": "10 days",
        "Note": "Spray early morning; avoid repeated same chemical."
    }
}

USER_CSV = os.path.join(BASE_DIR, "users.csv")
REPORT_CSV = os.path.join(BASE_DIR, "user_reports.csv")

if not os.path.exists(USER_CSV):
    pd.DataFrame(columns=["username", "password", "last_login"]).to_csv(USER_CSV, index=False)

if not os.path.exists(REPORT_CSV):
    pd.DataFrame(columns=["username", "image_name", "disease", "confidence", "timestamp", "latitude", "longitude"]).to_csv(REPORT_CSV, index=False)

st.title(" Tomato Crop Disease Detection System")
st.markdown("<h5 style='color:gray;'>AI-powered detection, Interactive dashboard, spray simulation & farmer tips </h5>", unsafe_allow_html=True)

def predict_disease(image: Image.Image):
    if model is None:
        raise RuntimeError("Model not loaded")
    img = image.convert("RGB").resize((224, 224))
    img_array = np.array(img)/255.0
    img_array = np.expand_dims(img_array, axis=0)
    preds = model.predict(img_array)
    class_idx = int(np.argmax(preds[0]))
    confidence = float(preds[0][class_idx])
    return CLASS_NAMES[class_idx], confidence, preds[0]

def save_report(username, image_name, disease, confidence, latitude=None, longitude=None):
    if not os.path.exists(REPORT_CSV):
        pd.DataFrame(columns=["username", "image_name", "disease", "confidence", "timestamp", "latitude", "longitude"]).to_csv(REPORT_CSV, index=False)
    df = pd.read_csv(REPORT_CSV)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    new_row = pd.DataFrame([[username, image_name, disease, confidence, timestamp, latitude, longitude]],
                           columns=["username", "image_name", "disease", "confidence", "timestamp", "latitude", "longitude"])
    df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv(REPORT_CSV, index=False)

def generate_pdf(username):
    df = pd.read_csv(REPORT_CSV)
    user_df = df[df['username'] == username]

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, f"Crop Disease Report for {username}", ln=True, align="C")
    pdf.ln(8)
    pdf.set_font("Arial", "", 12)

    if user_df.empty:
        pdf.cell(0, 8, "No records found.", ln=True)
    else:
        for idx, row in user_df.iterrows():
            pdf.cell(0, 8, f"{row['timestamp']} - Image: {row['image_name']} | Disease: {row['disease']} | Confidence: {row['confidence']*100:.2f}%", ln=True)
            if pd.notna(row.get('latitude')) and pd.notna(row.get('longitude')):
                pdf.cell(0, 8, f"   Location: {row['latitude']}, {row['longitude']}", ln=True)

    pdf_bytes = pdf.output(dest='S').encode('latin1')
    pdf_file = BytesIO(pdf_bytes)
    pdf_file.seek(0)
    return pdf_file

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
if 'username' not in st.session_state:
    st.markdown("## Create Account / Login")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Create Account")
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
                    df_users = pd.concat([df_users, pd.DataFrame([[new_user, new_pass, ""]], columns=df_users.columns)], ignore_index=True)
                    df_users.to_csv(USER_CSV, index=False)
                    st.success(" Account created successfully! Please login to continue.")

    with col2:
        st.markdown("###  Login")
        login_user = st.text_input("Username", key="login_user")
        login_pass = st.text_input("Password", type="password", key="login_pass")
        if st.button("Login"):
            df_users = pd.read_csv(USER_CSV)
            if login_user in df_users['username'].values:
                saved_pass = df_users[df_users['username'] == login_user]['password'].values[0]
                if login_pass == saved_pass:
                    df_users.loc[df_users['username'] == login_user, 'last_login'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    df_users.to_csv(USER_CSV, index=False)
                    st.session_state['username'] = login_user
                    st.success(" Login successful!")
                    st.experimental_rerun() if hasattr(st, "experimental_rerun") else st.rerun()
                else:
                    st.error("Incorrect password")
            else:
                st.error("User not found")
    st.stop()

username = st.session_state['username']
st.sidebar.success(f" **Welcome**, {username}!")
if st.sidebar.button("Logout"):
    del st.session_state['username']
    st.success("You have been logged out.")
    st.experimental_rerun() if hasattr(st, "experimental_rerun") else st.rerun()

tabs = st.tabs([" Detection", " Dashboard", " Spray Control", " Quiz", " Help & Tips"])
tab_detection, tab_dashboard, tab_spray, tab_quiz, tab_help = tabs
with tab_detection:
    st.subheader(" Upload Leaf Image for Detection")
    colA, colB = st.columns([2,1])

    with colA:
        uploaded_file = st.file_uploader("Upload Tomato Leaf Image (jpg/jpeg/png)", type=["jpg","jpeg","png"])
        st.markdown("Add farm location (latitude / longitude): ")
        lat = st.text_input("Latitude (e.g. 26.9124)", key="lat")
        lon = st.text_input("Longitude (e.g. 75.7873)", key="lon")

    with colB:
        st.markdown("### Info")
        st.info("Model required for detection must exist at model/tomato_disease_model.h5")
        st.write(" ")

    if uploaded_file:
        if model is None:
            st.error("Model file not found — detection disabled. Place model at model/tomato_disease_model.h5")
        else:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded image", use_container_width=True)

            confidence_threshold = st.slider(
                "Confidence Threshold for Auto Spray", 0.0, 1.0, 0.7, key="confidence_threshold"
            )

            with st.spinner(" Analyzing Image..."):
                time.sleep(0.6)
                predicted_class, confidence, all_probs = predict_disease(image)

            st.session_state['predicted_class'] = predicted_class
            st.session_state['confidence'] = confidence
            st.session_state['all_probs'] = all_probs

            st.success(f" Disease Detected: **{predicted_class}** ({confidence*100:.2f}%)")

            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=confidence * 100,
                title={'text': "Model Confidence (%)"},
                gauge={'axis': {'range': [0, 100]},
                       'bar': {'color': "#28a745"},
                       'steps': [
                           {'range': [0, 50], 'color': "#ffcccc"},
                           {'range': [50, 80], 'color': "#ffffb3"},
                           {'range': [80, 100], 'color': "#ccffcc"}]}
            ))
            st.plotly_chart(fig_gauge, use_container_width=True)

            prob_df = pd.DataFrame({'Disease': CLASS_NAMES, 'Probability': all_probs})
            fig_prob = px.bar(prob_df, x='Disease', y='Probability', color='Disease',
                              color_discrete_sequence=px.colors.qualitative.Vivid,
                              title="Prediction Probabilities")
            fig_prob.update_yaxes(range=[0, 1])
            st.plotly_chart(fig_prob, use_container_width=True)

            try:
                lat_val = float(lat) if lat.strip() != "" else None
                lon_val = float(lon) if lon.strip() != "" else None
            except:
                lat_val, lon_val = None, None

            save_report(username, uploaded_file.name, predicted_class, confidence, lat_val, lon_val)
            st.success("Report saved to your dashboard history.")

with tab_dashboard:
    st.subheader(" Your Reports Dashboard.")

    df_reports = pd.read_csv(REPORT_CSV)
    user_reports = df_reports[df_reports['username'] == username].copy()

    if not user_reports.empty and 'timestamp' in user_reports.columns:
        user_reports['timestamp'] = pd.to_datetime(user_reports['timestamp'])

    st.markdown("### Filters")
    cols = st.columns(3)
    with cols[0]:
        date_from = st.date_input("From", value=(user_reports['timestamp'].min().date() if not user_reports.empty else datetime.now().date()))
    with cols[1]:
        date_to = st.date_input("To", value=(user_reports['timestamp'].max().date() if not user_reports.empty else datetime.now().date()))
    with cols[2]:
        disease_options = ["All"] + CLASS_NAMES
        disease_filter = st.selectbox("Disease", options=disease_options)

    conf_min, conf_max = st.slider("Confidence range", 0.0, 1.0, (0.0, 1.0), key="conf_range")

    filtered = user_reports.copy()
    if not filtered.empty:
        filtered = filtered[(filtered['timestamp'].dt.date >= date_from) & (filtered['timestamp'].dt.date <= date_to)]
        if disease_filter != "All":
            filtered = filtered[filtered['disease'] == disease_filter]
        filtered = filtered[(filtered['confidence'] >= conf_min) & (filtered['confidence'] <= conf_max)]

    if filtered.empty:
        st.info("No reports match the selected filters.")
    else:
        st.markdown("####  Filtered Reports")
        st.dataframe(filtered.reset_index(drop=True), use_container_width=True)

        st.markdown("####  Detection Confidence Over Time")
        fig_hist = px.line(filtered, x='timestamp', y='confidence', markers=True, title="Confidence over time")
        fig_hist.update_yaxes(range=[0, 1])
        st.plotly_chart(fig_hist, use_container_width=True)

        st.markdown("####  Disease Frequency")
        freq = filtered['disease'].value_counts().reset_index()
        freq.columns = ['Disease', 'Count']
        fig_pie = px.pie(freq, names='Disease', values='Count', title="Disease distribution")
        st.plotly_chart(fig_pie, use_container_width=True)

        st.markdown("####  Export Data")
        st.download_button("Download CSV", data=filtered.to_csv(index=False).encode('utf-8'), file_name=f"{username}_reports_filtered.csv", mime="text/csv")
        towrite = BytesIO()
        with pd.ExcelWriter(towrite, engine="openpyxl") as writer:
            filtered.to_excel(writer, index=False, sheet_name="Reports")
        towrite.seek(0)
        st.download_button("Download Excel", data=towrite, file_name=f"{username}_reports_filtered.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        pdf_file = generate_pdf(username)
        st.download_button("Download Full PDF", pdf_file, file_name=f"{username}_report.pdf")

with tab_spray:
    st.markdown("<h3 style='color:#32CD32;'> Spray Simulation Panel</h3>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("<h4 style='color:#00BFFF;'> Manual Spray</h4>", unsafe_allow_html=True)

        if st.button("Start Manual Spray"):
            st.info("Starting manual spray process...")
            steps = 12
            delay = 0.06
            placeholder = st.empty()

            progress_bar = st.progress(0)
            for s in range(steps + 1):
                percent = int((s / steps) * 100)
                progress_bar.progress(percent)
                time.sleep(delay)

            placeholder.empty() 
            st.success(" Manual Spray Completed Successfully!")
            st.balloons()

    with col2:
        st.markdown("<h4 style='color:#1E90FF;'> Auto Spray Mode</h4>", unsafe_allow_html=True)
        auto_mode = st.checkbox("Enable Auto Spray Mode")

        if auto_mode:
            pc = st.session_state.get('predicted_class', None)
            conf = st.session_state.get('confidence', None)
            thr = st.session_state.get('confidence_threshold', 0.7)

            if pc and conf is not None and conf >= thr:
                st.info(f"Auto mode active — detected {pc} at {conf*100:.2f}%. Starting spray...")

                progress_bar_auto = st.progress(0)
                steps = 12
                delay = 0.05
                for s in range(steps + 1):
                    percent = int((s / steps) * 100)
                    progress_bar_auto.progress(percent)
                    time.sleep(delay)

                st.success(" Automatic Spray Completed Successfully!")
                st.snow()
            else:
                st.warning("Auto mode is ON, but no recent high-confidence detection found to trigger spray.")


with tab_quiz:
    st.subheader(" Tomato Health Quiz! ")

    if 'quiz_score' not in st.session_state:
        st.session_state['quiz_score'] = 0
        st.session_state['quiz_done'] = False

    if not st.session_state['quiz_done']:
        q1 = st.radio(
            "1) Which condition is commonly caused by fungal spores and shows concentric rings on leaves?",
            options=["Early blight", "Nutrient deficiency", "Overwatering"]
        )
        q2 = st.radio(
            "2) Spider mites usually cause which symptom on leaves?",
            options=["Yellow speckles", "Large holes", "White powder"]
        )
        q3 = st.radio(
            "3) Best time to spray pesticides for spider mites?",
            options=["Early morning", "Midday", "Late night"]
        )

        if st.button("Submit Quiz"):
            score = 0
            if q1 == "Early blight":
                score += 1
            if q2 == "Yellow speckles":
                score += 1
            if q3 == "Early morning":
                score += 1

            st.session_state['quiz_score'] = score
            st.session_state['quiz_done'] = True
            st.success(f" You scored {score}/3!")
    else:
        st.info(f"Your last quiz score: {st.session_state['quiz_score']}/3")
        if st.button("Retake Quiz"):
            st.session_state['quiz_done'] = False
            st.session_state['quiz_score'] = 0
            st.experimental_rerun()

with tab_help:
    st.subheader(" Help & Tips for Farmers")

    with st.expander(" Frequently Asked Questions (FAQ)"):
        st.markdown("""
        **Q:** How often should I check my plants?  
        **A:** Check daily for early signs, especially after rains.

        **Q:** When should I spray?  
        **A:** Prefer early morning or late evening; follow recommended intervals.

        **Q:** Can I use the app offline?  
        **A:** Detection requires the model and local processing; downloading the model allows offline use.
        """)

    st.markdown("---")
    st.subheader(" Crop Calendar — Tomato (general guide)")
    st.markdown("""
    - **Week 0–2 (Germination):** Keep seeds moist, moderate sunlight.  
    - **Week 3–6 (Seedling):** Harden off, provide balanced fertilizer.  
    - **Week 7–12 (Vegetative):** Prune lower leaves, watch for pests.  
    - **Week 12+ (Flowering & Fruit):** Monitor for blight; follow spray schedule if recommended.
    """)

    st.markdown("---")
    st.subheader(" Practical Tips")
    st.markdown("""
    - Keep field sanitation — remove infected debris.  
    - Rotate chemicals to prevent resistance.  
    - Maintain irrigation schedule; avoid overhead watering during disease-prone seasons.
    """)
