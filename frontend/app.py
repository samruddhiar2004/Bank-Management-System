import streamlit as st
import pandas as pd
import requests
import time

# 1. Page Configuration
st.set_page_config(
    page_title="FinAgent-360 | Enterprise AI",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 2. Custom CSS for a better look (The "Findability" Touch)
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
    }
    .stMetric {
        background-color: #1e2130;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #7030a0;
    }
    </style>
    """, unsafe_allow_html=True)

# 3. Sidebar: Model Intelligence
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2830/2830284.png", width=80)
    st.title("FinAgent Intelligence")
    st.success("🟢 AI Engine: Online")
    
    with st.expander("📊 Model Benchmarks", expanded=True):
        st.write("**Best Model:** Random Forest")
        st.progress(80)
        st.caption("Accuracy: 80.0%")
        
    st.markdown("---")
    st.info("Agent v1.0.4 | Baramati Hub")

# 4. Main Dashboard Header
col_header, col_logo = st.columns([4, 1])
with col_header:
    st.title("🏦 FinAgent-360: Loan Underwriting")
    st.write("Next-Generation Credit Risk Assessment using Multi-Model Ensembles.")

# 5. Input Section: Using 'Expander' to keep it clean
with st.form("application_form"):
    st.subheader("📝 Applicant Information")
    c1, c2, c3 = st.columns(3)
    with c1:
        name = st.text_input("Full Name", placeholder="e.g. Samruddhi")
    with c2:
        income = st.number_input("Annual Income (INR)", min_value=0, step=10000)
    with c3:
        score = st.slider("Credit Score", 300, 900, 720)
    
    submitted = st.form_submit_button("🚀 Run AI Analysis")

# 6. Results Area (The Real Connection)
if submitted:
    with st.spinner('FinAgent AI is analyzing credit risk...'):
        # Prepare the payload for the FastAPI "Nervous System"
        # We also need to add 'debt' since our brain expects it
        payload = {
            "income": float(income),
            "score": int(score),
            "debt": 0.2  # We can add a slider for this in the form later
        }
        
        try:
            # Send data to your FastAPI backend
            response = requests.post("http://127.0.0.1:8000/predict", json=payload)
            res = response.json()
            
            if res.get("status") == "Success":
                st.balloons()
                
                # Professional Metrics display using REAL AI data
                m1, m2, m3 = st.columns(3)
                m1.metric(label="Risk Status", value=res['decision'], delta="AI Decision")
                m2.metric(label="Confidence", value=res['confidence'], delta="Model Certainty")
                m3.metric(label="Agent Note", value=res['agent_note'])
            else:
                st.error("AI Engine returned an invalid response.")
                
        except Exception as e:
            st.error("❌ Connection Error: Is your FastAPI Backend running at http://127.0.0.1:8000?")
            st.info("Tip: Run 'uvicorn backend.main:app --reload' in your second terminal.")
# 7. Model Performance (Matching the image you shared)
st.markdown("---")
st.subheader("📈 Model Accuracy Comparison")
chart_data = pd.DataFrame({
    'Model': ['Logistic Regression', 'Decision Tree', 'Random Forest', 'XGBoost'],
    'Accuracy (%)': [79.0, 72.0, 80.0, 85.0]
})
st.bar_chart(chart_data.set_index('Model'))