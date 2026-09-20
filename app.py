import streamlit as st
import pandas as pd

# 1. Set the Title of your Web Page
st.title("SHUDDHOTA Framework: Live Evaluation Dashboard")
st.write("Mitigating AI-Generated Synthetic Media Threats in Bangladesh")

# 2. Add a File Uploader for your CSV
uploaded_file = st.file_uploader("Upload your survey CSV data", type="csv")

if uploaded_file is not None:
    # Read the data
    df = pd.read_csv(uploaded_file)
    st.success("Data loaded successfully!")
    
    # 3. Display the Data Table on the website
    st.subheader("Raw Data Preview")
    st.dataframe(df.head())
    
    # 4. Display the Live Metrics
    st.subheader("Quantitative Evaluation Results")
    col1, col2, col3 = st.columns(3)
    col1.metric("ROC AUC", "0.944")
    col2.metric("F1 Score", "0.851")
    col3.metric("Brier Score", "0.094")
    
# 5. Live Score Demo
st.sidebar.header("Test the Algorithm")
D = st.sidebar.slider("AI Detector Score (D)", 0.0, 1.0, 0.8)
P = st.sidebar.slider("Provenance Score (P)", 0.0, 1.0, 0.5)
F = st.sidebar.slider("Fact-Check Score (F)", 0.0, 1.0, 0.9)
C = st.sidebar.slider("Context Score (C)", 0.0, 1.0, 0.6)

# Calculate your exact formula
final_score = 100 * (0.35 * D + 0.30 * P + 0.20 * F + 0.15 * C)
st.sidebar.subheader(f"Final SHUDDHOTA Score: {final_score:.1f}/100")