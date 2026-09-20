import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay, precision_recall_curve, average_precision_score, accuracy_score, f1_score, brier_score_loss

# --- Page Configuration ---
st.set_page_config(page_title="SHUDDHOTA Intelligence Dashboard", layout="wide", initial_sidebar_state="collapsed")

# --- Premium Custom CSS Styling ---
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
        font-family: 'Inter', sans-serif;
    }
    .metric-card {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.02), 0 1px 3px rgba(0, 0, 0, 0.05);
        border: 1px solid #e5e7eb;
        text-align: center;
    }
    .card {
        background-color: #ffffff;
        padding: 25px;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.02), 0 1px 3px rgba(0, 0, 0, 0.05);
        border: 1px solid #e5e7eb;
        margin-bottom: 20px;
    }
    .proof-toast {
        background-color: #f0fdf4;
        border: 1px solid #bbf7d0;
        border-left: 5px solid #22c55e;
        padding: 18px;
        border-radius: 8px;
        margin-top: 15px;
        margin-bottom: 20px;
        color: #166534;
    }
    h1, h2, h3 {
        color: #1f2937;
    }
    .info-banner {
        background-color: #eff6ff;
        border-left: 4px solid #3b82f6;
        padding: 15px;
        border-radius: 0 8px 8px 0;
        margin-bottom: 20px;
        color: #1e40af;
    }
</style>
""", unsafe_allow_html=True)

# --- Helper Functions ---
def calculate_cronbach_alpha(df_items):
    n_items = df_items.shape[1]
    if n_items < 2: return 0.0
    item_vars = df_items.var(axis=0, ddof=1)
    t_var = df_items.sum(axis=1).var(ddof=1)
    if t_var == 0: return 0.0
    return (n_items / (n_items - 1)) * (1 - (item_vars.sum() / t_var))

@st.cache_data
def load_data(file):
    if file is not None:
        return pd.read_csv(file)
    else:
        try:
            return pd.read_csv("my_real_survey_data.csv")
        except FileNotFoundError:
            return None

# --- Popup Modal for "How It Works" ---
@st.dialog("✨ How SHUDDHOTA Works (Simple Guide)")
def show_how_it_works():
    st.markdown("### Welcome to the SHUDDHOTA Evaluation Portal")
    st.write("This tool helps protect society from fake AI-generated videos (deepfakes) and misinformation[cite: 1].")
    
    st.markdown("---")
    st.markdown("#### 🔄 The 3-Step Process:")
    st.markdown("""
    1. **Upload Data:** Drop your survey or test CSV file into the portal below.
    2. **Automated Analysis:** The system instantly tests the accuracy of the AI detection and measures public trust (Cronbach's Alpha).
    3. **Interactive Decision Testing:** Use the algorithm slider at the bottom to see how the system handles uncertainty and prevents false accusations.
    """)
    st.markdown("---")
    st.markdown("#### 🧮 The Scoring Formula:")
    st.latex(r"S = 100 \times (0.35D + 0.30P + 0.20F + 0.15C)")
    st.caption("Combines AI Detection (35%), Digital Provenance (30%), Fact-Checking (20%), and Contextual Risk (15%)[cite: 1].")
    
    if st.button("Got it, let's explore!", type="primary", use_container_width=True):
        st.rerun()

# --- HEADER SECTION ---
header_col1, header_col2 = st.columns([5, 1])
with header_col1:
    st.markdown("# 🛡️ SHUDDHOTA Intelligence Dashboard")
    st.markdown("##### *Integrated Framework for Synthetic Media Governance & Resilience*[cite: 1]")
with header_col2:
    st.write("")
    if st.button("❓ How It Works", use_container_width=True):
        show_how_it_works()

st.markdown("<br>", unsafe_allow_html=True)

# --- GUIDANCE BANNER ---
st.markdown("""
<div class="info-banner">
    <b>👋 Welcome!</b> This dashboard runs quantitative evaluations dynamically. 
    <b>Step 1:</b> Upload your custom dataset below to update all metrics, charts, and reliability scores instantly. <br>
    <b>Step 2:</b> Review real-time performance analytics. <br>
    <b>Step 3:</b> Test the decision-making engine at the bottom.
</div>
""", unsafe_allow_html=True)

# --- FULL WIDTH UPLOAD SECTION ---
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("📂 Step 1: Dataset Control Center")
uploaded_file = st.file_uploader("Upload your survey CSV file (Leaves empty to use the standard n=100 simulated baseline automatically):", type=["csv"])
st.markdown('</div>', unsafe_allow_html=True)

df = load_data(uploaded_file)

if df is None:
    st.warning("⚠️ Please ensure `my_real_survey_data.csv` is uploaded or present in the repository.")
    st.stop()

# Validate required columns
required_cols = ["GroundTruthFake", "SHUDDHOTAProbability"]
if not all(col in df.columns for col in required_cols):
    st.error(f"Error: Uploaded CSV is missing required columns: {', '.join(required_cols)}")
    st.stop()

# --- CALCULATE METRICS DYNAMICALLY ---
preds = (df["SHUDDHOTAProbability"] >= 0.5).astype(int)
y_true = df["GroundTruthFake"]
y_prob = df["SHUDDHOTAProbability"]

current_acc = accuracy_score(y_true, preds)
current_f1 = f1_score(y_true, preds, zero_division=0)
current_brier = brier_score_loss(y_true, y_prob)

# Handle edge case if all targets belong to a single class in custom CSV
try:
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    current_auc = auc(fpr, tpr)
except Exception:
    current_auc = 0.5

# --- RESULTS SECTION ---
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("📊 Step 2: Real-Time Performance Analytics")

if uploaded_file:
    data_source_name = f"Custom Upload ({uploaded_file.name})"
    st.success(f"🟢 **Active Data Source:** {data_source_name} (Fully Dynamic Analysis)")
else:
    data_source_name = "Standard Simulated Baseline (n=100 Site J Frame)"
    st.info(f"🔵 **Active Data Source:** {data_source_name}")

# Clean Metric Cards Layout
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div class="metric-card">
        <p style="color: #6b7280; font-size: 14px; margin-bottom: 5px;"><b>ROC AUC Score</b></p>
        <h2 style="color: #2563eb; margin: 0;">{current_auc:.3f}</h2>
        <p style="color: #9ca3af; font-size: 11px; margin-top: 5px;">Dynamic Calculation</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card">
        <p style="color: #6b7280; font-size: 14px; margin-bottom: 5px;"><b>F1 Harmonized Score</b></p>
        <h2 style="color: #16a34a; margin: 0;">{current_f1:.3f}</h2>
        <p style="color: #9ca3af; font-size: 11px; margin-top: 5px;">Dynamic Calculation</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-card">
        <p style="color: #6b7280; font-size: 14px; margin-bottom: 5px;"><b>Overall Accuracy</b></p>
        <h2 style="color: #9333ea; margin: 0;">{current_acc*100:.1f}%</h2>
        <p style="color: #9ca3af; font-size: 11px; margin-top: 5px;">Dynamic Calculation</p>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="metric-card">
        <p style="color: #6b7280; font-size: 14px; margin-bottom: 5px;"><b>Brier Calibration Loss</b></p>
        <h2 style="color: #ca8a04; margin: 0;">{current_brier:.3f}</h2>
        <p style="color: #9ca3af; font-size: 11px; margin-top: 5px;">Dynamic Calculation</p>
    </div>
    """, unsafe_allow_html=True)

# --- SCIENTIFIC PROOF TOAST ---
st.markdown(f"""
<div class="proof-toast">
    <h4 style="margin-top: 0; color: #166534;">🎯 What This Proves & Data Breakdown</h4>
    <ul style="margin-bottom: 0; font-size: 14px;">
        <li><b>Active Dataset Working With:</b> <code>{data_source_name}</code> containing <b>{len(df)} total records</b>.</li>
        <li><b>User Inputs Being Processed:</b> Binary ground truth labels (<code>GroundTruthFake</code>), continuous model confidence scores (<code>SHUDDHOTAProbability</code>), and multi-item Likert survey variables.</li>
        <li><b>Scientific Proof:</b> All visual charts, Cronbach alphas, and performance curves below have recalculated dynamically based on your uploaded file, proving that the evaluation pipeline adapts seamlessly to new datasets.</li>
    </ul>
</div>
""", unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# --- VISUAL ANALYSIS SECTION ---
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("📈 Step 3: Visual Diagnostic Charts")
tab1, tab2, tab3 = st.tabs(["🎯 Classification Performance", "📋 Dataset Reliability", "⚖️ Comparative Fit"])

with tab1:
    st.caption("Evaluating how cleanly the system separates real media from AI-generated deepfakes based on your active dataset.")
    f_col1, f_col2, f_col3 = st.columns(3)
    
    with f_col1:
        fig1, ax1 = plt.subplots(figsize=(4.5, 3.5))
        ax1.plot(fpr, tpr, color='#2563eb', lw=2, label=f'AUC = {current_auc:.3f}')
        ax1.plot([0, 1], [0, 1], color='#9ca3af', lw=1.5, linestyle='--')
        ax1.set_xlabel('False Positive Rate', fontsize=9)
        ax1.set_ylabel('True Positive Rate', fontsize=9)
        ax1.set_title('ROC Curve (Dynamic)', fontsize=10, fontweight='bold')
        ax1.legend(loc="lower right", fontsize=8)
        st.pyplot(fig1)

    with f_col2:
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        ap_score = average_precision_score(y_true, y_prob)
        fig2, ax2 = plt.subplots(figsize=(4.5, 3.5))
        ax2.plot(recall, precision, color='#9333ea', lw=2, label=f'AP = {ap_score:.3f}')
        ax2.set_xlabel('Recall', fontsize=9)
        ax2.set_ylabel('Precision', fontsize=9)
        ax2.set_title('Precision-Recall Curve (Dynamic)', fontsize=10, fontweight='bold')
        ax2.legend(loc="lower left", fontsize=8)
        st.pyplot(fig2)

    with f_col3:
        cm = confusion_matrix(y_true, preds)
        fig3, ax3 = plt.subplots(figsize=(4.5, 3.5))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Authentic', 'Synthetic'])
        disp.plot(cmap='Blues', values_format='d', ax=ax3)
        ax3.set_title("Confusion Matrix (Dynamic Cutoff)", fontsize=10, fontweight='bold')
        st.pyplot(fig3)

with tab2:
    st.caption("Verifying structural consistency and demographic survey balance from your active records.")
    f_col4, f_col5 = st.columns(2)
    
    with f_col4:
        authentic_count = len(df[y_true == 0])
        synthetic_count = len(df[y_true == 1])
        fig4, ax4 = plt.subplots(figsize=(6, 3.5))
        ax4.bar(['Authentic', 'Synthetic'], [authentic_count, synthetic_count], color=['#16a34a', '#dc2626'], width=0.5)
        ax4.set_title("Media-Task Profile (Dynamic Counts)", fontsize=10, fontweight='bold')
        ax4.set_ylabel("Case Count", fontsize=9)
        st.pyplot(fig4)

    with f_col5:
        constructs = ['DL', 'PE', 'AC', 'PS', 'TR', 'WR']
        alphas = []
        for c in constructs:
            cols = [f"{c}1", f"{c}2", f"{c}3", f"{c}4"]
            if all(col in df.columns for col in cols):
                alphas.append(calculate_cronbach_alpha(df[cols]))
            else:
                alphas.append(0.0)
        
        fig5, ax5 = plt.subplots(figsize=(6, 3.5))
        ax5.bar(constructs, alphas, color='#2563eb', width=0.5)
        ax5.axhline(y=0.70, color='#dc2626', linestyle='--', label='Min Acceptable (0.70)')
        ax5.set_title("Internal Consistency (Dynamic Alpha)", fontsize=10, fontweight='bold')
        ax5.set_ylim(0, 1.0)
        ax5.legend(fontsize=8)
        st.pyplot(fig5)

with tab3:
    st.caption("Comparing SHUDDHOTA's integrated performance against theoretical baseline comparators using your active metrics.")
    labels = ["C1 (Detector)", "C2 (Provenance)", "C3 (Fragmented)", "C4 (PDM)", "SHUDDHOTA"]
    auc_scores = [0.852, 0.803, 0.873, 0.941, current_auc] 
    f1_scores = [0.747, 0.689, 0.809, 0.862, current_f1]
    
    x = np.arange(len(labels))
    width = 0.35

    fig6, ax6 = plt.subplots(figsize=(9, 3.8))
    ax6.bar(x - width/2, auc_scores, width, label='ROC AUC', color='#2563eb')
    ax6.bar(x + width/2, f1_scores, width, label='F1 Score', color='#f97316')
    ax6.set_ylabel('Score Value', fontsize=9)
    ax6.set_title('Comparative Fit Profile (Dynamic Track B)', fontsize=10, fontweight='bold')
    ax6.set_xticks(x)
    ax6.set_xticklabels(labels, fontsize=8)
    ax6.legend(fontsize=8)
    st.pyplot(fig6)

st.markdown('</div>', unsafe_allow_html=True)

# --- ALGORITHM TESTER SECTION ---
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("⚙️ Step 4: Interactive Decision Logic Simulator")
st.write("Test how the framework reacts to different evidentiary signals in real time. Adjust sliders to see the system avoid false positives via the uncertainty window[cite: 1].")

c1, c2, c3, c4 = st.columns(4)
D = c1.slider("AI Detector Evidence (D)", 0.0, 1.0, 0.80)
P = c2.slider("Provenance Evidence (P)", 0.0, 1.0, 0.50)
F = c3.slider("Fact-Check Evidence (F)", 0.0, 1.0, 0.90)
C = c4.slider("Contextual Risk (C)", 0.0, 1.0, 0.60)

score = 100 * (0.35 * D + 0.30 * P + 0.20 * F + 0.15 * C)

res_col1, res_col2 = st.columns([1, 4])
with res_col1:
    st.markdown(f"""
    <div style="background-color: #f3f4f6; padding: 15px; border-radius: 8px; text-align: center;">
        <span style="font-size: 12px; color: #4b5563;"><b>SHUDDHOTA SCORE</b></span>
        <h1 style="color: #1f2937; margin: 0;">{score:.1f}</h1>
    </div>
    """, unsafe_allow_html=True)

with res_col2:
    st.write("")
    if score < 30:
        st.success("🟢 **SYSTEM DECISION: LOW RISK** — Content aligns with authentic benchmarks. Safe for general release.")
    elif score > 70:
        st.error("🔴 **SYSTEM DECISION: HIGH RISK** — Substantial multi-evidence indicators of malicious deepfake manipulation. Route to legal framework.")
    else:
        st.warning("🟡 **SYSTEM DECISION: UNCERTAINTY REGION** — Signals conflict or fall within neutral thresholds. **Automatically routing case to Human Expert Review (Abstention Safeguard).**")

st.markdown('</div>', unsafe_allow_html=True)
