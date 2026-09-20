import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay, precision_recall_curve, average_precision_score, accuracy_score, f1_score, brier_score_loss

# --- Page Configuration ---
st.set_page_config(page_title="SHUDDHOTA Dashboard", layout="wide", initial_sidebar_state="collapsed")

# --- Helper Functions ---
def calculate_cronbach_alpha(df_items):
    """Calculates internal consistency for survey items."""
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
@st.dialog("⚙️ How the SHUDDHOTA Pipeline Works")
def show_how_it_works():
    st.write("This dashboard mathematically demonstrates the SHUDDHOTA framework designed for Bangladesh.")
    st.markdown("""
    **1. The 6-Layer Architecture:**
    * **L1 - Helpline:** Citizen entry point.
    * **L2 - Score:** Multi-evidence output (Demonstrated below).
    * **L3 - Card:** Media literacy.
    * **L4 - Shokti:** Community support.
    * **L5 - Court:** Urgent protection.
    * **L6 - Fund:** Victim assistance.
    
    **2. Quantitative Processing:**
    * **Data Input:** The system reads the CSV containing Likert responses and binary ground truth.
    * **Reliability Check:** Calculates Cronbach's Alpha to ensure survey constructs are reliable (>0.70).
    * **Model Evaluation:** Compares the simulated AI probability against the ground truth to generate the ROC AUC, Precision-Recall, and Brier metrics.
    """)
    st.markdown("**Scoring Formula (Section 8.4.2):**")
    st.latex(r"S = 100 \times (0.35D + 0.30P + 0.20F + 0.15C)")
    if st.button("Close"):
        st.rerun()

# --- HEADER SECTION ---
col_title, col_button = st.columns([4, 1])
with col_title:
    st.title("SHUDDHOTA Framework Dashboard")
    st.markdown("Mitigating AI-Generated Synthetic Media Threats in Bangladesh")
with col_button:
    st.write("") # Spacing
    if st.button("ℹ️ How It Works (Popup)", use_container_width=True):
        show_how_it_works()

st.divider()

# --- FULL WIDTH UPLOAD SECTION ---
st.subheader("📂 1. Upload Dataset")
uploaded_file = st.file_uploader("Upload your survey CSV data to generate real-time metrics. (Defaults to simulated n=100 baseline if empty).", type=["csv"])

df = load_data(uploaded_file)

if df is None:
    st.warning("Please upload a valid CSV file to view the results.")
    st.stop()

# Validate required columns
required_cols = ["GroundTruthFake", "SHUDDHOTAProbability"]
if not all(col in df.columns for col in required_cols):
    st.error(f"Error: Uploaded CSV must contain columns: {', '.join(required_cols)}")
    st.stop()

# --- CALCULATE DYNAMIC METRICS ---
preds = (df["SHUDDHOTAProbability"] >= 0.5).astype(int)
y_true = df["GroundTruthFake"]
y_prob = df["SHUDDHOTAProbability"]

current_acc = accuracy_score(y_true, preds)
current_f1 = f1_score(y_true, preds)
current_brier = brier_score_loss(y_true, y_prob)
fpr, tpr, _ = roc_curve(y_true, y_prob)
current_auc = auc(fpr, tpr)

# --- RESULTS SECTION ---
st.subheader("📊 2. Evaluation Results")
if uploaded_file:
    st.success(f"Custom Data Loaded Successfully: Analyzing {len(df)} records.")
else:
    st.info(f"Using Baseline Simulated Data: Analyzing {len(df)} records.")

# Metrics Row
m1, m2, m3, m4 = st.columns(4)
m1.metric(label="ROC AUC", value=f"{current_auc:.4f}")
m2.metric(label="F1 Score", value=f"{current_f1:.4f}")
m3.metric(label="Accuracy", value=f"{current_acc:.4f}")
m4.metric(label="Brier Score", value=f"{current_brier:.4f}")

st.divider()

# --- GRAPHS SECTION (TABS) ---
st.subheader("📈 3. Visual Analysis")
tab1, tab2, tab3 = st.tabs(["Classification Performance", "Dataset & Reliability", "Comparative Analysis"])

with tab1:
    fig_col1, fig_col2, fig_col3 = st.columns(3)
    
    with fig_col1:
        fig1, ax1 = plt.subplots(figsize=(5, 4))
        ax1.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {current_auc:.3f}')
        ax1.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax1.set_xlabel('False Positive Rate')
        ax1.set_ylabel('True Positive Rate')
        ax1.set_title('ROC Curve')
        ax1.legend(loc="lower right")
        st.pyplot(fig1)

    with fig_col2:
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        ap_score = average_precision_score(y_true, y_prob)
        fig2, ax2 = plt.subplots(figsize=(5, 4))
        ax2.plot(recall, precision, color='purple', lw=2, label=f'AP = {ap_score:.3f}')
        ax2.set_xlabel('Recall')
        ax2.set_ylabel('Precision')
        ax2.set_title('Precision-Recall Curve')
        ax2.legend(loc="lower left")
        st.pyplot(fig2)

    with fig_col3:
        cm = confusion_matrix(y_true, preds)
        fig3, ax3 = plt.subplots(figsize=(5, 4))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Authentic', 'Synthetic'])
        disp.plot(cmap='Blues', values_format='d', ax=ax3)
        ax3.set_title("Confusion Matrix (Cutoff: 0.5)")
        st.pyplot(fig3)

with tab2:
    fig_col4, fig_col5 = st.columns(2)
    
    with fig_col4:
        authentic_count = len(df[y_true == 0])
        synthetic_count = len(df[y_true == 1])
        fig4, ax4 = plt.subplots(figsize=(6, 4))
        ax4.bar(['Authentic', 'Synthetic'], [authentic_count, synthetic_count], color=['#2ca02c', '#d62728'])
        ax4.set_title("Simulated Media-Task Profile")
        ax4.set_ylabel("Cases")
        st.pyplot(fig4)

    with fig_col5:
        constructs = ['DL', 'PE', 'AC', 'PS', 'TR', 'WR']
        alphas = []
        for c in constructs:
            cols = [f"{c}1", f"{c}2", f"{c}3", f"{c}4"]
            if all(col in df.columns for col in cols):
                alphas.append(calculate_cronbach_alpha(df[cols]))
            else:
                alphas.append(0.0)
        
        fig5, ax5 = plt.subplots(figsize=(6, 4))
        ax5.bar(constructs, alphas, color='#1f77b4')
        ax5.axhline(y=0.70, color='red', linestyle='--', label='Threshold (0.70)')
        ax5.set_title("Internal Consistency (Cronbach's Alpha)")
        ax5.set_ylim(0, 1.0)
        st.pyplot(fig5)

with tab3:
    labels = ["C1", "C2", "C3", "C4", "SHUDDHOTA"]
    auc_scores = [0.852, 0.803, 0.873, 0.941, current_auc] 
    f1_scores = [0.747, 0.689, 0.809, 0.862, current_f1]
    x = np.arange(len(labels))
    width = 0.35

    fig6, ax6 = plt.subplots(figsize=(8, 4))
    ax6.bar(x - width/2, auc_scores, width, label='ROC AUC', color='#1f77b4')
    ax6.bar(x + width/2, f1_scores, width, label='F1 Score', color='#ff7f0e')
    ax6.set_ylabel('Scores')
    ax6.set_title('Comparator Metric Profile (Track B)')
    ax6.set_xticks(x)
    ax6.set_xticklabels(labels)
    ax6.legend()
    st.pyplot(fig6)

st.divider()

# --- ALGORITHM TESTER (Moved to main body for clean full-width look) ---
st.subheader("⚙️ 4. Test SHUDDHOTA Decision Logic")
st.write("Adjust the evidence weights to see how the system handles uncertainty. The integrated model uses calibrated detector evidence alongside provenance, fact-check, and contextual risk.")

c1, c2, c3, c4 = st.columns(4)
D = c1.slider("AI Detector Evidence (D)", 0.0, 1.0, 0.80)
P = c2.slider("Provenance Evidence (P)", 0.0, 1.0, 0.50)
F = c3.slider("Fact-Check Evidence (F)", 0.0, 1.0, 0.90)
C = c4.slider("Contextual Risk (C)", 0.0, 1.0, 0.60)

score = 100 * (0.35 * D + 0.30 * P + 0.20 * F + 0.15 * C)

res_col1, res_col2 = st.columns([1, 3])
res_col1.metric(label="Calculated Score", value=f"{score:.1f} / 100")

with res_col2:
    st.write("") # Vertical alignment
    if score < 30:
        st.success("🟢 **DECISION:** LOW RISK. Supported as authentic.")
    elif score > 70:
        st.error("🔴 **DECISION:** HIGH RISK. Evidence of material synthesis.")
    else:
        st.warning("🟡 **DECISION:** UNCERTAINTY REGION. Escalate to Human Review.")
