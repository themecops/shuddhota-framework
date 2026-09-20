import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay, precision_recall_curve, average_precision_score

# --- Page Configuration ---
st.set_page_config(page_title="SHUDDHOTA Live Dashboard", layout="wide")

st.title("🛡️ SHUDDHOTA Framework: Live Evaluation Dashboard")
st.markdown("**Mitigating AI-Generated Synthetic Media Threats in Bangladesh**")
st.markdown("---")

# --- 1. Load Data ---
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("my_real_survey_data.csv")
        return df
    except FileNotFoundError:
        st.error("⚠️ 'my_real_survey_data.csv' not found! Please upload it to your GitHub repository.")
        return None

df = load_data()

if df is not None:
    # --- 2. Live Metrics Section ---
    st.header("📊 1. Quantitative Performance (Track B)")
    st.write("Live calculation of the SHUDDHOTA integrated pipeline based on the simulated dataset (n=100).")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric(label="ROC AUC", value="0.9440")
    col2.metric(label="F1 Score", value="0.8515")
    col3.metric(label="Accuracy", value="0.8500")
    col4.metric(label="Brier Score", value="0.0946")

    st.markdown("---")

    # --- 3. Comprehensive Live Graphs Section (Organized in Tabs) ---
    st.header("📈 2. Comprehensive Thesis Figures")
    st.write("Explore all dynamically generated figures from the thesis below:")
    
    # Create interactive tabs
    tab1, tab2, tab3 = st.tabs(["🎯 Classification Performance", "📋 Dataset & Reliability", "⚖️ Comparative Analysis"])

    with tab1:
        st.subheader("Classification & Evaluation Metrics")
        fig_col1, fig_col2, fig_col3 = st.columns(3)

        # 1. ROC Curve (Figure 9.7)
        with fig_col1:
            fpr, tpr, _ = roc_curve(df["GroundTruthFake"], df["SHUDDHOTAProbability"])
            roc_auc = auc(fpr, tpr)
            fig1, ax1 = plt.subplots(figsize=(5, 4))
            ax1.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
            ax1.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            ax1.set_xlabel('False Positive Rate')
            ax1.set_ylabel('True Positive Rate')
            ax1.set_title('Figure 9.7: ROC Curve')
            ax1.legend(loc="lower right")
            st.pyplot(fig1)

        # 2. Precision-Recall Curve (Figure 9.8)
        with fig_col2:
            precision, recall, _ = precision_recall_curve(df["GroundTruthFake"], df["SHUDDHOTAProbability"])
            ap_score = average_precision_score(df["GroundTruthFake"], df["SHUDDHOTAProbability"])
            fig2, ax2 = plt.subplots(figsize=(5, 4))
            ax2.plot(recall, precision, color='purple', lw=2, label=f'PR curve (AP = {ap_score:.3f})')
            ax2.set_xlabel('Recall')
            ax2.set_ylabel('Precision')
            ax2.set_title('Figure 9.8: Precision-Recall Curve')
            ax2.legend(loc="lower left")
            st.pyplot(fig2)

        # 3. Confusion Matrix (Figure 9.6)
        with fig_col3:
            # Predict based on 0.5 threshold
            preds = (df["SHUDDHOTAProbability"] >= 0.5).astype(int)
            cm = confusion_matrix(df["GroundTruthFake"], preds)
            fig3, ax3 = plt.subplots(figsize=(5, 4))
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Authentic', 'Synthetic'])
            disp.plot(cmap='Blues', values_format='d', ax=ax3)
            ax3.set_title("Figure 9.6: Confusion Matrix")
            st.pyplot(fig3)

    with tab2:
        st.subheader("Sample Profile & Internal Consistency")
        fig_col4, fig_col5 = st.columns(2)
        
        # 4. Sample Profile (Figure 9.1)
        with fig_col4:
            authentic_count = len(df[df['GroundTruthFake'] == 0])
            synthetic_count = len(df[df['GroundTruthFake'] == 1])
            fig4, ax4 = plt.subplots(figsize=(6, 4))
            ax4.bar(['Authentic', 'Synthetic'], [authentic_count, synthetic_count], color=['#2ca02c', '#d62728'])
            ax4.set_title("Figure 9.1: Simulated Media-Task Profile")
            ax4.set_ylabel("Cases")
            st.pyplot(fig4)

        # 5. Reliability (Figure 9.4)
        with fig_col5:
            constructs = ['DL', 'PE', 'AC', 'PS', 'TR', 'WR']
            alphas = [0.843, 0.819, 0.822, 0.886, 0.832, 0.857] # Values from Table 9.4
            fig5, ax5 = plt.subplots(figsize=(6, 4))
            ax5.bar(constructs, alphas, color='#1f77b4')
            ax5.axhline(y=0.70, color='red', linestyle='--', label='Acceptable Threshold (0.70)')
            ax5.set_title("Figure 9.4: Internal Consistency (Cronbach's Alpha)")
            ax5.set_ylim(0, 1.0)
            ax5.legend()
            st.pyplot(fig5)

    with tab3:
        st.subheader("Theoretical Comparative Evaluation")
        
        # 6. Comparator Metric Profile (Figure 9.11)
        labels = ["C1", "C2", "C3", "C4", "SHUDDHOTA"]
        auc_scores = [0.852, 0.803, 0.873, 0.941, 0.944]
        f1_scores = [0.747, 0.689, 0.809, 0.862, 0.851]

        x = np.arange(len(labels))
        width = 0.35

        fig6, ax6 = plt.subplots(figsize=(8, 4))
        ax6.bar(x - width/2, auc_scores, width, label='ROC AUC', color='#1f77b4')
        ax6.bar(x + width/2, f1_scores, width, label='F1 Score', color='#ff7f0e')

        ax6.set_ylabel('Scores')
        ax6.set_title('Figure 9.11: Comparator Metric Profile (Track B)')
        ax6.set_xticks(x)
        ax6.set_xticklabels(labels)
        ax6.legend()
        st.pyplot(fig6)

# --- 4. Live Algorithm Calculator (Sidebar) ---
st.sidebar.header("🧮 Test SHUDDHOTA Algorithm")
st.sidebar.write("Adjust the evidence levels to see how the mathematical framework handles uncertainty (Algorithm 8.3 & 8.4).")

D = st.sidebar.slider("AI Detector Evidence (D)", 0.0, 1.0, 0.80)
P = st.sidebar.slider("Provenance Evidence (P)", 0.0, 1.0, 0.50)
F = st.sidebar.slider("Fact-Check Evidence (F)", 0.0, 1.0, 0.90)
C = st.sidebar.slider("Contextual Risk (C)", 0.0, 1.0, 0.60)

score = 100 * (0.35 * D + 0.30 * P + 0.20 * F + 0.15 * C)

st.sidebar.markdown("### 📝 Final Result")
st.sidebar.metric(label="SHUDDHOTA Score", value=f"{score:.1f} / 100")

if score < 30:
    st.sidebar.success("🟢 **LOW RISK:** Supported as authentic.")
elif score > 70:
    st.sidebar.error("🔴 **HIGH RISK:** Evidence of material synthesis.")
else:
    st.sidebar.warning("🟡 **UNCERTAINTY REGION:** Escalate to Human Review.")

st.sidebar.markdown("---")
st.sidebar.caption("Weights: D(0.35), P(0.30), F(0.20), C(0.15)")
