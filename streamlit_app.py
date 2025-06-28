import datetime
import streamlit as st
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from fpdf import FPDF
import re

# ---------------- Load model & encoders ----------------

model = {
    "Logistic Regression": joblib.load("models/model.pkl"),
    "Random Forest": joblib.load("models/random_forest.pkl")
}
tfidf = joblib.load('models/vectorizer.pkl')
le = joblib.load('models/encoder.pkl')
structured_columns = joblib.load("models/structured_columns.pkl")
feature_length = joblib.load("models/feature_length.pkl")

# ---------------- Streamlit Setup ----------------

st.set_page_config(page_title='Incident Priority Predictor', layout="wide")
st.title("🚨 Smart Incident Detection & Prioritization System")
st.markdown("---")
model_name = st.selectbox("🔍 Choose ML Model", list(model.keys()))
# ---------------- Sidebar Project Details ----------------

with st.sidebar:
    st.header("📘 Project Overview")
    st.markdown("""
This system predicts the **priority of IT incidents** using NLP and structured features from incident logs.

**👨‍💻 Built By:** `Kunj Mehta`  
**📚 Domain:** AI/ML, NLP, Incident Management  

### 🔍 Use Case
Organizations receive thousands of incident tickets daily. This tool helps automatically assign **priority levels (High, Medium, Low)** based on structured + unstructured data.

### 🧠 Core Technologies
- **ML Models:** Logistic Regression, Random Forest  
- **Text Features:** TF-IDF Vectorization  
- **Structured Features:** Impact, Urgency, State, Contact Type  
- **Interface:** Streamlit Web App  
- **Report:** Auto PDF Generation  
- **Learning:** Feedback Loop + EDA  
""")

    st.markdown("### 🚀 Standout Features")
    st.markdown("""
- 🎯 Real-time incident text prediction  
- 📈 Interactive EDA (no upload needed)  
- 🧪 Model comparison (switch in sidebar)  
- 🧾 PDF Report Generation  
- 💬 Human feedback collection  
""")



# ---------------- Utility Functions ----------------

def clean_text(text):
    txt = text.lower()
    txt = re.sub(r"[^a-zA-Z0-9\s]", "", txt)
    return re.sub(r"\s+", " ", txt).strip()

def predict(df_struct, texts, model):
    for col in structured_columns:
        if col not in df_struct.columns:
            df_struct[col] = 0
    df_struct = df_struct[structured_columns]

    tfidf_mat = tfidf.transform(texts.apply(clean_text))
    full = np.hstack([tfidf_mat.toarray(), df_struct.values])

    if full.shape[1] < feature_length:
        pad = np.zeros((full.shape[0], feature_length - full.shape[1]))
        full = np.hstack([full, pad])
    elif full.shape[1] > feature_length:
        full = full[:, :feature_length]

    preds = model.predict(full)
    probs = model.predict_proba(full)
    return le.inverse_transform(preds), probs

# ---------------- Session State ----------------

if 'log' not in st.session_state:
    st.session_state.log = pd.DataFrame(columns=["incident_text", "predicted_priority"])

# ---------------- Tabs ----------------

tabs = st.tabs([
    "📍 Single Incident",
    "📊 EDA Dashboard",
    "📝 Feedback & Export"
])

# --- Tab 1: Single Incident ---

with tabs[0]:
    st.subheader("📍 Single Incident Prediction")

    col1, col2 = st.columns(2)
    with col1:
        impact = st.selectbox("Impact", ["1 - High", "2 - Medium", "3 - Low"])
        urgency = st.selectbox("Urgency", ["1 - High", "2 - Medium", "3 - Low"])
        contact = st.selectbox("Contact Type", ["Phone", "Email", "Self service"])
        state = st.selectbox("State", ["New", "In Progress", "Resolved", "Closed"])
    with col2:
        category = st.selectbox("Category", ["Category 10", "Category 20", "Category 30", "Category 40"])
        subcategory = st.selectbox("Subcategory", ["Subcategory 100", "Subcategory 150", "Subcategory 200"])
        note = st.text_area("Custom Text (Optional)", "")

    text = note or f"A {category} - {subcategory} issue was reported via {contact} with {impact} impact and {urgency} urgency."
    struct_df = pd.get_dummies(pd.DataFrame([{
        "impact": impact, "urgency": urgency, "contact_type": contact,
        "incident_state": state, "category": category, "subcategory": subcategory
    }]))

    pred_label, probs = predict(struct_df, pd.Series([text]), model[model_name])
    st.metric("🎯 Predicted Priority", pred_label[0])
    st.bar_chart(pd.DataFrame(probs, columns=le.classes_))

    new_row = pd.DataFrame([{"incident_text": text, "predicted_priority": pred_label[0]}])
    st.session_state.log = pd.concat([st.session_state.log, new_row], ignore_index=True)

# --- Tab 2: EDA Dashboard ---
with tabs[1]:
    st.subheader("📊 Interactive EDA Dashboard")

    if not st.session_state.log.empty:
        df = st.session_state.log
        selected_col = st.selectbox("Choose Feature to Analyze", ["predicted_priority"])
        fig, ax = plt.subplots()
        sns.countplot(data=df, x=selected_col, palette="coolwarm", ax=ax)
        ax.set_title(f"Distribution of {selected_col}")
        st.pyplot(fig)
    else:
        st.info("Make at least one prediction to see EDA.")

# --- Tab 3: Feedback & Export ---
with tabs[2]:
    st.subheader("📝 Feedback and Report Export")

    df = st.session_state.log
    if not df.empty:
        idx = st.number_input("Select Prediction Row", min_value=0, max_value=len(df)-1, step=1)
        true_lbl = st.selectbox("🔁 Correct Priority", ["High", "Medium", "Low"])
        if st.button("✅ Submit Feedback"):
            df.loc[idx, "corrected_priority"] = true_lbl
            st.success("Feedback recorded!")
        st.write(df.loc[[idx]])

        # Simple PDF Generator
        def gen_pdf(df):
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=12)
            pdf.cell(0, 10, "Incident Prediction Report", ln=1, align="C")
            pdf.ln(5)

            for i, row in df.iterrows():
                # Strip non-ASCII for safety
                text = ''.join(c if ord(c) < 128 else '?' for c in row["incident_text"])
                label = ''.join(c if ord(c) < 128 else '?' for c in row["predicted_priority"])
                pdf.multi_cell(0, 8, f"{i+1}. {text} -> {label}")

            now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
            pdf.set_font("Arial", size=10)
            pdf.cell(0, 10, f"Generated on {now}", ln=1)
            return pdf.output(dest='S').encode('latin-1')

        pdf_bytes = gen_pdf(df)
        st.download_button("📄 Download PDF Report", pdf_bytes, "incident_report.pdf", "application/pdf")
    else:
        st.info("Make a prediction to enable report export.")
