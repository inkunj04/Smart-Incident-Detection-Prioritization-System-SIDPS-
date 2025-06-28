# 🚨 Smart Incident Detection & Prioritization System (SIDPS)

An intelligent AI-powered Streamlit application that predicts the **priority of IT incidents** using both structured and unstructured data like ticket logs, urgency, impact, and more. Designed to optimize ITSM workflows by helping teams triage tickets faster and more accurately.

> 🎓 Built with love by **Kunj Mehta**  
> 🔍 Domain: AI/ML + NLP | Interface: Streamlit | Status: ✅ Completed  
> 📢 Ideal for Final Year Projects, Hackathons, AI Showcases, or Resume Portfolios

---

## 🧠 Problem Statement

Modern organizations receive **hundreds to thousands** of incident tickets every day through multiple sources (emails, calls, self-service portals). Manual triage is **time-consuming, error-prone**, and often inconsistent.

> **SIDPS** solves this by **automatically predicting ticket priority (High, Medium, Low)** using machine learning models on historical structured fields and incident description text.

---

## 🚀 Key Features

| Feature | Description |
|--------|-------------|
| 🎯 **Real-time Priority Prediction** | Input structured details + free text. Get instant prediction. |
| 🧪 **Model Comparison** | Toggle between `Logistic Regression` and `Random Forest`. |
| 📊 **Interactive EDA Dashboard** | Visualize your prediction logs with dynamic charts. |
| 🧾 **PDF Report Generation** | Generate a summary report of all predictions with one click. |
| 💬 **Feedback Loop** | Submit corrected priority to improve the model or analyze errors. |
| 🧠 **Combined Feature Learning** | Uses both NLP (TF-IDF) and categorical features (Impact, Urgency, etc.). |

---

## 🧰 Tech Stack

| Layer | Tools Used |
|------|------------|
| 🧠 ML Models | Logistic Regression, Random Forest |
| 📚 Feature Engineering | TF-IDF Vectorizer, Label Encoding, One-Hot Encoding |
| 📊 Data Viz | Matplotlib, Seaborn |
| 🖥 Interface | Streamlit |
| 📄 PDF Generator | FPDF |
| 📝 Others | Pandas, NumPy, Joblib |

---

## 📖 Project Explanation

### 🎯 Objective

The **Smart Incident Detection & Prioritization System (SIDPS)** aims to automate the process of assigning a **priority level** (High, Medium, Low) to IT service tickets using machine learning. It combines the power of structured data (e.g., urgency, impact) and unstructured textual information (e.g., ticket descriptions) to make accurate predictions.

---

### 🧠 How It Works

1. **Input Collection:**
   - The user selects structured ticket information such as `Impact`, `Urgency`, `State`, `Contact Type`, etc.
   - Optionally, the user can input a free-form incident description (text).

2. **Preprocessing & Feature Engineering:**
   - Text is cleaned and transformed into numerical features using **TF-IDF Vectorization**.
   - Structured categorical inputs are one-hot encoded.
   - Both are concatenated to form a combined feature vector.

3. **Prediction Engine:**
   - The user can choose between two pre-trained ML models (`Logistic Regression` or `Random Forest`).
   - The model predicts the most probable priority level and provides class probabilities.

4. **Post-Prediction Features:**
   - Predictions are logged and can be analyzed via an **interactive EDA dashboard**.
   - Users can submit **human feedback** if the model prediction was inaccurate.
   - A full **PDF report** can be downloaded containing all logged predictions.

---

### ✅ Why It Matters

- Saves time by **automating triage**.
- Enhances **consistency** across support agents.
- Reduces **risk** by surfacing high-priority tickets earlier.
- Easy to use, customizable, and extendable.

---

## 📁 Project Structure

```bash
📦 Smart Incident Detection & Prioritization System
│
├── 📜 streamlit_app.py            # Streamlit front-end logic
├── 📁 models/
│   ├── model.pkl                 # Logistic Regression model
│   ├── random_forest.pkl         # Random Forest model
│   ├── encoder.pkl               # Label Encoder
│   ├── vectorizer.pkl            # TF-IDF Vectorizer
│   ├── structured_columns.pkl    # Structured feature list
│   └── feature_length.pkl        # Combined feature vector length
│
├── 📓 incident.ipynb             # Notebook for data prep and model training
└── README.md                     # Project documentation
