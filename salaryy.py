import streamlit as st
import numpy as np
import pandas as pd
import time
from pathlib import Path

# ---------- Safe matplotlib import ----------
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

from sklearn.linear_model import LinearRegression

# ---------------- Page Config ----------------
st.set_page_config(
    page_title="Advanced Salary Predictor",
    page_icon="💼",
    layout="centered"
)

# ---------------- Load Dataset ----------------
@st.cache_data
def load_data():
    csv_path = Path("Salary_Data.csv")
    if not csv_path.exists():
        st.error("❌ CSV file not found. Please upload Salary_Data.csv to GitHub.")
        st.stop()
    return pd.read_csv(csv_path)

df = load_data()

# ---------------- Prepare Data ----------------
X = df.iloc[:, 0].values.reshape(-1, 1)   # Years of Experience
y = df.iloc[:, 1].values                 # Salary

# ---------------- Train Model ----------------
@st.cache_resource
def train_model(X, y):
    model = LinearRegression()
    model.fit(X, y)
    return model

model = train_model(X, y)

# ---------------- UI Header ----------------
st.title("💼 Advanced Salary Prediction System")
st.write("Real-time ML prediction trained directly from CSV data")

st.success("✅ Dataset loaded & model trained successfully")

# ---------------- User Input ----------------
years_experience = st.slider(
    "Select Years of Experience",
    float(X.min()),
    float(X.max()),
    5.0,
    0.5
)

# ---------------- Prediction ----------------
prediction = model.predict(np.array([[years_experience]]))[0]

st.subheader("🔴 Live Prediction")
st.info(f"Estimated Salary: ₹ {prediction:,.2f}")

# ---------------- Confidence Settings ----------------
CONFIDENCE_PERCENT = 0.10

# ---------------- Visualization ----------------
st.subheader("📈 Salary Trend with Confidence Band")

if HAS_MATPLOTLIB:
    plot_area = st.empty()

    x_full = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
    y_full = model.predict(x_full)

    upper = y_full * (1 + CONFIDENCE_PERCENT)
    lower = y_full * (1 - CONFIDENCE_PERCENT)

    for i in range(5, len(x_full) + 1):
        fig, ax = plt.subplots(figsize=(7, 4))

        ax.plot(x_full[:i], y_full[:i], label="Predicted Salary", linewidth=2)
        ax.fill_between(
            x_full[:i].flatten(),
            lower[:i],
            upper[:i],
            alpha=0.3,
            label="Confidence Range"
        )

        ax.scatter(years_experience, prediction, s=100, label="Your Input")

        ax.set_xlabel("Years of Experience")
        ax.set_ylabel("Salary")
        ax.set_title("Animated Salary Prediction")
        ax.legend()

        plot_area.pyplot(fig)
        plt.close(fig)
        time.sleep(0.03)
else:
    st.warning("Visualization unavailable (matplotlib missing).")

# ---------------- Final Output ----------------
st.subheader("🧠 Final Decision Output")

st.success(
    f"""
    **Final Salary Estimate:** ₹ {prediction:,.2f}  
    **Confidence Range:**  
    ₹ {(prediction * 0.9):,.2f} — ₹ {(prediction * 1.1):,.2f}
    """
)

# ---------------- Footer ----------------
st.markdown("---")
st.markdown("Built by **Anudhan Boss** | CSV-based ML Streamlit App")
