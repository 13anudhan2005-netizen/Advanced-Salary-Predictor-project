import streamlit as st
import pickle
import numpy as np
import matplotlib.pyplot as plt
import time

# ---------------- Page Config ----------------
st.set_page_config(
    page_title="Advanced Salary Predictor",
    page_icon="💼",
    layout="centered"
)

# ---------------- Load Model ----------------
model = pickle.load(open(r"Z:\FSDS\git\salaary\linear_regression_model.plk", 'rb'))


# ---------------- UI Header ----------------
st.title("💼 Advanced Salary Prediction System")
st.write("Real-time ML prediction with animated confidence visualization")

st.success("✅ Model loaded successfully")

# ---------------- Input ----------------
years_experience = st.slider(
    "Select Years of Experience",
    0.0, 50.0, 5.0, 0.5
)

# ---------------- Prediction ----------------
prediction = model.predict([[years_experience]])[0]

st.subheader("🔴 Live Prediction")
st.info(f"Estimated Salary: ₹ {prediction:,.2f}")

# ---------------- Confidence Settings ----------------
CONFIDENCE_PERCENT = 0.10   # ±10% uncertainty

# ---------------- Animation ----------------
st.subheader("📈 Real-Time Salary Trend with Confidence Band")

plot_area = st.empty()

x_full = np.arange(0, 51).reshape(-1, 1)
y_full = model.predict(x_full)

upper_band = y_full * (1 + CONFIDENCE_PERCENT)
lower_band = y_full * (1 - CONFIDENCE_PERCENT)

# Animate gradually
for i in range(2, len(x_full) + 1):
    fig, ax = plt.subplots(figsize=(7, 4))

    # Main prediction line
    ax.plot(
        x_full[:i],
        y_full[:i],
        linewidth=2,
        label="Predicted Salary"
    )

    # Confidence band
    ax.fill_between(
        x_full[:i].flatten(),
        lower_band[:i],
        upper_band[:i],
        alpha=0.3,
        label="Confidence Range"
    )

    # User input marker
    ax.scatter(
        years_experience,
        prediction,
        s=100,
        zorder=5,
        label="Your Input"
    )

    ax.set_xlim(0, 50)
    ax.set_ylim(0, max(upper_band) * 1.1)
    ax.set_xlabel("Years of Experience")
    ax.set_ylabel("Salary")
    ax.set_title("Animated Salary Prediction with Uncertainty")
    ax.legend()

    plot_area.pyplot(fig)
    plt.close(fig)

    time.sleep(0.04)

# ---------------- Final Confirmation ----------------
st.subheader("🧠 Final Decision Output")

st.success(
    f"""
    **Final Salary Estimate:** ₹ {prediction:,.2f}  
    **Confidence Range:**  
    ₹ {(prediction * 0.9):,.2f}  —  ₹ {(prediction * 1.1):,.2f}
    """
)

# ---------------- Footer ----------------
st.markdown("---")
st.markdown(
    "Built by **Anudhan Boss** | Advanced ML + Streamlit Application"
)
