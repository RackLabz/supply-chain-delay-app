import streamlit as st
import pandas as pd
import json
import requests
import joblib
import tempfile

st.set_page_config(layout="wide")

# -------------------------------
# LOAD MODEL FROM GOOGLE DRIVE
# -------------------------------
@st.cache_resource
def load_model():
    url = "https://drive.google.com/uc?export=download&id=1Idw3tCBakPP39q8G78RJevKG_fmj2NP8"

    response = requests.get(url)

    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(response.content)
        tmp_path = tmp.name

    model = joblib.load(tmp_path)
    return model

model = load_model()

# -------------------------------
# UI
# -------------------------------
st.title("Supply Chain Delay Prediction System")
st.write("Predict delay risk, simulate outcomes, and get recommendations")

st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Operational Inputs")

    delivery_days = st.slider("Delivery Days", 1, 10, 3)
    delay_gap = st.slider("Delay Gap", -2, 5, 0)

    quantity = st.number_input("Quantity", value=1)
    price = st.number_input("Product Price", value=50.0)

    shipping_mode = st.selectbox(
        "Shipping Mode",
        ["Standard Class", "Second Class", "First Class"]
    )

    region = st.selectbox(
        "Order Region",
        ["West", "East", "Central", "South"]
    )

with col2:
    st.subheader("Financial Inputs")

    sales = st.number_input("Sales", value=100.0)

st.markdown("---")

# -------------------------------
# PREDICTION
# -------------------------------
if st.button("Run Prediction"):

    input_df = pd.DataFrame([{
        "Order Item Quantity": quantity,
        "Order Item Product Price": price,
        "Order Item Discount": 0,
        "Order Item Profit Ratio": 0.1,
        "Sales": sales,
        "delivery_days": delivery_days,
        "delay_gap": delay_gap,
        "Shipping Mode": shipping_mode,
        "Order Region": region
    }])

    prob = model.predict_proba(input_df)[0][1]

    if prob < 0.3:
        risk = "Low"
    elif prob < 0.7:
        risk = "Medium"
    else:
        risk = "High"

    st.subheader("Prediction Result")

    c1, c2 = st.columns(2)

    with c1:
        st.metric("Delay Probability", f"{prob:.2f}")

    with c2:
        st.metric("Risk Level", risk)

    # -------------------------------
    # AUTOMATION
    # -------------------------------
    st.markdown("---")
    st.subheader("System Recommendation")

    if prob > 0.7:
        st.warning("High risk detected. Expedite shipment.")
    elif prob > 0.4:
        st.info("Moderate risk. Monitor closely.")
    else:
        st.success("Low risk. Proceed normally.")

    # -------------------------------
    # WHAT-IF SIMULATION
    # -------------------------------
    st.markdown("---")
    st.subheader("What-if Simulation")

    test_days = st.slider("Test Delivery Days", 1, 10, delivery_days)

    test_input = input_df.copy()
    test_input["delivery_days"] = test_days

    test_prob = model.predict_proba(test_input)[0][1]

    st.write(f"New Delay Probability: {test_prob:.2f}")

    # -------------------------------
    # DOWNLOAD REPORT
    # -------------------------------
    st.markdown("---")

    report = {
        "probability": float(prob),
        "risk": risk,
        "delivery_days": delivery_days,
        "delay_gap": delay_gap
    }

    st.download_button(
        label="Download Prediction Report",
        data=json.dumps(report, indent=2),
        file_name="prediction_report.json",
        mime="application/json"
    )
