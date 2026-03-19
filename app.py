import streamlit as st
import pandas as pd
import joblib

st.set_page_config(layout="wide")

model = joblib.load("model.pkl")

# Title
st.title("Supply Chain Delay Prediction")

st.write("Estimate the likelihood of shipment delay based on operational inputs.")

st.markdown("---")

# Layout
left, right = st.columns([1,1])

# LEFT SIDE
with left:
    st.subheader("Operational Inputs")

    delivery_days = st.slider("Delivery Duration (days)", 1, 10, 3)
    delay_gap = st.slider("Delay Gap (actual - scheduled)", -2, 5, 0)

    quantity = st.number_input("Order Quantity", value=1)
    price = st.number_input("Product Price", value=50.0)

    shipping_mode = st.selectbox(
        "Shipping Method",
        ["Standard Class", "Second Class", "First Class"]
    )

    region = st.selectbox(
        "Operational Region",
        ["West", "East", "Central", "South"]
    )

# RIGHT SIDE
with right:
    st.subheader("Financial Inputs")

    sales = st.number_input("Sales Value", value=100.0)

    st.subheader("Environmental Conditions")

    temperature = st.slider("Temperature", 0, 40, 25)
    humidity = st.slider("Humidity", 0.0, 1.0, 0.7)
    wind_speed = st.slider("Wind Speed", 0, 50, 10)

st.markdown("---")

# Prediction
if st.button("Run Prediction"):

    input_df = pd.DataFrame([{
        "Order Item Quantity": quantity,
        "Order Item Product Price": price,
        "Order Item Discount": 0,
        "Order Item Profit Ratio": 0.1,
        "Sales": sales,
        "delivery_days": delivery_days,
        "delay_gap": delay_gap,
        "temperature": temperature,
        "humidity": humidity,
        "wind_speed": wind_speed,
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

st.markdown("---")
st.caption("Model built using supply chain and environmental data")