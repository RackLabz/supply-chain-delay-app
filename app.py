import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

# Load dataset from Google Drive
df = pd.read_csv(
    "https://drive.google.com/uc?export=download&id=1gAW0M-orRx0BRh4eSsNk02bW9OqD_E7D",
    encoding="latin1"
)

# Target
df["delay_flag"] = df["Late_delivery_risk"]

# Feature engineering
df["order_date"] = pd.to_datetime(df["order date (DateOrders)"])
df["shipping_date"] = pd.to_datetime(df["shipping date (DateOrders)"])
df["delivery_days"] = (df["shipping_date"] - df["order_date"]).dt.days
df["delay_gap"] = df["Days for shipping (real)"] - df["Days for shipment (scheduled)"]

# Features
features = [
    "Order Item Quantity",
    "Order Item Product Price",
    "Order Item Discount",
    "Order Item Profit Ratio",
    "Sales",
    "delivery_days",
    "delay_gap",
    "Shipping Mode",
    "Order Region"
]

X = df[features]
y = df["delay_flag"]

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocessing
numeric_features = [
    "Order Item Quantity",
    "Order Item Product Price",
    "Order Item Discount",
    "Order Item Profit Ratio",
    "Sales",
    "delivery_days",
    "delay_gap"
]

categorical_features = ["Shipping Mode", "Order Region"]

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numeric_features),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
])

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier())
])

model.fit(X_train, y_train)

st.set_page_config(layout="wide")

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