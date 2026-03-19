import streamlit as st
import pandas as pd
import json

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(layout="wide")

# -------------------------------
# LOAD DATA (CACHED)
# -------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv(
        "https://drive.google.com/uc?export=download&id=1gAW0M-orRx0BRh4eSsNk02bW9OqD_E7D",
        encoding="latin1"
    )
    return df

# -------------------------------
# TRAIN MODEL (CACHED)
# -------------------------------
@st.cache_resource
def train_model(df):

    df["delay_flag"] = df["Late_delivery_risk"]

    df["order_date"] = pd.to_datetime(df["order date (DateOrders)"])
    df["shipping_date"] = pd.to_datetime(df["shipping date (DateOrders)"])

    df["delivery_days"] = (df["shipping_date"] - df["order_date"]).dt.days
    df["delay_gap"] = df["Days for shipping (real)"] - df["Days for shipment (scheduled)"]

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

    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

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

    return model

# -------------------------------
# LOAD SYSTEM
# -------------------------------
df = load_data()
model = train_model(df)

# -------------------------------
# DASHBOARD
# -------------------------------
st.title("Supply Chain Delay Prediction System")
st.write("Predict delays, get recommendations, and simulate outcomes")

st.markdown("---")

st.subheader("Historical Delay Distribution")
delay_counts = df["Late_delivery_risk"].value_counts()
st.bar_chart(delay_counts)

st.markdown("---")

# -------------------------------
# INPUT UI
# -------------------------------
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
