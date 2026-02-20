import streamlit as st

st.set_page_config(
    page_title="Airfare Price Prediction",
    page_icon="✈️",
    layout="centered"
)

st.title("✈️ Airfare Price Prediction")
st.write("Estimate flight ticket prices using Streamlit and Python")

st.divider()

airline = st.selectbox(
    "Select Airline",
    ["IndiGo", "Air India", "Vistara", "SpiceJet", "GoAir"]
)

stops = st.selectbox(
    "Number of Stops",
    [0, 1, 2]
)

duration = st.slider(
    "Flight Duration (hours)",
    1, 15, 2
)

st.divider()

airline_price = {
    "IndiGo": 3000,
    "Air India": 3500,
    "Vistara": 4000,
    "SpiceJet": 2800,
    "GoAir": 3200
}

stop_price = {
    0: 0,
    1: 1500,
    2: 3000
}

cost_per_hour = 500

if st.button("Predict Price 💰"):
    total_price = (
        airline_price[airline]
        + stop_price[stops]
        + duration * cost_per_hour
    )

    st.success(f"Estimated Airfare Price: ₹ {total_price}")

st.caption("Beginner Streamlit Project | Python Fundamentals")


