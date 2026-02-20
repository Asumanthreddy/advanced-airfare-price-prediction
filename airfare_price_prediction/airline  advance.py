import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Set page config
st.set_page_config(
    page_title="Advanced Airfare Price Prediction",
    page_icon="✈️",
    layout="wide"
)

# Sidebar for settings
st.sidebar.title("Settings")
st.sidebar.write("Adjust parameters for prediction.")
random_seed = st.sidebar.slider("Random Seed for Variability", 0, 100, 42)
peak_season_multiplier = st.sidebar.slider("Peak Season Multiplier", 1.0, 2.0, 1.5)

# Main title
st.title("✈️ Advanced Airfare Price Prediction")
st.write("Estimate flight ticket prices using machine learning and advanced features.")

st.divider()

# Function to generate dummy dataset
def generate_dummy_data(num_samples=1000):
    np.random.seed(random_seed)
    airlines = ["IndiGo", "Air India", "Vistara", "SpiceJet", "GoAir"]
    cities = ["Delhi", "Mumbai", "Bangalore", "Chennai", "Kolkata"]
    classes = ["Economy", "Business", "First"]
    
    data = {
        "Airline": np.random.choice(airlines, num_samples),
        "Stops": np.random.choice([0, 1, 2], num_samples),
        "Duration": np.random.uniform(1, 15, num_samples),
        "Departure": np.random.choice(cities, num_samples),
        "Arrival": np.random.choice(cities, num_samples),
        "Class": np.random.choice(classes, num_samples),
        "Date": pd.date_range(start="2023-01-01", periods=num_samples, freq="D"),
        "Price": np.random.uniform(2000, 15000, num_samples)
    }
    df = pd.DataFrame(data)
    # Ensure Departure != Arrival
    df["Arrival"] = df.apply(lambda row: np.random.choice([c for c in cities if c != row["Departure"]]), axis=1)
    # Adjust price based on factors
    df["Price"] += df["Stops"] * 1500 + df["Duration"] * 500
    df["Price"] *= np.where(df["Class"] == "Business", 2, 1)
    df["Price"] *= np.where(df["Class"] == "First", 3, 1)
    # Peak season (e.g., Dec-Mar)
    df["Month"] = df["Date"].dt.month
    df["Price"] *= np.where(df["Month"].isin([12, 1, 2, 3]), peak_season_multiplier, 1)
    return df

# Function to train model
def train_model(df):
    # Encode categorical variables
    le_airline = LabelEncoder()
    le_departure = LabelEncoder()
    le_arrival = LabelEncoder()
    le_class = LabelEncoder()
    
    df["Airline_encoded"] = le_airline.fit_transform(df["Airline"])
    df["Departure_encoded"] = le_departure.fit_transform(df["Departure"])
    df["Arrival_encoded"] = le_arrival.fit_transform(df["Arrival"])
    df["Class_encoded"] = le_class.fit_transform(df["Class"])
    
    # Features and target
    X = df[["Airline_encoded", "Stops", "Duration", "Departure_encoded", "Arrival_encoded", "Class_encoded", "Month"]]
    y = df["Price"]
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_seed)
    
    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    return model, le_airline, le_departure, le_arrival, le_class

# Load data and train model
df = generate_dummy_data()
model, le_airline, le_departure, le_arrival, le_class = train_model(df)

# User inputs in columns
col1, col2 = st.columns(2)

with col1:
    airline = st.selectbox(
        "Select Airline",
        ["IndiGo", "Air India", "Vistara", "SpiceJet", "GoAir"],
        help="Choose the airline for your flight."
    )
    stops = st.selectbox(
        "Number of Stops",
        [0, 1, 2],
        help="Number of layovers."
    )
    duration = st.slider(
        "Flight Duration (hours)",
        1.0, 15.0, 2.0,
        help="Estimated flight time."
    )

with col2:
    departure = st.selectbox(
        "Departure City",
        ["Delhi", "Mumbai", "Bangalore", "Chennai", "Kolkata"],
        help="Starting city."
    )
    arrival = st.selectbox(
        "Arrival City",
        ["Delhi", "Mumbai", "Bangalore", "Chennai", "Kolkata"],
        help="Destination city."
    )
    flight_class = st.selectbox(
        "Flight Class",
        ["Economy", "Business", "First"],
        help="Cabin class."
    )

travel_date = st.date_input(
    "Travel Date",
    min_value=datetime.today(),
    help="Select your travel date."
)

st.divider()

# Prediction logic
if st.button("Predict Price 💰"):
    # Validation
    if departure == arrival:
        st.error("Departure and Arrival cities cannot be the same!")
    elif travel_date < datetime.today().date():
        st.error("Travel date must be in the future!")
    else:
        with st.spinner("Calculating price..."):
            # Encode inputs
            airline_encoded = le_airline.transform([airline])[0]
            departure_encoded = le_departure.transform([departure])[0]
            arrival_encoded = le_arrival.transform([arrival])[0]
            class_encoded = le_class.transform([flight_class])[0]
            month = travel_date.month
            
            # Prepare input for model
            input_data = np.array([[airline_encoded, stops, duration, departure_encoded, arrival_encoded, class_encoded, month]])
            
            # Predict
            predicted_price = model.predict(input_data)[0]
            # Add some randomness for realism
            predicted_price += np.random.normal(0, 500)
            predicted_price = max(predicted_price, 1000)  # Minimum price
            
            st.success(f"Estimated Airfare Price: ₹ {predicted_price:.2f}")
            
            # Breakdown chart
            components = {
                "Base Price": predicted_price * 0.5,
                "Stops Adjustment": stops * 1500,
                "Duration Cost": duration * 500,
                "Class Multiplier": predicted_price * (0.5 if flight_class == "Economy" else 1.0 if flight_class == "Business" else 1.5),
                "Seasonal Adjustment": predicted_price * (peak_season_multiplier - 1) if month in [12, 1, 2, 3] else 0
            }
            fig, ax = plt.subplots()
            ax.bar(components.keys(), components.values())
            ax.set_ylabel("Price Contribution (₹)")
            ax.set_title("Price Breakdown")
            st.pyplot(fig)
            
            # Summary table
            st.subheader("Prediction Summary")
            summary = pd.DataFrame({
                "Input": ["Airline", "Stops", "Duration", "Departure", "Arrival", "Class", "Date"],
                "Value": [airline, stops, f"{duration} hours", departure, arrival, flight_class, travel_date]
            })
            st.table(summary)

# Reset button
if st.button("Reset Inputs 🔄"):
    st.experimental_rerun()

st.caption("Advanced Streamlit Project | Machine Learning Integration | Python Fundamentals")