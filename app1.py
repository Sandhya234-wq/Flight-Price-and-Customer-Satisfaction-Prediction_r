import streamlit as st
import pandas as pd
import joblib
from datetime import datetime

# Load trained pipeline model
model = joblib.load("flight_price_model.pkl")  # ← update with your actual model filename

# Streamlit App Configuration
st.set_page_config(page_title="✈️ Flight Price Predictor", layout="centered")
st.title("✈️ Flight Price Prediction App")
st.markdown("Predict the price of a flight based on journey details.")

# User Inputs
st.header("Enter Flight Details")

airline = st.selectbox("Airline", ['IndiGo', 'Air India', 'Jet Airways', 'SpiceJet', 'Vistara'])
source = st.selectbox("Source", ['Delhi', 'Kolkata', 'Mumbai', 'Chennai', 'Banglore'])
destination = st.selectbox("Destination", ['Cochin', 'Delhi', 'New Delhi', 'Hyderabad', 'Kolkata'])

date_of_journey = st.date_input("Date of Journey")
dep_time = st.time_input("Departure Time")
arr_time = st.time_input("Arrival Time")

total_stops = st.selectbox("Total Stops", ['non-stop', '1 stop', '2 stops', '3 stops', '4 stops'])

# Convert user inputs
journey_day = date_of_journey.day
journey_month = date_of_journey.month
dep_hour = dep_time.hour
dep_minute = dep_time.minute
arr_hour = arr_time.hour
arr_minute = arr_time.minute

# Duration calculation
duration_hours = abs(arr_hour - dep_hour)
duration_minutes = abs(arr_minute - dep_minute)

# Map total stops
stop_mapping = {'non-stop': 0, '1 stop': 1, '2 stops': 2, '3 stops': 3, '4 stops': 4}
total_stops_val = stop_mapping[total_stops]

# Prepare input DataFrame
input_df = pd.DataFrame({
    'Airline': [airline],
    'Source': [source],
    'Destination': [destination],
    'Total_Stops': [total_stops_val],
    'Journey_day': [journey_day],
    'Journey_month': [journey_month],
    'Dep_hour': [dep_hour],
    'Dep_minute': [dep_minute],
    'Arrival_hour': [arr_hour],
    'Arrival_minute': [arr_minute],
    'Duration_hours': [duration_hours],
    'Duration_minutes': [duration_minutes]
})

# Predict
if st.button("Predict Flight Price"):
    prediction = model.predict(input_df)[0]
    st.success(f"Estimated Flight Price: ₹{int(prediction):,}")
