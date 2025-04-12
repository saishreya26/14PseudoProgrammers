import streamlit as st
import pandas as pd
import requests
from datetime import datetime
from geopy.distance import geodesic
import joblib
from datetime import time

# Load model and encoder
model = joblib.load("model1.pkl")

label_encoder = joblib.load("label_encoder.pkl")

# Example
from datetime import datetime

# Geocode address using OpenStreetMap's Nominatim
def get_lat_lon(address):
    url = "https://nominatim.openstreetmap.org/search"
    params = {"q": address, "format": "json"}
    headers = {"User-Agent": "arrival-time-streamlit"}
    response = requests.get(url, params=params, headers=headers)
    data = response.json()
    if data:
        return float(data[0]["lat"]), float(data[0]["lon"])
    return None, None

def minutes_to_time(mins):
    hours = int(mins) // 60 % 24
    minutes = int(mins) % 60
    return f"{hours:02d}:{minutes:02d}"

# Prediction function
def predict_arrival(day, time_str, home_address, office_address):
    day_mapping = {
        'Monday': 0, 'Tuesday': 1, 'Wednesday': 2,
        'Thursday': 3, 'Friday': 4, 'Saturday': 5, 'Sunday': 6
    }
    day_code = day_mapping.get(day, 0)
    departure_hour = int(time_str.split(":")[0])
    traffic_encoded = 1
    home_coords = get_lat_lon(home_address)
    office_coords = get_lat_lon(office_address)

    if None in home_coords or None in office_coords:
        return "Error: Could not geocode one or more addresses."

    distance_km = geodesic(home_coords, office_coords).km

    input_df = pd.DataFrame([{
        'Day_of_Week_Code': day_code,
        'Direction': 0,
        'Departure_Hour': departure_hour,
        'distance_km': distance_km
    }])

    predicted_minutes = model.predict(input_df)[0]
    return minutes_to_time(predicted_minutes)

departure_dt = datetime.strptime("18:15", "%H:%M")
arrival_dt = datetime.strptime("19:00", "%H:%M")

# Make sure arrival is always *after* departure, maybe accounting for overnight if needed
duration_minutes = (arrival_dt - departure_dt).total_seconds() / 60

# Streamlit UI
st.title("🚗 Arrival Time Predictor")
st.markdown("Estimate your arrival time based on departure info, traffic, and distance.")

with st.form("prediction_form"):
    home_address = st.text_input("🏠 Home Address", "M G Road, Bengaluru")
    office_address = st.text_input("🏢 Office Address", "Electronic City, Bengaluru")
    day = st.selectbox("📅 Day of Week", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
    departure_time = st.time_input("🕒 Select Departure Time", value=time(8, 0))
    submitted = st.form_submit_button("Predict Arrival Time")

if submitted:
    time_str = departure_time.strftime("%H:%M")
    result = predict_arrival(day, time_str, home_address, office_address)
    st.success(f"🕓 Estimated Arrival Time: **{result}**")