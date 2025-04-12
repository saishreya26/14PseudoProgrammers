# Re-import required packages due to kernel reset
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingRegressor
import joblib

model = joblib.load("model1.pkl")
label_encoder = joblib.load("label_encoder.pkl")

geolocator = Nominatim(user_agent="arrival_time_predictor")

def get_lat_lon(address):
    location = geolocator.geocode(address)
    if location:
        return location.latitude, location.longitude
    else:
        return None, None

def minutes_to_time(mins):
    hours = int(mins) // 60
    minutes = int(mins) % 60
    return f"{hours:02d}:{minutes:02d}"

def predict_arrival_time(day, time_str, home_address, office_address):
    # Convert day to categorical code
    day_mapping = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3,
                   'Friday': 4, 'Saturday': 5, 'Sunday': 6}
    day_code = day_mapping.get(day, 0)

    home_lat, home_lon = get_lat_lon(home_address)
    office_lat, office_lon = get_lat_lon(office_address)
    if None in [home_lat, home_lon, office_lat, office_lon]:
        return "Could not geocode one or more addresses."

    distance_km = geodesic((home_lat, home_lon), (office_lat, office_lon)).km

    departure_hour = int(time_str.split(":")[0])

    traffic_encoded = label_encoder.transform(["moderate"])

    input_data = pd.DataFrame([{
        'Day_of_Week': day_code,
        'Direction': 0,
        'Departure_Hour': departure_hour,
        'distance_km': distance_km,
        'traffic_conditions': traffic_encoded
    }])

    predicted_minutes = model.predict(input_data)[0]
    return minutes_to_time(predicted_minutes)

predict_arrival_time("Wednesday", "06:12", "M G Road, Bengaluru", "Electronic City, Bengaluru")
