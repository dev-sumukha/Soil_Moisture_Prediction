import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime

@st.cache_data
def load_data():
    data = pd.read_csv('soil-moisture FINAL.csv')
    data['Date'] = data['Month'] + ' ' + data['Day'].astype(str)
    data['Date'] = pd.to_datetime(data['Date'] + ' 2023', format='%b %d %Y')
    data['Day_of_Year'] = data['Date'].dt.dayofyear
    data['Month_Num'] = data['Date'].dt.month
    data = data.drop(columns=['Month', 'Day', 'Date'])

    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    month_encoded = encoder.fit_transform(data[['Month_Num']])
    month_encoded_df = pd.DataFrame(month_encoded, columns=encoder.get_feature_names_out(['Month_Num']), index=data.index)
    data = pd.concat([data, month_encoded_df], axis=1)
    data = data.drop(columns=['Month_Num'])

    return data, encoder

data, encoder = load_data()
X = data.drop(columns=['avg_sm'])
y = data['avg_sm']

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

def predict_future_sm(avg_pm1, avg_pm2, avg_pm3, avg_am, avg_lum, avg_temp, avg_humd, avg_pres, day_of_year):
    try:
        month_num = (datetime(2023, 1, 1) + pd.Timedelta(days=day_of_year - 1)).month
        future_data = np.array([[avg_pm1, avg_pm2, avg_pm3, avg_am, avg_lum, avg_temp, avg_humd, avg_pres, day_of_year]])
        month_encoded = encoder.transform([[month_num]])
        future_data = np.concatenate([future_data, month_encoded], axis=1)
        feature_names = list(X.columns)
        future_df = pd.DataFrame(future_data, columns=feature_names)

        prediction = model.predict(future_df)
        return prediction[0]
    except ValueError as e:
        st.error(f"Error: {e}")
        return None

def categorize_prediction(predicted_value):
    if predicted_value is None:
        return "No prediction available"
    if predicted_value >= 7000:
        return "Best"
    elif 5000 <= predicted_value < 7000:
        return "Better"
    else:
        return "Worst"

st.title('Soil Moisture Prediction')

avg_pm1 = st.number_input("Enter avg_pm1:", min_value=0.0, value=0.0)
avg_pm2 = st.number_input("Enter avg_pm2:", min_value=0.0, value=0.0)
avg_pm3 = st.number_input("Enter avg_pm3:", min_value=0.0, value=0.0)
avg_am = st.number_input("Enter avg_am:", min_value=0.0, value=0.0)
avg_lum = st.number_input("Enter avg_lum:", min_value=0.0, value=0.0)
avg_temp = st.number_input("Enter avg_temp (°C):", min_value=-50.0, max_value=60.0, value=25.0)
avg_humd = st.number_input("Enter avg_humd (%):", min_value=0.0, max_value=100.0, value=50.0)
avg_pres = st.number_input("Enter avg_pres (Pa):", min_value=80000.0, max_value=110000.0, value=101325.0)
day_of_year = st.number_input("Enter day of the year (1-365):", min_value=1, max_value=365, value=180)

if st.button('Predict Soil Moisture'):
    if 1 <= day_of_year <= 365:
        predicted_sm = predict_future_sm(avg_pm1, avg_pm2, avg_pm3, avg_am, avg_lum, avg_temp, avg_humd, avg_pres, day_of_year)
        category = categorize_prediction(predicted_sm)

        if predicted_sm is not None:
            st.success(f"Predicted Soil Moisture: {predicted_sm:.2f}")
            st.info(f"The predicted soil moisture level is categorized as: {category}")
    else:
        st.error("Please enter a valid day of the year (1-365).")
