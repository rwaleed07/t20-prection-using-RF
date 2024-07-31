import streamlit as st
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load the trained model
model = joblib.load('t20model.joblib')

# Load the label encoder
label_encoder = LabelEncoder()

# Function to make predictions
def predict_winner(bat_first, bat_second):
    bat_first_encoded = label_encoder.transform([bat_first])[0]
    bat_second_encoded = label_encoder.transform([bat_second])[0]
    prediction = model.predict([[bat_first_encoded, bat_second_encoded]])
    predicted_winner = label_encoder.inverse_transform(prediction)[0]
    return predicted_winner

# Streamlit app layout
st.title("T20 Match Winner Predictor")

bat_first = st.selectbox('Select the team batting first', 
                         ['England', 'Australia', 'South Africa', 'New Zealand', 'Pakistan', 'Sri Lanka', 'WestIndies' , 'Zimbabwe', 'Kenya', 'Bangladesh','India'])  # Add all possible teams
bat_second = st.selectbox('Select the team batting second', 
                          ['England', 'Australia', 'South Africa', 'New Zealand', 'Pakistan', 'Sri Lanka', 'WestIndies' , 'Zimbabwe', 'Kenya', 'Bangladesh','India'])  # Add all possible teams

if st.button('Predict Winner'):
    result = predict_winner(bat_first, bat_second)
    st.write(f'The predicted winner is: {result}')
