import streamlit as st
import pickle
import numpy as np
import pandas as pd



st.set_page_config(page_title="California House Price Predictor" )
st.title(" California House Price Predictor")
st.write("Enter the details of the neighborhood to predict the median house value.")

@st.cache_resource
def load_assets():
    try:
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        return model, scaler
    except FileNotFoundError:
        st.error("Error: model.pkl or scaler.pkl not found!")
        return None, None

model, scaler = load_assets()


st.sidebar.header("Neighborhood Features")

median_income = st.sidebar.slider("Median Income ($10k)", min_value=0.5, max_value=15.0, value=5.0, step=0.1)
housing_median_age = st.sidebar.slider("Housing Median Age (Years)", min_value=1, max_value=50, value=25)
total_rooms = st.sidebar.number_input("Total Rooms in Block", min_value=10, value=1500)
total_bedrooms = st.sidebar.number_input("Total Bedrooms in Block", min_value=10, value=300)
population = st.sidebar.number_input("Total Population in Block", min_value=10, value=1000)
households = st.sidebar.number_input("Total Households", min_value=1, value=400)
latitude = st.sidebar.slider("Latitude", min_value=32.0, max_value=42.0, value=36.0)
longitude = st.sidebar.slider("Longitude", min_value=-125.0, max_value=-114.0, value=-119.0)


ocean_proximity = st.sidebar.selectbox("Ocean Proximity", ['<1H OCEAN', 'INLAND', 'ISLAND', 'NEAR BAY', 'NEAR OCEAN'])

if st.button("Predict House Price"):
    if model is not None and scaler is not None:
        
       
        rooms_per_household = total_rooms / households if households > 0 else 0
        bedrooms_per_room = total_bedrooms / total_rooms if total_rooms > 0 else 0
        population_per_household = population / households if households > 0 else 0
        

        prox_INLAND = 1 if ocean_proximity == 'INLAND' else 0
        prox_ISLAND = 1 if ocean_proximity == 'ISLAND' else 0
        prox_NEAR_BAY = 1 if ocean_proximity == 'NEAR BAY' else 0
        prox_NEAR_OCEAN = 1 if ocean_proximity == 'NEAR OCEAN' else 0
        
     
        input_data = pd.DataFrame({
            'longitude': [longitude],
            'latitude': [latitude],
            'housing_median_age': [housing_median_age],
            'total_rooms': [total_rooms],
            'total_bedrooms': [total_bedrooms],
            'population': [population],
            'households': [households],
            'median_income': [median_income],
            'rooms_per_household': [rooms_per_household],
            'bedrooms_per_room': [bedrooms_per_room],
            'population_per_household': [population_per_household],
            'ocean_proximity_INLAND': [prox_INLAND],
            'ocean_proximity_ISLAND': [prox_ISLAND],
            'ocean_proximity_NEAR BAY': [prox_NEAR_BAY],
            'ocean_proximity_NEAR OCEAN': [prox_NEAR_OCEAN]
        })
        
        try:
          
            input_scaled = scaler.transform(input_data)
            prediction = model.predict(input_scaled)
            
          
            predicted_price = prediction[0] * 100000
            
            st.success(f"### Estimated Median House Value: ${predicted_price:,.2f}")
            st.balloons() 
            
        except Exception as e:
            st.error(f"Prediction Error: {e}")