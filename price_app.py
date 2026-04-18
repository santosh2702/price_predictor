import streamlit as st
import joblib
import pandas as pd
import numpy as np

model = joblib.load('price_model.pkl')

st.title("🚗 Used Car Price Predictor")
st.write("Enter car details to estimate its market price.")

col1, col2 = st.columns(2)
with col1:
    year = st.slider("Year", 1990, 2024, 2015)
    mileage = st.number_input("Mileage (miles)", min_value=0, max_value=300000, value=50000)
with col2:
    fuel = st.selectbox("Fuel Type", ['Petrol', 'Diesel', 'CNG'])
    brand = st.selectbox("Brand", ['Toyota', 'Honda', 'Ford', 'BMW'])

if st.button("Predict Price"):
    input_df = pd.DataFrame([[year, mileage, fuel, brand]], 
                            columns=['year', 'mileage', 'fuel', 'brand'])
    prediction = model.predict(input_df)[0]
    st.success(f"💰 Estimated Price: **${prediction:,.0f}**")
    
    # Show confidence interval (simple approximation)
    st.caption("Note: This is an estimate. Actual prices vary based on condition, location, etc.")