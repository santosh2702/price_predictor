import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Cache the model so it trains only once
@st.cache_resource
def train_model():
    # Generate synthetic data (same as before)
    np.random.seed(42)
    n = 1000
    data = {
        'year': np.random.randint(1990, 2020, n),
        'mileage': np.random.randint(0, 200000, n),
        'fuel': np.random.choice(['Petrol', 'Diesel', 'CNG'], n),
        'brand': np.random.choice(['Toyota', 'Honda', 'Ford', 'BMW'], n),
        'price': np.random.normal(10000, 5000, n)
    }
    df = pd.DataFrame(data)
    df['price'] = df['price'] - (2020 - df['year']) * 300 + (df['mileage'] / 100) * -10
    df['price'] = df['price'].clip(500, 60000).astype(int)
    
    X = df.drop('price', axis=1)
    y = df['price']
    
    numeric_features = ['year', 'mileage']
    categorical_features = ['fuel', 'brand']
    
    preprocessor = ColumnTransformer([
        ('num', 'passthrough', numeric_features),
        ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_features)
    ])
    
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
    ])
    
    model.fit(X, y)
    return model

# Load or train the model (cached)
model = train_model()

# UI
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
    st.caption("Note: This is an estimate. Actual prices vary based on condition, location, etc.")