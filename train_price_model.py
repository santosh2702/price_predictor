import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib



# 1. Load dataset (I'll use a simple auto-mpg dataset but rename target to 'price')
# For real used cars, download from https://www.kaggle.com/datasets/orgesleka/used-cars-database
# But for learning, let's create a synthetic dataset that mimics car prices
np.random.seed(42)
n = 1000
data = {
    'year': np.random.randint(1990, 2020, n),
    'mileage': np.random.randint(0, 200000, n),
    'fuel': np.random.choice(['Petrol', 'Diesel', 'CNG'], n),
    'brand': np.random.choice(['Toyota', 'Honda', 'Ford', 'BMW'], n),
    'price': np.random.normal(10000, 5000, n)  # noisy base price
}
df = pd.DataFrame(data)
# Make price somewhat realistic: older cars cheaper, higher mileage cheaper
df['price'] = df['price'] - (2020 - df['year']) * 300 + (df['mileage'] / 100) * -10
df['price'] = df['price'].clip(500, 60000).astype(int)

print(df.head())
print(f"Price range: ${df['price'].min()} - ${df['price'].max()}")




# 2. Separate features and target
X = df.drop('price', axis=1)
y = df['price']
    



# 3. Preprocessing: numeric features scale, categorical features one-hot encode
numeric_features = ['year', 'mileage']
categorical_features = ['fuel', 'brand']

preprocessor = ColumnTransformer([
    ('num', 'passthrough', numeric_features),  # we'll scale later if needed
    ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_features)
])

# 4. Create pipeline with Random Forest
model = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

# 5. Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Train
model.fit(X_train, y_train)

# 7. Evaluate
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"\nMean Absolute Error: ${mae:.2f}")
print(f"R² Score: {r2:.4f}")

# 8. Save model
joblib.dump(model, 'price_model.pkl')
print("Model saved as price_model.pkl")