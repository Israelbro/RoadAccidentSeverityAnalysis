# Step 1: Import Libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import joblib

# Step 2: Load and Preprocess Data
df = pd.read_csv('road_accidents.csv')
print(df.head())

# Selecting features and target variable
X = df[['Time_of_Accident', 'Weather_Conditions', 'Road_Surface', 'Lighting', 'Vehicle_Type',
        'Speed', 'Driver_Age', 'Vehicles_Involved', 'Alcohol_Involvement', 'Road_Type', 'Intersection']]
y = df['Accident_Severity']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

joblib.dump(model, 'road_accident_severity_model.pkl')
print("Model trained and saved as 'road_accident_severity_model.pkl'")


y_pred = model.predict(X_test)

print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R^2 Score:", r2_score(y_test, y_pred))

with open('road_accident_severity_model.pkl', 'wb') as f:
    pickle.dump(model, f)
