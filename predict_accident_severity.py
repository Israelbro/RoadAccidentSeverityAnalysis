import pickle
import pandas as pd

with open('road_accident_severity_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Example input data based on the provided road accident data
example_data = pd.DataFrame({
    'Accident_Severity': [3],          
    'Time_of_Accident': [15],          
    'Weather_Conditions': [2],         
    'Road_Surface': [1],               
    'Lighting': [1],                    
    'Vehicle_Type': [1],                
    'Speed': [55],                      
    'Driver_Age': [34],                
    'Vehicles_Involved': [2],           
    'Alcohol_Involvement': [0],         
    'Road_Type': [1],                   
    'Intersection': [1]                 
})

predicted_severity = model.predict(example_data.drop(columns=['Accident_Severity']))
print("Predicted Accident Severity:", predicted_severity[0])
