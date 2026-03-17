import pandas as pd
import joblib

model = joblib.load("models/ev_battery_model.pkl")

sample = pd.DataFrame([{
    "Ambient_Temp_C": 24.3,
    "Anode_Overhang_mm": 1.162,
    "Electrolyte_Volume_ml": 104.84,
    "Internal_Resistance_mOhm": 1.65,
    "Capacity_mAh": 1042,
    "Retention_50Cycle_Pct": 95.88,
    "Production_Line": "Line_2",
    "Shift": "Morning",
    "Supplier": "VoltIndustries"
}])


prediction = model.predict(sample)

print("Prediction: ", prediction)