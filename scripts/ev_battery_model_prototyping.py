import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import mlflow.sklearn
import joblib
import mlflow

mlflow.set_tracking_uri('http://localhost:5000')

df = pd.read_parquet('data/ev_battery_p')

X = df[
    [
        "Ambient_Temp_C",
        "Anode_Overhang_mm",
        "Electrolyte_Volume_ml",
        "Internal_Resistance_mOhm",
        "Capacity_mAh",
        "Retention_50Cycle_Pct",
        "Production_Line",
        "Shift",
        "Supplier"
    ]
]

y = df["Defect_Type"]

categorical = ["Production_Line", "Shift", "Supplier"]
numeric = [
    "Ambient_Temp_C",
    "Anode_Overhang_mm",
    "Electrolyte_Volume_ml",
    "Internal_Resistance_mOhm",
    "Capacity_mAh",
    "Retention_50Cycle_Pct"
]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", "passthrough", numeric),
        ("cat", OneHotEncoder(), categorical)
    ]
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

with mlflow.start_run(run_name="EVBatteryModelPrototype") as run:
    mlflow.log_param("model_type", "RandomForest")
    mlflow.log_param("n_estimators", 100)
    model = Pipeline([
        ("preprocess", preprocessor),
        ("classifier", RandomForestClassifier(n_estimators=100))
    ])
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    print(classification_report(y_test, preds))

    joblib.dump(model, "models/ev_battery_model.pkl")
    mlflow.log_artifact("models/ev_battery_model.pkl")
    mlflow.sklearn.log_model(model, "ev_battery_model")

    print("MLflow Run ID:", run.info.run_id)
    