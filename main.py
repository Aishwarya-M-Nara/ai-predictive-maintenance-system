# ==============================
# AI Predictive Maintenance System
# ==============================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# ------------------------------
# STEP 1: Generate Synthetic Data
# ------------------------------
def generate_data():
    np.random.seed(42)

    data = pd.DataFrame({
        'temperature': np.random.normal(70, 10, 200),
        'vibration': np.random.normal(5, 2, 200),
        'pressure': np.random.normal(30, 5, 200)
    })

    # Failure condition
    data['failure'] = ((data['temperature'] > 85) | 
                       (data['vibration'] > 8)).astype(int)

    data.to_csv('sensor_data.csv', index=False)
    print("✅ Dataset generated as sensor_data.csv")


# ------------------------------
# STEP 2: Train ML Model
# ------------------------------
def train_model():
    data = pd.read_csv('sensor_data.csv')

    X = data[['temperature', 'vibration', 'pressure']]
    y = data['failure']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    joblib.dump(model, 'model.pkl')
    print("✅ Model trained and saved as model.pkl")


# ------------------------------
# STEP 3: Anomaly Detection
# ------------------------------
def detect_anomaly(temp, vib, pres):
    if temp > 85 or vib > 8 or pres > 40:
        return "⚠️ Anomaly Detected"
    return "✅ Normal"


# ------------------------------
# STEP 4: Maintenance Decision
# ------------------------------
def maintenance_action(prediction):
    if prediction == 1:
        return "🔧 Schedule Immediate Maintenance"
    else:
        return "🟢 Routine Monitoring"


# ------------------------------
# STEP 5: Cost Analysis
# ------------------------------
def cost_analysis(prediction):
    if prediction == 1:
        return "💰 Preventive maintenance is cheaper than breakdown!"
    return "💰 No immediate cost risk"


# ------------------------------
# STEP 6: Prediction System
# ------------------------------
def predict():
    model = joblib.load('model.pkl')

    print("\nEnter Sensor Values:")

    temp = float(input("Temperature: "))
    vib = float(input("Vibration: "))
    pres = float(input("Pressure: "))

    # Anomaly detection
    anomaly = detect_anomaly(temp, vib, pres)

    # Prediction
    prediction = model.predict([[temp, vib, pres]])[0]

    print("\n===== RESULTS =====")
    print("Anomaly Status:", anomaly)
    print("Failure Prediction:", "⚠️ Failure Likely" if prediction == 1 else "✅ No Failure")
    print("Maintenance Advice:", maintenance_action(prediction))
    print("Cost Insight:", cost_analysis(prediction))


# ------------------------------
# MAIN MENU
# ------------------------------
def main():
    while True:
        print("\n====== Predictive Maintenance System ======")
        print("1. Generate Data")
        print("2. Train Model")
        print("3. Predict Equipment Status")
        print("4. Exit")

        choice = input("Enter your choice: ")

        if choice == '1':
            generate_data()
        elif choice == '2':
            train_model()
        elif choice == '3':
            predict()
        elif choice == '4':
            print("Exiting...")
            break
        else:
            print("Invalid choice!")


if __name__ == "__main__":
    main()