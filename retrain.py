import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import os

def retrain_model():
    main_file = 'MainData.xlsx'
    new_file = 'NewData.xlsx'

    # Load both files
    main_df = pd.read_excel(main_file) if os.path.exists(main_file) else pd.DataFrame()
    new_df = pd.read_excel(new_file) if os.path.exists(new_file) else pd.DataFrame()

    # Combine both
    combined_df = pd.concat([main_df, new_df], ignore_index=True)

    # Drop rows without target
    combined_df = combined_df.dropna(subset=['target'])

    if combined_df.empty:
        print("No valid data with targets found.")
        return

    # Define features
    feature_columns = [
        "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
        "heartRate", "exang", "oldpeak", "BMI", "diaBP", "glucose", "Smkr"
    ]

    X = combined_df[feature_columns]
    y = combined_df['target']

    # Scale and train
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = RandomForestClassifier()
    model.fit(X_scaled, y)

    # Save model & scaler
    joblib.dump(model, 'best_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')

    print("✅ Model retrained and saved.")

    # Save updated full data to MainData.xlsx
    combined_df.to_excel(main_file, index=False)

    # Clear NewData.xlsx
    if os.path.exists(new_file):
        pd.DataFrame(columns=new_df.columns).to_excel(new_file, index=False)
        print("✅ New data merged into MainData.xlsx and cleared from NewData.xlsx.")
