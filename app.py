from flask import Flask, render_template, request
import pandas as pd
import joblib
import os
from retrain import retrain_model

app = Flask(__name__)
DATA_FILE = 'NewData.xlsx'
MAIN_FILE = 'MainData.xlsx'

def load_model_and_scaler():
    model = joblib.load('best_model.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/form')
def form():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    form_data = request.form.to_dict()
    name = form_data.get("name")
    email = form_data.get("email")

    try:
        features = [
            float(form_data.get("age")),
            float(form_data.get("sex")),
            float(form_data.get("cp")),
            float(form_data.get("trestbps")),
            float(form_data.get("chol")),
            float(form_data.get("fbs")),
            float(form_data.get("restecg")),
            float(form_data.get("heartRate")),
            float(form_data.get("exang")),
            float(form_data.get("oldpeak")),
            float(form_data.get("BMI")),
            float(form_data.get("diaBP")),
            float(form_data.get("glucose")),
            float(form_data.get("Smkr"))
        ]
    except (TypeError, ValueError):
        return "❌ Invalid input values. Please check the form."

    feature_columns = [
        "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
        "heartRate", "exang", "oldpeak", "BMI", "diaBP", "glucose", "Smkr"
    ]

    model, scaler = load_model_and_scaler()
    input_df = pd.DataFrame([features], columns=feature_columns)

    # --- Add interaction features (must match retrain.py) ---
    input_df["age_glucose"] = input_df["age"] * input_df["glucose"]
    input_df["age_BMI"] = input_df["age"] * input_df["BMI"]
    input_df["glucose_BMI"] = input_df["glucose"] * input_df["BMI"]
    input_df["heartRate_exang"] = input_df["heartRate"] * input_df["exang"]
    input_df["chol_fbs"] = input_df["chol"] * input_df["fbs"]

    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)[0]
    overall_result = "At Risk" if prediction == 1 else "Not At Risk"

    new_row = input_df.copy()
    new_row["target"] = prediction

    images = {}

    try:
        if os.path.exists(MAIN_FILE):
            main_df = pd.read_excel(MAIN_FILE)
        else:
            main_df = pd.DataFrame(columns=feature_columns + ['target'])

        # --- Add interactions to main_df if missing (ensure alignment) ---
        for col in ["age_glucose", "age_BMI", "glucose_BMI", "heartRate_exang", "chol_fbs"]:
            if col not in main_df.columns:
                main_df[col] = main_df["age"] * main_df["glucose"]
                main_df.drop(columns=[col], inplace=True)

        main_df["age_glucose"] = main_df["age"] * main_df["glucose"]
        main_df["age_BMI"] = main_df["age"] * main_df["BMI"]
        main_df["glucose_BMI"] = main_df["glucose"] * main_df["BMI"]
        main_df["heartRate_exang"] = main_df["heartRate"] * main_df["exang"]
        main_df["chol_fbs"] = main_df["chol"] * main_df["fbs"]

        main_df_comp = main_df.drop(columns=['target'], errors='ignore')
        new_row_comp = new_row.drop(columns=['target'], errors='ignore')

        main_df_comp, new_row_comp = main_df_comp.align(new_row_comp, axis=1, fill_value=None)

        is_duplicate = any((main_df_comp == new_row_comp.iloc[0]).all(axis=1))

        if not is_duplicate:
            if os.path.exists(DATA_FILE):
                existing = pd.read_excel(DATA_FILE)
                if existing.empty or existing.isna().all().all():
                    updated = new_row.copy()
                else:
                    updated = pd.concat([existing, new_row], ignore_index=True)
            else:
                updated = new_row

            updated.to_excel(DATA_FILE, index=False)
            print("✅ New user data saved to NewData.xlsx")

            # ✅ Only retrain when 20 or more new entries
            if len(updated) >= 2:
                print("🔁 2 or more new records found. Starting retraining...")
                images = retrain_model()
            else:
                print(f"⏳ Not enough new data to retrain. {20 - len(updated)} more record(s) needed.")

    except Exception as e:
        print(f"❌ Error saving or processing data: {e}")

    def convert_user_data(data):
        cp_map = ['Typical Angina', 'Atypical Angina', 'Non-anginal Pain', 'Asymptomatic']
        ecg_map = ['Normal', 'ST-T Abnormality', 'LV Hypertrophy']

        mapped_data = {}
        for key, value in data.items():
            try:
                val = int(value)
            except:
                mapped_data[key] = value
                continue

            if key == 'sex':
                mapped_data[key] = 'Male' if val == 1 else 'Female'
            elif key == 'Smkr':
                mapped_data[key] = 'Yes' if val == 1 else 'No'
            elif key == 'fbs':
                mapped_data[key] = 'Yes' if val == 1 else 'No'
            elif key == 'cp':
                mapped_data[key] = cp_map[val]
            elif key == 'restecg':
                mapped_data[key] = ecg_map[val]
            elif key == 'exang':
                mapped_data[key] = 'Yes' if val == 1 else 'No'
            else:
                mapped_data[key] = value

        return mapped_data

    display_data = convert_user_data(form_data)

    return render_template(
        "result.html",
        name=name,
        email=email,
        user_data=display_data,
        overall_result=overall_result,
        prediction=prediction,
        images=images
    )

if __name__ == '__main__':
    app.run(debug=True)
