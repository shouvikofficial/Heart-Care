from flask import Flask, render_template, request
import pandas as pd
import joblib
import os
import tempfile
import shutil
from retrain import retrain_model

app = Flask(__name__)
DATA_FILE = 'NewData.xlsx'  # Store only new user data

def load_model_and_scaler():
    model = joblib.load('best_model.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    form_data = request.form.to_dict()
    name = form_data.get("name")
    email = form_data.get("email")

    # Extract features
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

    # Load model and scaler
    model, scaler = load_model_and_scaler()
    input_data = scaler.transform([features])
    prediction = model.predict(input_data)[0]
    overall_result = "At Risk" if prediction == 1 else "Not At Risk"

    # Save user data + prediction to Excel safely
    new_row = pd.DataFrame([{
        "age": features[0],
        "sex": features[1],
        "cp": features[2],
        "trestbps": features[3],
        "chol": features[4],
        "fbs": features[5],
        "restecg": features[6],
        "heartRate": features[7],
        "exang": features[8],
        "oldpeak": features[9],
        "BMI": features[10],
        "diaBP": features[11],
        "glucose": features[12],
        "Smkr": features[13],
        "target": prediction
    }])

    try:
        if os.path.exists(DATA_FILE):
            existing = pd.read_excel(DATA_FILE)
            updated = pd.concat([existing, new_row], ignore_index=True)
        else:
            updated = new_row

        with tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx') as tmp:
            temp_path = tmp.name
        updated.to_excel(temp_path, index=False)
        shutil.move(temp_path, DATA_FILE)
    except Exception as e:
        print(f"Error writing to Excel: {e}")

    # Retrain model
    retrain_model()

    # Prepare result
    user_data = {
        "Name": name,
        "Email": email,
        "Age": form_data.get("age"),
        "Gender": "Male" if form_data.get("sex") == "1" else "Female",
        "Chest Pain Type": form_data.get("cp"),
        "Resting Blood Pressure (in mm/Hg)": form_data.get("trestbps"),
        "Cholesterol Level": form_data.get("chol"),
        "Fasting Blood Sugar > 120 mg/dl": form_data.get("fbs"),
        "Resting ECG Result": form_data.get("restecg"),
        "Max Heart Rate": form_data.get("heartRate"),
        "Exercise Induced Angina": form_data.get("exang"),
        "Oldpeak": form_data.get("oldpeak"),
        "BMI": form_data.get("BMI"),
        "Diastolic BP": form_data.get("diaBP"),
        "Glucose": form_data.get("glucose"),
        "Smoker": form_data.get("Smkr")
    }

    return render_template(
        "result.html",
        name=name,
        email=email,
        user_data=user_data,
        overall_result=overall_result,
        prediction=prediction
    )

if __name__ == '__main__':
    app.run(debug=True)
