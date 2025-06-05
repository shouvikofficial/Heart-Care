import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_auc_score
)
from sklearn.model_selection import train_test_split
import joblib
import os
import warnings

def retrain_model():
    main_file = 'MainData.xlsx'
    new_file = 'NewData.xlsx'

    # Load data
    main_df = pd.read_excel(main_file) if os.path.exists(main_file) else pd.DataFrame()
    new_df = pd.read_excel(new_file) if os.path.exists(new_file) else pd.DataFrame()

    # Combine datasets
    combined_df = pd.concat([main_df, new_df], ignore_index=True)

    # Drop rows without target
    combined_df = combined_df.dropna(subset=['target'])

    if combined_df.empty:
        warnings.warn("⚠️ No valid data with targets found. Retraining aborted.")
        return

    # Features and target
    feature_columns = [
        "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
        "heartRate", "exang", "oldpeak", "BMI", "diaBP", "glucose", "Smkr"
    ]

    try:
        X = combined_df[feature_columns]
        y = combined_df["target"]

        # Class distribution
        class_counts = y.value_counts()
        if len(class_counts) < 2:
            warnings.warn("⚠️ Only one class present in data. Retraining aborted.")
            return

        imbalance_ratio = max(class_counts) / min(class_counts)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        if imbalance_ratio >= 2.1:
            print(f"⚠️ Imbalance ratio = {imbalance_ratio:.2f}, applying SMOTE...")
            sm = SMOTE()
            X_scaled, y = sm.fit_resample(X_scaled, y)
        else:
            print(f"✅ Imbalance ratio = {imbalance_ratio:.2f}, no SMOTE applied.")

        # Split for evaluation
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

        # Initialize models
        models = {
            "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
            "RandomForest": RandomForestClassifier(random_state=42),
            "XGBoost": XGBClassifier(eval_metric='logloss', random_state=42),
        }

        results = []

        # Train and evaluate each model
        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]

            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, zero_division=0)
            rec = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            roc_auc = roc_auc_score(y_test, y_proba)
            cm = confusion_matrix(y_test, y_pred)

            results.append({
                "Model": name,
                "Accuracy": acc,
                "Precision": prec,
                "Recall": rec,
                "F1 Score": f1,
                "ROC AUC": roc_auc,
                "Confusion Matrix": cm,
                "Model Object": model
            })

        # Print comparison table
        print("\nModel Performance Comparison:")
        print(f"{'Model':<18} {'Accuracy':<9} {'Precision':<10} {'Recall':<8} {'F1 Score':<9} {'ROC AUC':<8}")
        for r in results:
            print(f"{r['Model']:<18} {r['Accuracy']:<9.4f} {r['Precision']:<10.4f} {r['Recall']:<8.4f} {r['F1 Score']:<9.4f} {r['ROC AUC']:<8.4f}")
            print(" Confusion Matrix:")
            print(r["Confusion Matrix"])
            print()

        # Select best model by F1 Score
        best_result = max(results, key=lambda x: x['F1 Score'])
        best_model = best_result['Model Object']
        best_model_name = best_result['Model']

        print(f"✅ Best model based on F1 Score: {best_model_name} with F1 Score = {best_result['F1 Score']:.4f}")

        # Save best model and scaler
        joblib.dump(best_model, 'best_model.pkl')
        joblib.dump(scaler, 'scaler.pkl')
        print(f"✅ Saved best model to 'best_model.pkl' and scaler to 'scaler.pkl'")

        # Merge and save data
        combined_df.to_excel(main_file, index=False)
        pd.DataFrame(columns=feature_columns + ['target']).to_excel(new_file, index=False)
        print("✅ New data merged into MainData.xlsx and cleared from NewData.xlsx.")

    except Exception as e:
        warnings.warn(f"❌ Error during retraining: {e}")
