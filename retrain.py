import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_auc_score, roc_curve
)
from sklearn.model_selection import train_test_split, cross_val_score
import joblib
import os
import warnings
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import io
import base64

def retrain_model():
    enable_outlier_removal = True
    enable_shap = True
    enable_roc = True
    enable_cv = True

    main_file = 'MainData.xlsx'
    new_file = 'NewData.xlsx'

    # Create folder for plots - keep this (even if we won't save files)
    plot_dir = "plots"
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    # Function to convert current plot to base64 string
    def plot_to_base64():
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close()
        buf.seek(0)
        return base64.b64encode(buf.read()).decode('utf-8')

    try:
        main_df = pd.read_excel(main_file) if os.path.exists(main_file) else pd.DataFrame()
        new_df = pd.read_excel(new_file) if os.path.exists(new_file) else pd.DataFrame()

        combined_df = pd.concat([main_df, new_df], ignore_index=True)
        combined_df = combined_df.dropna(subset=['target'])

        if combined_df.empty:
            warnings.warn("⚠️ No valid data with targets found. Retraining aborted.")
            return {}

        feature_columns = [
            "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
            "heartRate", "exang", "oldpeak", "BMI", "diaBP", "glucose", "Smkr"
        ]

        X = combined_df[feature_columns]
        y = combined_df["target"]

        # ---------------- Null Handling ----------------
        if X.isnull().sum().sum() > 0:
            print("⚠️ Null values found. Filling with mean...")
            X.fillna(X.mean(numeric_only=True), inplace=True)

        # ---------------- Outlier Removal ----------------
        if enable_outlier_removal:
            Q1 = X.quantile(0.25)
            Q3 = X.quantile(0.75)
            IQR = Q3 - Q1
            outlier_summary = {}
            for col in X.select_dtypes(include=['float64', 'int64']).columns:
                lower = Q1[col] - 1.5 * IQR[col]
                upper = Q3[col] + 1.5 * IQR[col]
                outlier_summary[col] = X[(X[col] < lower) | (X[col] > upper)].shape[0]

            print("🔍 Outlier count per feature (before removal):")
            for k, v in outlier_summary.items():
                print(f"   {k}: {v}")

            mask = ~((X < (Q1 - 1.5 * IQR)) | (X > (Q3 + 1.5 * IQR))).any(axis=1)
            outlier_count = (~mask).sum()
            X = X[mask]
            y = y[mask]
            print(f"✅ Outlier removal completed using IQR. Removed {outlier_count} rows. Remaining samples: {len(X)}")
        else:
            print("🚫 Outlier removal skipped")

        # ---------------- SMOTE ----------------
        class_counts = y.value_counts()
        if len(class_counts) < 2:
            warnings.warn("⚠️ Only one class present. Aborting.")
            return {}

        imbalance_ratio = max(class_counts) / min(class_counts)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        if imbalance_ratio >= 2.1:
            print(f"⚠️ Imbalance ratio = {imbalance_ratio:.2f}. Applying SMOTE...")
            sm = SMOTE()
            X_scaled, y = sm.fit_resample(X_scaled, y)
        else:
            print(f"✅ Imbalance ratio = {imbalance_ratio:.2f}. No SMOTE applied.")

        # ---------------- Split ----------------
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

        models = {
            "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
            "RandomForest": RandomForestClassifier(random_state=42),
            "XGBoost": XGBClassifier(eval_metric='logloss', random_state=42)
        }

        results = []

        # ---------------- Train Models ----------------
        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]

            results.append({
                "Model": name,
                "Accuracy": accuracy_score(y_test, y_pred),
                "Precision": precision_score(y_test, y_pred, zero_division=0),
                "Recall": recall_score(y_test, y_pred, zero_division=0),
                "F1 Score": f1_score(y_test, y_pred, zero_division=0),
                "ROC AUC": roc_auc_score(y_test, y_proba),
                "Confusion Matrix": confusion_matrix(y_test, y_pred),
                "Model Object": model
            })

        # ---------------- Print Summary ----------------
        print("\nModel Performance Comparison:")
        print(f"{'Model':<18} {'Accuracy':<9} {'Precision':<10} {'Recall':<8} {'F1 Score':<9} {'ROC AUC':<8}")
        for r in results:
            print(f"{r['Model']:<18} {r['Accuracy']:<9.4f} {r['Precision']:<10.4f} {r['Recall']:<8.4f} {r['F1 Score']:<9.4f} {r['ROC AUC']:<8.4f}")
            print(" Confusion Matrix:")
            print(r["Confusion Matrix"])
            print()

        # ---------------- Best Model ----------------
        best_result = max(results, key=lambda x: x['F1 Score'])
        best_model = best_result["Model Object"]
        best_model_name = best_result["Model"]
        print(f"✅ Best model: {best_model_name} (F1 Score: {best_result['F1 Score']:.4f})")

        images = {}

        # ---------------- SHAP ----------------
        if enable_shap:
            try:
                plt.figure(figsize=(8,6))  # <-- Add this line so SHAP plot renders correctly
                if best_model_name == "LogisticRegression":
                    explainer = shap.Explainer(best_model.predict, X_train)
                else:
                    explainer = shap.Explainer(best_model, X_train)
                shap_values = explainer(X_test[:100])
                shap.summary_plot(shap_values, X_test[:100], show=False)
                images['shap'] = plot_to_base64()  # <-- Key 'shap' (was 'shap_summary') to match your template
                print("✅ SHAP summary plot created.")
            except Exception as e:
                print(f"⚠️ SHAP failed: {e}")

        # ---------------- Confusion Matrix Heatmap ----------------
        try:
            plt.figure(figsize=(5, 4))
            sns.heatmap(confusion_matrix(y_test, best_model.predict(X_test)), annot=True, fmt='d', cmap='Blues')
            plt.title(f'Confusion Matrix ({best_model_name})')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.tight_layout()
            images['confusion_matrix'] = plot_to_base64()
            print("✅ Confusion matrix heatmap created.")
        except Exception as e:
            print(f"⚠️ Heatmap failed: {e}")

        # ---------------- ROC Curve ----------------
        if enable_roc:
            try:
                y_score = best_model.predict_proba(X_test)[:, 1]
                fpr, tpr, thresholds = roc_curve(y_test, y_score)
                plt.figure()
                plt.plot(fpr, tpr, label=f'{best_model_name} (AUC = {roc_auc_score(y_test, y_score):.2f})')
                plt.plot([0, 1], [0, 1], 'k--')
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title('ROC Curve')
                plt.legend()
                plt.tight_layout()
                images['roc_curve'] = plot_to_base64()
                print("✅ ROC curve created.")
            except Exception as e:
                print(f"⚠️ ROC curve failed: {e}")

        # ---------------- Cross Validation ----------------
        if enable_cv:
            try:
                scores = cross_val_score(best_model, X_scaled, y, cv=5, scoring='f1')
                print(f"✅ Cross-validation F1 scores: {scores}")
                print(f"✅ Mean F1 Score: {scores.mean():.4f}")
            except Exception as e:
                print(f"⚠️ Cross-validation failed: {e}")

        # Save best model and scaler
        joblib.dump(best_model, 'best_model.pkl')
        joblib.dump(scaler, 'scaler.pkl')
        print("✅ Model and scaler saved.")

        # Save combined data and clear new data file
        combined_df.to_excel(main_file, index=False)
        pd.DataFrame(columns=feature_columns + ['target']).to_excel(new_file, index=False)
        print("✅ Data merged and new data cleared.")

        return images

    except Exception as e:
        warnings.warn(f"❌ Error during retraining: {e}")
        return {} 