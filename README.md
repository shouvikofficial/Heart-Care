# 🫀 HeartCare: Intelligent Heart Failure Risk Prediction System

> A smart, scalable, and explainable ML-powered web application that predicts heart failure risk and evolves with new data — built with Flask, pandas, and Random Forest, Logistic Regression, XGBoost.

---

## 📖 Overview

HeartCare is a heart failure risk prediction system developed using machine learning and data science. It takes key patient health indicators and provides a real-time risk prediction. It also adapts over time by retraining itself with new user inputs, offering continuous learning and model improvement. Visual explanations and data insights (via SHAP, ROC curves, and more) make the predictions understandable and transparent.

---

## ✨ Key Features

- ✅ **Live heart failure prediction** via a user-friendly web interface  
- 📊 **Real-time data collection** and retraining with new input data  
- ⚖️ **Class imbalance handled** using **SMOTE**  
- 📈 **Outlier detection** using **IQR**  
- 🧠 **Multiple model comparison**: Logistic Regression, Random Forest, XGBoost  
- 📊 **Interpretability** with SHAP, confusion matrix, and ROC curve  
- 💾 Saves inputs to `MainData.xlsx` and `NewData.xlsx` automatically  
- 🛡️ **Robust error handling** and duplicate prevention  
- 📁 Full backend and frontend integration using Flask + HTML/CSS + JS  

---

## 🏗️ Tech Stack

| Layer        | Tools & Libraries |
|--------------|------------------|
| **Frontend** | HTML, CSS, JavaScript |
| **Backend**  | Python (Flask, pandas, joblib) |
| **ML Models**| Scikit-learn, Logistic Regression, RandomForest, XGBoost, imbalanced-learn |
| **Visualization** | Matplotlib, Seaborn, SHAP |
| **Deployment** | Google Colab (for model retraining), Localhost |
| **Dataset**   | `MainData.xlsx`, `NewData.xlsx` |

---

## 📦 Installation

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/HeartCare.git
cd HeartCare

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Run the app
python app.py

# 4. Access the app
Go to http://127.0.0.1:5000 in your browser
