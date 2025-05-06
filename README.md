# 🏦 Loan Eligibility Predictor

A **Streamlit-based machine learning web app** that predicts loan approval status based on user input. This app uses both **Logistic Regression** and **Random Forest** models trained on cleaned and encoded loan application data.

## 📌 Project Highlights

- 🔍 **EDA & Preprocessing:** Includes proper handling of missing values, outlier detection, and categorical encoding.
- 🧠 **Models Used:** Logistic Regression and Random Forest (trained via Scikit-learn).
- 🧮 **Encoders:** Includes support for both OneHotEncoder and pandas' `get_dummies`.
- 🎯 **Batch Prediction Support:** Accepts CSV file uploads for bulk predictions.
- ⚙️ **Model Deployment:** Deployed using Streamlit for a clean, interactive interface.
- 📦 **Preprocessor Saved:** Encoder pipeline saved separately and reused during prediction to avoid mismatched columns.

---

## 🚀 Features

- Single or bulk prediction (via CSV upload)
- Handles categorical input robustly (including "3+" → 3 conversion for dependents)
- Data preprocessing pipeline serialized for consistent input transformation
- Switch between Logistic Regression and Random Forest
- Displays prediction results on screen and optionally downloadable as CSV

---

## 🧪 Models Used

- `logistic_model.pkl` — Logistic Regression model
- `rf_model.pkl` — Random Forest model
- `preprocessor.pkl` — Column transformer with encoding logic

---

## 🛠 Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/loan_eligibility_predictor.git
cd loan_eligibility_predictor
### 2. Create and Activate Virtual Environment

python -m venv venv
source venv/bin/activate      # For Linux/macOS
venv\Scripts\activate         # For Windows

### 3. Install Dependencies

pip install -r requirements.txt

### 4. Run the App

streamlit run app.py

### Author

Vineet Kumar Saini
Data Science Enthusiast
