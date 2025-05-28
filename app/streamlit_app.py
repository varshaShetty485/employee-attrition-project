import streamlit as st 
import pandas as pd
import joblib

# Load the trained model and feature names
model = joblib.load("app/final_logreg_model.pkl")
feature_names = joblib.load("app/feature_names.pkl")

st.title("🧠 Employee Attrition Predictor")

st.markdown("Fill in the details in the sidebar to predict if an employee is likely to leave the company.")

# Sidebar inputs (only collect what you care to expose)
age = st.sidebar.slider("Age", 18, 60, 30)
monthly_income = st.sidebar.number_input("Monthly Income", min_value=1000, value=5000)
distance = st.sidebar.slider("Distance From Home", 1, 30, 5)
years_with_manager = st.sidebar.slider("Years With Current Manager", 0, 20, 3)
overtime = st.sidebar.selectbox("OverTime", ["Yes", "No"])
job_role = st.sidebar.selectbox("Job Role", [
    'Human Resources', 'Laboratory Technician', 'Manager', 'Manufacturing Director',
    'Research Director', 'Research Scientist', 'Sales Executive', 'Sales Representative'
])
department = st.sidebar.selectbox("Department", [
    'Research & Development', 'Sales', 'Human Resources'
])
business_travel = st.sidebar.selectbox("Business Travel", [
    'Travel_Frequently', 'Travel_Rarely', 'Non-Travel'
])
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
marital_status = st.sidebar.selectbox("Marital Status", ["Married", "Single", "Divorced"])
education_field = st.sidebar.selectbox("Education Field", [
    'Life Sciences', 'Medical', 'Marketing', 'Technical Degree', 'Other'
])

# Build input data dictionary
input_dict = {
    'Age': age,
    'MonthlyIncome': monthly_income,
    'DistanceFromHome': distance,
    'YearsWithCurrManager': years_with_manager,
    'OverTime_Yes': 1 if overtime == "Yes" else 0
}

# One-hot encoded fields based on selected values
input_dict[f'JobRole_{job_role}'] = 1
input_dict[f'Department_{department}'] = 1
if business_travel != "Non-Travel":
    input_dict[f'BusinessTravel_{business_travel}'] = 1
if gender == "Male":
    input_dict['Gender_Male'] = 1
if marital_status in ["Married", "Single"]:
    input_dict[f'MaritalStatus_{marital_status}'] = 1
input_dict[f'EducationField_{education_field}'] = 1

# Convert to DataFrame
input_df = pd.DataFrame([input_dict])

# Fill missing features with 0 and reorder
for col in feature_names:
    if col not in input_df.columns:
        input_df[col] = 0
input_df = input_df[feature_names]

# Predict and display
if st.button("Predict"):
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    st.subheader("Prediction Result")
    if prediction == 1:
        st.error(f"⚠️ High Risk of Attrition — Probability: {probability:.2f}")
    else:
        st.success(f"✅ Low Risk of Attrition — Probability: {probability:.2f}")


