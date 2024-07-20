import streamlit as st
import joblib
import numpy as np

# Load the model and scaler
model = joblib.load('heart_disease_model2.pkl')
scaler = joblib.load('scaler.pkl')

# UI for inputting health metrics
st.title('Heart Disease Risk Assessment')
age = st.number_input('Age', min_value=1, max_value=100, value=25)
sex = st.selectbox('Sex', [0, 1])
cp = st.selectbox('Chest Pain Type (cp)', [0, 1, 2, 3])
trestbps = st.number_input('Resting Blood Pressure (trestbps)', min_value=80, max_value=200, value=120)
chol = st.number_input('Cholesterol (chol)', min_value=100, max_value=400, value=200)
fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl (fbs)', [0, 1])
restecg = st.selectbox('Resting Electrocardiographic Results (restecg)', [0, 1, 2])
thalach = st.number_input('Maximum Heart Rate Achieved (thalach)', min_value=60, max_value=200, value=150)
exang = st.selectbox('Exercise Induced Angina (exang)', [0, 1])
oldpeak = st.number_input('ST Depression Induced by Exercise (oldpeak)', min_value=0.0, max_value=10.0, value=1.0)
slope = st.selectbox('Slope of the Peak Exercise ST Segment (slope)', [0, 1, 2])
ca = st.selectbox('Number of Major Vessels Colored by Fluoroscopy (ca)', [0, 1, 2, 3, 4])
thal = st.selectbox('Thalassemia (thal)', [1, 2, 3])

# Make prediction
if st.button('Predict'):
    input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
    input_data = scaler.transform(input_data)
    prediction = model.predict(input_data)[0]
    if prediction == 0:
        st.write("The person is at low risk of heart disease.")
    else:
        st.write("The person is at high risk of heart disease.")
