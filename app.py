import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import pickle


st.set_page_config(page_title="Bank Churn Predictor", page_icon="🏦")


@st.cache_resource
def load_models():

    model = tf.keras.models.load_model('churn_model.h5')
    
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('column_transformer.pkl', 'rb') as f:
        ct = pickle.load(f)
    
    return model, scaler, ct

try:
    model, sc, ct = load_models()
except FileNotFoundError:
    st.error("Error: 'churn_model.h5', 'scaler.pkl', or 'column_transformer.pkl' not found. Please export them from your notebook first!")
    st.stop()


st.title("Banking Sector: Customer Churn Prediction")
st.markdown("Developed by: **Vaibhav**")
st.write("Enter customer details below to calculate the probability of them leaving the bank.")


with st.container():
    col1, col2 = st.columns(2)
    
    with col1:
        credit_score = st.number_input("Credit Score", 300, 850, 600)
        geography = st.selectbox("Geography", ["France", "Spain", "Germany"])
        gender = st.selectbox("Gender", ["Female", "Male"])
        age = st.slider("Age", 18, 95, 35)
        tenure = st.slider("Tenure (Years)", 0, 10, 5)

    with col2:
        balance = st.number_input("Account Balance ($)", 0.0, 300000.0, 50000.0)
        num_products = st.selectbox("Number of Products", [1, 2, 3, 4])
        has_crcard = st.selectbox("Has Credit Card?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        is_active = st.selectbox("Is Active Member?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        salary = st.number_input("Estimated Salary ($)", 0.0, 250000.0, 75000.0)


if st.button("Calculate Churn Risk"):
    
    raw_data = [[credit_score, geography, gender, age, tenure, balance, num_products, has_crcard, is_active, salary]]
    
 
    input_arr = np.array(raw_data)
    input_arr[:, 2] = 1 if gender == "Male" else 0
    
 
    input_transformed = ct.transform(input_arr)
    
   
    input_final = sc.transform(input_transformed)
    
  
    prediction_prob = model.predict(input_final)[0][0]
    is_churn = prediction_prob > 0.5

    
    st.divider()
    if is_churn:
        st.error(f"High Risk!** This customer is likely to churn.")
    else:
        st.success(f"Low Risk.** This customer is likely to stay.")
    
    st.metric(label="Churn Probability", value=f"{prediction_prob:.2%}")
    st.progress(float(prediction_prob))