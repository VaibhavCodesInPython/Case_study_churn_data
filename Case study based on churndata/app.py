import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

@st.cache_resource
def load_and_prep_transformers():
    df = pd.read_csv('Artificial_Neural_Network_Case_Study_data.csv')
    X = df.iloc[:, 3:-1].values
    
    
    le = LabelEncoder()
    X[:, 2] = le.fit_transform(X[:, 2])
    
    ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
    X = np.array(ct.fit_transform(X))
    

    sc = StandardScaler()
    sc.fit(X)
    
    return le, ct, sc

@st.cache_resource
def load_ann_model():
    
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(units=128, activation='relu'),
        tf.keras.layers.Dense(units=64, activation='relu'),
        tf.keras.layers.Dense(units=1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    return model


st.set_page_config(page_title="Bank Churn Predictor")
st.title("Customer Churn Prediction")
st.markdown("Enter customer details below to predict the likelihood of them leaving the bank.")

le, ct, sc = load_and_prep_transformers()
model = load_ann_model()

col1, col2 = st.columns(2)

with col1:
    geography = st.selectbox("Geography", ["France", "Spain", "Germany"])
    gender = st.selectbox("Gender", ["Female", "Male"])
    age = st.slider("Age", 18, 100, 35)
    tenure = st.number_input("Tenure (Years)", 0, 10, 5)
    credit_score = st.number_input("Credit Score", 300, 850, 600)

with col2:
    balance = st.number_input("Account Balance", 0.0, 300000.0, 50000.0)
    num_products = st.selectbox("Number of Products", [1, 2, 3, 4])
    has_crcard = st.radio("Has Credit Card?", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
    is_active = st.radio("Is Active Member?", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
    estimated_salary = st.number_input("Estimated Salary", 0.0, 250000.0, 75000.0)

if st.button("Predict Churn Probability"):
   
    input_data = [[credit_score, geography, gender, age, tenure, balance, num_products, has_crcard, is_active, estimated_salary]]
    
    input_array = np.array(input_data)
    
   
    input_array[:, 2] = le.transform(input_array[:, 2])
    
  
    input_array = np.array(ct.transform(input_array))
    
    input_scaled = sc.transform(input_array)
    
    prediction_prob = model.predict(input_scaled)[0][0]
    prediction = prediction_prob > 0.5
    
    st.divider()
    if prediction:
        st.error(f"**High Risk!** Probability of Churn: {prediction_prob:.2%}")
    else:
        st.success(f"**Low Risk.** Probability of Churn: {prediction_prob:.2%}")
        
    st.progress(float(prediction_prob))