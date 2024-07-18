# import libraries
import streamlit as st
import pandas as pd
from pycaret.regression import load_model, predict_model

# load model from .pkl file
model = load_model("insurance_dt_model")

st.header("Insurance Charge Predictions")

# inputs for prediction
st.write("Please fill out the form below for a prediction of your insurance charges")
age = st.number_input("Age", 1, 100)
sex = st.radio("Sex",["male","female"])
bmi = st.number_input("BMI", 1, 100)
children = st.number_input("Children", 0, 10)
smoker = st.radio("Smoker", ["yes","no"])
region = st.selectbox("Region", ["northwest", "southwest", "northeast", "southeast"])

# make prediction when button is pressed
if st.button("Predict"):
    #input_dict = {'age':20,'sex':'male','bmi':20,'children':2,'smoker':'yes','region':'southwest'}
    input_dict = {'age':age,'sex':sex,'bmi':bmi,'children':children,'smoker':smoker,'region':region}
    input_df = pd.DataFrame([input_dict])
    predictions_df = predict_model(estimator=model, data=input_df)
    prediction = predictions_df.iloc[0]['prediction_label']
    st.markdown("Your insurance charges are predicted to be: ")
    st.markdown(prediction)