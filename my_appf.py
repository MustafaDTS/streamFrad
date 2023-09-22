import streamlit as st
import pandas as pd
import numpy as np

st.title("Welcome to our free transaction prediction site")
st.text("Please fill the following inputs. Depending of your input our model will predic \n whether the transaction fraudal or nonfradual.\n You can use left-sidebar to specify extra specifications. ")

# To load machine learning model
import pickle

de_05_fraud_model = pickle.load(open("xgb_final_model", "rb"))


v14 = st.sidebar.slider("Chose your v14 value:", min_value=-19.21, max_value=10.53, value=-5.0,step=0.5)
v4 = st.sidebar.slider("Chose your v4 value:", min_value=-5.68, max_value=16.88, value=5.0,step=0.5)
v12 = st.sidebar.slider("Chose your v12 value:", min_value=-18.68, max_value=7.85, value=-10.0,step=0.5)
v10 = st.sidebar.slider("Chose your v10 value:", min_value=-24.59, max_value=23.75, value=0.0,step=0.5)
v27 = st.sidebar.slider("Chose your v27 value:", min_value=-22.57, max_value=31.61, value=2.0,step=0.5)
v17 = st.sidebar.slider("Chose your v17 value:", min_value=-25.16, max_value=9.25, value=-5.0,step=0.5)
v7 = st.sidebar.slider("Chose your v7 value:", min_value=-43.56, max_value=120.59, value=30.0,step=0.5)
v28 = st.sidebar.slider("Chose your v28 value:", min_value=-15.43, max_value=33.85, value=0.0,step=0.5)
v11 = st.sidebar.slider("Chose your v11 value:", min_value=-4.80, max_value=12.02, value=3.0,step=0.5)


my_dict = {
    "v14": v14,
    "v4": v4,
    "v12": v12,
    "v10": v10,
    'v27': v27,
    "v17": v17, 
    'v7':v7,
    "v28": v28, 
    "v11": v11
    }

my_dict2 = {
    "v14 level": v14,
    "v4 level": v4,
    "v12 level": v12,
    "v10 level": v10,
    'v27 level':v27,
    "v17 level": v17, 
    'v7 level':v7,
    "v28 level": v28, 
    "v11 level": v11
    }

df2 = pd.DataFrame.from_dict([my_dict2])
st.write("The configuration of your selections is below") 
st.table(df2)


df = pd.DataFrame.from_dict([my_dict])

st.subheader("Press predict if your configuration is okay")

# Prediction with user inputs
predict = st.button("Predict")
result = de_05_chur_model.predict(df)
if predict:
    st.write("Based on our model your prediction is:")
    if int(result[0]) == 0:
        st.success(result[0], icon="✅")
        st.write("You can relax! This is a nonfradual transaction.")
    else:
        st.error(result[0], icon="🚨")
        st.write("Bad news! This is a nonfradual transaction.")
    