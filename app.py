import streamlit as st
import pandas as pd
import numpy as np
import joblib


#Title
st.header("Streamlit Machine Learning App")

#Input Bar 1
height = st.number_input("Enter height")

#Input bar 2
weight = st.number_input("Enter weight")

#Dropdown Input
eyes= st.selectbox("Select Eye Colour",("Blue","Brown"))

#Button is pressed
if st.button("Submit"):

    clf = joblib.load('clf.pkl')

    X = pd.DataFrame([[height,weight,eyes]],columns=["Height","Weight","Eye"])
    X = X.replace(['Brown','Blue'],[1,0])

    prediction = clf.predict(X)[0]

    st.text(f"This instance is a {prediction}")