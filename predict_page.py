import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn import linear_model
import seaborn as sns

# def loading():
#     with open('saved_steps.pkl', 'rb') as file:
#         data = pickle.load(file)
#     return data

# data = loading()

df = pd.read_csv("IBD_Diseases.csv")

Xval = df.drop("Condition", axis = 1)
Yval = df["Condition"]
logreg = linear_model.LogisticRegression()
logreg.fit(Xval, Yval)
y_pred = logreg.predict(Xval)

def show_predict_page():
    st.title("TraCed")

    st.write("""### Tracking Crohn's Disease in Inflammatory Bowel Disease Based on Micronutrient Concentration in Hair Samples""")

    patientName = st.text_input("Patient's Name")
    mgConc = st.number_input("Magnesium Concentration %")
    sConc = st.number_input("Sulfur Concentration %")
    feConc = st.number_input("Iron Concentration %")
    znConc = st.number_input("Zinc Concentration %")
    caConc = st.number_input("Calcium Concentration %")

    risk = st.button("Determine Risk")

    if risk:
        X = np.array([mgConc, sConc, feConc, znConc, caConc])

        valu = logreg.predict(X.reshape(1,-1))
        if valu[0] == 1:
            st.subheader(f"Hi {patientName}, Based on the your hair sample, you may not be at risk for Crohn's Disease or Ulcerative Collitis")
        elif valu[0] == 2:
            st.subheader(f"Hi {patientName}, Based on the your hair sample, you may be at risk for Crohn's Disease")
        else:
            st.subheader(f"Hi {patientName}, Based on the your hair sample, you may be at risk for Ulcerative Collitis")

        

        

    