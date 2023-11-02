import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pickle
import warnings
import sklearn
warnings.filterwarnings('ignore')
from sklearn.ensemble import AdaBoostClassifier

st.title('Medical Diagnostic Web App 🆘')
st.subheader('Does the paitent have diabetes?')
df = pd.read_csv('diabetes.csv')

if st.sidebar.checkbox('View Data', False):
    st.write(df)
if st.sidebar.checkbox('View Distributions', False):
    df.hist()
    plt.tight_layout()
    st.pyplot()

#Step1: Load the pickled model
model=open('rfc.pickle','rb')
clf=pickle.load(model)
model.close()

#Step2:Get the front end user input
pregs=st.number_input('Pregnancies',0,20,0)
plas=st.slider('Glucose',40,200,40)
BP=st.slider('BloodPressure',20,150,20)
skin=st.slider('SkinThickness',7,99,7) 
insulin=st.slider('Insulin',14,850,14)
bmi=st.slider('BMI',18,70,18) 
dpf=st.slider('DiabetesPedigreeFunction',0.05,2.05,0.05)
Age=st.slider('Age',21,90,21)

#Step3: Get the model input
input_data=[[pregs,plas,BP,skin,insulin,bmi,dpf,Age]]

#Step4: Get the prediction and print the result
prediction=clf.predict(input_data)[0]
if st.button('Predict'):
    if prediction==0:
        st.subheader('Non Diabetic')
    else:
        st.subheader('Diabetic')
