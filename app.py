#!/usr/bin/env python
# coding: utf-8

# In[3]:


import streamlit as st
import pickle
import numpy as np
import pandas as pd


# In[4]:


model = pickle.load(open('model.bin','rb'))

def predict_price(area_type,availability,location,dist_mainroad,street_type,park_facility,total_sqft,bath,balcony,bedroom,price_per_sqft):
    input = np.array([[area_type,availability,location,dist_mainroad,street_type,park_facility,total_sqft,bath,balcony,bedroom,price_per_sqft]]).astype(np.float64)
    prediction = model.predict(input)
    return float(prediction)
     
def main():
    html_temp=""" <div style="background-color:#025247; padding:10px"> <h2 style="color:white;text-align:center;">Bengaluru Real Estate Price Prediction</h2></div> """
    st.markdown(html_temp, unsafe_allow_html=True)
    area_type = st.selectbox("select area type",("Built-up Area","Carpet Area","Plot Area","Super built-up Area"))
    if area_type == 'Super built-up  Area':
        area_type=0
    elif area_type == 'Plot Area':
        area_type=1
    elif area_type == 'Built-up Area':
        area_type=2
    else:
        area_type=3    
    availability = st.selectbox("select availability date",("Immediate Possession","Ready To Move","Others"))
    if availability == 'Immediate Possession':
        availability=0
    elif availability == 'Ready To Move':
        availability=1
    else:
        availability=2
    location = st.selectbox("select location type",("Highly populated","moderately populated","less populated"))
    if location == "Highly populated":
        location=0
    elif location == "moderately populated":
        location=1
    else:
        location=2
    dist_mainroad = st.slider("select distance from main road",0,500)
    street_type = st.selectbox("select streat type",("Gravel","Paved","No Access"))
    if street_type == 'Gravel':
        street_type=0
    elif street_type == 'Paved':
        street_type=1
    else:
        area_type=2 
    park_facility = st.selectbox("select parking facility",("Yes","No"))
    if park_facility == 'Yes':
        park_facility=0
    else:
        park_facility=1
    total_sqft = st.slider("select total sqft area of house",5,70000)
    bedroom = st.slider("select number of bedrooms in house",1,50)
    bath = st.slider("select number of bathrooms in house",1,50)
    balcony = st.slider("select number of balconies in house",0,5)
    price_per_sqft = st.bedroom = st.slider("select price per sqft area of house",20,50000)
    
    if st.button('PREDICT'):
        output = predict_price(area_type,availability,location,dist_mainroad,street_type,park_facility,total_sqft,bath,balcony,bedroom,price_per_sqft)
        st.success('The predicted price of house with above features is {}'.format(output))

if __name__=='__main__':
    main()


# In[ ]:




