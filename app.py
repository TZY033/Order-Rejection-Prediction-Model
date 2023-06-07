#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import pandas as pd
import numpy as np
from prediction import predict

st.title('Order Rejection Model')

st.header("Plant Features")
col1, col2 = st.columns(2)

with col1:
    st.text("Please select method of fulfillment")
    inp_fulfillment = st.radio('method of fulfillment:',np.unique(data['fulfilment']))
    
    st.text("Please select shipment service level")
    inp_shipment_service_level = st.radio('shipment service level:',np.unique(data['shipment_service_level']))
    
    st.text("Please select category")
    inp_category = st.radio('category:',np.unique(data['category']))
    
    st.text("Please select size")
    inp_size = st.radio('size:',np.unique(data['size']))

    st.text("Please select amount")
    inp_amount = st.slider('amount:',1.0,2000.0,0.01)
    
    st.text("Please select shipping state")
    inp_shipping_state = st.radio('shipping state:',np.unique(data['shipping_state']))
    
    st.text("Please select region")
    inp_region = st.radio('region:',np.unique(data['region']))

st.text('')
if st.button("Predict Rejection"):
    result = predict(
        np.array([[inp_fulfillment, inp_shipment_service_level, inp_category, inp_size,inp_amount,inp_shipping_state,inp_region]]))
    st.text(result[0])


st.text('')
st.text('')


# In[ ]:




