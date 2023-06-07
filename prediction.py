#!/usr/bin/env python
# coding: utf-8

# In[1]:


import joblib


def predict(data):
    clf = joblib.load("rf_model.sav")
    return clf.predict(data)


# In[ ]:




