#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
pd.set_option('display.max_columns', None)


# In[2]:


# Import necessary libraries for modelling & evaluation
from sklearn.model_selection import cross_validate, GridSearchCV # Cross Validation & Grid Search
from sklearn.linear_model import LogisticRegression              # Logistic Regression
from sklearn.ensemble import RandomForestClassifier              # Random Forest
from sklearn.naive_bayes import GaussianNB                       # Naive Bayes
from sklearn.neighbors import KNeighborsClassifier               # KNN
from sklearn.svm import SVC                                      # SVM

from matplotlib import pyplot
from sklearn.tree import plot_tree

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, precision_recall_fscore_support #Metrics Libraries
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report                          #Metrics Libraries


# In[3]:


df = pd.read_csv('Amazon_Sales_Report.csv')


# In[4]:


df.head()


# In[5]:


# Split the dataset into training & testing set with 30-70% split

def split_data(df):
    #test train split
    y = df['rejected'].values # Target
    X = df.drop(['rejected'], axis=1).values # Attributes
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42) #30%-70% split with random state (seed) of 42

    print("Training set size: ", len(y_train), "\nTesting set size: ", len(y_test))
    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = split_data(df)


# In[6]:


def RandomF():
    # Random Forest
    # randF = RandomForestClassifier(random_state=42) # Random state (seed) of 42
    randF = RandomForestClassifier(max_depth=5, random_state=42)
    randF_result = randF.fit(X_train, y_train) # Fit the model
    y_pred_RF = randF.predict(X_test) # Predict class labels for testing set

    # Performance metrics
    print("\nRandom Forest model\n", "=" * 80)
    print("\nModel performance\n", "-" * 80, "\nAccuracy score\t: ", accuracy_score(y_test, y_pred_RF))
    print("Precision score\t: ", precision_score(y_test, y_pred_RF))
    print("Recall score\t: ", recall_score(y_test, y_pred_RF))
    print("F-score\t\t: ", f1_score(y_test, y_pred_RF))

    # Plot decision tree
    fig = plt.figure(figsize=(10, 10))
    plot_tree(randF.estimators_[0],
              feature_names=df.drop(['rejected'], axis=1).columns,
              class_names='rejected',
              filled=True, impurity=True,
              rounded=True)
    joblib.dump(randF, "rf_model.sav")
    return randF_result, y_pred_RF


RandomForestModel, y_pred_RF = RandomF()


# In[ ]:




