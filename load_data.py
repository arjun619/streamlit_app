import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import streamlit as st
from sklearn.preprocessing import LabelEncoder

@st.cache(persist=True)
def load_data():
    data=pd.read_csv("C:\\Users\\arjun\\Downloads\\mushroom_csv.csv")
    label=LabelEncoder()
    data.drop(['stalk-root'],axis=1,inplace=True)
    columns=data.columns
    for col in range(len(columns)):
        print(columns[col])
        print(type(columns[col]))
        print(col)
        data[columns[col]]=label.fit_transform(data[columns[col]])
    return data

def split(data):
    y=data['class']
    X=data.drop(['class'],axis=1)
    x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)
    return x_train,x_test,y_train,y_test



