import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import streamlit as st
from sklearn.preprocessing import LabelEncoder
from sklearn.datasets import load_iris

@st.cache(persist=True,suppress_st_warning=True)
def load_data(dataset):
    if dataset=='Mushroom Dataset':
        data=pd.read_csv("C:\\Users\\arjun\\Downloads\\mushroom_csv.csv")
        label=LabelEncoder()
        data.drop(['stalk-root'],axis=1,inplace=True)
        columns=data.columns
        for col in range(len(columns)):
            #print(columns[col])
            #print(type(columns[col]))
            #print(col)
            data[columns[col]]=label.fit_transform(data[columns[col]])
    if dataset=='Iris':
        x=load_iris()
        data=x['data']
        y=x['target']
        data=pd.DataFrame(data)
        data.columns=['Sepal Length','Sepal Width','Petal Length','Petal Width']
        data['target']=y


    return data

def split(data,dataset):
    if dataset=='Mushroom Dataset':    
        y=data['class']
        X=data.drop(['class'],axis=1)
    if dataset=='Iris':
        y=data['target']
        X=data.drop(['target'],axis=1)
    x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)
    
    
    return x_train,x_test,y_train,y_test

def column_names(data):
    column_names=data.columns
    return column_names



