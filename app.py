import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_roc_curve,plot_confusion_matrix,plot_precision_recall_curve   
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score,precision_score,recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import app_algorithm
import plotter
import load_data


st.title('binary classification for mushrooms')
st.sidebar.title("binary classification for mushrooms")
data=app_algorithm.choose_dataset()
#dataset=st.sidebar.selectbox("choose the dataset",("Mushroom Dataset","other"),key='dataset')
#@st.cache(persist=True)

#data=load_data.load_data(dataset)
#x_train,x_test,y_train,y_test=load_data.split(data)
#class_names=['edible','poisonous']"""

st.sidebar.subheader('Choose Classifier')
classifier=st.sidebar.selectbox("Classfier",("SVM","Logistic Regression","Random Forest"))


if classifier=="SVM":
    app_algorithm.svm_display(data)

if classifier=="Logistic Regression":
    app_algorithm.LogisticRegression_display(data)

if classifier=="Random Forest":
    app_algorithm.random_forest_display(data)


if st.sidebar.checkbox("Show Dataset",False):   
    st.subheader("SHOWING DATA ")
    st.write(data)

flag=st.sidebar.checkbox("Visualize Data?",key="vis",value=False)
if flag:
    column_names=load_data.column_names(data)
    plotter.bi_variate_graph(data,column_names)



