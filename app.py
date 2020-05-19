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

st.title('binary classification for mushrooms')
st.sidebar.title("binary classification for mushrooms")

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

@st.cache(persist=True)
def split(data):
    y=data['class']
    X=data.drop(['class'],axis=1)
    x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)
    return x_train,x_test,y_train,y_test

def plot_metrics(metrics_list):
    if 'ROC Curve' in metrics_list:
        st.subheader("ROC Curve")
        plot_roc_curve(model,x_test,y_test)
        st.pyplot()
    if 'Confusion Matrix' in metrics_list:
        st.subheader("Confusion Matrics")
        plot_confusion_matrix(model,x_test,y_test,display_labels=class_names)
        st.pyplot()
    if 'Precision Recall Curve' in metrics_list:
        st.subheader("Precision Recall Graph")
        plot_precision_recall_curve(model,x_test,y_test)
        st.pyplot()
data=load_data()
x_train,x_test,y_train,y_test=split(data)
class_names=['edible','poisonous']

st.sidebar.subheader('Choose Classifier')
classifier=st.sidebar.selectbox("Classfier",("SVM","Logistic Regression","Random Forest"))

if classifier=="SVM":
    c=st.sidebar.number_input("regularization parameter",0.01,10.0,step=0.1,key='c')
    kernel=st.sidebar.radio("kernel",("rbf","linear"),key="kernel")
    gamma=st.sidebar.radio("gamma",("auto","scale"),key="gamma")

    metrics=st.sidebar.multiselect("metrics",('ROC Curve','Confusion Matrix','Precision Recall Curve'),key="metrics")

    if st.sidebar.button("CLASSIFY",key="classifer"):
        st.subheader("SVM RESULTS")
        model=SVC(kernel=kernel,gamma=gamma,C=c)
        model.fit(x_train,y_train)
        accuracy=model.score(x_test,y_test)
        y_pred=model.predict(x_test)
        st.write("Accuracy :",accuracy.round(2))
        st.write("Precision:",precision_score(y_test,y_pred))
        st.write("Recall Score",recall_score(y_test,y_pred))
        plot_metrics(metrics)

if classifier=="Logistic Regression":
    st.sidebar.subheader("Model Hyperparameters")
    lambdas=st.sidebar.number_input("regularization parameter",0.01,10.0,step=0.1,key='lambda')
    max_iter=st.sidebar.slider("max_iter",1,1000,key="max_iter") 
    #gamma=st.sidebar.radio("gamma",("auto","scale"),key="gamma")

    metrics=st.sidebar.multiselect("metrics",('ROC Curve','Confusion Matrix','Precision Recall Curve'),key="metrics")

    if st.sidebar.button("CLASSIFY",key="classifer"):
        st.subheader("LOGISTIC REGRESSION RESULTS")
        model=LogisticRegression(max_iter=max_iter,C=lambdas)
        model.fit(x_train,y_train)
        accuracy=model.score(x_test,y_test)
        y_pred=model.predict(x_test)
        st.write("Accuracy :",accuracy.round(2))
        st.write("Precision:",precision_score(y_test,y_pred))
        st.write("Recall Score",recall_score(y_test,y_pred))
        plot_metrics(metrics)

if classifier=="Random Forest":
    st.sidebar.subheader("Model Hyperparameters")
    n_estimators=st.sidebar.number_input("trees in forest",10,500,step=5,key="n_estimators")
    max_depth=st.sidebar.slider("depth",5,20,step=1,key="max_depth")
    bootstrap=st.sidebar.radio("bootstrap",("True","False"),key="bootstrap")
    metrics=st.sidebar.multiselect("metrics",('ROC Curve','Confusion Matrix','Precision Recall Curve'),key="metrics")

    if st.sidebar.button("CLASSIFY",key="classifer"):
        st.subheader("RANDOM FOREST RESULTS")
        model=RandomForestClassifier(n_estimators=n_estimators,max_depth=max_depth,bootstrap=bootstrap)
        model.fit(x_train,y_train)
        accuracy=model.score(x_test,y_test)
        y_pred=model.predict(x_test)
        st.write("Accuracy :",accuracy.round(2))
        st.write("Precision:",precision_score(y_test,y_pred))
        st.write("Recall Score",recall_score(y_test,y_pred))
        plot_metrics(metrics)


if st.sidebar.checkbox("Show Dataset",False):   
    st.subheader("SHOWING DATA ")
    st.write(data)

