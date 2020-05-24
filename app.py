import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import streamlit as st
from sklearn.metrics import precision_score,accuracy_score,recall_score
import load_data
from sklearn.metrics import plot_roc_curve,plot_confusion_matrix,plot_precision_recall_curve

data=load_data.load_data()
x_train,x_test,y_train,y_test=load_data.split(data)
class_names=['edible','poisonous']

def svm_display():
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
        plot_metrics(model,metrics)

def LogisticRegression_display():
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
        plot_metrics(model,metrics)

def random_forest_display():
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
        plot_metrics(model,metrics)

def plot_metrics(model,metrics_list):
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
    