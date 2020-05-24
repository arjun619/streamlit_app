import plotly.express as px
import pandas as pd
import streamlit as st


def bi_variate_graph(data,column_names):
    column_names=tuple(column_names)
    feature1=st.selectbox("Select first feature",column_names,key="selector")
    feature2=st.selectbox("Select second feature",column_names,key="selector1")
    flag_vis=st.checkbox("visualize",value=False)

    if flag_vis:
        fig=px.scatter(data,x=feature1,y=feature2)
        st.plotly_chart(fig)

    
