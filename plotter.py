import plotly.express as px
import pandas as pd
import streamlit as st


def bi_variate_graph(data,column_names):
    column_names=tuple(column_names)
    feature1=st.selectbox("Select first feature",column_names,key="selector")
    feature2=st.selectbox("Select second feature",column_names,key="selector1")
    flag_vis=st.checkbox("visualize",value=False)
    bar_flag=False
    scatter_flag=False
    if flag_vis:
        bar_flag=st.checkbox("Do you want a bar plot?",value=False)
        scatter_flag=st.checkbox("Do you want a scatter plot?",value=False)
    if bar_flag:
        fig_bar=px.bar(data,x=feature1,y=feature2)
        st.plotly_chart(fig_bar)
    if scatter_flag:
        fig_scatter=px.scatter(data,x=feature1,y=feature2)
        st.plotly_chart(fig_scatter)

    
