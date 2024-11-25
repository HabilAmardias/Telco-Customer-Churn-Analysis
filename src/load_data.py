import pandas as pd
import streamlit as st

@st.cache_data
def load_data():
    data = pd.read_csv('data/WA_Fn-UseC_-Telco-Customer-Churn.csv')
    return data