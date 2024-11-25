from statsmodels.iolib.smpickle import load_pickle
import streamlit as st
import pickle

@st.cache_resource
def load_logistic():
    model = load_pickle('src/notebook/logistic.pickle')
    return model

@st.cache_resource
def load_xgb():
    with open('src/notebook/xgb.pkl','rb') as f:
        model = pickle.load(f)
    return model
