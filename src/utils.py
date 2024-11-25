import pandas as pd
import streamlit as st
import pickle
from sklearn.inspection import PartialDependenceDisplay

@st.cache_data
def preprocessing(X:pd.DataFrame,y:pd.Series | None)->pd.DataFrame:
    num_col = ['SeniorCitizen','tenure','MonthlyCharges','TotalCharges']
    cat_col = list(set(X.columns)-set(num_col))
    X1=X.copy()
    for col in cat_col:
        if col == 'gender':
            mapping = {'Female':0,'Male':1}
        elif col == 'Contract':
            mapping = {'Month-to-month':0, 'One year':1, 'Two year':2}
        elif col in ['Dependents','PaperlessBilling','PhoneService','Partner']:
            mapping = {'No':0, 'Yes':1}
        elif col == 'MultipleLines':
            mapping = {'No phone service':0,'No':1,'Yes':2}
        elif col in ['DeviceProtection','TechSupport','OnlineSecurity','StreamingTV','StreamingMovies','OnlineBackup']:
            mapping = {'No internet service':0,'No':1,'Yes':2}
        elif col == 'InternetService':
            mapping = {'No':0,'DSL':1,'Fiber optic':2}
        elif col == 'PaymentMethod':
            continue
        X1[col] = X1[col].apply(lambda x: mapping[x] if x in mapping else -1)
    if isinstance(y,pd.Series):
        y1 = y.apply(lambda x: 1 if x=='Yes' else 0)
        return X1,y1
    else:
        return X1

@st.cache_resource
def get_transformer():
    with open('src/notebook/transformer.pkl','rb') as f:
        transformer = pickle.load(f)
    return transformer

def clean_data(data:pd.DataFrame)->pd.DataFrame:
    data.loc[data['TotalCharges']==' ','TotalCharges'] = data.loc[data['TotalCharges']==' ','tenure'] * data.loc[data['TotalCharges']==' ','MonthlyCharges']
    data['TotalCharges']=data['TotalCharges'].astype(float)
    return data

@st.cache_data
def get_pdd(_xgb,X,column):
    return PartialDependenceDisplay.from_estimator(_xgb,X,[column])