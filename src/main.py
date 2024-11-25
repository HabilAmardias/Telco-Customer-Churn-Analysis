import streamlit as st
import plotly.express as px
from load_data import load_data
from load_model import load_logistic, load_xgb
import numpy as np
import pandas as pd
from utils import get_transformer,preprocessing,clean_data,get_pdd
import plotly.graph_objects as go

st.set_page_config('Customer Churn Analysis',
                   layout='wide')
st.title('Customer Churn Analysis')
st.markdown("""
The Telco customer churn data contains information about a telco company that provided home phone and Internet services to 7043 customers in California in Q3. 
It indicates which customers have left, or stayed for their service. Multiple important features are included for each customer. This data is taken from
[Kaggle page here](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
""")

data = load_data()
data = clean_data(data)
st.dataframe(data.head())
with st.expander("Data Description",expanded=False):
    st.write("""
             gender: Whether the customer is a male or a female\n
             SeniorCitizen: Whether the customer is a senior citizen or not (1, 0)\n
             Partner: Whether the customer has a partner or not (Yes, No)\n
             Dependents: Whether the customer has dependents or not (Yes, No)\n
             tenure: Number of months the customer has stayed with the company\n
             PhoneService: Whether the customer has a phone service or not (Yes, No)\n
             MultipleLines: Whether the customer has multiple lines or not (Yes, No, No phone service)\n
             InternetService: Customer’s internet service provider (DSL, Fiber optic, No)\n
             OnlineSecurity: Whether the customer has online security or not (Yes, No, No internet service)\n
             OnlineBackup: Whether the customer has online backup or not (Yes, No, No internet service)\n
             DeviceProtection: Whether the customer has device protection or not (Yes, No, No internet service)\n
             TechSupport: Whether the customer has tech support or not (Yes, No, No internet service)\n
             StreamingTV: Whether the customer has streaming TV or not (Yes, No, No internet service)\n
             StreamingMovies: Whether the customer has streaming movies or not (Yes, No, No internet service)\n
             Contract: The contract term of the customer (Month-to-month, One year, Two year)\n
             PaperlessBilling: Whether the customer has paperless billing or not (Yes, No)\n
             PaymentMethod: The customer’s payment method (Electronic check, Mailed check, Bank transfer (automatic), Credit card\n
             MonthlyCharges: The amount charged to the customer monthly\n
             TotalCharges: The total amount charged to the customer\n
             Churn: Whether the customer churned or not (Yes or No)\n
             """)


col1,col2=st.columns([2,1],vertical_alignment='center')
with col1:
    contingency = pd.crosstab(data['PaymentMethod'],data['Churn'])
    tab = pd.DataFrame({
        "PaymentMethod": ["Bank transfer (automatic)", "Credit card (automatic)", 
                        "Electronic check", "Mailed check"] * 2,
        "Churn": ["No", "No", "No", "No", "Yes", "Yes", "Yes", "Yes"],
        "Count": contingency.values.T.reshape((-1))
    })
    payment_churn_contingency_tab = px.bar(tab, 
                 x="PaymentMethod", 
                 y="Count",  
                 color="Churn",
                 labels={"PaymentMethod": "Payment Method", "Churn": "Churn", "Count": "Number of Customers"})
    payment_churn_contingency_tab.update_layout(
        height=800
    )
    st.plotly_chart(payment_churn_contingency_tab,use_container_width=True)
with col2:
    st.write("This plot shows association between Churn Behaviour and Payment Method")
    st.write("""
            A lot of churned customer using Electronic Check as payment method compared
            to other method. There are some possible reasons\n
            1. Electronic Checks may offer customers more flexibility to cancel payments 
                compared to other methods like credit cards. This convenience 
                might encourage customers who are considering churn to proceed with 
                cancellation.\n
            2. Electronic Check payments could be prone to issues like failed transactions, 
                delays, or discrepancies, which could lead to dissatisfaction and, ultimately, 
                churn.
            """)

col1,col2=st.columns([2,1],vertical_alignment='center')
with col1:
    churn_monthly_box = px.box(data_frame=data,
                               x='Churn',
                               y='MonthlyCharges',
                               color='Churn')
    churn_monthly_box.update_layout(xaxis_title="Churn",
                                    yaxis_title="MonthlyCharges",
                                    height=800)
    st.plotly_chart(churn_monthly_box,use_container_width=True)
with col2:
    st.write("""
            The median monthly charges for churned customers are visibly higher than for 
            loyal customers, suggesting that churned customers tend to pay more. 
            While there is some overlap in the ranges of monthly charges between the two groups, churned customers show a concentration at higher values.
            It is possible that expensive monthly charges is one of the reason why customer
            churn.
            """)
    

col1,col2=st.columns([2,1],vertical_alignment='center')
with col1:
    tab = pd.crosstab(data['Contract'],data['Churn'])
    heatmap = px.imshow(tab,text_auto=True)
    heatmap.update_layout(height=800)
    st.plotly_chart(heatmap,use_container_width=True)
with col2:
    st.write("This plot show correlation between contract term and churn behaviour")
    st.write("""
            As seen from the plot, churned customer tend to have shorter contract term. 
            Possible explanations for
            this behaviour are:\n
             1. Trying out the service without a long-term intention to stay\n
             2. More likely to compare alternatives and switch providers when they find better 
             options.
            """)

st.markdown("""
<h3>Now lets see how each service affect customer behaviour</h3>
""",unsafe_allow_html=True)

service_cols = ['PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
       'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
       'StreamingMovies']
cols = service_cols + ['Churn']
services = data[cols].copy(True)
services = preprocessing(X=services,y=None)


col1,col2 = st.columns([2,1],vertical_alignment='center')
with col1:
    logistic = load_logistic()
    importances = np.exp(logistic.params.sort_values(ascending=False))[:-1]
    indices = importances.index
    logistic_coef=px.bar(importances,orientation='h',color=indices)
    logistic_coef.update_layout(
        xaxis_title="Importances*",
        yaxis_title="Columns",
        height=800
    )
    st.plotly_chart(logistic_coef,use_container_width=True)
with col2:
    st.write("""
    This plot shows association between churn behaviour and each service
    """)
    st.write("""
    It can be seen that there is a strong association between Internet Service 
    and Customer Churn compared to other services. This shows that it is possible one of the reason customer 
    churned is because customer is not satisfied with the internet service.
    """)
    st.markdown('<small>*exponent of logistic regression model coefficient with Churn column as dependent variable</small>',
                unsafe_allow_html=True)

col1,col2 = st.columns([2,1],vertical_alignment='center')
with col1:
    services.drop('Churn',axis=1,inplace=True)
    services['MonthlyCharges'] = data['MonthlyCharges']
    service_monthly_corr = services.corr('kendall')['MonthlyCharges'].sort_values(ascending=False)[1:]
    service_monthly_barplot = px.bar(service_monthly_corr,
                                     orientation='h',
                                     color=service_monthly_corr.index)
    service_monthly_barplot.update_layout(
        xaxis_title="Kendall Tau Correlation Coefficient",
        yaxis_title="Columns",
        height=800
    )
    st.plotly_chart(service_monthly_barplot,use_container_width=True)

with col2:
    st.write("""
    This plot shows correlation between MonthlyCharges and each service
    """)
    st.write("""
    From the plot we can see that Internet Service contribute a lot to
    monthly charge amount. This high contribution maybe one of the reason
    customers quit.
    """)


X,y = data.drop('Churn',axis=1),data['Churn']
X,y = preprocessing(X,y)
transformer = get_transformer()
xgb = load_xgb()
X = transformer.transform(X)
columns = X.columns.values

st.markdown("""
<h3>Let's try model the data with XGBoost and see each features importance</h3>
""",unsafe_allow_html=True)

selected_features = st.selectbox('Select a Feature',list(columns))
pdd = get_pdd(xgb,X,selected_features)
fig = go.Figure()
fig.add_trace(go.Scatter(x=pdd.pd_results[0]['grid_values'][0], 
                        y=pdd.pd_results[0]['average'][0], 
                        mode='lines', 
                        name=f'{selected_features}'))
fig.update_layout(
    title=f"{selected_features} Partial Dependence Plot (XGBoost)"
)
st.plotly_chart(fig,use_container_width=True)

xgb_importances_df = pd.DataFrame({
    'Importances': xgb.feature_importances_,
    'Columns':columns
}).sort_values(by='Importances',ascending=False)
xgb_importances_bar=px.bar(
    xgb_importances_df,
    orientation='h',
    x='Importances',
    y='Columns',
    title='XGBoost Feature Importance',
    color_discrete_sequence=px.colors.qualitative.Dark24,
    color='Columns'
)
xgb_importances_bar.update_layout(
    height=800
)
st.plotly_chart(xgb_importances_bar,use_container_width=True)

with st.container():
    st.markdown("""
    <h3>Conclusion</h3>
    """,unsafe_allow_html=True)
    st.write("""
    Based on the analysis of the data, several key factors were identified as having a significant impact on churn behavior. Specifically, the following features were found to be 
    highly correlated with customer churn:\n

    1. Contract Type: Customers with shorter-term contracts tend to have a higher likelihood of churning. This suggests that offering long-term contracts or incentives for contract 
    renewals could reduce churn rates.\n

    2. Internet Service: Customers with specific types of internet service (e.g., Fiber optic) are more likely to churn. Enhancing internet service offerings or lowering the monthly 
    charge for internet service could improve customer retention.\n

    3. Electronic Payment Method: Customers who use electronic payment methods (such as electronic checks) show a higher tendency to churn. 
    Encouraging customers to switch to more convenient payment methods or enhancing electronic payment method may help in reducing churn.\n
    """)
with st.container():
    st.markdown("""
    <h3>Suggestions</h3>
    """,unsafe_allow_html=True)
    st.write("""
            1. Explore how other features like tenure, Partner, Dependent and PaperBilling affecting churn behaviour.\n
            2. Use other machine learning model for analysis
            """)