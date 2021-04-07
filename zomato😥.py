# IMPORT STATEMENTS
import streamlit as st
import pandas as pd
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import plotly.figure_factory as ff
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import seaborn as sns

st.write("""
# Zomato Price Prediction Developed By **PREM KUMAR**
This app predicts the **restaurant ratings**!
""")
st.write('---')








#df = (r"C:\Users\Admin\Downloads\Zomato_df.csv")

@st.cache

def load_data():
  data = pd.read_csv('https://github.com/Premkumar7090/zomatopredictions/blob/projects/Zomato_df.csv', encoding = 'ISO-8859-1')
  return data
data=load_data()


df1 = data.head(200)  




# HEADINGS
st.title('zomato ratings')
st.sidebar.header('zomato Data')
st.subheader('Training Data Stats')
st.write(df1.describe())


# X AND Y DATA
x=df1.drop('rate',axis=1)
y=df1['rate']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.3,random_state=10)


# FUNCTION
def user_report():
  OR = st.sidebar.slider('online_order', 0,1 )
  BT = st.sidebar.slider('book_table', 0,1 )
  V = st.sidebar.slider('votes',  float(x.votes.min()),float(x.votes.max()), float(x.votes.mean()))
  LO = st.sidebar.slider('location',  float(x.location.min()),float(x.location.max()), float(x.location.mean()))
  RT = st.sidebar.slider('rest_type',  float(x.rest_type.min()),float(x.rest_type.max()), float(x.rest_type.mean()))
  CU = st.sidebar.slider('cuisines',  float(x.cuisines.min()),float(x.cuisines.max()), float(x.cuisines.mean()))
  CO = st.sidebar.slider('cost',  float(x.cost.min()),float(x.cost.max()), float(x.cost.mean()))
  MI = st.sidebar.slider('menu_item',  float(x.menu_item.min()),float(x.menu_item.max()), float(x.menu_item.mean()) )
  
  user_report_data = {
      'online_order':OR,
      'book_table':BT,
      'votes':V,
      'location':LO,
      'rest_type':RT,
      'cuisines':CU,
      'cost':CO,
      'menu_item':MI
      
  }
  report_data = pd.DataFrame(user_report_data, index=[0])
  return report_data




# PATIENT DATA
user_data = user_report()
st.subheader('restaurant details')
st.write(user_data)



# MODEL
from sklearn.ensemble import  ExtraTreesRegressor
ET_Model=ExtraTreesRegressor(n_estimators = 120)
ET_Model.fit(x_train,y_train)


y_predict=ET_Model.predict(user_data)

st.header('Prediction of zomato')
st.write(y_predict)
st.write('---')

