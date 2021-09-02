import pandas as pd
import numpy as np
import matplotlib as plt
import streamlit as st
st.title(" ----------------->    Welcome <----------------")
name=st.text_input("Enter your name here")
day=st.text_input(" how many days of prediction you want ")
c=float(int(day))
df=pd.read_excel(r"DEXINUS (1).xls")
data=df.copy()
# filling null value by privious row
df.fillna(method='ffill', inplace=True)
# splitting data
train_df=df.iloc[:12529]
test_df=df.iloc[12529:]
# AR modele creation
from statsmodels.tsa.arima_model import ARMA
model_ar1 = ARMA(df.DEXINUS, order = (1,0))
results_ar1 = model_ar1.fit(disp = 0)
future_dates = pd.date_range(start = df.observation_date.max() + pd.DateOffset(1), end = df.observation_date.max() + pd.DateOffset(c), freq = 'D')
future_df = pd.DataFrame()
future_df['Month'] = [i.month for i in future_dates]
future_df['Year'] = [i.year for i in future_dates]
future_df['Day'] = [i.day for i in future_dates]
future_dates_df=pd.DataFrame(index=future_dates[1:],columns=df.columns)
df_future=pd.concat([df,future_dates_df])
prdct_future=results_ar1.predict(start=12649, end=len(df_future),dynamic=False)
if(st.button(" click here for Predict")):
    result=pd.DataFrame({'months':future_dates,'DEXINUS':prdct_future})
    st.write(result)
    st.write('Thanks',name)

