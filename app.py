import streamlit as st
import FinanceDataReader as fdr
nasdaq_df=fdr.DataReader('IXIC','2020')
st.write(nasdaq_df)
