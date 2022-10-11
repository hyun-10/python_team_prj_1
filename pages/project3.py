import streamlit as st
from utils import project3_desc as p3d


def app():
	st.write('''
		## FinanceDataReader(금융 데이터 리더기)
		'''
		)
	p3d.desc()
