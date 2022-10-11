import streamlit as st
from utils import project4_desc as p4d


def app():
	st.write('''
		## FinanceDataReader를 활용한 최근 경제 흐름 분석
	
		'''
		)
	p4d.desc()
