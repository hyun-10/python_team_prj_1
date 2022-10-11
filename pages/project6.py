import streamlit as st
from utils import project6_desc as p6d

def app():
	st.write('''
		## 원하는 주식 예측 데이터 만들기
		'''
		)
	p6d.desc()
