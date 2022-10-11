import streamlit as st
from utils import project8_desc as p8d

def app():
	st.write('''
		## backtrader(백트레이더)이용하여 모의투자 해보기
		'''
		)
	p8d.desc()
