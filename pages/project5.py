import streamlit as st
from utils import project5_desc as p5d

def app():
	st.write('''
		## 현재 시장 현황 알아보기
		'''
		)
	p5d.desc()
