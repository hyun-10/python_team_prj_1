import streamlit as st
from utils import project2_desc as p2d


def app():
	st.write('''
		## PyKrx(증권 데이터 수집 라이브러리)
		'''
		)
	p2d.desc()
