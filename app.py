import streamlit as st
from Make_page import make_page
from pages import project1 as p1
from pages import project2 as p2
from pages import project3 as p3
from pages import project4 as p4
from pages import project5 as p5
from pages import project6 as p6
from pages import project7 as p7
from pages import project8 as p8

app = make_page()

# 타이틀
st.title('Project')

# 추가할 페이지
app.add_page("lstm 란?", p1.app)
app.add_page("pykrx 소개", p2.app)
app.add_page("FinanceDataReader소개", p3.app)
app.add_page("FinanceDataReader를 활용한 최근 경제 흐름 분석", p4.app)
app.add_page("현재 시장 현황", p5.app)
app.add_page("원하는 주식 예측 데이터 만들기", p6.app)
app.add_page("backtrader 소개", p7.app)
app.add_page("backtrader 활용한 모의투자 해보기", p8.app)
app.run()
