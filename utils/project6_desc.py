import streamlit as st
import Make_data as mk
import FinanceDataReader as fdr
from pykrx import stock
import pandas as pd
import module_anlstock as anl


def desc():
 with st.container():
    st.subheader('[1] 종목 조회')  
    code=st.text_input('매매하고자 하는 종목코드를 입력해주세요(ex:005930 (삼성전자))/2020년 1월이후 데이터')
    if st.button('검색'):
        (df,nasdaq_df)=mk.search_data(code)
        anl_play=anl.anl_stock(code,df,nasdaq_df)
        
        st.subheader('[2] 선택한 기업의 미래 예측 데이터')
        st.write('해당 종목의 과거지수를 토대로 훈련한 모델로 예측한 결과입니다.')
        anl_play.anl_code()
        
        
        st.subheader('[3] (나스닥지수 포함)선택한 기업의 미래 예측 데이터')
        st.write('코스피의 선행지수라고 판단될 수 있는 미국 나스닥 지수와 해당 종목의 과거를 토대로 훈련한 모델로 예측한 결과입니다.')
        anl_play.anl_code_nasdaq()
    
    

                                       