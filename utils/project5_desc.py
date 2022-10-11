import streamlit as st
import Make_data as mk
import FinanceDataReader as fdr
from pykrx import stock
import pandas as pd

KRX= pd.read_csv("stock_data/KRX_table.csv")
KOSPI= pd.read_csv("stock_data/KOSPI_table.csv")
KOSDAQ= pd.read_csv("stock_data/KOSDAQ_table.csv")
market_price_change= pd.read_csv("stock_data/market_price_change_table.csv")

def desc():
 with st.container():
    mk.main()
    st.subheader('[1] 코스피, 코스닥 현황 그래프')

    if st.checkbox('코스피'):
        KOSPI_=KOSPI['Close']
        st.line_chart(KOSPI_)

    if st.checkbox('코스닥'):
        KOSDAQ_=KOSDAQ['Close']
        st.line_chart(KOSDAQ_)

    st.subheader('[2] KRX 기업정보 현황')

    if st.checkbox('전체 기업 현황 보기'):
        st.dataframe(KRX)

    if st.checkbox('특정 기업 정보 보기'):
        name_=st.selectbox('보고자 하는 기업명을 입력해주세요',tuple(KRX['Name']))

        try :
            if st.button('검색'):
                    
                tmp=KRX[KRX.Name==name_]
                st.dataframe(tmp)
        except Exception as e :
            st.warning('찾고자 하는 기업 정보가 없습니다.')

    st.subheader('[3] KRX 기업 등락 현황')

    if st.checkbox('KRX 기업 등락률 보기'):

        status = st.radio('정렬을 선택하세요', ['오름차순정렬','내림차순정렬'])
        if status == '오름차순정렬' :
            st.dataframe( market_price_change.sort_values('등락률',ascending=False) )
        elif status == '내림차순정렬' :
            st.dataframe( market_price_change.sort_values('등락률'))
                                          