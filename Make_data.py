import streamlit as st
import FinanceDataReader as fdr
from pykrx import stock
import pandas as pd

def Make_data(): #사용할 것들 자동갱신 파일화

     KRX_table = fdr.StockListing('KRX')

     KOSPI_table = fdr.DataReader('KS11', start='2020')

     KOSDAQ_table = fdr.DataReader('KQ11',start='2020')

     market_price_change_table = stock.get_market_price_change("20220801", "20220820")

     KRX_table.to_csv("stock_data/KRX_table.csv")

     KOSPI_table.to_csv("stock_data/KOSPI_table.csv")

     KOSDAQ_table.to_csv("stock_data/KOSDAQ_table.csv")

     market_price_change_table.to_csv("stock_data/market_price_change_table.csv")

def make_data2(code): # 8페이지 검색하면 8원하는 기업 주식 정보 파일로 저장
    
    if st.button('검색'):
        df = fdr.DataReader(code,'2020-01-01')
        df.to_csv("stock_data/df2.csv")


def search_data(code): # 6페이지 검색하면 실행, 원하는 기업 주식 정보 출력
    df = fdr.DataReader(code,'2020')
    df.to_csv("stock_data/df.csv")
    tmp=df.index
    df= pd.read_csv("stock_data/df.csv")
    
    df=pd.DataFrame(data=df.Close)
    df.index=tmp
    st.write(f'[1-1] {code} 의 DataFrame')  
    st.dataframe(df.sort_index(ascending=False))

    st.write(f'[1-2] {code} 의 Linechart') 
    st.line_chart(df['Close'])
    
    
    #나스닥 지수 df생성        
    nasdaq_df=fdr.DataReader('IXIC','2020')
    nasdaq_df.to_csv('stock_data/nasdaq_df.csv')
    tmp2=nasdaq_df.index
    nasdaq_df=pd.read_csv('stock_data/nasdaq_df.csv')
    nasdaq_df.index=tmp2
    return df, nasdaq_df
        
def main(): #누르면 데이터 갱신 , 많이하면 블락당함
    if st.button('데이터 갱신하기'):
        Make_data()

if __name__ == '__main__' :
    main()
