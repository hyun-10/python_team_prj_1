import streamlit as st
import Make_data as mk
import FinanceDataReader as fdr
from pykrx import stock
import pandas as pd
import backtrader as bt
import Trading_Test as tt
from IPython.display import display, Image
import Open_Img as oi

count = 0

def desc():
 with st.container():
    st.write('''

#### 1. 사용한 투자 전략 소개

    1) 단순 이동 평균(SMA) 크로스 전략
    2) 상대적 강도 지수(RSI) 전략
    3) 볼린저 밴드(Bollinger_Band) 전략
	''')
    st.subheader('[1] 원하는 기업 찾기')

    code=st.text_input('모의 투자 하고 싶은 기업 종목코드를 입력해주세요(ex:005930 (삼성전자))/2020년 1월이후 데이터')
    
    if st.button('검색'):
        df = fdr.DataReader(code,'2020-01-01')
        df.to_csv("stock_data/df2.csv")

    st.subheader('[2] 초기 자본금과 매매단위 정하기')
  
    start_money= st.number_input('투자할 초기 금액을 매매단위를 고려하여 입력하세요 (10000000원 추천)',0)

    trading_count= st.number_input('매매단위를 입력하세요',0)   

    st.subheader('[3] 투자 전략을 선택하세요 (중복 선택 불가능)')

    if st.checkbox('단순 이동 평균(SMA) 크로스 전략'):
        st.write('''

    ##### 1) 단순 이동 평균(SMA) 크로스 전략)
         
        모든 투자자들이 사용하는 지표라고 할 수 있을 정도로 가장 대중적인 지표이다.
        이동 평균은 주식 가격의 추세를 파악하는데 사용 된다.
        단점은 추세의 빠른 변화를 바로 읽지 못한다.

        ''')
        count = 1

    if st.checkbox('상대적 강도 지수(RSI) 전략'):
        st.write('''

    ##### 2) 상대적 강도 지수(RSI) 전략
         
        과매수, 과매도를 판단하기 위한 지표로서 과매수는 주식 상승으로 인해 많이 매수된
        상태를 말하고, 과매도는 주식 하락으로 인해 많이 매도된 상태를 말한다.
        RSI = 일정기간 동안의 상승폭 합계 / (일정기간 동안의 상승폭 합계 + 일정기간 동안의 하락폭 합계)
        로 구하며 기본적으로 과매수 70 과매도 30 으로 잡는다.
        단점은 주가가 같은 방향으로 오래 지속되면 거짓 신호를 보낼 확률이 있다.
        
        ''')
        count = 2

    if st.checkbox('볼린저 밴드(Bollinger_Band) 전략'):
        st.write('''

    ##### 3) 볼린저 밴드(Bollinger_Band) 전략
         
        이동평균에 변동성을 결합한 그래프이다. 이동평균선이 밴드의 중심축이 되고, 밴드의
        상단과 하단은 이동평균 과 표준편차의 합과 차로 표시한다.
        기본적으로 20일의 이동평균선을 기준으로 하며
        볼린저밴드 상단선: 20일 이동평균선+2*표준편차
        볼린저밴드 하단선: 20일 이동평균선-2*표준편차
        이 기본 식이다.
        하단선에서 매수, 상한선에서 매도 하는 방식이다.
        단점은 단타 투자에 적합한 방식으로 기간이 길면 손해를 보기 쉬운 전략이다.
        
        ''')
        count = 3   

    if st.checkbox('캘트너 채널(Keltner) 전략'):
        st.write('''

    ##### 4) 캘트너 채널(Keltner) 전략
         
        볼린저 밴드의 후에 나온 투자 방식으로 기존의 볼린저 밴드에서 중단선이 추가된 형태로
        캘트너 채널 상단선: 20일 이동평균선+2*표준편차
        캘트너 채널 중단선: 20일 이동평균선
        캘트너 채널 하단선: 20일 이동평균선-2*표준편차
        볼린저 밴드보다 시장 변동성에 대한 빠른 반응이 가능한 장점이 있다.
        단점은 역시 단타 투자에 적합한 방식으로 기간이 길면 손해를 보기 쉬운 전략이다.
        
        ''')
        count = 4        

    st.subheader('[4] 모의 투자 결과 확인')

    if st.button('투자 시작하기'):

        # 세레브로 가져오기

        cerebro = bt.Cerebro()

        # 금융 데이터 불러오기
        data = bt.feeds.GenericCSVData(dataname="stock_data/df2.csv", dtformat=('%Y-%m-%d'))

        # 데이터 추가하기
        cerebro.adddata(data)

        # 전략 추가하기

        if count == 1:
            cerebro.addstrategy(tt.SmaCross)
            st.write("### 단순 이동 평균(SMA) 크로스 전략 실행")

        if count == 2:
            cerebro.addstrategy(tt.RSI)
            st.write("### 상대적 강도 지수(RSI) 전략 실행")

        if count == 3:
            cerebro.addstrategy(tt.BollingerBand)
            st.write("### 볼린저 밴드(Bollinger_Band) 전략 실행")

        if count == 4:
            cerebro.addstrategy(tt.Keltner)
            st.write("### 캘트너 채널(Keltner) 전략 전략 실행")    

        # 브로거 설정(초기금액)
        cerebro.broker.setcash(start_money)

        # 매매 단위 설정하기

        cerebro.addsizer(bt.sizers.SizerFix, stake=trading_count) 

        # 초기 투자금 가져오기

        init_cash = cerebro.broker.getvalue()

        # 세레브로 실행하기

        cerebro.run()

        # 최종 금액 가져오기

        final_cash = int(cerebro.broker.getvalue())

        
        cerebro.plot()[0][0].savefig('Img\plot.png', dpi=100)
        display(Image(filename='Img\plot.png'))
        oi.open_img('Img\plot.png')

        st.write("### 초기 투자금 : ", init_cash, "원")
        st.write("### 결과 총액: ", final_cash, "원")
        st.write("### 수익금 : ", final_cash - init_cash, "원")
        st.write("### 수익률 : ", round(float(final_cash - init_cash)/float(init_cash) * 100.,3),"퍼센트")



