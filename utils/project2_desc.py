import streamlit as st

def desc():

	st.write(

#### 1. PyKrx 소개

    Naver와 한국 거래소(KRX)에서 유가 증권 데이터를 스크래핑 후 데이터 프레임으로 반환 하는
    오픈 소스로서 현재도 유지/개발 중인 라이브러리이다.
	
#### 2. PyKrx 설치 방법

    설치 : !pip install pykrx
	
#### 3. PyKrx 장/단점

######
###### 1) 장점

    한국 거래소(KRX)에서 데이터를 수집하기 때문에 데이터의 신뢰성이 높다.

######
###### 2) 단점

    데이터의 범주가 국내 주식 데이터에 한정되어 있다.
	PER, PBR, 배당수익률과 같은 지표는 신뢰성이 떨어진다. (실시간 수정을 하지 않기 때문)

#### 4. PyKrx 실행 및 기능

######
###### 1) 주식 API

    한국 거래소(KRX)의 주식 데이터를 다루는 API이다.
    from pykrx import stock로 실행

###### 2) 채권 API

    한국 거래소(KRX)의 채권 데이터를 다루는 API이다.
    from pykrx import bond로 실행

###### 3) 기능

    주식과 채권의 전반적인 데이터를 다룬다.
    https://github.com/sharebook-kr/pykrx 참조
	)

	
