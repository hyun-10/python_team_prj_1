import streamlit as st

def desc():

	st.write(

#### 1. FinanceDataReader 소개

    pandas-datareader의 단점들을 보완하기 위한 목적으로 만들어진 금융 데이터 수집
	라이브러리이다.
	
#### 2. FinanceDataReader 설치 방법

    설치 : !pip install finance-datareader
	
#### 3. FinanceDataReader 장/단점

######
###### 1) 장점

    단순히 주식 가격 데이터만을 가져오는 것이 아니라 주가지수, 환율, 채권, 암호화폐 등의
	다양한 금융 데이터를 받을 수 있고 데이터가 국내에 한정되어 있지 않다.

######
###### 2) 단점

    fdr 함수의 호출 횟수가 시간 복잡도를 많이 잡아 먹는다.
	여러 개의 함수를 동시에 호출하면 시간이 오래걸린다.

#### 4. FinanceDataReader 실행 및 기능

######
###### 1) StockListing : 주식 종목 코드 조회

    거래소마다 해당하는 주식 종목의 코드와 각종 정보를 가져오는 함수이다.

###### 2) EtfListing : ETF 목록 조회

    StockListing 함수와 비슷한 기능을 하면서 국가별 ETF 리스트를 조회하는 함수이다.

###### 3) DataListing : 주식, 주가지수, 환율, 채권 등 각종 가격 데이터 조회

    수정주가(Adjusted Price)를 사용하는 각종 가격 데이터를 불러올 수 있다. 

###### 4) 기능

    https://github.com/FinanceData/FinanceDataReader 에서 확인 할 수 있다.


	)

	
