import streamlit as st

def desc():

	st.write('''

#### 1. backtrader 소개

    backtesting 방식으로 한 명의 트레이더 또는 투자자로서 새로운 시장과 전략을 탐색할 때
    트레이딩 전략이 타당한지, 이를 통해 향후 수익을 창출할 수 있는지를 알아보기위해 사용하는
    라이브러리 이다.
	
#### 2. backtrader 설치 방법

    설치 : !pip install backtrader
	
#### 3. backtrader 장/단점

###### 1) 장점

    실제 돈이 들지 않고 가상으로 투자 전략을 실행 할 수 있다.
    자신의 투자 전략의 성능 테스트에 도움이 된다.

###### 2) 단점

    시장 상황과 완벽히 일치할 수 없어 정확한 결과로서는 볼 수 없다.
    편향된 관점을 가지는 경우가 생긴다. (무조건적인 전략의 맹신)

#### 4. backtrader실행 과정
      
    1) 시작일, 종료일을 설정한 주식 데이터 설정
    2) 투자 전략 설정
    3) 초기금 설정
    4) 실행
    5) 결과 산출
    위와 같은 5번의 과정을 거친다.


#### 5. backtrader결과를 통해 알 수 있는 정보

######
###### 1) 최대 수익과 손실

    여러가지 전략을 실행해 가면서 어떤 전략이 최대 수익을 벌고 어떤 전략이 
    손실액이 큰지를 알수 있다.

######
###### 2) 전략의 수익률

    본인이 실행하고자 하는 전략의 수익률을 알 수 있다. 

######
###### 3) 전체 포트폴리오에서 전략을 위해 할당해야하는 자본의 양 (초기금)

    1) 과 2) 에서 알아낸 정보를 통해 이 주식에 얼마를 투자해야 할지를 정할수 있다.


	''')                                   