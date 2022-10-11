import streamlit as st
import backtrader as bt

# 1. 단순 이동 평균(SMA) 크로스 전략

class SmaCross(bt.Strategy):
    # list of parameters which are configurable for the strategy
    params = dict(
        pfast=10,  # period for the fast moving average
        pslow=30   # period for the slow moving average
    )

    def __init__(self):
        sma1 = bt.ind.SMA(period=self.p.pfast)  # fast moving average
        sma2 = bt.ind.SMA(period=self.p.pslow)  # slow moving average
        self.crossover = bt.ind.CrossOver(sma1, sma2)  # crossover signal

    def next(self):
        if not self.position:  # not in the market
            if self.crossover > 0:  # if fast crosses slow to the upside
                self.buy()  # enter long

        elif self.crossover < 0:  # in the market & cross to the downside
            self.close()  # close long position

# 2. 상대적 강도 지수(RSI) 전략 

class RSI(bt.Strategy):

    def __init__(self):
        self.rsi = bt.indicators.RSI(self.data.close)

    def next(self):
        if not self.position: # 아직 주식을 사지 않았다면
            if self.rsi < 30:
                self.order = self.buy()

        elif self.rsi > 70:
            self.order = self.sell()

#1.3 볼린저 밴드 단타 전략
# 볼린저밴드는 20일 이동편균선을 기준으로 주가가 어느 위치에 있는지 알려줌
# 볼린저밴드 상단선: 20일 이동평균선+2*표준편차
# 볼린저밴드 하단선: 20일 이동평균선-2*표준편차
# 전략 1 볼린저 밴드 하단선에서 매수하고 볼린저 밴드 중간선에서 매도 

class BollingerBand(bt.Strategy): 
    params = (
        ("period", 20),
        ("devfactor", 2),
        ("debug", False)
    )

    def __init__(self):
        self.boll = bt.indicators.BollingerBands(period=self.p.period, devfactor=self.p.devfactor, plot=True)
    
    def next(self):
        if not self.position: # not in the market
            if self.data.low[0] < self.boll.lines.bot[0]:
                self.order = self.buy() #매수
        else:
            if self.data.high[0] > self.boll.lines.mid[0]:
                self.order = self.sell() #매도

class Keltner(bt.Strategy): 
    params = (
        ("period", 20),
        ("devfactor", 2),
        ("debug", False)
    )

    def __init__(self):
        self.boll = bt.indicators.BollingerBands(period=self.p.period, devfactor=self.p.devfactor, plot=True)
    
    def next(self):
        if not self.position: # not in the market
            if self.data.low[0] < self.boll.lines.bot[0] and self.data.low[1]>self.data.low[0]:
                self.order = self.buy() #매수
        else:
            if self.data.high[0] > self.boll.lines.mid[0] and self.data.low[1]<self.data.low[0]:
                self.order = self.sell() #매도                