import streamlit as st
import FinanceDataReader as fdr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime as dt
from datetime import timedelta
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from PIL import Image



# DF생성시 인덱스 설정 위한 date_index 만들기
def date_to_str(start,end) :
    start=start.strftime('%Y-%m-%d')
    end=end.strftime('%Y-%m-%d')
    dates=pd.date_range(start,end)
    return dates

class anl_stock():
    code=''
    df=''
    nasdaq_df=''
    def __init__(self,code,df,nasdaq_df):
        self.code=code
        self.df=df    
        nasdaq_df.rename(columns={'Close':'Nasdaq'},inplace=True)
        nasdaq_df=pd.DataFrame(nasdaq_df.Nasdaq,index=nasdaq_df.index)
        self.nasdaq_df=nasdaq_df
        
    #검색 종목에 대한 (현재+1일 ~ 현재+20일) 예측치 구하기
    def anl_code(self):
        '''
        X_train : 과거 40일치 시계열 데이터
        y_train : 과거 40일치 데이터 직후 과거 20일 시계열 데이터
        X_val : [-60일~-20일]시계열 데이터로 [-19일 ~ 현시점] 백테스팅을 위함
        y_val : [-19일 ~ 현시점] 벡스팅 label 자료
        X_test : [-40일~현시점] 시계열 데이터로 [현시점 ~ 20일]
        '''
        code=self.code
        df=self.df
        #데이터셋 구성
        X_train, y_train, X_test, y_val, X_val, date_idx, scaler = self.ppc_df(df)        
        
        #모델 구성 및 학습된 모델 불러오기(없을경우 신규생성)
        model_name='predst.h5'
        model=self.get_model(model_name,X_train,y_train)                
        
        #백테스팅     
        #y_val=y_val.reshape(1,20)
        #evl=model.evaluate(X_val[-1:],y_val[-1:])        
        #print(evl)
        pred_now=model.predict(X_val[-1:])# -60일~-20일 데이터로 -19일전 ~ 현시점 예측
        t_pred_now=scaler.inverse_transform(pred_now) #정규화 -> 원래값 회복
        
        
        #백테스팅(-20일~현시점) df 구성
        now=date_idx[-1]#현시점 Date
        delta=now-timedelta(19)#현시점-20일 Date
        dates=date_to_str(delta, now)#dateindex 구성
        now_df_anl=pd.DataFrame(t_pred_now[0],index=dates,columns=['price'])#-20일~현재        
        
        #예측        
        pred=model.predict(X_test[-1:])# -40일~현시점 데이터로 현시점+1일 ~ 현시점+20일후 예측
        t_pred=scaler.inverse_transform(pred) #정규화 -> 원래값 회복
                      
        #예측(현시점+1일 ~ 현시점+20일) df 구성
        now2=date_idx[-1]+timedelta(1)#현시점+1일 Date
        delta2=now2+timedelta(19)
        dates2=pd.date_range(now2,delta2) #마지막 종가+1일 ~ 마지막종가++20일
        df_anl=pd.DataFrame(t_pred[0],index=dates2,columns=['price'])#마지막종가 +1일 ~ 20일예측
        
        #(그래프 비교용)-20일 ~ 현시점 ~ +20일 df 구성               
        new_df=pd.concat([now_df_anl,df_anl])
        
        #그래프 그리기(df:과거 Close 전체, new_df:백테스팅값(label) 및 예측값(+20일))        
        self.set_graph(df,new_df,graph_name='Stock_pred.jpg')


    #검색 종목 + 나스닥 지수 요인 포함한 데이터로 학습 후 (현재+1일 ~ 현재+20일) 예측치 구하기    
    def anl_code_nasdaq(self):    
        '''
        X_train : 과거 40일치 시계열 데이터
        y_train : 과거 40일치 데이터 직후 과거 20일 시계열 데이터
        X_val : [-60일~-20일]시계열 데이터로 [-19일 ~ 현시점] 백테스팅을 위함
        y_val : [-19일 ~ 현시점] 벡스팅 label 자료
        X_test : [-40일~현시점] 시계열 데이터로 [현시점 ~ 20일]
        new_df : 해당 code의 df와 nasdaq의 df join 
         * 기준 : code_df.index 우선 / nasdaq_df는 na 발생시 ffill로 대체
        '''
        code=self.code
        df=self.df
        nasdaq_df=self.nasdaq_df
        
        #데이터셋 구성
        new_df, X_train, y_train, X_test, y_val, X_val, date_idx = self.set_train_data(df,nasdaq_df)

        #모델 구성 및 학습된 모델 불러오기(없을경우 신규생성)
        model_name='predst_nasdaq.h5'
        model=self.get_model(model_name,X_train,y_train,nasdaq='ok')  
        
        
        
        '''
        scaler 재생성
        기존 scaler : ( , 2)  2가지 요인으로 fit(code의 Close, Nasdaq의 Close) : 훈련데이터용
        아래 scaler : ( , 1) 1가지 요인으로 fit(code의 Close) predict값 회복용
        '''        
        train_data2=np.array(new_df.Close.values)
        train_data2=train_data2.reshape(-1,1)        
        scaler2=MinMaxScaler()
        scaled_data2=scaler2.fit_transform(train_data2)
        
        #백테스팅   
        #y_val=y_val.reshape(1,20)
        #evl=model.evaluate(X_val[-1:],y_val[-1:])        
        #print(evl)
        pred_now=model.predict(X_val[-1:])# -60일~-20일 데이터로 -19일전 ~ 현시점 예측
        t_pred_now=scaler2.inverse_transform(pred_now) #정규화 -> 원래값 회복
        
        #백테스팅(-20일~현시점) df 구성
        now=date_idx[-1]#현시점 Date
        delta=now-timedelta(19)#현시점-20일 Date
        dates=date_to_str(delta, now)#dateindex 구성
        now_df_anl=pd.DataFrame(t_pred_now[0],index=dates,columns=['price'])#-20일~현재        
        
        #예측 
        pred=model.predict(X_test[-1:])# -40일~현시점 데이터로 현시점+1일 ~ 현시점+20일후 예측
        t_pred=scaler2.inverse_transform(pred)#정규화 -> 원래값 회복
        
        
        #예측(현시점+1일 ~ 현시점+20일) df 구성
        now2=date_idx[-1]+timedelta(1)#현시점+1일 Date
        delta2=now2+timedelta(19)
        dates2=pd.date_range(now2,delta2) #마지막 종가+1일 ~ 마지막종가++20일
        df_anl=pd.DataFrame(t_pred[0],index=dates2,columns=['price'])#마지막종가 +1일 ~ 20일예측
        
        #(그래프 비교용)-20일 ~ 현시점 ~ +20일 df 구성               
        new_df=pd.concat([now_df_anl,df_anl])
        
        #그래프 그리기(df:과거 Close 전체, new_df:백테스팅값(label) 및 예측값(+20일))        
        self.set_graph(df,new_df,graph_name='Stock_pred_Nasdaq.jpg')   





    
    #정규화 및 데이터셋 구성
    def ppc_df(self,df):
        date_idx=df.index
        train_data=np.array(df.values)
        scaler=MinMaxScaler()
        scaled_data=scaler.fit_transform(train_data)
        X_train=[]
        y_train=[]
        date_i=[]
        for i in range(40,len(scaled_data)-20):
            X_train.append(scaled_data[i-40:i])
            date_i.append(date_idx[i-40:i])
            y_train.append(scaled_data[i:i+20])
        X_train, y_train=np.array(X_train), np.array(y_train)
        X_test=np.array(scaled_data[-40:])[np.newaxis,:,:]
        y_val=np.array(scaled_data[-20:])[np.newaxis,:,:]
        X_val=np.array(scaled_data[-60:-20])[np.newaxis,:,:]
        return X_train, y_train, X_test, y_val, X_val, date_idx, scaler
        
    #모델구성 및 훈련 모델 저장
    def set_model(self,X_train, y_train):        
        model=Sequential()
        print('X_train=',X_train.shape)
        print('y_train=',y_train.shape)
        model.add(LSTM(40,activation='relu',input_shape=(X_train.shape[1],1),return_sequences=True))
        model.add(LSTM(40,return_sequences=True))
        model.add(LSTM(40,return_sequences=False))

        model.add(Dense(20))
        earlystop=EarlyStopping(monitor='loss',patience=15)
        model.compile(loss='mse',optimizer='rmsprop',metrics=['mae'])
        model.fit(X_train[:-1],y_train[:-1],epochs=1000,batch_size=10,callbacks=[earlystop])
        model.save('predst.h5')
        return model
    
    #미리 학습된 model 불러오기
    def get_model(self, model_name,X_train,y_train, nasdaq=None):        
        if nasdaq==None:
            try :
                model=load_model(model_name)
            #저장된 모델 없을 경우
            except :
                st.warning('모델이 설정되지 않은 관계로 모델 재설정 중입니다. 기다려주세요')
                model=self.set_model(X_train, y_train)
            return model
        else :
            try :
                model=load_model(model_name)
            #저장된 모델 없을 경우
            except :
                st.warning('모델이 설정되지 않은 관계로 모델 재설정 중입니다. 기다려주세요')
                model=self.set_model_nasdaq(X_train, y_train)
            return model
    
    #그래프 그리기
    def set_graph(self,df,new_df,graph_name):     
        plt.figure(figsize=(16,12))
        plt.plot(df.index,df.values,label='true') # 기존가
        plt.plot(new_df.index,new_df.values,label='pred') #예측가
        plt.xlabel('Date',fontsize=25)
        plt.ylabel('Price',fontsize=25)
        plt.tick_params(length=5, labelsize=7)
        plt.title('Stock price',fontsize=25)
        plt.tight_layout()
        plt.grid()
        plt.legend(loc='upper right')
        plt_name=graph_name
        plt.savefig(plt_name)
        image=Image.open(plt_name)         
        rp=new_df.astype('int')
        now_price=df.iloc[-1].values[0] #마지막 종가측정 가격 확인
        af5_price=rp.iloc[-16].values[0] #마지막 종가일로부터 +5일
        af20_price=rp.iloc[-1].values[0] # 마지막 종가일로부터 20일가격
        now_af20_ir=(af20_price-now_price) #마지막 종가일로부터 20일가격-마지막 종가)
        st.write('''
                 2020년 1월 이후의 데이터로 1일에서 40일까지의 데이터로 , 41일에서 60일까지 데이터를 예측하는 훈련을 진행한 딥러닝 AI프로그램으로써,
                 사용시 주식 매매에 유의하여 주세요
                 ''' )
        st.image(image,caption=self.code)
        st.write('### 현재(마지막 종가발표일자) 기준으로 예측합니다')
        st.write('#### 1) 현재일자 :', df.index[-1].strftime('%Y-%m-%d'))
        st.write('#### 2) 현재종가 :', now_price,'원')
        st.write('#### 3) 5일 후 예상 금액 : ',af5_price)
        st.write('#### 4-1) 20일 후 예상 금액 : ',af20_price)
        st.write('#### 4-2) 20일 후 수익률:',np.round(now_af20_ir/now_price*100,2),'%')
        st.write('#### 4-3) 20일 후 수익금액(1주당):',int(now_af20_ir),'원')
        st.write('### ')

        
    #나스닥 지수포함한 데이터 구성(정규화 포함)
    def set_train_data(self,df,df2):                

        new_df=df.join(df2,how='left')
        new_df.Nasdaq.fillna(method='ffill',inplace=True)
        
        train_data=np.array(new_df.values)

        scaler=MinMaxScaler()
        scaled_data=scaler.fit_transform(train_data)

        date_idx=new_df.index

        X_train=[]
        y_train=[]
        date_i=[]
        scaled_data2=[]

        for i in range(len(scaled_data)):    
            scaled_data2.append(scaled_data[i][0])
        scaled_data2=np.array(scaled_data2)
        for i in range(40,len(scaled_data)-20):
            X_train.append(scaled_data[i-40:i])
            date_i.append(date_idx[i-40:i])
            y_train.append(scaled_data2[i:i+20])

        X_train, y_train=np.array(X_train), np.array(y_train)
        X_test=np.array(scaled_data[-40:])[np.newaxis,:,:]
        y_val=np.array(scaled_data2[-20:])[np.newaxis,:]

        X_val=np.array(scaled_data[-60:-20])[np.newaxis,:,:]

        return new_df,X_train, y_train, X_test, y_val, X_val, date_idx

    #나스닥 지수포함한 모델로 훈련
    def set_model_nasdaq(self,X_train,y_train):
        model=Sequential()

        model.add(LSTM(40,activation='relu',input_shape=(X_train.shape[1],2),return_sequences=True))
        model.add(LSTM(40,return_sequences=True))
        model.add(LSTM(40,return_sequences=False))

        model.add(Dense(20))
        earlystop=EarlyStopping(monitor='loss',patience=15)
        model.compile(loss='mse',optimizer='rmsprop',metrics=['mae'])
        model.fit(X_train[:-1],y_train[:-1],epochs=1000,batch_size=10,callbacks=[earlystop])
        model_nasdaq='predst_nasdaq.h5'
        model.save(model_nasdaq)
        return model
    


