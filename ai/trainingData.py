# -*- coding: utf-8 -*-
"""
Created on Fri Jun  9 10:54:48 2023

@author: ozasd
"""
import random
import numpy as np
import pandas as pd
import tensorflow as tf
import yfinance as yf
from datetime import datetime, timedelta
from finta import TA
from sklearn.preprocessing import MinMaxScaler
import pickle
import warnings
from tqdm import tqdm
from keras.utils.np_utils import to_categorical
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Dropout, Lambda, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPool2D, AvgPool2D
from tensorflow.keras.optimizers import Adam 
from keras.callbacks import ReduceLROnPlateau , EarlyStopping
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from sklearn.metrics import classification_report 
from keras.models import load_model
import datetime as dt
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

def val_acc(yval,y_pred):
    t = []
    for i,j in zip(yval,y_pred):
        t.append(np.allclose(i,j))
    return np.mean(t)

def triple_barrier_signal(price,ub,lb,t):
  '''
  triple-barrier包含2個horizontal barrier，1個vertical barrier。
  首先解釋一下這3个barrier：
  根據3个barrier中第一個被touch的進行label
  barrier 1 (the upper barrier)首先達到，label 1
  barrier 2（the lower barrier)首先達到，label -1
  如果barrier 1和barrier 2都没有達到，则barrier 3達到，label 0
  '''
  signal = []
  for i in range(len(price)-t):
    # 情況1.如果price[i:i+t+1]這段序列有任何元素的值大於price[i]*ub則signal[i] = 1
    if max(price[i:i+t+1]) > price[i] * ub:
      signal.append(1)
    # 情況2.如果price[i:i+t+1]這段序列有任何元素的值低於price[i]*lb則signal[i] = -1
    elif min(price[i:i+t+1]) < price[i] * lb:
      signal.append(-1)
    # 如果以上情況1和情況2都沒有發生則signal[i] = 0
    else:
      signal.append(0)
  return signal


def main(stock_id,period="12mo",漲幅 = 1.02 ,跌幅 = 0.98):
    # 參數設定
    #================================================================================
    print("執行 main 主程式")
    seed = 89
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    y_symbol = stock_id+'.TW' # 股票代碼 ^TWII 台灣加權指數 美股蘋果 AAPL 台股台積電 2330.TW
    漲幅  = 1.03 # 漲3%出場
    跌幅 = 0.98 # 跌2%出場
    持有時間 = 20 #預期要持有多長時間
    period = period
    # 資料下載ˋ
    #================================================================================
    start_date = (dt.datetime.now() - dt.timedelta(days=150)).strftime("%Y-%m-%d")
    end_date = dt.datetime.now().strftime("%Y-%m-%d")
    df = yf.download(y_symbol,period=period)
    
    # triple_barrier_signal
    #================================================================================
    ret = triple_barrier_signal(df.Close.values.tolist(),漲幅,跌幅,持有時間)
    df = df.head(len(ret))
    df['triple_barrier_signal'] = ret
    
    # 建立ohlc
    #================================================================================
    ohlcv = df[['Open','High','Low','Close','Volume']]
    ohlcv.columns = ['open','high','low','close','volume']
    
    # 建立指標
    #================================================================================
    df['RSI'] = TA.RSI(ohlcv)
    df['Williams %R'] = TA.WILLIAMS(ohlcv)
    df['SMA'] = TA.SMA(ohlcv)
    df['EMA'] = TA.EMA(ohlcv)
    df['WMA'] = TA.WMA(ohlcv)
    df['HMA'] = TA.HMA(ohlcv)
    df['TEMA'] = TA.TEMA(ohlcv)
    df['CCI'] = TA.CCI(ohlcv)
    df['CMO'] = TA.CMO(ohlcv)
    df['MACD'] = TA.MACD(ohlcv)['MACD'] - TA.MACD(ohlcv)['SIGNAL']
    df['PPO'] = TA.PPO(ohlcv)['PPO'] - TA.PPO(ohlcv)['SIGNAL']
    df['ROC'] = TA.ROC(ohlcv)
    df['CFI'] = TA.CFI(ohlcv)
    df['DMI'] = TA.DMI(ohlcv)['DI+'] - TA.DMI(ohlcv)['DI-']
    df['SAR'] = TA.SAR(ohlcv)
    df = df.dropna(axis=0)#刪除有缺失的row,會缺失主要因為用時間rolling計算技術指標導致,正常的
    features = ['RSI','Williams %R','SMA','EMA','WMA','HMA','TEMA','CCI','CMO','MACD','PPO','ROC','CFI','DMI','SAR']
    y_name = 'triple_barrier_signal' #當作labels
    df = df[features+[y_name]]
    min_max_scaler = MinMaxScaler()
    df_minmax = df.copy()
    df_minmax[features] = min_max_scaler.fit_transform(df_minmax[features])#縮放到0-1之間
    with open(f'StockModel/{stock_id}.pkl', 'wb') as f:
        pickle.dump(min_max_scaler, f)
        
    # x,y pair
    #================================================================================
    #定義觀察天數,起始index(0),結束index(資料筆數-觀察天數)
    days = 15
    start_index = 0
    end_index = len(df)-days
    
    #特徵欄位
    features = df.drop(y_name,axis=1).columns.tolist()
    
    #待存放序列
    Xs = []
    ys = []
    indexs = []
    
    '''
    若資料筆數100,days=15天,f_index=85,i只會跑到84,i+days=99,features只會跑到98天.
    若資料筆數100,days=15天,f_index=85+1,i會跑到85,i+days=100,features會跑到99天.
    '''
    for i in tqdm(range(start_index ,end_index+1 ,1)):#每次i都會遞增1
       X = df.iloc[i:i+days,:][features] #ex:若i為0,則i+days為15因此數據index為0...14(不含15)之features
       y = df.iloc[i+days-1:i+days,:][y_name]#ex:若i為0days為15則[i+days-1:i+days]為[14:15]相當於index[14]之y_name('triple_barrier_signal')
       X = np.array(X) # 轉成np_array
       Xs.append(X) #加入至list
       ys.append(y) #加入至list
       indexs.append((df.iloc[[i]].index,df.iloc[[i+days-1]].index)) #加入資料日期
    #轉換成np_array
    Xs = np.array(Xs)
    ys = np.array(ys)
    print('準備完成')
    print('資料筆數:',len(Xs))
    print('第一筆的index開始和結束:{}-{}'.format(indexs[0][0].date[0],indexs[0][1].date[0]))
    print('最後一筆的index開始和結束:{}-{}'.format(indexs[-1][0].date[0],indexs[-1][1].date[0]))
    Xs = Xs.reshape(-1,days,len(features),1)
    ys = to_categorical(ys, num_classes = 3)

   
    # 測試資料切割
    #================================================================================
    X_test = Xs[-40:] #最後40天features當作test資料
    X_train,y_train = Xs[:-40],ys[:-40] #其他當作訓練
    
    
    # 調整採樣
    #================================================================================
    X_train2維 = X_train.reshape(X_train.shape[0],-1)
    y_train數字 = np.array([ np.argmax(i) for i in y_train])
    ros = RandomUnderSampler()
    X_train平衡 ,y_train平衡 = ros.fit_resample(X_train2維 ,y_train數字)    
    X_train = X_train平衡.reshape(X_train平衡.shape[0],15,15,1)#資料數,15,15,1
    y_train = y_train平衡.reshape(y_train平衡.shape[0],1)#資料數,1
    y_train = to_categorical(y_train , num_classes = 3)#one_hot

    # Split training and val sets
    #================================================================================
    xtrain, xval, ytrain, yval = train_test_split(
                                                    X_train,
                                                    y_train, 
                                                    test_size = 0.2,
                                                    random_state = 42,
                                                    shuffle = True,#是否洗牌
                                                    stratify = y_train #是否根據y欄位做分層取樣
                                                    )
    # 訓練模型
    #================================================================================
    # 創建一個序列模型
    model = Sequential()
    
    # 添加第一個卷積層，使用32個3x3的卷積核，使用ReLU激活函數，並指定輸入形狀
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(15, 15, 1)))
    
    # 添加第二個卷積層，使用64個3x3的卷積核，使用ReLU激活函數
    model.add(Conv2D(64, (3, 3), activation='relu'))
    
    # 添加第三個卷積層，使用128個3x3的卷積核，使用ReLU激活函數
    model.add(Conv2D(128, (3, 3), activation='relu'))
    
    # 添加平坦層，將卷積層的輸出展開為一維數組
    model.add(Flatten())
    
    # 添加dropout層，防止過度擬合
    model.add(Dropout(0.5))
    
    # 添加全連接層，使用softmax激活函數，輸出3個類別
    model.add(Dense(3, activation='softmax'))
    
    # 編譯模型，使用交叉熵損失函數和Adam優化器
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    # 打印模型結構
    model.summary()
    
    # 定義訓練過程的早停法機制
    #================================================================================
    es = EarlyStopping(monitor='val_accuracy',#驗證集acc
                   min_delta=0.0001, 
                   patience=20,  
                   mode='max',
                   restore_best_weights=True)#恢復最佳權重
    
    history = model.fit(
    xtrain,
    ytrain,
    batch_size = 128,
    epochs = 200,
    validation_data=(xval,yval),
    verbose=1,callbacks=[es]
    )
          
    #預測的y
    ypred_onehot = model.predict(xval)
    #轉換預測的y: [0 0 1 0 0 ...] --> 2
    ypred = np.argmax(ypred_onehot,axis=1)
    #轉換真實的y
    ytrue = np.argmax(yval,axis=1)
    #計算 confusion matrix
    

    y_pred = model.predict(xval)
    y_pred = [np.argmax(i) for i in y_pred]
    y_pred = to_categorical(y_pred, num_classes = 3)
    target_names = ['Hold','BUY','SELL']
    print(classification_report(yval,y_pred,target_names=target_names))
    ac = val_acc(yval,y_pred)
    print("模型精準度",ac)
    answer = model.predict(X_test)
    answer = [ np.argmax(i) for i in answer]
    print(len(answer))
    C = pd.DataFrame()
    C['Close'] = yf.download(y_symbol,period=period)['Close']
    C['SIGNAL'] = 0
    C = C.tail(len(answer))
    C['SIGNAL'] = answer 
    buy = C[C['SIGNAL']==1]['Close']
    sell = C[C['SIGNAL']==2]['Close']
    C['Close'].plot()
    #print(C)
    plt.scatter(list(buy.index),list(buy.values),color='red',marker="^")
    plt.scatter(list(sell.index),list(sell.values),color='green',marker="^")       
    pd.set_option('display.float_format', '{:.4f}'.format)
    np.set_printoptions(suppress=True)
    C['HOLD%'] = model.predict(X_test)[:,0]
    C['BUY%'] = model.predict(X_test)[:,1]
    C['SELL%'] = model.predict(X_test)[:,2]
    C.drop(['SIGNAL'],axis=1).tail(20)
    model.save(f'StockModel/{stock_id}.h5')
    del model
    from keras.models import load_model
    model = load_model(f'StockModel/{stock_id}.h5')
    predict = model.predict(X_test)
    predict[-5:]
    
    C.to_csv(f"StockModel/{stock_id}.csv",encoding='utf-8',index=False)

    return ac




if __name__ == "__main__":
    df = main('2443',period="120mo")