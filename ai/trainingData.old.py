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

warnings.filterwarnings("ignore")


def getData(stock_id,period):
    stock_id = str(stock_id) + ".TW"
    #data = yf.Ticker(stock_id)
    #ohlc = yf.download(stock_id,period=period)
    #ohlc = data.history(period=period)
    start_date = dt.datetime.strptime('2017-12-31', "%Y-%m-%d") 
    end_date = dt.datetime.strptime('2022-12-31', "%Y-%m-%d")

    print(start_date,end_date)
    #start_date = (dt.datetime.now() - dt.timedelta(days=150)).strftime("%Y-%m-%d")
    #end_date = dt.datetime.now().strftime("%Y-%m-%d")

    ohlc = yf.download(stock_id, start=start_date, end=end_date)
    ohlc = ohlc.loc[:, ["Open", "High", "Low", "Close", "Volume"]]
    
    if len(ohlc) > 0 :
        print("完成股票資料抓取 !")
        return ohlc
    else:
        raise KeyError("抓取資料發生錯誤，你可能輸入錯誤的股票代碼 !")
        #print("抓取資料發生錯誤，你可能輸入錯誤的股票代碼 !")
def triple_barrier_signal(df,price,ub,lb,t):
   print("添加漲跌訊號 !")
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
      
   df = df.head(len(signal))
   df['triple_barrier_signal'] = signal
   return df

def Stock_Index(triple_barrier_signal , ohlcv):
    print("計算各項指標 !")
    # 技術指標添加
    #print(df,ohlcv)
    df = pd.DataFrame()
    
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
    df['triple_barrier_signal'] = triple_barrier_signal
    

    return df

    

def features_scaling(df,stock_id):
    print("執行特徵縮放 !")
    df = df.dropna(axis=0)#刪除有缺失的row,會缺失主要因為用時間rolling計算技術指標導致,正常的
    features = ['RSI','Williams %R','SMA','EMA','WMA','HMA','TEMA','CCI','CMO','MACD','PPO','ROC','CFI','DMI','SAR']
    y_name = 'triple_barrier_signal' #當作labels
    df = df[features+[y_name]]
    min_max_scaler = MinMaxScaler()
    df_minmax = df.copy()
    df_minmax[features] = min_max_scaler.fit_transform(df_minmax[features])#縮放到0-1之間
    with open(f'model/{stock_id}.pkl', 'wb') as f:
        pickle.dump(min_max_scaler, f)
    df = df_minmax
    print('特徵縮放完成!')
    return df

def training(stock_id,df,ohlcv,period):
    y_name = 'triple_barrier_signal' #當作labels
    #定義觀察天數,起始index(0),結束index(資料筆數-觀察天數)
    days = 15
    start_index = 0
    end_index = len(df)-days
    #特徵欄位
    features = df.drop(y_name,axis=1).columns.tolist()
    #print(features)
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
    Xs = np.array(Xs)
    ys = np.array(ys)
    print('準備完成')
    print('資料筆數:',len(Xs))
    print('第一筆的index開始和結束:{}-{}'.format(indexs[0][0].date[0],indexs[0][1].date[0]))
    print('最後一筆的index開始和結束:{}-{}'.format(indexs[-1][0].date[0],indexs[-1][1].date[0]))
    Xs = Xs.reshape(-1,days,len(features),1)
    ys = to_categorical(ys, num_classes = 3)
    X_test = Xs[-40:] #最後40天features當作test資料
    X_train,y_train = Xs[:-40],ys[:-40] #其他當作訓練
    X_train2維 = X_train.reshape(X_train.shape[0],-1)
    y_train數字 = np.array([ np.argmax(i) for i in y_train]) 
    from imblearn.under_sampling import RandomUnderSampler
    ros = RandomUnderSampler()
    X_train平衡 ,y_train平衡 = ros.fit_resample(X_train2維 ,y_train數字)  
    X_train = X_train平衡.reshape(X_train平衡.shape[0],15,15,1)#資料數,15,15,1
    y_train = y_train平衡.reshape(y_train平衡.shape[0],1)#資料數,1
    y_train = to_categorical(y_train , num_classes = 3)#one_hot
    try:
        xtrain, xval, ytrain, yval = train_test_split(
                                                        X_train,
                                                        y_train, 
                                                        test_size = 0.2,
                                                        random_state = 42,
                                                        shuffle = True,#是否洗牌
                                                        stratify = y_train #是否根據y欄位做分層取樣
                                                        )
    except Exception:
        raise KeyError("樣本數過少")
        
        
    
    
    
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
    
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
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
    #print("預測的結果",ypred)
    #print("真實的結果",ytrue)
    y_pred = model.predict(xval)
    y_pred = [np.argmax(i) for i in y_pred]
    y_pred = to_categorical(y_pred, num_classes = 3)
    target_names = ['Hold','BUY','SELL']
    print(classification_report(yval,y_pred,target_names=target_names))
    
    t = []
    for i,j in zip(yval,y_pred):
        t.append(np.allclose(i,j))
    print("驗證集精準度",np.mean(t))
    model.save(f'model/{stock_id}.h5', save_format='h5')
    del model
    import pickle
    load_name = "model/"+stock_id+'.h5'
    model = load_model(load_name)
    predict = model.predict(X_test)
    ypred = np.argmax(predict,axis=1)
    #print("預測的結果",ypred)
    table = pd.DataFrame({
        'model':[],
        'accuracy':[]
        })
    table.loc[0, 'model'] = stock_id
    table.loc[0, '訓練資料(月)'] = period
    table.loc[0, 'accuracy'] = np.mean(t)
    #table.to_csv(f"模型評估表/{stock_id}模型評估表.csv",encoding='utf-8',index=False)
    
    start_date = dt.datetime.strptime('2017-12-31', "%Y-%m-%d") 
    end_date = dt.datetime.strptime('2022-12-31', "%Y-%m-%d")
    pd.set_option('display.float_format', '{:.4f}'.format)
    np.set_printoptions(suppress=True)
    answer = model.predict(X_test)
    answer = [ np.argmax(i) for i in answer]
    print(len(answer))
    C = pd.DataFrame()
    data = yf.download(stock_id+".TW", start = start_date,end = end_date)
    print(data)
    #C['Date'] = data.index

    print(data)
    C['Close'] = data['Close']

    C['SIGNAL'] = 0
    C = C.tail(len(answer))
    C['SIGNAL'] = answer 
    buy = C[C['SIGNAL']==1]['Close']
    sell = C[C['SIGNAL']==2]['Close']
    C['HOLD%'] = model.predict(X_test)[:,0]
    C['BUY%'] = model.predict(X_test)[:,1]
    C['SELL%'] = model.predict(X_test)[:,2]
    C.drop(['SIGNAL'],axis=1).tail(20)
    print(C)
    C.to_csv(f"{stock_id}.csv",encoding='utf-8',index=False)

    return np.mean(t)

        
    

    

def main(stock_id,period="12mo"):
    print("執行 main 主程式")
    seed = 89
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    y_symbol = stock_id # 股票代碼 ^TWII 台灣加權指數 美股蘋果 AAPL 台股台積電 2330.TW
    漲幅  = 1.02 # 漲3%出場
    跌幅 = 0.98 # 跌3%出場
    持有時間 = 10 #預期要持有多長時間
    period = period
    df = getData(stock_id,period)
    df = triple_barrier_signal(df,df.Close.values.tolist(),漲幅,跌幅,持有時間)
    ohlcv = df[['Open','High','Low','Close','Volume']]
    ohlcv.columns = ['open','high','low','close','volume']
    df = Stock_Index(df['triple_barrier_signal'],ohlcv)
    df = features_scaling(df,stock_id)
    df = training(stock_id,df,ohlcv,period)
    print("訓練完成 !")
    return df

   
         
        


if __name__ == "__main__":
    df = main('2408',period="120mo")
    