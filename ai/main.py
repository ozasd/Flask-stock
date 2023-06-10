#此份檔案程式碼主要架構參考自chatgpt:https://sharegpt.com/c/kZOYTX1
'''
幫我寫一個程式功能如下:
有一個函數叫send_to_telegram(message)用來發送message
有一個函數叫process(df,min_max_scaler,model)用來根據輸入的參數產生交易信號
有一個函數叫main()此做幾件事情:
1.載入model.h5模型
2.載入scaler.pkl
3.使用yfinance取得最近180天^TWII資料稱為df
4.套用process(df,min_max_scaler,model)取得signal
5.當signal == 0,send_to_telegram('不動作')
6.當signal == 1,send_to_telegram('買進')
7.當signal == 2,send_to_telegram('賣出')

使用schedule套件讓這個程式每天早上8點執行一次
'''
import yfinance as yf
from finta import TA
from keras.models import load_model
import pickle
import datetime as dt
import requests
import schedule
import time
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

original_cash = 0

# 發電報函數
def send_to_telegram(message):
    apiToken = '5850662274:AAGeKZqM1JfQfh3CrSKG6BZ9pEvDajdBUqs'
    chatID = '1567262377'
    apiURL = f'https://api.telegram.org/bot{apiToken}/sendMessage'
    try:
        response = requests.post(apiURL, json={'chat_id': chatID, 'text': message})
        print(response.text)
    except Exception as e:
        print(e)

# 計算金融技術指標函數
def calculate_ta(df):
    ta_functions = [TA.RSI, TA.WILLIAMS, TA.SMA, TA.EMA, TA.WMA, TA.HMA, TA.TEMA, TA.CCI, TA.CMO, TA.MACD, TA.PPO, TA.ROC, TA.CFI, TA.DMI, TA.SAR]
    ta_names = ['RSI', 'Williams %R', 'SMA', 'EMA', 'WMA', 'HMA', 'TEMA', 'CCI', 'CMO', 'MACD', 'PPO', 'ROC', 'CFI', 'DMI', 'SAR']
    for i, ta_func in enumerate(ta_functions):
        try:
            df[ta_names[i]] = ta_func(df)
        except:
            if ta_names[i] == 'MACD':
                df[ta_names[i]] = ta_func(df)['MACD']-ta_func(df)['SIGNAL']
            if ta_names[i] == 'PPO':
                df[ta_names[i]] = ta_func(df)['PPO']-ta_func(df)['SIGNAL']  
            if ta_names[i] == 'DMI':
                df[ta_names[i]] = ta_func(df)['DI+']-ta_func(df)['DI-']
    #print(df)
    return df

# 給定歷史數據,scaler,model,輸出買賣訊號的函數
def process(df,min_max_scaler,model):
    df = calculate_ta(df)
    df = df.dropna(axis=0)
    features = ['RSI', 'Williams %R', 'SMA', 'EMA', 'WMA', 'HMA', 'TEMA', 'CCI', 'CMO', 'MACD', 'PPO', 'ROC', 'CFI', 'DMI', 'SAR']
    Close = df[['Close']]
    df = df[features]
    # 數據轉換
    df[features] = min_max_scaler.transform(df[features])
    # 製作X(15x15's array)
    days = 15
    start_index = 0
    end_index = len(df)-days
    Xs = []
    indexs = []
    for i in tqdm(range(start_index ,end_index+1 ,1)):
        X = df.iloc[i:i+days,:][features]
        X = np.array(X)
        Xs.append(X)
        indexs.append((df.iloc[[i]].index,df.iloc[[i+days-1]].index))
    Xs = np.array(Xs)
    # 模型預測買賣訊號
    singal = model.predict(Xs)
    singal = [ np.argmax(i) for i in singal]
    # 繪圖
    Close = Close.iloc[-len(Xs):,:]
    Close['SIGNAL'] = singal
    
    #print("------------------------------------------")
    #print(Close)
    buy = Close[Close['SIGNAL']==1]['Close']
    sell = Close[Close['SIGNAL']==2]['Close']
    
    Close['Close'].plot()
    plt.scatter(list(buy.index),list(buy.values),color='red',marker="^")
    #plt.show()
    
    Close['Close'].plot()
    plt.scatter(list(sell.index),list(sell.values),color='green',marker='v')
    plt.show()

    return singal
def trade(trade_df):
    trade_df["Buy"] = None
    trade_df["Sell"] = None
    trade_df["Num"] = None
    fee = 0.001425
    tax = 0.003    
    hold = 0
    hold_num = 0
    
    last_index = trade_df.index[-1]
    
    global cash
    cash = original_cash
    for index, row in trade_df.copy().iterrows():
        if index == last_index: 
            if hold == 1:
                price = row["Close"]
                trade_df.at[index, "Sell"] = price
                trade_df.at[index, "Num"] = hold_num
                cash += price * hold_num * (1 - fee - tax) * 1000
                hold_num = 0
                hold = 0
            break
        if (row["signal"] == 1) and hold == 0:
                price = row["Close"]
                trade_df.at[index, "Buy"] = price
                hold_num = cash / price * (1 + fee)
                hold_num = hold_num // 1000
                cash -= price * hold_num * 1000 * (1 + fee)
                hold = 1
        elif (row["signal"] == 2) and hold == 1:
                price = row["Close"]
                trade_df.at[index, "Sell"] = price
                trade_df.at[index, "Num"] = hold_num
                cash += price * hold_num * (1 - fee - tax) * 1000
                hold_num = 0
                hold = 0
    #print(trade_df)
    return trade_df

def get_KPI(df):
    # 將買賣價格配對
    global original_cash 

    #print(df)
    record_df = pd.DataFrame()
    record_df["Buy"] = df["Buy"].dropna().to_list()
    record_df["Sell"] = df["Sell"].dropna().to_list()
    record_df["Num"] = df["Num"].dropna().to_list() # 持有數量
    record_df["Buy_fee"] = record_df["Buy"] * 0.001425 * record_df["Num"] * 1000
    record_df["Sell_fee"] = record_df["Sell"] * 0.001425 * record_df["Num"] * 1000
    record_df["Sell_tax"] = record_df["Sell"] * 0.003 * record_df["Num"] * 1000
    record_df.to_excel("record_df.xlsx")
    
    # 交易次數
    trade_time = record_df.shape[0] 

    # 總報酬
    record_df["profit"] = (record_df["Sell"] - record_df["Buy"]) * record_df["Num"] * 1000  - record_df["Buy_fee"] - record_df["Sell_fee"] - record_df["Sell_tax"]
    total_profit = record_df["profit"].sum()
    
    # 累計報酬率，若沒有賠錢，分母為0，會出現ZeroDivisionError，則顯示空值
    #print(total_profit , original_cash)
    acc_ROI = total_profit / original_cash
    
    # 成敗次數
    win_times = (record_df["profit"] >= 0).sum()
    loss_times = (record_df["profit"] < 0).sum()

    # 勝率
    if trade_time > 0:
        win_rate = win_times / trade_time
    else:
        win_rate = 0
    
    # 獲利/虧損金額
    win_profit = record_df[ record_df["profit"] >= 0 ]["profit"].sum()
    loss_profit = record_df[ record_df["profit"] < 0 ]["profit"].sum()
    
    # 獲利因子
    profit_factor =abs(win_profit / loss_profit)
    
    # 平均獲利金額
    if win_times > 0:
        avg_win_profit = win_profit / win_times
    else:
        avg_win_profit = 0

    # 平均虧損金額
    if loss_times > 0:
        avg_loss_profit = loss_profit / loss_times
    else:
        avg_loss_profit = 0
    
    # 賺賠比
    if loss_times > 0:
        profit_rate = abs(avg_win_profit / avg_loss_profit)
    else:
        profit_rate = 0
    
    import math

    
    max_profit = record_df["profit"].max()
    
    if max_profit < 0 or math.isnan(max_profit):
        max_profit = 0
        
        

    
    max_loss = record_df["profit"].min()
    if  max_loss > 0 or math.isnan(max_loss):
        max_loss = 0 

    # 最大回落MDD
    record_df["acu_profit"] = record_df["profit"].cumsum() # 累積報酬
    MDD = 0
    peak = 0
    for i in record_df["acu_profit"]:
        if i > peak:
            peak = i
        diff = peak - i
        if diff > MDD:
            MDD = diff
    KPI_df = pd.DataFrame()
    KPI_df.at["交易次數", "數值"] = trade_time
    KPI_df.at["累計報酬率", "數值"] = acc_ROI
    KPI_df.at["成功次數", "數值"] = win_times
    KPI_df.at["虧損次數", "數值"] = loss_times
    KPI_df.at["勝率", "數值"] = win_rate
    KPI_df.at["獲利總金額", "數值"] = win_profit
    KPI_df.at["虧損總金額", "數值"] = loss_profit
    KPI_df.at["獲利因子", "數值"] = profit_factor
    KPI_df.at["平均獲利金額", "數值"] = avg_win_profit
    KPI_df.at["平均虧損金額", "數值"] = avg_loss_profit
    KPI_df.at["賺賠比", "數值"] = profit_rate
    KPI_df.at["最大單筆獲利", "數值"] = max_profit
    KPI_df.at["最大單筆虧損", "數值"] = max_loss
    KPI_df.at["MDD", "數值"] = MDD
    KPI_df.to_csv("KPI.csv")
   
    kpi = {
        "交易次數":str(round(trade_time,2)),
        "累計報酬率":str(round(acc_ROI,2)),
        "成功次數":str(round(win_times,2)),
        "虧損次數":str(round(loss_times,2)),
        "勝率":str(round(win_rate,2)),
        "獲利總金額":str(round(win_profit,2)),
        "虧損總金額":str(round(loss_profit,2)),
        "獲利因子":str(round(profit_factor,2)),
        "平均獲利金額":str(round(avg_win_profit,2)),
        "平均虧損金額":str(round(avg_loss_profit,2)),
        "賺賠比":str(round(profit_rate,2)),
        "最大單筆獲利":str(round(max_profit,2)),
        "最大單筆虧損":str(round(max_loss,2)),
        "MDD":str(round(MDD,2))
      
        
        
        
        
        }

    print(kpi)

    return kpi
    
def predict(stock_id = '2204',cash = 10000000):
    # 載入模型和scaler
    
    symbol = stock_id + '.TW'

    model = load_model(f'StockModel/{stock_id}.h5')
    with open(f'StockModel/{stock_id}.pkl', 'rb') as f:
        min_max_scaler = pickle.load(f)
    # 資料時間範圍設定
   
    start_date = (dt.datetime.now() - dt.timedelta(days=180)).strftime("%Y-%m-%d")
    end_date = dt.datetime.now().strftime("%Y-%m-%d")
    print(start_date,end_date)

    df = yf.download(symbol, start=start_date, end=end_date)
    # 取得交易訊號
    signal = process(df,min_max_scaler,model)
    df = df[-len(signal):]
    df = df[["Open","High","Low","Close"]]
    df['signal'] = signal
    #df['signal'] = signal
    
    global original_cash
    original_cash = cash

    df = trade(df)
    kpi = get_KPI(df)

    df.to_csv('trade.csv')
    signal = signal[-1] #抓最後一筆的訊號
    if signal == 0:
        action = '不動作'
    if signal == 1:
        action = '建議買入'
    if signal == 2:
        action = '建議賣出'
    # 發送消息至telegram
    current_time = dt.datetime.now().strftime("%Y-%m-%d")
    #message = f"現在時間:{current_time} 投資項目:{symbol} 當前建議操作:{action}"

    message = {"現在時間":current_time,"當前建議操作":action}
    message.update(kpi)

    #send_to_telegram(message)
    print(message)
    return message

if __name__ == '__main__':
    predict(stock_id='2443')
    # 如果你有local python環境或是上雲端python環境可以使用以下代碼讓他定時執行main()函數
    #schedule.every().day.at('08:00').do(main)#每日八點準時執行
    #while True:
    #    schedule.run_pending()
    #    time.sleep(1)
