# -*- coding: utf-8 -*-
"""
Created on Sun Apr 10 11:47:56 2022

@author: user
"""

import talib
import pandas as pd
import yfinance as yf
pd.set_option('display.float_format',lambda x : '%.2f' % x)

origin_cash = 10000000

def get_yf(stock_id, period="12mo"):
    stock_list = []
    stock_id = str(stock_id)
    period = str(period)
    print("執行 yf")
    
    stock_id = str(stock_id) + ".TW"
    data = yf.Ticker(stock_id)
    ohlc = data.history(period=period)
    ohlc = ohlc.loc[:, ["Open", "High", "Low", "Close", "Volume"]]
    #ohlc.index = ohlc.index.tz_localize(None)
    #ohlc.to_excel("stock_Kbar.xlsx")
    #ohlc.to_csv("stock_Kbar.csv")
    #print(ohlc)
    print("完成 yf")
    #{date: '03/01', open: '100.00', high: '100.87', low: '88.25', close: '93.32'}
    for i,row in ohlc.iterrows():
        date = f'{i}'.split('-')
        date = date[0]+'/'+date[1]+'/'+date[2][0:2]

        a ={'date': date, 'open': row['Open'], 'high': row['High'], 'low': row['Low'], 'close': row['Close'],'volume':row['Volume']}
        stock_list.append(a)
   

    return stock_list
# 取得技術分析(Technical Analysis)指標
def get_TA(stock_id, period = "12mo"):  #period = "12mo" 有修改
    # 取得資料
    df = pd.DataFrame(get_yf(stock_id,period))

    # 配合程式更改欄位名稱
    print(df)
    df.rename(columns = {"open" : "開盤價", 
               "high" : "最高價",
               "low" : "最低價",
               "close" : "收盤價",
               "volume" : "交易量"}
              , inplace = True)
    
    df["Buy"] = None
    df["Sell"] = None
    
    df["RSI_5"] = talib.RSI(df["收盤價"], 5) # 計算RSI
    df["RSI_5_T1"] = df["RSI_5"].shift(1) # 前一日的RSI
    df["RSI_5_T2"] = df["RSI_5"].shift(2) # 前兩日的RSI
    
    df["SMA_20"] = talib.SMA(df["收盤價"], 20) # 計算 SMA20
    df["SMA_20_T1"] = df["SMA_20"].shift(1) # 前一日的 SMA20
    
    # 設定 MACD 的參數
    fastperiod = 12
    slowperiod = 26
    signalperiod = 9
    
    # 參考：https://fxlittw.com/what-is-macd-indicator/
    # macd = EMA_12 - EMA_26，即DIF值，也稱為快線
    # signal = DIF值取EMA_9，也稱為慢線
    # hist = 柱狀圖，快線 - 慢線
    df["macd"], df["signal"], df["hist"] = talib.MACD(df["收盤價"], 
                                    fastperiod=fastperiod, 
                                    slowperiod=slowperiod, 
                                    signalperiod=signalperiod)
    # MACD為慢線
    #df = df.dropna() # 去除空值
   
    return df

def trade(df):
    
    df["Buy"] = None
    df["Sell"] = None
    df["Num"] = None
    fee = 0.001425 # 手續費
    tax = 0.003 # 稅金
    global cash 
    cash = origin_cash # 原始現金
    
    last_index = df.index[-1]
    hold = 0 # 是否持有
    hold_num = 0 # 持有數量
    
    for index, row in df.copy().iterrows():
        # 最後一天不交易，並將部位平倉
        if index == last_index: 
            if hold == 1: # 若持有部位，平倉
                price = row["收盤價"]
                df.at[index, "Sell"] = price # 紀錄賣價
                df.at[index, "Num"] = hold_num # 在賣出時記錄數量
                cash += price * hold_num * (1 - fee - tax) * 1000
                hold_num = 0
                hold = 0
            break # 跳出迴圈

        # 買進條件
        # 買進條件1 前一日RSI5<=前二日RSI5 and 前一日RSI5<今日RSI5
        # 買進條件2 今日SMA20>前一日SMA20
        buy_condition_1 = (row["RSI_5_T1"] <= row["RSI_5_T2"]) and (row["RSI_5_T1"] < row["RSI_5"])
        buy_condition_2 = row["SMA_20"] > row["SMA_20_T1"]
        
        # 賣出條件
        # 賣出條件1 前一日SMA5>=前二日SMA5 and 前一日SMA5>今日SMA5
        # 賣出條件2 今日DIF<0 and今日MACD<0 
        sell_condition_1 = (row["RSI_5_T1"] >= row["RSI_5_T2"]) and (row["RSI_5_T1"] > row["RSI_5"])
        sell_condition_2 = row["macd"] < 0 and row["signal"] < 0
        
        # 符合兩個買進條件，沒有持有股票，符合以上條件買入
        if (buy_condition_1 and buy_condition_2) and hold == 0:
            price = row["收盤價"]
            df.at[index, "Buy"] = price # 記錄買價
            hold_num = cash / price * (1 + fee)
            hold_num = hold_num // 1000
            
            cash -= price * hold_num * 1000 * (1 + fee)
            hold = 1
        
        # 符合兩個賣出條件，有持有股票，符合以上條件賣出
        elif (sell_condition_1 and sell_condition_2) and hold == 1:
            price = row["收盤價"]
            df.at[index, "Sell"] = price # 紀錄賣價
            df.at[index, "Num"] = hold_num # 在賣出時記錄數量
            cash += price * hold_num * (1 - fee - tax) * 1000
            hold_num = 0
            hold = 0

    return df

def main(stock_id, period = "12mo"):
    df = get_TA(stock_id, period) # 取得資料
    df = trade(df) # 交易
    return df
    print(df)

if __name__ == "__main__":
    df = main(2330, "12mo")
    
    print(df)
    #print(KPI_df)
    

