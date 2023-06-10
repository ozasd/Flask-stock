# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 23:42:12 2023

@author: ozasd
"""

import yfinance as yf
import pandas as pd
import talib
import numpy as np
#from MA_strategy import *
pd.set_option('mode.chained_assignment', None)

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

def getMA(stock_id, period="12mo",ma=20):
    print("執行Ma",ma)
    df = pd.DataFrame(get_yf(stock_id,period))
    df.rename(columns = {
        "open" : "開盤價", 
        "high" : "最高價",
        "low" : "最低價",
        "close" : "收盤價",
        "volume" : "交易量"}
    , inplace = True)
    df['成交價'] =  df['收盤價'] #成交價用今日收盤
    df["MA5"] = talib.SMA(df["收盤價"],5)
    df[f"MA20"] = talib.SMA(df["收盤價"],ma)
    df["diff"] = df["MA5"]- df["MA20"]
    df = df[ma-1:]
    df["Sign"] = df["diff"].map(lambda x:"買" if x > 0  else "賣")
    df["last_Sign"] = df["Sign"].shift(1)

    print("執行 trade")
    df["Buy"] = None
    df["Sell"] = None
    hold = 0 #是否持有
    last_index = df.index[-1]
    last_data = None

    for i , row in df.iterrows():
        
        if i == last_index and hold == 1:
            # 最後一天清倉
            df["Sell"][i] = df["成交價"][i]
            hold = 0 
            break

        if row["Sign"] == "買" and hold == 0 and row["last_Sign"] == "賣" and i != last_index:
            df["Buy"][i] = df["成交價"][i] 
            hold = 1
        if row["Sign"] == "買" and hold == 0 and row["last_Sign"] == "賣" :
            last_data = [row['date'],row['Sign'],row['成交價']]
        elif row["Sign"] == "賣" and hold == 1:
            df["Sell"][i] = df["成交價"][i]
            hold = 0
        
 
    df.to_excel("trade.xlsx")
    record_df = pd.DataFrame()


    record_df["Buy"] = df["Buy"].dropna().to_list()
    record_df["Sell"] = df["Sell"].dropna().to_list()

    record_df["Buy_fee"] = record_df["Buy"] * 0.001425
    record_df["Sell_fee"] = record_df["Sell"] *0.001425
    record_df["Sell_tax"] = record_df["Sell"] * 0.003

    trade_time = record_df.shape[0]
    #print(f"trade_time:{record_df}")
    
    # 總報酬
    record_df["profit"] = (record_df["Sell"] - record_df["Buy"] -record_df["Buy_fee"] -record_df["Sell_tax"]) * 1000
    total_profit = record_df["profit"].sum()
    #print(f"total_profit:{total_profit}")
    
    win_times = (record_df["profit"] >= 0).sum()
    loss_times = (record_df["profit"] < 0).sum()
    
    
        
    win_profit = record_df[record_df["profit"] >= 0 ]["profit"].sum()
    loss_profit = record_df[record_df["profit"] < 0 ]["profit"].sum()

    
    
    if win_times > 0:
        avg_win_profit = win_profit / win_times
    else:
        avg_win_profit = 0
        
    if loss_times > 0:
        avg_loss_profit = loss_profit / loss_times
    else:
        avg_loss_profit = 0 
        
    if avg_loss_profit != 0:
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
        
    record_df["acu_profit"] = record_df["profit"].cumsum()
    
    MDD = 0
    peak = 0
    
    for i in record_df["acu_profit"]:
        if i > peak:
            peak = i 
        diff = peak - i
        if diff > MDD:
            MDD = diff
            
    sign = None
    if last_data == None:
        for i , row in df.iterrows():
            #print(row)
            if row['Sign'] == '買' and sign == None:
                sign = row['Sign']
                last_data = [row['date'],row['Sign'],row['成交價']]
            else:
                if row['Sign'] != sign:
                    sign = row['Sign']
                    last_data = [row['date'],row['Sign'],row['成交價']]

    #print(last_data)
        

    kpi = {
        "收益":float(round(total_profit,2)),
        "最大單筆收益":float(round(max_profit,2)),
        "最大單筆虧損":float(round(max_loss,2)),
        f"MA20":float(df[f"MA20"][df.index[-1]].round(2)),
        "MA5":float(df[f"MA5"][df.index[-1]].round(2)),
        "交叉點":last_data,
        "平均獲利":float(round(avg_win_profit,2)),
        "平均虧損":float(round(avg_loss_profit,2)),
        "交易次數":int(trade_time),
        "MDD":float(round(MDD,2)),
        "勝率":int(win_times),
        "敗率":int(loss_times)

        


        
        }
    
    print(kpi)


    return kpi

if __name__ == "__main__":

    df = getMA('2542','12mo',20)
    data = {"狀態": "成功", "資料": df}
   