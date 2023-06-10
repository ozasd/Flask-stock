# -*- coding: utf-8 -*-
"""
Created on Fri May 26 14:25:26 2023

@author: ozasd
"""

import yfinance as yf
import pandas as pd

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

        a ={'date': date, 'open': row['Open'], 'high': row['High'], 'low': row['Low'], 'close': row['Close']}
        stock_list.append(a)
   

    return stock_list

if __name__ == "__main__":

    df = pd.DataFrame(get_yf(3008,'12mo'))
    last_column = df.iloc[-1, :]
