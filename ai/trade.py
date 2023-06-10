# -*- coding: utf-8 -*-

import pandas as pd
from get_KPI import get_KPI

trade_df = pd.read_csv("trade.csv",encoding="utf-8")
trade_df = pd.DataFrame(trade_df)
original_cash = 10000000
pd.set_option('display.float_format',lambda x : '%.2f' % x)
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
    return trade_df

def main(trade_df):
    df = trade(trade_df)
    KPI_df = get_KPI(df,original_cash)
    KPI_df.at["剩餘資金", "數值"] = cash
    return df, KPI_df

if __name__ == "__main__":
    trade_df, KPI_df = main(trade_df)
    trade_df.to_excel("new_trade.xlsx")
    KPI_df.to_excel("new_KPI.xlsx")
    print(trade_df)
    print(KPI_df)