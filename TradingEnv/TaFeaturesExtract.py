import numpy as np
import ta
import pandas as pd

raw_mapping = {
                "open_time": 0,
                "open":1,
                "high":2,
                "low":3,
                "close":4,
                "volume":5,
                "close_time":6,
                "quote_volume":7,
                "count":8,
                "taker_buy_volume":9,
                "taker_buy_quote_volume":10,
                "ignore":11,
                "rsi": 12,
                "macd": 13,
                "cci": 14, 
                "adx": 15
            }

file_to_open = "15m.csv"
file_to_save = "15m-w-ta-feature.csv"

data = np.loadtxt(file_to_open,
                  delimiter=",", dtype=float)

data_pd = pd.DataFrame(data)
#print(data_pd)

rsi = ta.momentum.RSIIndicator(data_pd[raw_mapping["close"]]).rsi()
data_pd[raw_mapping["rsi"]] = rsi

macd = ta.trend.MACD(data_pd[raw_mapping["close"]]).macd()
data_pd[raw_mapping["macd"]] = macd

cci = ta.trend.cci(data_pd[raw_mapping["high"]], data_pd[raw_mapping["low"]], data_pd[raw_mapping["close"]])
data_pd[raw_mapping["cci"]] = cci

adx = ta.trend.adx(data_pd[raw_mapping["high"]], data_pd[raw_mapping["low"]], data_pd[raw_mapping["close"]])
data_pd[raw_mapping["adx"]] = adx



#Ignore first 100 datas
data_pd.iloc[100:].to_csv(file_to_save, index=False, header=False)
