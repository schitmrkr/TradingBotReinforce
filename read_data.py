# open_time,open,high,low,close,volume,close_time,quote_volume,count,taker_buy_volume,taker_buy_quote_volume,ignore

import numpy as np
from numpy import random
import matplotlib.pyplot as plt
import matplotlib

import scipy

from scipy.fftpack import fft
 
# using loadtxt()
data = np.loadtxt("15m.csv",
                 delimiter=",", dtype=float)

mapping = {"open_time": 0,
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
            "ignore":11}

batch_size = 20
block_size = 50

print(data[:block_size,mapping["open"]])
print(data.shape)

#plt.plot(data[:100, mapping["close_time"]], data[:100, mapping["close"]])
#plt.show()

strat_signal = np.zeros((data.shape[0],2),dtype=np.int32)
strat_signal_short = np.zeros((data.shape[0],2),dtype=np.int32)
num_check_block = 20
for i in range(data.shape[0]):
    buy_price = data[i, mapping["close"]]
    stop_loss = 0.995 * buy_price
    take_profit = 1.015 * buy_price
    stop_loss_short = 1.01 * buy_price
    take_profit_short = 0.98 * buy_price
    j = i+1
    k = 1
    if j < data.shape[0]:
        while True:
            if data[j, mapping["low"]] <= stop_loss:
                strat_signal[i, 0] = -1
                strat_signal[i, 1] = k
                break
            if data[j, mapping["high"]] >= take_profit:
                strat_signal[i, 0] = 1
                strat_signal[i, 1] = k
                break

            if data[j, mapping["high"]] >= stop_loss_short:
                strat_signal_short[i, 0] = -1
                strat_signal_short[i, 1] = k
                break
            if data[j, mapping["low"]] >= take_profit_short:
                strat_signal_short[i, 0] = 1
                strat_signal_short[i, 1] = k
                break

            if k > num_check_block:
                strat_signal[i, 0] = 0
                strat_signal_short[i, 0] = 0
                strat_signal[i, 1] = k
                break
            j += 1
            k += 1
            if j+1 > data.shape[0]:
                strat_signal[i, 0] = 0
                strat_signal_short[i, 0] = 0
                break


print(strat_signal.shape)

random_num = random.randint(data.shape[0])
print(strat_signal[random_num:random_num+100, 0])

def get_batch():
    return 0

colors = ['red','green','blue']

timepoints = np.arange(100)

plt.subplot(1,1,1)
plt.plot(timepoints,data[random_num:random_num+100, mapping["close"]], color='blue')
for t, v, l in zip(timepoints, data[random_num:random_num+100, mapping["close"]], strat_signal[random_num:random_num+100, 0]):
    if l == 0:
        plt.scatter(t, v, color='black', marker='o')
    elif l == 1:
        plt.scatter(t, v, color='green', marker='^')
    else:
        plt.scatter(t, v, color='red', marker='x')
plt.xlabel('Time')
plt.ylabel('Value')
plt.title('Time Series with Labels')
plt.legend()
plt.grid(True)
"""
plt.subplot(2,1,2)
plt.plot(timepoints,data[random_num:random_num+100, mapping["close"]], color='blue')
for t, v, l in zip(timepoints, data[random_num:random_num+100, mapping["close"]], strat_signal_short[random_num:random_num+100, 0]):
    if l == 0:
        plt.scatter(t, v, color='black', marker='o')
    elif l == 1:
        plt.scatter(t, v, color='green', marker='^')
    else:
        plt.scatter(t, v, color='red', marker='x')
plt.xlabel('Time')
plt.ylabel('Value')
plt.title('Time Series with Labels')
plt.legend()
plt.grid(True)
"""
plt.show()
"""


#Fourier Analysis
x = data[10000:10000+block_size,mapping["open"]]

y = fft(x)
y_amp = abs(y)
print(y_amp)

#plt.plot(range(block_size-1),y_amp[1:])
#plt.show()

#Percentage change


data_2 = data.copy()
print(data_2.shape)

new_data = (data_2[1:data_2.shape[0], mapping["close"]] - data_2[:data_2.shape[0]-1, mapping["close"]]) / data_2[:data_2.shape[0]-1, mapping["close"]]

print(new_data.shape)


timepoint = 33333
plt.subplot(2,1,1)
plt.axhline(0)
plt.plot(new_data[timepoint:timepoint+block_size])

plt.subplot(2,1,2)
plt.plot(data_2[timepoint:timepoint+block_size, mapping["close"]])

plt.show()
"""