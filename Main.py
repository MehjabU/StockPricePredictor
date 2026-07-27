import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import root_mean_squared_error

# using cuda if user has gpu available, else utilize cpu for resources
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# if you want to choose the stock, replace line 17 with user input
ticker = 'MSFT'
df = yf.download(ticker, '2023-01-01')
#print(df) #to visualize table in console

scaler = StandardScaler()
df['Close'] = scaler.fit_transform(df['Close'])

seq_length = 90
data = []

#overlap some to aid next prediction
for i in range(len(df)-seq_length):
    data.append(df.Close[i:i+seq_length])

data = np.array(data)
train_size = int(0.8 * len(data)) #taking 80% of data to use for training

x_train = torch.from_numpy(data[:train_size, :-1, :])
y_train = torch.from_numpy(data[:train_size, -1, :])
x_test = torch.from_numpy(data[train_size:, -1, :])
y_test = torch.from_numpy(data[train_size:, -1, :])

#print(x_train)

#print(df)

#goal at this point is to look at past x days in a given time period to lookup data for prediction tmr

# df.Close.plot()
# plt.ylabel("Close Price")
# #plt.savefig("name of graph") if running code w/o graphical display capabilities
# plt.show()


