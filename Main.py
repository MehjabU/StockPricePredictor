import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import root_mean_squared_error

from Model import Model
#ignore this line

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

#this is for predicting one day ahead
x_train = torch.from_numpy(data[:train_size, :-1, :]).type(torch.Tensor).to(device)
y_train = torch.from_numpy(data[:train_size, -1, :]).type(torch.Tensor).to(device)
x_test = torch.from_numpy(data[train_size:, :-1, :]).type(torch.Tensor).to(device)
y_test = torch.from_numpy(data[train_size:, -1, :]).type(torch.Tensor).to(device)

#5 days ahead
x5_train = torch.from_numpy(data[:train_size, :-5, :]).type(torch.Tensor).to(device)
y5_train = torch.from_numpy(data[:train_size, -1, :]).type(torch.Tensor).to(device)
x5_test = torch.from_numpy(data[train_size:, :-5, :]).type(torch.Tensor).to(device)
y5_test = torch.from_numpy(data[train_size:, -1, :]).type(torch.Tensor).to(device)

#20 days ahead
x20_train = torch.from_numpy(data[:train_size, :-20, :]).type(torch.Tensor).to(device)
y20_train = torch.from_numpy(data[:train_size, -1, :]).type(torch.Tensor).to(device)
x20_test = torch.from_numpy(data[train_size:, :-20, :]).type(torch.Tensor).to(device)
y20_test = torch.from_numpy(data[train_size:, -1, :]).type(torch.Tensor).to(device)

model1 = Model(1, 32, 1, 2, device).to(device)
model5 = Model(1, 32, 1, 2, device).to(device)
model20 = Model(1, 32, 1, 2, device).to(device)

criterion = nn.MSELoss()
optimizer1 = optim.Adam(model1.parameters(), lr=0.01)
optimizer5 = optim.Adam(model5.parameters(), lr=0.01)
optimizer20 = optim.Adam(model20.parameters(), lr=0.01)


#optimizer = optim.Adam(model.parameters(), lr=0.01)

num_epochs = 20

#visualizing model on training data 1 day ahead
for i in range(num_epochs):
    y_train_prediction = model1(x_train)
    loss = criterion(y_train_prediction, y_train) #comparing prediction to actual

    # if i % 25 == 0:
    #     print(i, loss.item())

    optimizer1.zero_grad()
    loss.backward() #back propogation
    optimizer1.step()

#for 5 days ahead
for j in range(num_epochs):
    y5_train_prediction = model5(x5_train)
    loss = criterion(y5_train_prediction, y5_train) #comparing prediction to actual

    optimizer5.zero_grad()
    loss.backward() #back propogation
    optimizer5.step()

#for 20 days ahead
for k in range(num_epochs):
    y20_train_prediction = model20(x20_train)
    loss = criterion(y20_train_prediction, y20_train) #comparing prediction to actual

    optimizer20.zero_grad()
    loss.backward() #back propogation
    optimizer20.step()

#visualizing model on unseen data
model1.eval()
y_test_prediction = model1(x_test)

y_train_prediction = scaler.inverse_transform(y_train_prediction.detach().cpu().numpy())
y_train = scaler.inverse_transform(y_train.detach().cpu().numpy())
y_test_prediction = scaler.inverse_transform(y_test_prediction.detach().cpu().numpy())
y_test = scaler.inverse_transform(y_test.detach().cpu().numpy())

train_rmse = root_mean_squared_error(y_train[:, 0], y_train_prediction[:, 0])
test_rmse = root_mean_squared_error(y_test[:, 0], y_test_prediction[:, 0])

#unseen data 5 days ahead
model5.eval()
y5_test_prediction = model5(x5_test)
y5_train_prediction = scaler.inverse_transform(y5_train_prediction.detach().cpu().numpy())
y5_train = scaler.inverse_transform(y5_train.detach().cpu().numpy())
y5_test = scaler.inverse_transform(y5_test.detach().cpu().numpy())
y5_test_prediction = scaler.inverse_transform(y5_test_prediction.detach().cpu().numpy())

train5_rmse = root_mean_squared_error(y_train[:, 0], y5_train_prediction[:, 0])
test5_rmse = root_mean_squared_error(y_test[:, 0], y5_test_prediction[:, 0])

#unseen data 20 days ahead
model20.eval()
y20_test_prediction = model20(x20_test)
y20_train_prediction = scaler.inverse_transform(y20_train_prediction.detach().cpu().numpy())
y20_train = scaler.inverse_transform(y20_train.detach().cpu().numpy())
y20_test = scaler.inverse_transform(y20_test.detach().cpu().numpy())
y20_test_prediction = scaler.inverse_transform(y20_test_prediction.detach().cpu().numpy())

train20_rmse = root_mean_squared_error(y_train[:, 0], y20_train_prediction[:, 0])
test20_rmse = root_mean_squared_error(y_test[:, 0], y20_test_prediction[:, 0])

#print(train_rmse, test_rmse)

fig = plt.figure(figsize = (12,12))
gs = fig.add_gridspec(4, 1)
ax1 = fig.add_subplot(gs[:3, 0])
#data from beginning to end
ax1.plot(df.iloc[-len(y_test):].index, y_test, color = 'blue', label = 'Actual Price')
ax1.plot(df.iloc[-len(y_test):].index, y_test_prediction, color = 'green', label = 'Predicted Price One Day Ahead')
ax1.plot(df.iloc[-len(y5_test):].index, y5_test_prediction, color = 'purple', label = 'Predicted Price Five Days Ahead')
ax1.plot(df.iloc[-len(y20_test):].index, y20_test_prediction, color = 'orange', label = 'Predicted Price 20 Days Ahead')
ax1.legend()
plt.title(f"{ticker} Stock Price Prediction")
plt.xlabel("Date")
plt.ylabel("Price")

ax2 = fig.add_subplot(gs[3, 0])
ax2.axhline(test_rmse, color = 'blue', linestyle = '--', label = 'RMSE')
ax2.plot(df[-len(y_test):].index, abs(y_test - y_test_prediction), color = 'red', label = 'Prediction Error')

ax2.axhline(test5_rmse, color = 'blue', linestyle = '-.', label = 'RMSE')
ax2.plot(df[-len(y5_test):].index, abs(y5_test - y5_test_prediction), color = 'pink', label = 'Prediction Error')

ax2.axhline(test20_rmse, color = 'blue', linestyle = ':', label = 'RMSE')
ax2.plot(df[-len(y20_test):].index, abs(y20_test - y20_test_prediction), color = 'black', label = 'Prediction Error')
ax2.legend()
plt.ylabel("Error")
plt.title("Prediction Error")
plt.xlabel("Date")

plt.tight_layout()
plt.show()


