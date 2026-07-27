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

x_train = torch.from_numpy(data[:train_size, :-1, :]).type(torch.Tensor).to(device)
y_train = torch.from_numpy(data[:train_size, -1, :]).type(torch.Tensor).to(device)
x_test = torch.from_numpy(data[train_size:, :-1, :]).type(torch.Tensor).to(device)
y_test = torch.from_numpy(data[train_size:, -1, :]).type(torch.Tensor).to(device)

model = Model(input_dim = 1, hidden_dim = 32, output_dim = 1, num_layers = 2, device=device).to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

num_epochs = 200

#visualizing model on training data

for i in range(num_epochs):
    y_train_prediction = model(x_train)
    loss = criterion(y_train_prediction, y_train) #comparing prediction to actual

    if i % 25 == 0:
        print(i, loss.item())

    optimizer.zero_grad()
    loss.backward() #back propogation
    optimizer.step()


#visualizing model on unseen data
model.eval()
y_test_prediction = model(x_test)

y_train_prediction = scaler.inverse_transform(y_train_prediction.detach().cpu().numpy())
y_train = scaler.inverse_transform(y_train.detach().cpu().numpy())
y_test_prediction = scaler.inverse_transform(y_test_prediction.detach().cpu().numpy())
y_test = scaler.inverse_transform(y_test.detach().cpu().numpy())

train_rmse = root_mean_squared_error(y_train[:, 0], y_train_prediction[:, 0])
test_rmse = root_mean_squared_error(y_test[:, 0], y_test_prediction[:, 0])

print(train_rmse, test_rmse)

fig = plt.figure(figsize = (7,7))
gs = fig.add_gridspec(4, 1)
ax1 = fig.add_subplot(gs[:3, 0])
#data from beginning to end
ax1.plot(df.iloc[-len(y_test):].index, y_test, color = 'blue', label = 'Actual Price')
ax1.plot(df.iloc[-len(y_test):].index, y_test_prediction, color = 'green', label = 'Predicted Price')
ax1.legend()
plt.title(f"{ticker} Stock Price Prediction")
plt.xlabel("Date")
plt.ylabel("Price")

ax2 = fig.add_subplot(gs[3, 0])
ax2.axhline(test_rmse, color = 'blue', linestyle = '--', label = 'RMSE')
ax2.plot(df[-len(y_test):].index, abs(y_test - y_test_prediction), color = 'red', label = 'Prediction Error')
ax2.legend()
plt.ylabel("Error")
plt.title(f"{ticker} Stock Price Prediction")
plt.xlabel("Date")

plt.tight_layout()
plt.show()


