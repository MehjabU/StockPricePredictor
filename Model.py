import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import root_mean_squared_error




#from Main import device


class Model(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, device):
        super(Model, self).__init__()

        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.device = device

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout = 0.15)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(
            self.num_layers, x.size(0), self.hidden_dim,
            device=self.device
        )
        c0 = torch.zeros(
            self.num_layers, x.size(0), self.hidden_dim,
            device=self.device
        )

        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.fc(out[:, -1, :])

        return out