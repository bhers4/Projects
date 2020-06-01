import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam

# Get the data
stock_csv = open("AAPL.csv", "r")
header = stock_csv.readline()
data = {}
for item in header.split(","):
    data[item.strip()] = []
for line in stock_csv:
    line = line.split(",")
    data['Date'].append(line[0])
    data['Open'].append(float(line[1]))
    data['High'].append(float(line[2]))
    data['Low'].append(float(line[3]))
    data['Close'].append(float(line[4]))
    data['Adj Close'].append(float(line[5]))
    data['Volume'].append(int(line[6].strip()))

for item in data.keys():
    print(item, ": ", len(data[item]))


class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_units, num_stacked, output_size):
        super(LSTMClassifier, self).__init__()
        self.input_size = input_size
        self.hidden_units = hidden_units
        self.num_stacked = num_stacked
        self.output_size = output_size
        self.lstm1 = nn.LSTM(input_size, hidden_units, num_stacked, batch_first=True).double()
        self.fcn = nn.Linear(in_features=hidden_units, out_features=output_size)
        return

    def forward(self, x):
        out, (hn, cn) = self.lstm1(x)
        out = self.fcn(out[:, -1, :])
        return out

class StockData(Dataset):
    def __init__(self, data, seq_len, input_size):
        self.data = data
        self.seq_len = seq_len
        self.input_size = input_size

    def __len__(self):
        return len(self.data['Open'])-self.seq_len

    def __getitem__(self, item):
        print("Item: ", item)
        five_data = np.zeros((self.seq_len, self.input_size))
        counter = 0
        for i in range(item, item+self.seq_len):
            open_data = data['Open'][i]
            close_data = data['Close'][i]
            high_data = data['High'][i]
            low_data = data['Low'][i]
            volume_data = data['Volume'][i]
            adjclose_data = data['Adj Close'][i]
            five_data[counter] = np.array([open_data, close_data, high_data, low_data, volume_data, adjclose_data])
            counter += 1
        five_data = torch.tensor(five_data.astype(dtype=np.double))
        return five_data


# Set up network
sequence_len = 5  # Number of days to look at, like predict mondays price given last mon-fri
batch_size = 1
input_size = 6
hidden_units = 10
num_stacked = 1  # Number of Stacked LSTM's so second LSTM takes input from first
output_size = 1

# Model
model = LSTMClassifier(input_size, hidden_units, num_stacked, output_size).double()

# Loss
loss = nn.MSELoss()
# Optimizer
lr = 0.001
optim = Adam(model.parameters(), lr=lr)
# Dataloaders
train_data = StockData(data, sequence_len, input_size)
train_dataset = DataLoader(train_data, batch_size=4, shuffle=False)

for i, data in enumerate(train_dataset, 0):
    print("data: ", data.shape)
    out = model(data.double())
    print("Out: ", out)

