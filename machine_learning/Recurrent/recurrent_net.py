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
    
    def init_hidden(self, batch_size):
        # This method generates the first hidden state of zeros which we'll use in the forward pass
        # We'll send the tensor holding the hidden state to the device we specified earlier as well
        hidden = torch.zeros(self.num_stacked, batch_size, self.hidden_units)
        return hidden

class StockData(Dataset):
    def __init__(self, data, seq_len, input_size):
        self.data = data
        self.seq_len = seq_len
        self.input_size = input_size

    def __len__(self):
        return len(self.data['Open'])-self.seq_len-1

    def __getitem__(self, item):
        five_data = np.zeros((self.seq_len, self.input_size))
        counter = 0
        for i in range(item, item+self.seq_len):
            open_data = self.data['Open'][int(i)]
            close_data = self.data['Close'][i]
            high_data = self.data['High'][i]
            low_data = self.data['Low'][i]
            # volume_data = self.data['Volume'][i]
            # adjclose_data = self.data['Adj Close'][i]
            five_data[counter] = np.array([open_data, close_data, high_data, low_data])
            counter += 1
        five_data = torch.tensor(five_data.astype(dtype=np.double))
        open_price = self.data['Open'][item+self.seq_len+1]

        return five_data, open_price


# Set up network
sequence_len = 40  # Number of days to look at, like predict mondays price given last mon-fri
batch_size = 30
input_size = 4
hidden_units = 30
num_stacked = 2  # Number of Stacked LSTM's so second LSTM takes input from first
output_size = 1
normalize_factor = 200

# Model
model = LSTMClassifier(input_size, hidden_units, num_stacked, output_size).double()

# Loss
loss = nn.MSELoss()
# Optimizer
lr = 0.0005
optim = Adam(model.parameters(), lr=lr)
# Dataloaders
train_data = StockData(data, sequence_len, input_size)
train_dataset = DataLoader(train_data, batch_size=batch_size, shuffle=False)

errors = []
epochs = 30
for epoch in range(epochs):
    ave_error = []
    for i, data in enumerate(train_dataset, 0):
        actual_data, open_price = data
        actual_data = actual_data / normalize_factor
        open_price = open_price / normalize_factor
        # print("Actual data: ", actual_data.shape)
        # print("Open price: ", open_price)
        optim.zero_grad()
        out = model(actual_data.double())
        # print("Out: ", out)
        error = loss(out, open_price)
        
        ave_error.append(error.item())
        error.backward()
        optim.step()
        # print("Error: ", error.item())
    print("Epoch: ", epoch, " Average Error: ", np.average(ave_error))
    errors.append(np.average(ave_error))

from matplotlib import pyplot as plt

plt.plot(errors)
plt.show()

train_dataset = DataLoader(train_data, batch_size=1, shuffle=False)
actual = []
predicted = []
model.eval()
with torch.no_grad():
    for i, data in enumerate(train_dataset, 0):
        actual_data, open_price = data
        actual_data = actual_data / normalize_factor
        open_price = open_price / normalize_factor
        out = model(actual_data.double())
        actual.append(open_price.item())
        predicted.append(out.item())
plt.plot(predicted, c='b', label='predicted')
plt.plot(actual, c='r', label='actual')
plt.legend()
plt.show()

