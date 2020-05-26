import torch
import torch.nn as nn
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
    print(len(data[item]))
print("Hello")
