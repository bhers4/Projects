"""
    Author: Ben Hers
    Description: 2 layer neural network using tanh activation function and sigmoid output layer
    for 0-1 output
"""
import numpy as np
from matplotlib import pyplot as plt

# Sigmoid function
def sigmoid(X):
    return 1/(1+np.exp(-X))

# Initialize Parameters
def initialize_parameters(nX, nH):
    # nX is number of input features
    # nH is number of hidden units
    W1 = np.random.normal(0, 0.01, (nH, nX))
    B1 = np.zeros((nH, 1))
    W2 = np.random.normal(0, 0.01, (1, nH))
    B2 = np.zeros((1, 1))
    return W1, B1, W2, B2


def forward_propagation(X, W1, B1, W2, B2):
    # W1, B1 is weights and bias for first layer of 2 layer network
    # W2, B2 is the weights and biases for second layer of neural network
    Z1 = np.matmul(W1, X)+B1
    A1 = np.tanh(Z1)
    Z2 = np.matmul(W2, A1)+B2
    A2 = sigmoid(Z2)
    return A1, A2


def compute_cost(A2, Y):
    r = np.matmul(Y, np.log(A2).flatten())
    s = np.matmul((1-Y), np.log(1-A2).flatten())
    cost = (-1)*np.sum(r+s)/len(Y)
    return cost


def backward_propagation(X, Y, W2, A1, A2):
    dZ2 = A2-Y
    dW2 = np.matmul(dZ2, A1.T)/Y.size
    dB2 = np.sum(dZ2, axis=1)/Y.size
    first = np.matmul(W2.T, dZ2)
    second = 1-np.power(A1, 2)
    dZ1 = first*second
    dW1 = np.matmul(dZ1, X.T)/Y.size
    dB1 = (1/Y.size)*np.sum(dZ1, axis=1)
    dB1 = np.reshape(dB1, (dB1.size, 1))
    return dW1, dB1, dW2, dB2


def train_neural_network(X, Y, num_hidden, num_iterations, learning_rate):
    W1, B1, W2, B2 = initialize_parameters(X.shape[0], num_hidden)
    costs = []
    for i in range(num_iterations):
        A1, A2 = forward_propagation(X, W1, B1, W2, B2)
        dW1, dB1, dW2, dB2 = backward_propagation(X, Y, W2, A1, A2)
        W1 = W1-learning_rate*dW1
        W2 = W2-learning_rate*dW2
        B1 = B1-learning_rate*dB1
        B2 = B2-learning_rate*dB2
        cost = compute_cost(A2, Y)
        costs.append(cost)
    return W1, B1, W2, B2, costs


def predict(X, W1, B1, W2, B2):
    A1 = np.tanh(np.matmul(W1, X)+B1)
    A2 = np.matmul(W2, A1)+B2
    predictions = np.rint(sigmoid(A2))
    return predictions


def compute_accuracy(X, Y, W1, B1, W2, B2):
    Y_predicted = predict(X, W1, B1, W2, B2)
    accuracy = np.mean(Y_predicted == Y)
    return accuracy


def plot_decision_boundary(X, Y, W1, B1, W2, B2, subplot=plt):
    # Set min and max values and give it some padding
    x_min, x_max = X[0, :].min() - 1, X[0, :].max() + 1
    y_min, y_max = X[1, :].min() - 1, X[1, :].max() + 1

    # Generate a grid of points with distance h between them
    h = 0.01
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Predict the function value for the whole grid
    Z = predict(np.c_[xx.ravel(), yy.ravel()].T, W1, B1, W2, B2)
    Z = Z.reshape(xx.shape)

    # Plot the contour and training examples
    subplot.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    subplot.scatter(X[0, :], X[1, :], c=Y, cmap=plt.cm.Spectral)

from a1_tools import load_noisy_circles
np.random.seed(0)
X, Y = load_noisy_circles()
n_iters = 5000
learning_rate = 0.05

plt.scatter(X[0, :], X[1,:], c=Y, cmap=plt.cm.Spectral)
plt.title("Data")
plt.xlabel("Feature X1")
plt.ylabel("Feature X2")
plt.show()

W1, B1, W2, B2, costs = train_neural_network(X, Y, 5, n_iters, learning_rate)
plt.plot(costs)
plt.title('Training Cost')
plt.xlabel('Iteration')
plt.ylabel("Cost")
plt.show()

plot_decision_boundary(X, Y, W1, B1, W2, B2)
plt.show()
compute_accuracy(X, Y, W1, B1, W2, B2)