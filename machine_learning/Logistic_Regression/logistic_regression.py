"""
    Author: Ben Hers
    In this python script, we implement logisitic regression and gradient descent from scratch
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

def load_defect_data():
    with np.load('a1_data.npz') as data:
        x = data['defect_x']
        y = data['defect_y']
    return x, y


def load_swirls():
    with np.load('a1_data.npz') as data:
        x = data['swirls_x']
        y = data['swirls_y']
    return x, y


def load_noisy_circles():
    with np.load('a1_data.npz') as data:
        x = data['circles_x']
        y = data['circles_y']
    return x, y


def load_noisy_moons():
    with np.load('a1_data.npz') as data:
        x = data['moons_x']
        y = data['moons_y']
    return x, y


def load_partitioned_circles():
    with np.load('a1_data.npz') as data:
        x = data['partitioned_circles_x']
        y = data['partitioned_circles_y']
    return x, y


# Sigmoid Function
def sigmoid(x):
    '''
    Performs sigmoid function on numpy vectors
    :param x: x is an numpy array
    :return: numpy array
    '''
    s = 1/(1+np.exp(-x))
    return s

# Initialize weights of logistic regression classifier
def initialize_parameters(n):
    '''
    Initializes all the weights to start at 0
    :param n: n is the number of features
    :return: (nx1) array w of zeros and single float
    '''
    w = np.zeros((n,1))
    b = float(0)
    return w,b

# Hypothesis Function
def hypothesis(X, w, b):
    '''
    Calculates the sigmoid function of the inputs
    :param X: (n,m) array of input data
    :param w: (n,1) array of weights for each feature
    :param b: float for the bias
    :return: (1,m) numpy array of the hypothesis for each data point
    '''
    A = sigmoid(np.matmul((w.T),X)+b)
    return A

# Compute the cost function for the hypothesis
def compute_cost(A, Y):
    '''
    Logistic Regression Cost Function
    :param A: (1,m) array of the hypothesis
    :param Y: (m,) with correct label
    :return: cost(float)
    '''
    log_loss = np.sum(Y*np.log(A))+np.sum((1-Y)*np.log(1-A))
    cost = (-1/Y.shape[0])*log_loss
    return cost

# Calculate the gradient
def compute_gradients(A, X, Y):
    """
    Calculates the gradient of the loss function so we can update our weights at each iteration
    :param A: (1,m) array with the hypothesis
    :param X: (n,m) array of the input data
    :param Y: (m,) array of the labels of the data
    :return: 2 arrays, one with cost gradient for weights and one with cost gradient for bias
    """
    dw = (1/Y.shape[0])*np.matmul(X, (A-Y).T)
    db = (1/Y.shape[0])*np.sum(A-Y)
    return dw, db

def gradient_descent(X, Y, num_iterations, learning_rate, print_costs=True):
    """
    Trains a logistic regression classifier on data through gradient descent
    :param X: Input Data
    :param Y: Labels for the data
    :param num_iterations: Number of training iterations
    :param learning_rate: Learning rate for updating weights
    :param print_costs: flag to print cost at each iteration
    :return:
    """
    w,b = initialize_parameters(X.shape[0])
    costs = []
    for i in range(num_iterations):
        A = hypothesis(X, w, b)
        cost = compute_cost(A, Y)
        dw, db = compute_gradients(A, X, Y)
        w = w-learning_rate*dw
        b = b-learning_rate*db
        costs.append(cost)
        if print_costs and i % 1000 == 0:
            print("Iteration: %d - Cost: %f" % (i, cost))
    print("Iteration: %d - Cost: %f" % (i, cost))
    return w, b, costs

def predict(X, w, b):
    """
    Based on the provided weights and bias, uses logistic regression to provide a hypothesis
    :param X: (n,m) Input data matrix
    :param w: (n,1) Input array with weights for each feature
    :param b: bias parameter
    :return: (m,) predictions
    """
    y_pred = np.rint(hypothesis(X, w, b))
    return y_pred

def plot_decision_boundary(X, Y, w, b, save=False, fig_name=""):
    plt.scatter(X[0,:], X[1,:], c=Y, cmap=colors.ListedColormap(["blue", "red"]))
    x_0 = np.array([min(X[0, :]) - 1, max(X[0, :]) + 1])
    x_1 = - (w[0] * x_0 + b) / w[1]
    plt.plot(x_0, x_1, label="Decision_Boundary")
    if save:
        plt.savefig("figures/"+fig_name)
    plt.show()


X,Y = load_defect_data()
color_map = ['blue', 'red']
plt.scatter(X[0,:], X[1,:], c=Y, cmap=colors.ListedColormap(color_map))
plt.xlabel('X1')
plt.ylabel('X0')
plt.title('Chip Defects')
save_first=True
if save_first:
    plt.savefig("figures/defect_data.png")
plt.show()


w, b, costs = gradient_descent(X,Y,num_iterations=100000,learning_rate=0.002)
plt.plot(costs)
plt.xlabel('Iteration')
plt.ylabel('Cost')
plt.title('Logistic Regression Cost vs Iterations')
if save_first:
    plt.savefig("figures/cost.png")
plt.show()
plot_decision_boundary(X, Y, w, b, save=True, fig_name="DecisionBoundary.png")
# Test effect of learning rate
num_points = 40
rates = np.linspace(0.001,0.003,num_points)
learning_costs = []
for i in range(num_points):
    w, b, costs = gradient_descent(X, Y, num_iterations=80000, learning_rate=rates[i], print_costs=False)
    learning_costs.append(costs[-1]) # Gets last cost and adds it to rate
plt.plot(rates, learning_costs)
plt.xlabel('Learning Rate')
plt.ylabel('Cost')
plt.title('Cost vs Learning Rate')
if save_first:
    plt.savefig("figures/learning_costs.png")
plt.show()
