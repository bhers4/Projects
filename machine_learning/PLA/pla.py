from matplotlib import pyplot as plt
import numpy as np


def loadData():
    with np.load('notMNIST.npz') as data:
        Data, Target = data['images'], data['labels']
        posClass = 2
        negClass = 9
        dataIndx = (Target == posClass) + (Target == negClass)
        Data = Data[dataIndx] / 255.0
        Target = Target[dataIndx].reshape(-1, 1)
        Target[Target == posClass] = 1
        Target[Target == negClass] = 0
        np.random.seed(1)
        randIndx = np.arange(len(Data))
        np.random.shuffle(randIndx)
        Data, Target = Data[randIndx], Target[randIndx]
        trainData, trainTarget = Data[:3500], Target[:3500]
        validData, validTarget = Data[3500:3600], Target[3500:3600]
        testData, testTarget = Data[3600:], Target[3600:]
    return trainData, validData, testData, trainTarget, validTarget, testTarget


def ErrorRate (w, x, y):
    predictions = (np.matmul(w, x.T)) # Will give a vector of predictions
    predictions = predictions>0
    # Now compare against y
    error = np.sum(predictions.T==y)
    error = 1-(float(error)/y.shape[0])
    assert(predictions.shape==(1,3500))
    return error


def PLA(w, x, y, maxIter):
    plot=False
    for i in range(maxIter):
        # Get Error, no sense predicting in both PLA() and Error()
        error = ErrorRate(w, x, y)
        predictions = (np.matmul(w, x.T)) # Will give a vector of predictions
        predictions = predictions-128
        predict0 = predictions
        predictions = predictions>0
        predict1 = predictions
        predictions = predictions.flatten()
        if plot:
            n = np.linspace(0, predict0.size, predict0.size)
            plt.scatter(n, predict0)
            plt.scatter(n, predict1)
            plt.title("Predict0 and Predict1")
            plt.show()
        length = len(predictions.flatten())
        for i in range(len(predictions)):
            if predictions[i] != y[i]:
                if y[i] == 0: # Set to -1
                    w = w-x[i]
                else:
                    w = w+x[i]
                break
    return w


def pocket(x, y, T):
    flattenedLength = len(x[0].flatten())
    numSubIterations = 1
    w = np.zeros((1, flattenedLength))
    wBest = np.zeros_like(w)
    bestError = 1
    for i in range(T):
        w = PLA(w, x, y, numSubIterations)
        errorI = ErrorRate(w, x, y)
        if errorI < bestError:
            wBest = w
            bestError = errorI
        print("Best Error: ", bestError)
    return wBest


def sigmoid(x):
    return (1/(1+np.exp(-x)))

def crossEntropyLoss(w, x, y, reg):
    # First calculate the matrix multiplication of x and w
    l2norm = (reg/2)*np.square(np.linalg.norm(w, ord=2))
    pred = np.matmul(w, x.T)
    yhat = sigmoid(pred)
    yhat = np.clip(yhat, 0.00001, 0.99999, out=yhat)
    # print("Y: ", y.shape, " YHAT: ", yhat.shape)
    first_term = np.matmul(y.flatten(), np.log(yhat).flatten())
    temp0 = (1-y).flatten()
    temp1 = np.log(1-yhat).flatten()
    second_term = np.matmul(temp0, temp1)
    # second_term = np.sum((1-y)*np.log(1-yhat))
    cost = (-1/y.shape[0])*(first_term+second_term)
    # Gets the L2 norm and squares it and multiplies by half reg term
    cost = cost + l2norm
    # print("Cost: ", cost, " First: ", first_term, " Second: ", second_term)
    return cost


def gradCE(w, x, y, reg):
    # YOUR CODE HERE
    pred = np.matmul(w, x.T)
    sig = sigmoid(pred)
    diff = sig-y.T
    dw = (1/y.shape[0])*(np.matmul(diff, x))
    dw = dw+(reg*w)
    return dw


def grad_descent(w, x, y, eta, iterations, reg, error_tol):
    # YOUR CODE HERE
    costs = []
    for i in range(iterations):
        gradients = gradCE(w, x, y, reg)
        gradients = gradients/np.linalg.norm(gradients)
        # print("Gradient: ", gradients.shape, " Max: ", np.max(gradients), " Min: ", np.min(gradients))
        xEntropyLoss = crossEntropyLoss(w, x, y, reg)
        w_old = w
        w = w-eta*gradients
        w_diff = np.linalg.norm(w-w_old)
        diff2 = np.abs(np.linalg.norm(w)-np.linalg.norm(w_old))
        costs.append(xEntropyLoss)
        # print("Error: ", xEntropyLoss, " w_diff: ", w_diff, " wDiff2: ", diff2)
        if w_diff <= error_tol:
            print("Below or equal to error tolerance")
            break
    return w

# Load data from notMNIST dataset using provided function
trainData, validData, testData, trainTarget, validTarget, testTarget = loadData()
# Number of Training Samples
numTrainingSamples = len(trainData)
# Initialize weights
flattenedLength = len(trainData[0].flatten())
w = np.zeros((1, flattenedLength+1))
numIterations = 100 # Supposed to be 100
# Create x vector that is 28^2 by number of samples
numRows = len(trainData)
numCols = len(trainData[0].flatten())+1

x = np.zeros((numRows, numCols))
for i in range(numRows):
    x[i] = np.concatenate([np.ones(1), trainData[i].flatten()])
# We got 3500 rows, 785 columns one for each input plus a bias
w = PLA(w, x, trainTarget, numIterations)

# Evaluate on test data
print("Test Set Shapes: ", testData.shape, " Target: ", testTarget.shape)
print(testData[0].size)
testX = np.zeros((145,785))
counter = 0
for item in testData:
    # np.concatenate([np.ones(1), trainData[i].flatten()])
    img = np.concatenate([np.ones(1), item.flatten()])
    testX[counter] = img
    counter = counter+1
predictions = (np.matmul(w, testX.T)-128)>0
numCorrect = np.sum(testTarget==predictions.T)
accuracyVanillaPLA = numCorrect/len(predictions.flatten())
# Pocket Algorithm
wPocket = pocket(x, trainTarget, 100)
predictions = (np.matmul(wPocket, testX.T)-128)>0
numCorrect = np.sum(testTarget==predictions.T)
accuracyPocket = numCorrect/len(predictions.flatten())
print("Reg: " , accuracyVanillaPLA, " Pocket: ", accuracyPocket)

import time
time.sleep(1)


wPocket = pocket(x, trainTarget, 100)
predictions = (np.matmul(wPocket, testX.T)-128)>0
numCorrect = np.sum(testTarget==predictions.T)
accuracyPocket = numCorrect/len(predictions.flatten())
print("Reg: " , accuracyVanillaPLA, " Pocket: ", accuracyPocket)

w = np.random.normal(0,1, flattenedLength+1)
w_005 = grad_descent(w, x, trainTarget, 0.005, 5000, 0, 10^-7)
w_001 = grad_descent(w, x, trainTarget, 0.001, 5000, 0, 10^-7)
w_0001 = grad_descent(w, x, trainTarget, 0.0001, 5000, 0, 10^-7)

pred = np.matmul(w_005, testX.T)
pred[pred>0.5] = 1
pred[pred<=0.5] = 0
numCorrect = np.sum(pred==testTarget.T)
print("005: NumCorrect: ", numCorrect, " Num: ", pred.size, " Accuracy:", numCorrect/pred.size)
pred = np.matmul(w_001, testX.T)
pred[pred>0.5] = 1
pred[pred<=0.5] = 0
numCorrect = np.sum(pred==testTarget.T)
print("001: NumCorrect: ", numCorrect, " Num: ", pred.size, " Accuracy:", numCorrect/pred.size)
pred = np.matmul(w_0001, testX.T)
pred[pred>0.5] = 1
pred[pred<=0.5] = 0
numCorrect = np.sum(pred==testTarget.T)
print("0001: NumCorrect: ", numCorrect, " Num: ", pred.size, " Accuracy:", numCorrect/pred.size)

# reg_001 = grad_descent(w, x, trainTarget, 0.005, 5000, 0.001, 10^-7)
# reg_01 = grad_descent(w, x, trainTarget, 0.005, 5000, 0.01, 10^-7)
# reg_1 = grad_descent(w, x, trainTarget, 0.005, 5000, 0.1, 10^-7)