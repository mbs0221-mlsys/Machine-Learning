import math
import numpy as np
import matplotlib.pyplot as plt


def loadDataSet():

    data, labels = [], []
    file = open('./dataset/LR.txt')
    for line in file.readlines():
        words = line.strip().split()
        data.append([1.0, float(words[0]), float(words[1])])
        labels.append(int(words[2]))

    return data, labels


def sigmoid(X):
    return 1.0/(1+np.exp(-X))


def BatchGradAscent(data, labels):

    dataMat = np.mat(data)
    labelsMat = np.mat(labels).transpose()
    m, n = np.shape(dataMat)

    alpha = 0.001
    maxCycles = 5000
    W = np.ones((n, 1))
    for k in range(maxCycles):
        H = sigmoid(dataMat * W)
        E = labelsMat - H
        W = W + alpha*dataMat.transpose()*E

    return W


def StocGradAscent(data, labels):

    dataMat = np.array(data)
    m, n = np.shape(dataMat)
    print("%d, %d"%(m, n))
    alpha = 0.01
    weights = np.ones(n)
    for i in range(m):
        h = sigmoid(sum(dataMat[i]*weights))
        e = labels[i] - h
        weights = weights + alpha * e * dataMat[i]

    return weights


def plotBestFit(W):

    data, labels = loadDataSet()
    x0, y0 = [], []
    x1, y1 = [], []
    for i in range(len(labels)):
        if int(labels[i]) == 0:
            x0.append(data[i][1])
            y0.append(data[i][2])
        else:
            x1.append(data[i][1])
            y1.append(data[i][2])

    fig = plt.figure()
    # plot raw data
    ax = fig.add_subplot(111)
    ax.scatter(x0, y0, s=3, c='cyan', marker='s')
    ax.scatter(x1, y1, s=3, c='blue')
    # plot regression line
    x = np.arange(-4.0, 4.0, 0.1)
    y = (-W[0]-W[1]*x)/W[2]
    x = x.reshape((80, 1))
    y = y.reshape((80, 1))
    ax.plot(x, y)
    # X-Y coordinates
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.savefig('./images/LR_BGD.jpg')
    plt.show()

data, labels = loadDataSet()
W = BatchGradAscent(data, labels)
plotBestFit(W)