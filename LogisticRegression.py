import math
import numpy as np
import matplotlib.pyplot as plt
import random as rd


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

    m, n = np.shape(data)

    alpha = 0.001
    maxCycles = 5000
    W = np.ones((n, 1))
    for k in range(maxCycles):
        H = sigmoid(data * W)
        E = labels - H
        W = W + alpha*data.transpose()*E

    return W


def StocGradAscent(data, labels, num_iter=500):
    """
    随机梯度上升算法

    :param data: 数据集
    :param labels: 标签
    :param num_iter: 迭代次数
    :return:
    """

    data_mat = np.array(data)
    m, n = np.shape(data_mat)
    alpha = 0.01 # 程序清单5-3 随机梯度上升算法
    weights = np.ones(n)
    for j in range(num_iter):
        for i in range(m):
            # 程序清单5-4 改进的随机梯度上升算法
            alpha = 4 / (1.0 + j + i) + 0.01 # alpha每次迭代时需要调整
            rand_index = int(rd.uniform(0, len(labels))) # 随机选取更新
            h = sigmoid(sum(data_mat[rand_index]*weights))
            e = labels[rand_index] - h
            weights = weights + alpha * e * data_mat[rand_index]

    return weights


def classifyVector(X, weights):
    """
    程序清单5-5 Logistic回归分类函数

    :param X: 样本
    :param weights: 权重向量
    :return: 分类标签
    """
    prob = sigmoid(sum(X*weights))
    if prob > 0.5:
        return 1.0
    else:
        return 0.0


def plotBestFit(W):
    """
    程序清单5-2 画出数据集和Logistic回归最佳拟合直线的函数
    :param W: 权重向量
    :return: 无
    """

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
    # test classifier
    error = 0.0
    for i in range(len(labels)):
        if classifyVector(data[i], W) != labels[i]:
            error += 1
    print("error rate:%.2f"%(error/len(labels)))
    # plot regression line
    x = np.arange(-4.0, 4.0, 0.1)
    y = (-W[0]-W[1]*x)/W[2]
    x = x.reshape((80, 1))
    y = y.reshape((80, 1))
    ax.plot(x, y)
    # X-Y coordinates
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.savefig('./images/LC_SGD_Improve.jpg')
    plt.show()


data, labels = loadDataSet()
W = StocGradAscent(data, labels, 40)
plotBestFit(W)