import numpy as np


def loadDataSet(file):

    dataMat = []
    fin = open(file)

    for line in fin.readlines():
        words = line.strip().split()
        floatLine = [float(words[0]), float(words[1])]
        dataMat.append(floatLine)

    return dataMat


def distEuclid(X, Y):
    """
    Calculate Euclid distance between X and Y

    :param X: array type X
    :param Y: array type Y
    :return: The Euclid distance between X and Y
    """
    return np.sqrt(np.sum(np.power(np.subtract(X, Y), 2)))


def randomCenter(dataSet, k):
    """
    程序清单10-1 K-均值聚类支持函数

    :param dataSet: 数据集
    :param k:
    :return:
    """
    m, n = np.shape(dataSet)

    centroids = np.zeros((k, n))

    for j in range(n):
        minJ = np.min(dataSet, axis=j)
        rangeJ = np.max(np.array(dataSet), axis=j) - minJ
        print(rangeJ)
        rangeJ = np.float(rangeJ)
        centroids[:, j] = minJ + rangeJ * np.random.rand(k, 1)

    return centroids


def kMeans(dataSet, k, measure=distEuclid, createCenter=randomCenter):
    """
    K-Means algorithm

    :param dataSet: 数据集
    :param k: 分类数
    :param measure: 测度
    :param createCenter: 初始聚类中心
    :return: 聚类中心，簇划分
    """
    m, n = np.shape(dataSet)
    clusterAssign = np.zeros((m, 2))
    centroids = createCenter(dataSet, k)
    clusterChanged = True

    while clusterChanged:

        clusterChanged = False
        # 对于每一个样例
        for i in range(m):
            minDist, minIndex = np.inf, -1
            # 寻找最近的质心
            for j in range(k):
                distJI = measure(centroids[j, :], dataSet[i, :])
                if distJI < minDist:
                    minDist = distJI, minIndex = j

            if clusterAssign[i][0] != minIndex:
                clusterChanged = True
            clusterAssign[i, :] = minIndex, minDist ** 2

        print(centroids)
        # 更新质心的位置
        for center in centroids:
            pointInCluster = dataSet[np.nonzero(clusterAssign[:,0] == center)]
            centroids[center, :] = np.mean(pointInCluster, axis=0)

    return centroids, clusterAssign


dist = distEuclid([0.1, 0.1, 0.1], [0.3, 0.3, 0.3])
print(dist)

data = loadDataSet('./dataset/LR.txt')
print(data)

centroids, clusterAssign = kMeans(data, 4)
print(centroids)
print(clusterAssign)