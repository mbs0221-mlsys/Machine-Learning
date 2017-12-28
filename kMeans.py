import numpy as np


def loadClsDataSet(file, delim=','):
    """
    加载分类数据集

    :param file: 文件名
    :param delim: 数值分隔符
    :return: 数据集
    """
    fp = open(file)
    strArr = fp.readlines()
    data, labels = [], []
    for line in strArr[1:]:
        words = line.strip().split(delim)
        data.append(np.array(words[1:], dtype=float))
        labels.append(np.int(words[0]))
    data = np.array(data)
    labels = np.array(labels)
    return data, labels


def distEuclid(X, Y):
    """
    Calculate Euclid distance between X and Y

    :param X: array type X
    :param Y: array type Y
    :return: The Euclid distance between X and Y
    """
    return np.sqrt(np.sum(np.power(np.subtract(X, Y), 2)))


def randomCenter(data_set, k):
    """
    程序清单10-1 K-均值聚类支持函数

    :param data_set: 数据集
    :param k: 随机质心数目
    :return: k个随机生成的质心
    """

    m, n = np.shape(data_set)
    center = np.zeros((k, n))

    for j in range(n):
        minJ = np.min(data_set[:, j])
        rangeJ = np.float(np.max(data_set[:, j]) - minJ)
        center[:, j] = minJ + rangeJ * np.random.rand(1, k)

    return center


def kMeans(dataSet, k, measure=distEuclid, createCenter=randomCenter):
    """
    程序清单10-2 K-均值聚类算法

    :param dataSet: 数据集
    :param k: 簇数
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
                    minDist, minIndex = distJI, j

            # 簇划分发生变化
            if clusterAssign[i][0] != minIndex:
                clusterChanged = True
            clusterAssign[i, :] = minIndex, minDist ** 2

        # 更新质心的位置
        for i in range(k):
            pointInCluster = dataSet[np.nonzero(clusterAssign[:, 0] == i)[0]]
            centroids[i, :] = np.mean(pointInCluster, axis=0)

    return centroids, clusterAssign


def biKMeans(dataSet, k, measure=distEuclid):
    """
    程序清单10-3 二分K-均值聚类算法

    :param dataSet: 数据集
    :param k: 簇数
    :param measure: 度量
    :return: 聚类中心，簇划分列表
    """
    m, n = np.shape(dataSet)
    clusterAssign = np.mat(np.zeros((m, 2)))
    # 1-创建一个初始簇
    center = np.mean(dataSet, axis=0).tolist()[0]
    centers = [center]
    for j in range(m):
        clusterAssign[j, 1] = measure(np.mat(center), dataSet[j, :]) ** 2
    while len(centers) < k:
        lowestSSE = np.inf
        for i in range(len(centers)):
            # 2-尝试划分每一簇
            ptInCurrCluster = dataSet[np.nonzero(clusterAssign[:, 0] == i)[0], :]
            centerMat, splitClusterAss = kMeans(ptInCurrCluster, 2, measure)
            sseSplit = sum(splitClusterAss[:, 1])
            sseNotSplit = sum(clusterAssign[np.nonzero(clusterAssign[:, 0] != i)[0], i])
            print('split and not split: ', sseSplit, sseNotSplit)
            if sseSplit + sseNotSplit < lowestSSE:
                bestCentToSplit = i
                bestNewCenters = centerMat
                bestClusterAss = splitClusterAss.copy()
                lowestSSE = sseSplit + sseNotSplit
        # 3-更新簇的分配结果
        bestClusterAss[np.nonzero(bestClusterAss[:, 0] == 1)[0], 0] = len(centers)
        bestClusterAss[np.nonzero(bestClusterAss[:, 0] == 0)[0], 0] = bestCentToSplit
        print('the best center to split is : ', bestCentToSplit)
        print('the len of best cluster assign is : ', len(bestClusterAss))
        centers[bestCentToSplit] = bestNewCenters[0, :]
        centers.append(bestNewCenters[1, :])
        clusterAssign[np.nonzero(clusterAssign[:, 0] == bestCentToSplit)[0], :] = bestClusterAss
    return np.mat(centers), clusterAssign


# 在wine数据集上做聚类测试
data, labels = loadClsDataSet('./dataset/wine/wine.data')
centroids, clusterAssign = kMeans(data, 3)
print(centroids)
print(clusterAssign)
centroids, clusterAssign = biKMeans(data, 3)
print(centroids)
print(clusterAssign)
