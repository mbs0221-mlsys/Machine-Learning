import numpy as np
import urllib
import json
from time import sleep
import matplotlib
import matplotlib.pyplot as plt


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


def geoGrab(address, city):
    """
    程序清单10-4 Yahoo! PlaceFinder API

    :param address: 地址
    :param city: 城市
    :return: JSON数据
    """
    apiStem = 'http://where.yahooapis.com/geocode?'
    params = {
        'flags': 'J',
        'appid': 'ppp68N8t',
        'location': '%s %s' % (address, city)
    }
    url_params = urllib.urlencode(params)
    yahooApi = apiStem + url_params
    print(yahooApi)
    c = urllib.open(yahooApi)
    return json.load(c.read())


def massPlaceFind(filename):
    """
    程序清单10-4 Yahoo! PlaceFinder API

    :param filename: 文件名
    :return: 无
    """
    fr = open(filename)
    fw = open('places.txt', 'w')
    for line in fr.readlines():
        line = line.strip()
        other, address, city = line.split('\t')
        retDict = geoGrab(address, city)
        ResultSet = retDict['ResultSet']
        if ResultSet['Error'] == 0:
            Results = ResultSet['Results']
            lat = np.float(Results[0]['latitude'])
            long = np.float(Results[0]['longitude'])
            print(other, lat, long)
            fw.write('%s\t%f\t%f\n' % (line, lat, long))
        else:
            print('error fetching')
        sleep(1)
    fw.close()
    fr.close()


def distSLC(vecA, vecB):
    """
    程序清单10-5 球面距离计算

    :param vecA: 球面坐标A
    :param vecB: 球面坐标B
    :return: 球面距离
    """
    a = np.sin(vecA[0, 1] * np.pi / 180) * np.sin(vecB[0, 1] * np.pi / 180)
    b = np.cos(vecA[0, 1] * np.pi / 180) * np.cos(vecB[0, 1] * np.pi / 180)
    return np.arccos(a + b) / 6371.0


def clusterClubs(numCluster=5):
    """
    程序清单10-5 聚类测试

    :param numCluster: 簇数
    :return: 无
    """
    # 读取文件数据
    fr = open('places.txt')
    data = []
    for line in fr.readlines():
        strArr = line.split('\t')
        data.append([float(strArr[4]), float(strArr[3])])
    dataMat = np.mat(data)
    # 二分K-均值聚类
    centers, clusterAssign = biKMeans(dataMat, numCluster, measure=distSLC)
    # 图表绘制
    fig = plt.figure()
    # 设置区域、标记
    rect = [0.1, 0.1, 0.8, 0.8]
    scatterMarkers = ['s', 'o', '^', '8', 'p', 'd', 'v', 'h', '>', '<']
    axprops = dict(xticks=[], yticks=[])
    # 绘制背景
    ax0 = fig.add_axes(rect, label='ax0', **axprops)
    imgP = plt.imread('Portland.png')
    ax0.imshow(imgP)
    # 绘制数据点
    ax1 = fig.add_axes(rect, label='ax1', frameon=False)
    for i in range(numCluster):
        pstInCurrCluster = dataMat[np.nonzero(clusterAssign[:, 0] == i), :]
        markerStyle = scatterMarkers[i % len(scatterMarkers)]
        ax1.scatter(pstInCurrCluster[:, 0].flatten().A[0], pstInCurrCluster[:, 1].flatten().A[0], marker=markerStyle, s=90)
    # 绘制聚类中心
    ax1.scatter(centers[:, 0].flatten().A[0], centers[:, 1].flatten().A[0], marker='+', s=300)
    plt.show()


# 在wine数据集上做聚类测试
data, labels = loadClsDataSet('./dataset/wine/wine.data')
centroids, clusterAssign = kMeans(data, 3)
print(centroids)
print(clusterAssign)
centroids, clusterAssign = biKMeans(data, 3)
print(centroids)
print(clusterAssign)
