import numpy as np


def loadDataSet(file, delim=' '):
    fp = open(file)
    strArr = [line.strip().split(delim) for line in fp.readlines()]
    fltArr = [np.array(line, dtype=float) for line in strArr]
    return np.mat(fltArr)


def PCA(dataSet, topNfeat=9999999):
    """
    程序清单13-1 PCA算法

    :param dataSet: 数据集
    :param topNfeat: 前N个特征
    :return: 低秩矩阵，重构矩阵
    """
    data = np.array(dataSet)
    # 1-去平均值
    meanVals = data.mean(axis=0)
    meanRemoved = data - meanVals
    # 计算协方差矩阵
    covMat = np.cov(meanRemoved, rowvar=0)
    # 计算协方差矩阵的特征值与特征向量
    eigVals, eigVects = np.linalg.eig(covMat)
    # 2-从小到大对N个特征值进行排序
    eigValInd = np.argsort(eigVals)
    eigValInd = eigValInd[:-(topNfeat + 1):-1]
    redEigVects = eigVects[:, eigValInd]
    # 3-将数据转换到新空间
    lowDDataMat = meanRemoved * redEigVects
    reconMat = (lowDDataMat * redEigVects.T) + meanVals
    return lowDDataMat, reconMat


def replaceNanWithMean():
    """
    程序清单13-2 将NaN替换成平均值的函数
    :return: 替换NaN之后的数据集
    """
    dataMat = loadDataSet('secm.data', ' ')
    numFeat = np.shape(dataMat)[1]
    for i in range(numFeat):
        # 1-计算所有非NaN的平均值
        meanVal = np.mean(dataMat[np.nonzero(~np.isnan(dataMat[:, i].A))[0], i])
        # 2-将所有NaN置位平均值
        dataMat[np.nonzero(np.isnan(dataMat[:, i].A)), i] = meanVal
    return dataMat
