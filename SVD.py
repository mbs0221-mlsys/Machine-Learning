import numpy as np
from numpy import linalg as la


def loadExData():
    return [
        [1, 1, 1, 0, 0],
        [2, 2, 2, 0, 0],
        [1, 1, 1, 0, 0],
        [5, 5, 5, 0, 0],
        [1, 1, 0, 2, 2],
        [0, 0, 0, 3, 3],
        [0, 0, 0, 1, 1]
    ]


def EuclidSim(X, Y):
    """
    程序清单14-1 欧氏距离

    :param X:
    :param Y:
    :return:
    """
    return 1.0 / (1.0 + la.norm(X - Y))


def PearsonSim(X, Y):
    """
    程序清单14-1 皮尔逊相关系数

    :param X:
    :param Y:
    :return:
    """
    if len(X) < 3: return 1 / 0
    return 0.5 + 0.5 * np.corrcoef(X, Y, rowvar=False)[0][1]


def CosSim(X, Y):
    """
    程序清单14-1 余弦相似度

    :param X:
    :param Y:
    :return:
    """
    num = float(X.T * Y)
    norm = la.norm(X) * la.norm(Y)
    return 0.5 + 0.5 * (num / norm)


def StandardEst(data, user, sim, item):
    """
    程序清单14-2 基于物品相似度的推荐引擎

    :param data:
    :param user: 用户
    :param sim: 相似度度量
    :param item: 商品条目
    :return:
    """
    m, n = np.shape(data)
    simTotal, ratSimTotal = 0.0, 0.0
    for j in range(n):
        userRating = data[user, j]
        if userRating == 0:
            continue
        # 1-寻找两个用户都评级的物品
        overLap = np.nonzero(np.logical_and(data[:, item].A > 0, data[:, j].A > 0))[0]
        if len(overLap) == 0:
            similarity = 0
        else:
            similarity = sim(data[overLap, item], data[overLap, j])
        print('the %d and %d similarity is: %f' % (item, j, similarity))
        simTotal += similarity
        ratSimTotal += similarity * userRating
    if simTotal == 0:
        return 0
    else:
        return ratSimTotal / simTotal


def SvdEst(data, user, sim, item):
    """
    程序清单14-3 基于SVD的评分估计

    :param data:
    :param user: 用户编号
    :param sim: 相似度度量
    :param item: 商品编号
    :return:
    """
    m, n = np.shape(data)
    simTotal, ratSimTotal = 0.0, 0.0
    U, Sigma, VT = np.linalg.svd(data)
    # 1-建立对角矩阵
    Sig4 = np.mat(np.eye(4) * Sigma[:4])
    # 2-构建转换后的物品
    xFormedItems = data.T * U[:, :4] * Sig4.I
    for j in range(n):
        userRating = data[user, j]
        if userRating == 0 or j == item:
            continue
        similarity = sim(xFormedItems[item, :].T, xFormedItems[j, :].T)
        print('the %d and %d similarity is: %f' % (item, j, similarity))
        simTotal += similarity
        ratSimTotal += similarity * userRating
    if simTotal == 0:
        return 0
    else:
        return ratSimTotal / simTotal


def Recommend(data, user, N=3, sim=CosSim, est=StandardEst):
    """
    程序清单14-2 基于物品相似度的推荐引擎

    :param data:
    :param user: 用户
    :param N:
    :param sim: 相似度度量
    :param est: 评估方法
    :return:
    """
    # 2-寻找未评级的物品
    unratedItems = np.nonzero([data[user, :].A == 0])[1]
    if len(unratedItems) == 0:
        return 'you are rated everything!'
    itemScores = []
    for item in unratedItems:
        estScore = est(data, user, sim, item)
        itemScores.append((item, estScore))
    # 3-寻找前N个未评级物品
    return sorted(itemScores, key=lambda item: item[1], reverse=True)[:N]


matrix = loadExData()
mat = np.mat(matrix)

print('est=SvdEst')
rec = Recommend(mat, 1, est=SvdEst)
print('Recommend')
print(rec)
print('est=SvdEst, sim=PearsonSim')
rec = Recommend(mat, 1, est=SvdEst, sim=PearsonSim)
print('Recommend')
print(rec)

# e1 = EuclidSim(mat[:, 0], mat[:, 4])
# e2 = EuclidSim(mat[:, 0], mat[:, 0])
# print(e1, e2)
#
# e1 = CosSim(mat[:, 0], mat[:, 4])
# e2 = CosSim(mat[:, 0], mat[:, 0])
# print(e1, e2)
#
# e1 = PearsonSim(mat[:, 0], mat[:, 4])
# e2 = PearsonSim(mat[:, 0], mat[:, 0])
# print(e1, e2)
