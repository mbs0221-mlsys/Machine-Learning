def loadDataSet():
    pass


def createC1(dataSet):
    """
    程序清单11-1 Apriori算法中的辅助函数

    :param dataSet:
    :return:
    """
    C1 = []
    for transaction in dataSet:
        for item in transaction:
            if not [item] in C1:
                C1.append(item)
    C1.sort()
    # 1-对C1中每个项构建不变集合
    return map(frozenset, C1)


def scanD(D, Ck, minSupport):
    """
    程序清单11-1 Apriori算法的辅助函数

    :param D:
    :param Ck:
    :param minSupport:
    :return:
    """
    ssCnt = {}
    for tid in D:
        for can in Ck:
            if can.issubset(tid):
                if not ssCnt.has_key(can):
                    ssCnt[can] = 1
    numItems = float(len(D))
    retList = []
    supportData = {}
    # 2-计算所有项集的支持度
    for key in ssCnt:
        support = ssCnt[key] / numItems
        if support > minSupport:
            retList.insert(0, key)
        supportData[key] = support
    return retList, supportData


def aprioriGen(Lk, k):
    """
    程序清单11-2 Apriori算法

    :param Lk:
    :param k:
    :return:
    """
    pass


def apriori(dataSet, minSupport=0.5):
    """
    程序清单11-2 Apriori算法

    :param dataSet: 数据集
    :param minSupport: 最小支持度
    :return:
    """
    pass
