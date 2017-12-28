def loadDataSet():
    return [
        [1, 3, 4],
        [2, 3, 5],
        [1, 2, 3, 5],
        [2, 5]
    ]


def createC1(dataSet):
    """
    程序清单11-1 Apriori算法中的辅助函数

    :param dataSet:
    :return: 单个物品的项集列表
    """
    C1 = []
    for transaction in dataSet:
        for item in transaction:
            if not [item] in C1:
                C1.append(item)
    C1.sort()
    # 1-对C1中每个项构建不变集合
    return map(frozenset, C1)


def scanD(dataSet, Ck, minSupport):
    """
    程序清单11-1 Apriori算法的辅助函数

    :param dataSet: 数据集
    :param Ck: 候选项集列表，Candidate K
    :param minSupport: 感兴趣项集的最小支持度
    :return: 满足最小支持度的项集，频繁项集的支持度
    """
    subsetCnt = {}
    for transaction in dataSet:
        for item in Ck:
            if item.issubset(transaction):
                if subsetCnt.has_key(item):
                    subsetCnt[item] += 1
                else:
                    subsetCnt[item] = 1
    numItems = float(len(dataSet))  # 数据集大小
    L1 = []  # 将满足最小支持度的项集构成集合L1
    supportData = {}  # 支持度的字典
    # 2-计算所有项集的支持度
    for key in subsetCnt:
        support = subsetCnt[key] / numItems
        if support > minSupport:
            L1.insert(0, key)
        supportData[key] = support
    return L1, supportData


def aprioriGen(Lk, k):
    """
    程序清单11-2 创建候选项集Ck

    :param Lk: 频繁项集列表Lk
    :param k: 项集元素个数
    :return: 候选项集Ck
    """
    Ck = []
    lenLk = len(Lk)
    for i in range(lenLk):
        for j in range(i + 1, lenLk):
            # 1-前k-2个项相同时，将两个集合合并
            L1 = list(Lk[i])[:k - 2], L2 = list(Lk[j])[:k - 2]
            L1.sort(), L2.sort()
            if L1 == L2:
                Ck.append(Lk[i] | Lk[j])
    return Ck


def apriori(dataSet, minSupport=0.5):
    """
    程序清单11-2 Apriori算法

    :param dataSet: 数据集
    :param minSupport: 最小支持度
    :return:
    """
    C1 = createC1(dataSet)
    D = map(set, dataSet)
    L1, supportData = scanD(dataSet, C1, minSupport)
    L = [L1]
    k = 2
    while len(L[k - 2]) > 0:
        Ck = aprioriGen(L[k - 2], k)
        # 2-扫描数据集，从Ck得到Lk
        Lk, SupK = scanD(D, Ck, minSupport)
        supportData.update(SupK)
        L.append(Lk)
        k += 1
    return L, supportData


def calcConf(freqSet, H, supportData, br1, minConf=0.7):
    """
    程序清单11-3 计算规则的可信度

    :param freqSet: 频繁项集
    :param H:
    :param supportData:
    :param br1:
    :param minConf:
    :return:
    """
    pass


def rulesFromConseq(freqSet, H, supportData, br1, minConf=0.7):
    """
    程序清单11-3

    :param freqSet: 频繁项集
    :param H: 可以出现在规则右部的元素列表H
    :param supportData:
    :param br1:
    :param minConf:
    :return:
    """
    pass


def generateRules(L, supportData, minConf=0.7):
    """
    程序清单11-3 关联规则生成函数

    :param L: 频繁项集列表
    :param supportData: 频繁项集支持度字典
    :param minConf: 最小可信度阈值
    :return: 一个包含可信度的规则列表
    """
    pass
