class TreeNode:
    def __init__(self, nameValue, numOcurr, parentNode):
        self.name = nameValue
        self.count = numOcurr
        self.nodeLink = None
        self.parent = parentNode
        self.children = {}

    def inc(self, numOccur):
        self.count += numOccur

    def display(self, ind=1):
        print(' ' * ind, self.name, self.count)
        for child in self.children.values():
            child.display(ind + 1)


def update_header(nodeToTest, targetNode):
    """
    更新头结点

    :param nodeToTest: 待测试节点
    :param targetNode: 要更新的头结点列表
    :return:
    """
    while nodeToTest.nodeLink is not None:
        nodeToTest = nodeToTest.nodeLink
    nodeToTest.nodeLink = targetNode


def update_tree(records, inTree, headerTable, count):
    """
    更新FP树

    使用循环迭代代替原书中的尾递归

    :param records: 有序频繁记录列表
    :param inTree: 树根节点
    :param headerTable: 头指针表
    :param count: 出现频率
    :return:
    """
    items = records
    root = inTree
    while len(items) > 1:
        if items[0] in root.children:
            root.children[items[0]].inc(count)
        else:
            # 创建子节点
            root.children[items[0]] = TreeNode(items[0], count, inTree)
            # 更新头指针表
            if headerTable[items[0]][1] is None:
                headerTable[items[0]][1] = root.children[items[0]]
            else:
                update_header(headerTable[items[0]][1], root.children[items[0]])
        # 在剩下的项集中迭代
        items.pop(0)
        root = root.children[items[0]]
        # iterate on remaining items: replaced tail-recursion with loop-iterating
        # if len(items) > 1:
        #     updateTree(items[1:], inTree.children[items[0]], headerTable, count)


def create_tree(dataSet, minSup=1):
    """
    程序清单 12-2 FP树构建函数

    :param dataSet:
    :param minSup:
    :return:
    """
    headerTable = {}
    for trans in dataSet:
        for item in trans:
            headerTable[item] = headerTable.get(item, 0) + dataSet[trans]

    # 1-移出不满足最小支持度的元素项
    for k in headerTable.keys():
        if headerTable[k] < minSup:
            del (headerTable[k])

    # 转换频繁项集
    freqItemSet = set(headerTable.keys())

    # 2-如果没有元素项满足要求，则退出
    if (len(freqItemSet)) == 0: return None, None

    # create root-node contains empty set
    for k in headerTable:
        headerTable[k] = [headerTable[k], None]

    retTree = TreeNode('Null Set', 1, None)
    for tranSet, count in dataSet.items():
        # 3-根据全局频率对每个事务中的元素进行排序
        localD = {}
        for item in tranSet:
            # 将每一个满足最小支持度的记录项添加到localD
            if item in freqItemSet:
                localD[item] = headerTable[item][0]
        # 4-使用排序后的频率项对树进行填充
        if (len(localD)) > 0:
            orderedItems = [v[0] for v in sorted(localD.items(), key=lambda p: p[1], reverse=True)]
            update_tree(orderedItems, retTree, headerTable, count)

    return retTree, headerTable


def loadSimpleDataSet():
    """
    创建简单数据集

    :return: 数据集
    """
    dataSet = [
        ['r', 'z', 'h', 'j', 'p'],
        ['z', 'y', 'x', 'w', 'v', 'u', 't', 's'],
        ['z'],
        ['r', 'x', 'n', 'o', 's'],
        ['y', 'r', 'x', 'z', 'q', 't', 'p'],
        ['y', 'z', 'x', 'e', 'q', 's', 't', 'm']
    ]

    return dataSet


def createInitSet(dataSet):
    """
    初始数据集

    :param dataSet: 原始数据集
    :return: 初始数据集
    """
    retDict = {}
    for trans in dataSet:
        retDict[frozenset(trans)] = 1
    return retDict


dataSet = loadSimpleDataSet()
initSet = createInitSet(dataSet)
fpTree, headerTable = create_tree(initSet, 3)
fpTree.display()
