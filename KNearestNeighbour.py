import numpy as np
import operator


def loadDataSet():
    group = np.array([
        [1.0, 1.1], [1.0, 1.0], [0.9, 0.8], [0.8, 0.8],
        [0.5, 0.5], [0.4, 0.5], [0.5, 0.6], [0.6, 0.6],
        [0.0, 0.0], [0.0, 0.1], [0.2, 0.1], [0.1, 0.1],
    ])
    labels = [
        'A', 'A', 'A', 'A',
        'B', 'B', 'B', 'B',
        'C', 'C', 'C', 'C'
    ]
    return group, labels


def KNN(X, dataSet, labels, k):
    """
    K Nearest Neighbour algorithm

    :param X:
    :param dataSet: array of dimensions m*n
    :param labels: training labels of
    :param k: the number of nearest neighbour
    :return: which class X belongs to
    """
    dataSetSize = dataSet.shape[0]

    # calculate distances between X and object in dataSet
    diffMat = np.tile(X, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    print(distances)

    # select the top k nearest neighbours
    sortedDistIndicies = np.argsort(distances)
    print(sortedDistIndicies)
    classCount = {}
    for i in range(k):
        # vote for label i
        voteILabel = labels[sortedDistIndicies[i]]
        classCount[voteILabel] = classCount.get(voteILabel, 0) + 1

    print(classCount)
    # sort the vote result
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)

    # return the best support class label
    return sortedClassCount[0][0]


data, labels = loadDataSet()
label = KNN((0.1, 0.1), data, labels, 3)
print(label)