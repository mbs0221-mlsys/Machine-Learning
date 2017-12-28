import numpy as np
import operator

from StatModal import StatModal


class KNN(StatModal):
    def __init__(self, n_neighbours):
        self.dataSet = None
        self.n_neighbours = n_neighbours

    def fit(self, dataSet):
        self.dataSet = dataSet

    def predict(self, X):
        """
        K Nearest Neighbour algorithm

        :param X:
        :param dataSet: array of dimensions m*n
        :param labels: training labels of
        :param k: the number of nearest neighbour
        :return: which class X belongs to
        """
        dataSetSize = self.dataSet.shape[0]

        # calculate distances between X and object in dataSet
        diffMat = np.tile(X, (dataSetSize, 1)) - self.dataSet
        sqDiffMat = diffMat ** 2
        sqDistances = sqDiffMat.sum(axis=1)
        distances = sqDistances ** 0.5

        # select the top k nearest neighbours
        sortedDistIndices = np.argsort(distances)
        classCount = {}
        for i in range(self.n_neighbours):
            # vote for label i
            voteILabel = labels[sortedDistIndices[i]]
            classCount[voteILabel] = classCount.get(voteILabel, 0) + 1

        # sort the vote result
        sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)

        # return the best support class label
        return sortedClassCount[0][0]


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


data, labels = loadDataSet()
label = KNN((0.1, 0.1), data, labels, 3)
print(label)
