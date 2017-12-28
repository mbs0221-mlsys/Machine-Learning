import operator
import numpy as np


class KNN():
    def __init__(self, n_neighbours):
        self.dataSet = None
        self.n_neighbours = n_neighbours

    def fit(self, data_set):
        self.dataSet = data_set

    def predict(self, X):
        """
        K Nearest Neighbour algorithm

        :param X:
        :return: which class X belongs to
        """
        data, labels = self.dataSet
        m, n = np.shape(data)

        # calculate distances between X and object in dataSet
        diffMat = np.tile(X, (m, 1)) - data
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


def loadFile(file, delim=' '):
    fp = open(file)
    strArr = fp.readlines()
    data, labels = [], []
    for line in strArr:
        words = line.strip().split(delim)
        data.append(np.array(words[1:], dtype=float))
        labels.append(words[0])
    return data, labels


data, labels = loadFile('dataset/wine/wine.data', delim=',')
X = [
    14.37, 11.95, 12.5, 16.8, 113,
    13.85, 13.49, 10.24, 12.18, 17.8,
    10.86, 13.45, 1480
]

clf = KNN(12)
clf.fit((data, labels))
result = clf.predict(X)
print('The prediction of X is ', result)
