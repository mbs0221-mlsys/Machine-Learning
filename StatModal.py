import numpy as np


class StatModal:
    def fit(self):
        pass

    def predict(self):
        pass


class Measure:
    def __init__(self, name):
        self.name = name

    @staticmethod
    def __eular__(X: np.array, Y: np.array):
        diffMat = X - Y
        sqDiffMat = diffMat ** 2
        sqDistances = sqDiffMat.sum(axis=1)
        return sqDistances ** 0.5

    def measure(self, X, Y):
        if self.name == 'euler':
            self.__eular__(X, Y)
        else:
            return self.__eular__(X, Y)
