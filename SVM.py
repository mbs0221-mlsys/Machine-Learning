import numpy as np
import math


def svkernel(X, A, ker, sigma):
    m, n = np.shape(X)
    K = np.zeros((m, 1))
    if ker == 'linear':
        K = X * A.T
    elif ker == 'rbf':
        for i in range(m):
            deltaRow = X[i, :] - A
            K[i] = deltaRow * deltaRow.T
        K = np.exp(K / (-1 * sigma ** 2))
    else:
        raise NameError('Houston, We have a problem!')

    return K


def SVM(X, Y, kernel, C):
    m, n = np.shape(X)
    epsilon = 1e-10

    # Construct the H matrix and c vector
    H = np.zeros(m, m)
    for i in range(m):
        for j in range(m):
            H[i][j] = svkernel(X[i], X[j], kernel[0], sigma=kernel[1])

    c = -np.ones(n, 1)

    # Add small amount of zeros order regularization to avoid problem when Hessian is badly conditioned
    cond = np.linalg.cond(H)
    if abs(cond) > 1e+10:
        print("Hessian is badly conditioned, regularizing...")
        print("Old condition number is: %4.2f" % (cond))
        H = H + 1e-8 * np.eye(m, m)
        cond = np.linalg.cond(H)
        print("New condition number is: %4.2f" % (cond))

    # Set up the parameter for the Optimisation problem
    vlb = np.zeros(n, 1) # Set the bounds: alpha >= 0
    vub = C * np.ones(n, 1) #              alpha <=C
    x0 = [] # The start point is  [0, ..., 0]
    # neqcstr =


