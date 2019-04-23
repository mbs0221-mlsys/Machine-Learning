import cvxpy as cvx

H = cvx.Variable(shape=2,2)
x = cvx.Variable()
p = cvx.Problem(cvx.Minimize(x), [])
