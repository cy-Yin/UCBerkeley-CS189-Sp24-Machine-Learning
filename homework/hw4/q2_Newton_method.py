import numpy as np

x1 = np.array([0.2, 3.1, 1])
x2 = np.array([1.0, 3.0, 1])
x3 = np.array([-0.2, 1.2, 1])
x4 = np.array([1.0, 1.1, 1])

y1 = 1
y2 = 1
y3 = 0
y4 = 0

X = np.array([x1, x2, x3, x4])
y = np.array([y1, y2, y3, y4])

w0 = np.array([-1, 1, 0])

# sigmoid function: s(y) =  1 / (1 + exp(-y))
# Newton's method: w <- w - (Hessian)^-1 * Jacobi
# Here according to q1_1 and q1_2,
# Jacobi = X^T (s - y)
# Hessian = X^T diag(s_i * (1 - s_i)) X

s0 = 1 / (1 + np.exp(- X @ w0))
print("s0 = ", s0)

Hessian_0 = X.T @ np.diag(s0 * (1 - s0)) @ X
Jacobi_0 = X.T @ (s0 - y)
w1 = w0 - np.linalg.inv(Hessian_0) @ Jacobi_0
print("w1 = ", w1)

s1 = 1 / (1 + np.exp(-X @ w1))
print("s1 = ", s1)

Hessian_1 = X.T @ np.diag(s1 * (1 - s1)) @ X
Jacobi_1 = X.T @ (s1 - y)
w2 = w1 - np.linalg.inv(Hessian_1) @ Jacobi_1
print("w2 = ", w2)