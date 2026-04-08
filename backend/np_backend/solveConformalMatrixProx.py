import numpy as np
from numpy.linalg import svd
from numpy.linalg import eigh


def projectOnToSimplex(y, C):
    """
    Solves the following convex problem
    Minimize 1/2 * ||x - y ||_2^2
         St. x >= 0
         St. 1^T x <= C
    """
    d = len(y)
    if C < 0:
        raise ValueError("The radius of the simplex is not large enough")
    y_sorted = np.sort(y)[::-1]
    y_cumsum = y_sorted.cumsum()
    if (y_cumsum[d - 1] <= C) & (y_sorted[d - 1] >= 0):
        return y
    if y_sorted[y_sorted > 0].sum() <= C:
        return np.maximum(y, 0.0)

    taus = (y_cumsum - C) / (np.arange(1, d + 1))
    ind = np.where(y_sorted - taus < 0)[0]
    if len(ind) == 0:
        ind = d - 1
    else:
        ind = ind[0] - 1
    tau = (y_cumsum[ind] - C) / (ind + 1)
    return np.maximum(y - tau, 0)


# Solves the following convex problem
# Minimize 1/2 * ||x - y ||_F^2
#          St. x >= eps
#          St. 1^T x <= C
def projectOnToVectorL1Ball(y, epsilon, C):
    d = len(y)
    return projectOnToSimplex(y - epsilon, C - epsilon * d) + epsilon


# Solves the following convex problem
# Minimize 1/2 * ||N - M ||_F^2
#          St. N >= eps * Id
#          St. tr(N) <= C
def projectOnToMatrixL1Ball(M, epsilon, C, isSymmetric=None):
    if isSymmetric is None:
        isSymmetric = np.allclose(M, M.T)
    if isSymmetric:
        s, u = eigh(M)
        vh = u.T
    else:
        u, s, vh = np.linalg.svd(M, full_matrices=False)
    d = projectOnToVectorL1Ball(s, epsilon, C)
    return (u * d) @ vh


# Solves the following problem
# Minimize lbda/2 * ||M - M_0||^2 + max(<N,M>-t, 0.0)
#          St. M >= epsilon * I
#          St. tr(M) <= C
def solveAProxStep(M_0, N, epsilon, C, t, lbda=1.0, isSymmetric=None, tol=1e-6):
    if isSymmetric is None:
        isSymmetric = (np.allclose(M_0, M_0.T)) & (np.allclose(N, N.T))

    N_norm = np.linalg.norm(N)**2
    N_M_0_prod = (N * M_0).sum()

    alpha_min = 0.0
    alpha_max = 1.0 / lbda
    alpha = 0.5 * alpha_min + 0.5 * alpha_max
    while alpha_max - alpha_min > tol:
        M_alpha = projectOnToMatrixL1Ball(
            M_0 - alpha * N, epsilon, C, isSymmetric=isSymmetric
        )
        alpha_grad = np.sum(M_alpha * N) - t
        if alpha_grad > 0:
            alpha_min = alpha
            alpha = 0.5 * alpha_min + 0.5 * alpha_max
        else:
            alpha_max = alpha
            alpha = 0.5 * alpha_min + 0.5 * alpha_max
    return M_alpha


# Solve the following convex problem
# Minimize lbda/2 * ||gamma - gamma_0||^2 + alpha * gamma^T  x + max(err - gamma^T x,0)
def solveAProxLinearQuantile(x, gamma_0, err, alpha=0.05, lbda=1.0, tol=1e-8):
    if np.linalg.norm(x) < tol:
        return gamma_0
    mu = alpha + lbda / (x @ x) * (err - gamma_0 @ x)
    return gamma_0 - (alpha - np.clip(mu, 0.0, 1.0)) / lbda * x


#     if mu < 0:
#         return gamma_0 - alpha / lbda * x
#     else if mu > 1:
#         return gamma_0 - (alpha - 1.0) / lbda * x
#     else:
#         return gamma_0 - (1)