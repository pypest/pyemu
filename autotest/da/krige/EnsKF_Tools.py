import numpy as np


def cov(correlation_scale, stand_dev, distMat):
    """
    :param correlation_scale: correlation scale
    :param stand_dev:
    :param dist:  a mtarix of distances
    :return:
    """
    mat = stand_dev * stand_dev * np.exp(-1.0 * distMat / correlation_scale)
    return mat

def EnsKF_evenson(H = None, K = None, d = None, err_perc = 0.0, thr_perc = 0.1):

    # get matrices dimensions
    n,N = K.shape
    m = len(d)

    # compute anomolies
    H_dash = (H - H.mean(axis = 1).reshape(m,1))/np.power((N-1), 0.5)
    K_dash = (K - K.mean(axis = 1).reshape(n,1))/np.power((N-1), 0.5)

    # Add noise to measurements, error is assumed to be a percentage of prior std
    err_std = (err_perc/100.0) *  H.std(axis=1).reshape(m,1)
    D = d[np.newaxis].T + (err_std / 100.0) * np.random.randn(m, N)
    D_dash = D - H

    u, s, v = np.linalg.svd(H_dash)

    s_perc = 100.0 * s/np.sum(s)

    s_ = np.power(s, -2.0)
    s_[s_perc < thr_perc] = 0.0
    s_ = np.diag(s_)

    x1 = np.dot(s_, u.T)
    x2 = np.dot(x1, D_dash)
    x3 = np.dot(u, x2)
    x4 = np.dot(H_dash.T, x3)

    Ka = K + np.dot(K_dash, x4)

    return Ka

def svd_random_generator(covM=None, seed = 6543, nreal = 100):
    u, s, v = np.linalg.svd(covM)
    us = np.dot(u, np.power(np.diag(s), 0.5))
    usut = np.dot(us, v)
    np.random.seed(seed)
    z = np.random.randn(len(np.diag(s)), nreal)
    ens = np.dot(usut, z)

    return  ens

def pinv2(self, a, cut_percentage):
    """
    Compute the pseudoinverse of the observation covariance matrix
    :param a:
    :param cut_percentage:
    :return:
    """

    a = a.conjugate()
    u, s, vt = np.linalg.svd(a, 0)
    eig_val = np.copy(s)
    m = u.shape[0]
    n = vt.shape[1]
    s_sum = np.sum(s)
    cutoff = s_sum * cut_percentage / 100.0

    for i in range(np.min([n, m])):
        if s[i] > cutoff:
            s[i] = 1. / s[i]
        else:
            s[i] = 0.
    res = np.dot(np.transpose(vt), np.multiply(s[:, np.newaxis], np.transpose(u)))

    return res, eig_val
