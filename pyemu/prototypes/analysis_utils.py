import numpy as np
from scipy.linalg import blas as blas
from scipy.linalg import lapack as lap
import matplotlib.pyplot as plt


def svdS(S, nrobs, nrens, nrmin, truncation):
    # Compute SVD of S=HA`  ->  U0, sig0
    lwork = 2 * max(3 * nrens + nrobs, 5 * nrens)

    S0 = S
    sig0 = 0.0
    U0, sig0, VT0, ierr = lap.dgesvd(S0)
    if (ierr != 0):
        ValueError('svdS: ierr from call dgesvd 0= {}'.format(ierr))

    sigsum = 0.0
    for i in range(nrmin):
        sigsum = sigsum + np.power(sig0[i], 2.0)

    sigsum1 = 0.0
    # Significant eigenvalues.
    nrsigma = 0
    for i in range(nrmin):
        if (sigsum1 / sigsum < truncation):
            nrsigma = nrsigma + 1
            sigsum1 = sigsum1 + np.power(sig0[i], 2.0)
        else:
            sig0[i:] = 0.0
            break

    print('      analysis svdS: dominant sing. values and'
          ' share {}, {}'.format(nrsigma, sigsum1 / sigsum))

    for i in range(nrsigma):
        if sig0[i] == 0:
            sig0[i] = 0
        else:
            sig0[i] = 1.0 / (sig0[i])

    return U0, sig0


def lowrankE(S, E, nrobs, nrens, nrmin, truncation):
    # Compute SVD of S=HA`  ->  U0, sig0
    U0, sig0 = svdS(S, nrobs, nrens, nrmin, truncation)

    # Compute X0=sig0^{*T} U0^T E
    # X0= U0^T R
    X0 = blas.dgemm(alpha=1, a=U0.T, b=E)
    for j in range(nrens):
        for i in range(nrmin):
            X0[i, j] = sig0[i] * X0[i, j]

    # Compute singular value decomposition  of X0(nrmin,nrens)
    eig = 0.0
    U1, eig, VT1, ierr = lap.dgesvd(X0)
    if (ierr != 0):
        ValueError('mod_anafunc (lowrankE): ierr from call dgesvd 1= {}'.format(ierr))

    for i in range(nrmin):
        eig[i] = 1.0 / (1.0 + np.power(eig[i], 2.0))

    # W = U0 * sig0^{-1} * U1
    for j in range(nrmin):
        for i in range(nrmin):
            U1[i, j] = sig0[i] * U1[i, j]

    X0 = blas.dgemm(alpha=1, a=U0.T, b=E)
    W = blas.dgemm(alpha=1, a=U0, b=U1)
    return W, eig


def lowrankCee(nrmin, nrobs, nrens, R, U0, sig0):
    # Compute B=sig0^{-1} U0^T R U0 sig0^{-1}

    # X0= U0^T R
    X0 = blas.dgemm(alpha=1.0, a=U0.T, b=R)

    # B= X0 U0
    B = blas.dgemm(alpha=1.0, a=X0, b=U0)
    if False:  # the two are the same
        cccc = np.dot(np.diag(sig0), B)
        B = np.dot(cccc, (np.diag(sig0)).T)
    if True:
        for j in range(nrmin):
            for i in range(nrmin):
                B[i, j] = sig0[i] * B[i, j]

        for j in range(nrmin):
            for i in range(nrmin):
                B[i, j] = sig0[j] * B[i, j]

    B = float(nrens - 1) * B

    return B


def eigsign(eig, truncation):
    nrobs = len(eig)
    np.savetxt('eigenvalues.dat', eig)
    # Significant eigenvalues
    sigsum = np.sum(eig)
    sigsum1 = 0.0
    nrsigma = 0
    for i in range(len(eig)):

        if (sigsum1 / sigsum < truncation):
            nrsigma = nrsigma + 1
            sigsum1 = sigsum1 + eig[i]
            eig[i] = 1.0 / eig[i]
        else:
            eig[i:] = 0.0
            break

        print('       analysis: Number of dominant eigenvalues: '
              '{} of {}'.format(nrsigma, nrobs))
        print('       analysis: Share (and truncation)        : {} , ({})'.format(sigsum1 / sigsum, truncation))
    return eig


def lowrankCinv(S, R, nrobs, nrens, nrmin, truncation):
    # Compute SVD of S=HA`  ->  U0, sig0
    U0, sig0 = svdS(S, nrobs, nrens, nrmin, truncation)

    # Compute B = sig0^{-1} U0^T R U0 sig0^{-1}
    # X0= U0^T R
    X0 = blas.dgemm(alpha=1.0, a=U0.T, b=R)
    # B= X0 U0
    B = blas.dgemm(alpha=1.0, a=X0, b=U0)
    if True:
        for j in range(nrmin):
            for i in range(nrmin):
                B[i, j] = sig0[i] * B[i, j]

        for j in range(nrmin):
            for i in range(nrmin):
                B[i, j] = sig0[j] * B[i, j]

    B = float(nrens - 1) * B

    # Compute eigenvalue decomposition  of B(nrmin,nrmin)
    eig, Z, info = lap.dsbev(B)

    # Compute inverse diagonal of (I+Lamda)
    for i in range(nrmin):
        eig[i] = 1.0 / (1.0 + eig[i])

    # W = U0 * sig0^{-1} * Z
    for j in range(nrmin):
        for i in range(nrmin):
            Z[i, j] = sig0[i] * Z[i, j]

    W = blas.dgemm(alpha=1, a=U0, b=Z)
    return W, eig


def genX3(nrobs, nrmin, eig, W, D):
    #X1 = np.empty(shape=(nrmin, nrobs))
    X1 = eig * W.T
    if False:
        for i in range(nrmin):
            for j in range(nrobs):
                X1[i, j] = eig[i] * W[j, i]

    #     X2=matmul(X1,D)
    X2 = blas.dgemm(alpha=1, a=X1, b=D)

    # X3=matmul(W,X2)
    X3 = blas.dgemm(alpha=1, a=W, b=X2)
    return X3


def meanX5(nrens, nrobs, nrmin, S, W, eig, innov):
    if (nrobs == 1):
        y1 = W[0, 0] * innov[0]
        y2 = eig[0] * y1[0]
        y3 = W[0, 0] * y2[0]
        y4 = y3[0] * S
    else:
        y1 = blas.dgemv(alpha=1, a=W.T, x=innov)
        y2 = eig * y1
        y3 = blas.dgemv(alpha=1, a=W, x=y2)
        y4 = blas.dgemv(alpha=1, a=S.T, x=y3)

    X5 = np.empty(shape=(nrens, nrens))
    for i in range(nrens):
        X5[:, i] = y4

    # X5=enN + (I - enN) X5  = enN + X5
    X5 = 1.0 / float(nrens) + X5

    return X5


def genX2(nrens, nrobs, idim, S, W, eig):
    # Generate X2= (I+eig)^{-0.5} * W^T * S
    X2 = blas.dgemm(alpha=1, a=W.T, b=S)
    for j in range(nrens):
        for i in range(idim):
            X2[i, j] = np.power((eig[i]), 0.5) * X2[i, j]
    return X2


def randrot(nrens):
    B = np.random.rand(nrens, nrens)
    A = np.random.rand(nrens, nrens)
    tiny = np.finfo(float).tiny
    Q = np.sqrt(-2. * np.log(A + tiny)) * np.cos(2. * np.pi * B)

    # QR factorization
    Q, sigma, work, ierr = lap.dgeqrf(a=Q)

    if (ierr != 0): ValueError('randrot: dgeqrf ierr= {}'.format(ierr))

    # Construction of Q
    Q, work, ierr = lap.dorgqr(a=Q, tau=sigma)
    if (ierr != 0): ValueError('randrot: dorgqr ierr={}'.format(ierr))

    return Q


def mean_preserving_rotation(Up, nrens):
    # Generates the mean preserving random rotation for the EnKF SQRT algorithm
    # using the algorithm from Sakov 2006-07.  I.e, generate rotation Up suceh that
    # Up*Up^T=I and Up*1=1 (all rows have sum = 1)  see eq 17.
    # From eq 18,    Up=B * Upb * B^T
    # B is a random orthonormal basis with the elements in the first column equals 1/sqrt(nrens)
    # Upb = | 1  0 |
    #       | 0  U |
    # where U is an arbitrary orthonormal matrix of dim nrens-1 x nrens-1  (eq. 19)

    # Generating the B matrix
    # Starting with a random matrix with the correct 1st column
    B = np.random.rand(nrens, nrens)
    B[:, 0] = 1.0 / np.sqrt(float(nrens))

    # with overwriting of B
    for k in range(nrens):
        R = np.sqrt(np.dot(B[:, k], B[:, k]))
        B[:, k] = B[:, k] / R[k, k]
        for j in range(k + 1, nrens):
            R[k, j] = np.dot(B[:, k], B[:, j])
            B[:, j] = B[:, j] - B[:, k] * R[k, j]

    # Creating the orthonormal nrens-1 x nrens-1 U matrix
    U = randrot(nrens - 1)

    # Creating the orthonormal nrens x nrens Upb matrix
    Upb = np.empty(shape=(nrens, nrens))
    Upb[1:nrens, 1:nrens] = U[0:nrens - 1, 0:nrens - 1]
    Upb[0, 0] = 1.0
    Upb[1:nrens, 0] = 0.0
    Upb[0, 1:nrens] = 0.0

    # Creating the random orthonormal mean preserving nrens x nrens Upb matrix: Up=B^T Upb B
    Q = blas.dgemm(alpha=1.0, a=B, b=Upb)
    Up = blas.dgemm(alpha=1.0, a=Q, b=B.T)

    return Up


def X5sqrt(X2, nrobs, nrens, nrmin, X5, update_randrot, mode):
    update_randrot = False

    print('      analysis (X5sqrt): update_randrot= {}'.format(update_randrot))
    if (update_randrot):
        ROT = np.empty((nrens, nrens))
        ROT = mean_preserving_rotation(ROT, nrens)
    else:
        ROT = np.eye(nrens, nrens)

    # SVD of X2
    lwork = 2 * max(3 * nrens + nrens, 5 * nrens)
    sig = 0.0
    U, sig, VT, ierr = lap.dgesvd(X2)
    if (ierr != 0):
        ValueError('X5sqrt: ierr from call dgesvd = {}'.format(ierr))

    if (mode == 21): nrmin = min(nrens, nrobs)
    isigma = np.ones(nrmin)

    for i in range(nrmin):
        if (sig[i] > 1.0): print('X5sqrt: WARNING (m_X5sqrt): sig > 1 {} {}'.format(i, sig[i]))
        isigma[i] = np.sqrt(max(1.0 - (sig[i] ** 2), 0.0))

    X3 = np.empty(shape=(nrens, nrens))
    if 0:
        for j in range(nrens):
            X3[:, j] = VT[j, :]
    X3 = VT.T

    for j in range(nrmin):
        X3[j, :] = X3[j, :] * isigma[j]

    # Multiply  X3* V' = (V*sqrt(I-sigma*sigma) * V' to ensure symmetric sqrt and
    # mean preserving rotation.   Sakov paper eq 13
    X33 = blas.dgemm(alpha=1.0, a=X3, b=VT)
    if update_randrot:
        X4 = blas.dgemm(alpha=1.0, a=X33, b=ROT)
    else:
        X4 = X33

    IenN = (-1.0 / float(nrens)) * np.ones(shape=(nrens, nrens))
    for i in range(nrens):
        IenN[i, i] = IenN[i, i] + 1.0

    X5 = blas.dgemm(alpha=1, a=IenN, b=X4, beta=1, c= X5)

    return X5


def multa(A, X, ndim, nrens, iblkmax):
    for ia in range(0, ndim, iblkmax):
        ib = min(ia + iblkmax - 1, ndim)
        v = A[ia:ib, 0:nrens]
        A[ia:ib, 0:nrens] = blas.dgemm(alpha=1, a=v, b=X)

    return A
