"""
This is a direct translation of Evenson's code
https://enkf.nersc.no/Code/Analysis/analysis.F90

"""

from .analysis_utils import  *


def analysis(A, R = None, E = None, S = None, D = None, innov = None,
              verbose = True, truncation = 0.99, mode = 12 ,
              update_randrot = False):

    # analysis(A, R, E, S, D, innov, ndim, nrens, nrobs, verbose, truncation, mode, update_randrot)
    # Computes the analysed ensemble for A using the EnKF or square root schemes.
    # ndim   ! dimension of model state
    # nrens   ! number of ensemble members
    # nrobs    ! number of observations
    # A(ndim, nrens)    ! ensemble matrix
    # R(nrobs, nrobs)   ! matrix holding R(only used if mode =?1 or ?2)
    # D(nrobs, nrens)   ! matrix holding perturbed measurments D = D - HA
    # E(nrobs, nrens)   ! matrix holding perturbations(only used if mode =?3)
    # S(nrobs, nrens)   ! matrix holding HA`
    # innov(nrobs)      ! vector holding d - H * mean(A)
    # verbose           ! Printing some diagnostic output
    # truncation        ! The ratio of variaince retained in pseudo inversion(0.99)
    # mode              ! first integer means(EnKF=1, SQRT=2) !
    #                     Second integer is pseudo inversion
    #                   !  1 = eigen  value pseudo inversion of SS '+(N-1)R ,
    #                      2 = SVD subspace pseudo inversion of SS '+(N-1)R !  3 = SVD
    #                          subspace pseudo inversion of SS'+EE'
    # update_randrot   ! Normally true; false for all but first grid point
    #                  ! updates when using local analysis since all grid
    #                  ! points need to use the same rotation.

    # Get dimensions
    ndim, nrens = A.shape
    nrobs = D.shape[0]
    innov = np.mean(D, axis=1)

    lreps = False
    if (verbose):
        print('analysis: verbose is on')

    # ! Pseudo inversion of C = SS ' +(N-1)*R
    print('      analysis: Inversion of C:')

    if (nrobs == 1):
        # single observation
        nrmin = 1
        eig = np.dot(S, S.T) + float(nrens - 1) * R
        eig = 1.0 / eig
        W = 1.0

    else:
        # multiple observations
        if mode in [11, 21]:
            nrmin = nrobs

            # Evaluate R = S * S ` + (nrens - 1) * R
            R = blas.dgemm(alpha=1, a=S, b=S.T, beta=(nrens-1),  c=R)

            # ! Compute eigenvalue decomposition of R -> W * eig * W
            eig, W = np.linalg.eigh(R)
            eig = eigsign(eig, truncation)

        elif mode in [12, 22]:
            nrmin = min(nrobs, nrens)
            W, eig = lowrankCinv(S, R, nrobs, nrens, nrmin, truncation)

        elif mode in [13, 23]:
            nrmin = min(nrobs, nrens)
            W, eig = lowrankE(S, E, nrobs, nrens, nrmin, truncation)
        else:
            print('analysis: Unknown mode: {}'.format(mode))

    # Generation of X5 (or representers in EnKF case with few measurements)
    print('      analysis: Generation of X5:')

    if mode in [11, 12, 13]:
        if (nrobs > 1):
            X3 = genX3(nrobs, nrmin, eig, W, D)
        else:
            X3 = D * eig

        if (2 * ndim * nrobs < nrens * (nrobs + ndim)):
            #       Code for few observations ( m<nN/(2n-N) )
            if (verbose): print('analysis: Representer approach is used')
            lreps = True
            #        Reps=matmul(A,transpose(S))
            Reps = blas.dgemm(alpha=1, a=A, b=S.T)

        else:
            if (verbose):
                print('analysis: X5 approach is used')
            X5 = blas.dgemm(alpha=1, a=S.T, b=X3)

            for i in range(nrens):
                X5[i, i] = X5[i, i] + 1.0



    elif mode in [21, 22, 23]:
        # Mean part of X5
        X5 = meanX5(nrens, nrobs, nrmin, S, W, eig, innov)

        # Generating X2
        X2 = np.empty(shape=(nrmin, nrens))
        X2 = genX2(nrens, nrobs, nrmin, S, W, eig)

        # Generating X5 matrix
        x5 = X5sqrt(X2, nrobs, nrens, nrmin, X5, update_randrot, mode)

    else:
        ValueError('analysis: Unknown flag for mode: {}'.format(mode))

    # Final ensemble update
    print('      analysis: Final ensemble update:')
    if (lreps):
        #A= A+ multa(Reps,X3)
        #A = blas.dgemm(alpha=1.0, a=Reps, b=X3)
        #iblkmax = min(ndim, 200)
        A = blas.dgemm(alpha=1.0, a=Reps, b=X3, beta=1, c= A)
        np.savetxt('X3.uf', X3)
        np.savetxt('S_.uf', S)
    else:
        iblkmax = min(ndim, 200)
        A = multa(A, X5, ndim, nrens, iblkmax)
        np.savetxt('X5.uf', X5)
        np.savetxt('S_.uf', S)

    return A


def analysis2(K = None, H = None, R =None, D= None, E = None,
             verbose=True, truncation=0.99, mode=22,
             update_randrot=False):
    # K(ndim, nrens)  is parameter/state to be updated
    # H(nrobs, nrens) is observable parameters/states
    # Get dimensions


    A = np.vstack((H, K))
    S = H - H.mean(axis = 1)[:, np.newaxis]
    ndim, nrens = A.shape
    nrobs = D.shape[0]
    D_dash = D - H

    Aa = analysis(A, R=R, E=E, S=S, D= D_dash, innov=None,
             verbose=True, truncation= truncation, mode=mode,
             update_randrot=False)
    Aaa = Aa[nrobs:,:]
    return Aaa

