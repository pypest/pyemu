import os
import numpy as np
from pyemu.utils.geostats import GeoStruct

"""
This class handles the generation of ensemble of realizations based on information in the parameter group
"""

class Parstat():

    def __init__(self, name=None, mu=0.0, sigma=1.0, coord=None, ext_generator=None):
        """
            mu : (n x 1) vector of the parameter group mean, where n is the number of parameters in the group.
            sigma: is
                - (scalar) standard devitation if mu is scalar
                - (n x1) is the standard deviation of each parameter, the correlation assumed to be zero
                - (n x n) full covariance matrix,
                - a file name that contains a full n by n covariance
                -

            coord: if a geostatistical structure is proivded then the coordinate of each of each parameter must be provided,
            the coordinate can be 1d, 2d,...nd
        """
        # todo: in case of numpy array check the shape and force to be n  x 1
        self.name = name
        self.mu = np.array(mu)
        self.sigma = np.array(sigma)
        self.coord = np.array(coord)
        self.ext_generator = ext_generator

    def generate_ensemble(self, nreal, seed=123):
        """
        a method for Par_pdf class that is used to generate random realization(s)
        :param nreal: number of realizations
        :return:
        """

        if not (self.ext_generator is None):
            # TODO: in case external random generator is used, the result should be read in as a table (preferably
            # TODO:     csv file, where each column is a realization)
            # use external random generator
            pass

        elif isinstance(self.sigma, np.ndarray):
            if self.sigma.ndim == 1:
                self.Cov = np.diag(self.sigma)
            elif self.sigma.ndim == 2:
                self.Cov = np.copy(self.sigma)
            else:
                raise ValueError("The covariance matrix has more than 2 dimensions....")

            # make sure the mean field (mu) has similiar dimension as covariance
            self.Mean = np.zeros_like(self.Cov[0, :]) + self.mu

            # generate the ensemble
            np.random.seed(seed=seed)
            self.Ens = np.random.multivariate_normal(self.mu, self.cov, nreal)

        elif isinstance(self.sigma, str):
            if os.path.isfile(self.sigma):
                # TODO: add code to read covaraince file
                pass  # read the covariance file
            else:
                raise ValueError("Error reading a file: {} file doe not exist....".format(self.sigma))

        elif isinstance(self.sigma, GeoStruct):
            # todo: use the geostat
            # generate uncorrelated realizations
            pass
        else:
            raise ValueError("Cannot an ensemble for parameter {}".format(str(self.name)))

        return self.Ens
