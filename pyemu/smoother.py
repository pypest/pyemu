from __future__ import print_function, division
import os
import copy
import math
import numpy as np
import pandas as pd
from pyemu.en import ParameterEnsemble,ObservationEnsemble
from pyemu.mat import Cov, Matrix
from pyemu.pst import Pst


class EnsembleSmoother():

    def __init__(self,pst,parcov=None,obscov=None):
        if isinstance(pst,str):
            pst = Pst(pst)
        assert isinstance(pst,Pst)
        self.pst = pst
        if parcov is not None:
            assert isinstance(parcov,Cov)
        else:
            parcov = Cov.from_parameter_data(self.pst)
        if obscov is not None:
            assert isinstance(obscov,Cov)
        else:
            obscov = Cov.from_observation_data(pst)

        self.parcov = parcov
        self.obscov = obscov

        self.__initialized = False
        self.num_reals = 0
        self.half_parcov_diag = None
        self.half_obscov_diag = None
        self.delta_par_prior = None
        self.iter_num = 0

    def initialize(self,num_reals):
        '''
        (re)initialize the process
        '''
        self.num_reals = int(num_reals)
        self.parensemble_0 = ParameterEnsemble(self.pst)
        self.parensemble_0.draw(cov=self.parcov,num_reals=num_reals)
        self.parensemble = self.parensemble_0.copy()
        self.parensemble_0.to_csv("parensemble.0.csv")


        self.obsensemble_0 = ObservationEnsemble(self.pst)
        self.obsensemble_0.draw(cov=self.obscov,num_reals=num_reals)
        self.obsensemble = self.obsensemble_0.copy()
        self.obsensemble_0.to_csv("obsensemble.0.csv")

        if self.parcov.isdiagonal:
            self.half_parcov_diag = self.parcov.inv.sqrt
        else:
            self.half_parcov_diag = Cov(x=np.diag(self.parcov.x),
                                        names=self.parcov.col_names,
                                        isdiagonal=True).inv.sqrt
        #if self.obscov.isdiagonal:
        #self.half_obscov_inv = self.obscov.inv.sqrt
       # else:
        #    self.half_obscov_diag = Cov(x=np.diag(self.obscov.x),
        #                                names=self.obscov.col_names,
        #                                isdiagonal=True)

        self.delta_par_prior = self._calc_delta_par()

        u,s,v = self.delta_par_prior.pseudo_inv_components()
        self.Am = u * s.inv


        self.__initialized = True

    def _calc_delta_par(self):
        '''
        calc the scaled parameter ensemble differences from the mean
        '''
        mean = np.array(self.parensemble.mean(axis=0))
        delta = self.parensemble.as_pyemu_matrix()
        for i in range(self.num_reals):
            delta.x[i,:] -= mean
        #delta = Matrix(x=(self.half_parcov_diag * delta.transpose()).x,
        #               row_names=self.parensemble.columns)
        delta = self.half_parcov_diag * delta.T
        return delta * (1.0 / np.sqrt(float(self.num_reals - 1.0)))

    def _calc_delta_obs(self):
        '''
        calc the scaled observation ensemble differences from the mean
        '''

        mean = np.array(self.obsensemble.mean(axis=0))
        delta = self.obsensemble.as_pyemu_matrix()
        for i in range(self.num_reals):
            delta.x[i,:] -= mean
        delta = self.obscov.inv.sqrt * delta.T
        return delta * (1.0 / np.sqrt(float(self.num_reals - 1.0)))

    def _calc_obs(self):
        '''
        propagate the ensemble forward...
        '''
        self.parensemble.to_csv(os.path.join("sweep_in.csv"))
        #os.chdir("smoother")
        #print(os.listdir('.'))
        os.system("sweep {0}".format(self.pst.filename))
        #os.chdir('..')
        obs = ObservationEnsemble.from_csv(os.path.join('sweep_out.csv'))
        obs.columns = [item.lower() for item in obs.columns]
        self.obsensemble = ObservationEnsemble.from_dataframe(df=obs.loc[:,self.obscov.row_names],pst=self.pst)
        #todo: modifiy sweep to be interactive...
        return

    @property
    def current_lambda(self):
        return 1.0

    def update(self):
        if not self.__initialized:
            raise Exception("must call initialize() before update()")

        delta_obs = self._calc_delta_obs()

        u,s,v = delta_obs.pseudo_inv_components()
        scaled_par_diff = self._calc_delta_par()
        obs_diff = self.obsensemble.as_pyemu_matrix() -\
               self.obsensemble_0.as_pyemu_matrix()
        scaled_ident = (self.current_lambda*Cov.identity_like(s) + s**2).inv

        x1 = u.T * self.obscov.inv.sqrt * obs_diff.T
        x1.autoalign = False
        x2 = scaled_ident * x1
        x3 = v * s * x2
        upgrade_1 = -1.0 *  (self.half_parcov_diag * scaled_par_diff *\
                             x3).to_dataframe()
        upgrade_1.index.name = "parnme"
        upgrade_1.T.to_csv("upgrade_1.{0}.csv".format(self.iter_num))
        self.parensemble += upgrade_1.T
        if self.iter_num > 1:
            par_diff = (self.parensemble - self.parensemble_0).\
                as_pyemu_matrix().T
            x4 = self.Am.T * self.half_parcov_diag * par_diff
            x5 = self.Am * x4
            x6 = scaled_par_diff.T * x5
            x7 = v * scaled_ident * v.T * x6
            upgrade_2 = -1.0 * (self.half_parcov_diag *
                                scaled_par_diff * x7).to_dataframe()
            upgrade_2.index.name = "parnme"
            upgrade_2.T.to_csv("upgrade_2.{0}.csv".format(self.iter_num))
            self.parensemble += upgrade_2.T
        self.parensemble.to_csv("parensemble.{0}.csv".format(self.iter_num))

        self._calc_obs()
        self.obsensemble.to_csv("obsensemble.{0}.csv".format(self.iter_num))
        self.iter_num += 1




