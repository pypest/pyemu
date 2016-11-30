from __future__ import print_function, division
import os
import threading
import time
import numpy as np
import pandas as pd
import pyemu
from pyemu.en import ParameterEnsemble,ObservationEnsemble
from pyemu.mat import Cov
from pyemu.pst import Pst

"""this is a prototype ensemble smoother based on the LM-EnRML
algorithm of Chen and Oliver 2013.  It requires the pest++ "sweep" utility
 to propagate the ensemble forward.
"""

class EnsembleSmoother():

    def __init__(self,pst,parcov=None,obscov=None,num_slaves=0,use_approx=True,
                 restart_iter=0):
        self.num_slaves = int(num_slaves)
        self.use_approx = bool(use_approx)
        self.paren_prefix = ".parensemble.{0:04d}.csv"
        self.obsen_prefix = ".obsensemble.{0:04d}.csv"
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
        self.restart = False

        if restart_iter > 0:
            self.restart_iter = restart_iter
            paren = self.pst.filename+self.paren_prefix.format(restart_iter)
            assert os.path.exists(paren),\
                "could not find restart par ensemble {0}".format(paren)
            obsen0 = self.pst.filename+self.obsen_prefix.format(0)
            assert os.path.exists(obsen0),\
                "could not find restart obs ensemble 0 {0}".format(obsen0)
            obsen = self.pst.filename+self.obsen_prefix.format(restart_iter)
            assert os.path.exists(obsen),\
                "could not find restart obs ensemble {0}".format(obsen)
            self.restart = True


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
        if self.restart:
            print("restarting...ignoring num_reals")
        if self.restart:
            df = pd.read_csv(self.pst.filename+self.paren_prefix.format(self.restart_iter))
            self.parensemble_0 = ParameterEnsemble.from_dataframe(df=df,pst=self.pst)
            self.parensemble = self.parensemble_0.copy()
            df = pd.read_csv(self.pst.filename+self.obsen_prefix.format(0))
            self.obsensemble_0 = ObservationEnsemble.from_dataframe(df=df,pst=self.pst)
            df = pd.read_csv(self.pst.filename+self.obsen_prefix.format(self.restart_iter))
            self.obsensemble = ObservationEnsemble.from_dataframe(df=df,pst=self.pst)
            assert self.parensemble.shape[0] == self.obsensemble.shape[0]
            self.num_reals = self.parensemble.shape[0]

        else:
            self.num_reals = int(num_reals)
            self.parensemble_0 = ParameterEnsemble(self.pst)
            self.parensemble_0.draw(cov=self.parcov,num_reals=num_reals)
            self.parensemble_0.enforce()
            self.parensemble = self.parensemble_0.copy()
            self.parensemble_0.to_csv(self.pst.filename+".parensemble.0000.csv")
            self.obsensemble_0 = ObservationEnsemble(self.pst)
            self.obsensemble_0.draw(cov=self.obscov,num_reals=num_reals)
            self.obsensemble = self.obsensemble_0.copy()
            self.obsensemble_0.to_csv(self.pst.filename+".obsensemble.0000.csv")

        # if using the approximate form of the algorithm, let
        # the parameter scaling matrix be the identity matrix
        if self.use_approx:
            self.half_parcov_diag = 1.0
        else:
            # if self.parcov.isdiagonal:
            #     self.half_parcov_diag = self.parcov.sqrt.inv
            # else:
            #     self.half_parcov_diag = Cov(x=np.diag(self.parcov.x),
            #                                 names=self.parcov.col_names,
            #                                 isdiagonal=True).inv.sqrt
            self.half_parcov_diag = 1.0
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
        delta = self.half_parcov_diag * delta.T
        delta *= (1.0 / np.sqrt(float(self.num_reals - 1.0)))
        return delta

    def _calc_delta_obs(self):
        '''
        calc the scaled observation ensemble differences from the mean
        '''

        mean = np.array(self.obsensemble.mean(axis=0))
        delta = self.obsensemble.as_pyemu_matrix()
        for i in range(self.num_reals):
            delta.x[i,:] -= mean
        delta = self.obscov.inv.sqrt * delta.T
        delta *= (1.0 / np.sqrt(float(self.num_reals - 1.0)))
        return delta

    def _calc_obs(self):
        '''
        propagate the ensemble forward...
        '''
        self.parensemble.to_csv(os.path.join("sweep_in.csv"))
        if self.num_slaves > 0:
            port = 4004
            def master():
                os.system("sweep {0} /h :{1}".format(self.pst.filename,port))
            master_thread = threading.Thread(target=master)
            master_thread.start()
            time.sleep(1.5) #just some time for the master to get up and running to take slaves
            pyemu.utils.start_slaves("template","sweep",self.pst.filename,self.num_slaves,slave_root='.',port=port)
            master_thread.join()
        else:
            os.system("sweep {0}".format(self.pst.filename))

        obs = ObservationEnsemble.from_csv(os.path.join('sweep_out.csv'))
        obs.columns = [item.lower() for item in obs.columns]
        self.obsensemble = ObservationEnsemble.from_dataframe(df=obs.loc[:,self.obscov.row_names],pst=self.pst)
        self.obsensemble.to_csv(self.pst.filename+self.obsen_prefix.format(self.iter_num))

        return

    @property
    def current_lambda(self):
        return 1.0

    def update(self):
        self.iter_num += 1
        if not self.__initialized:
            raise Exception("must call initialize() before update()")
        if self.restart and self.iter_num == 1:
            pass
        else:
            self._calc_obs()
        scaled_delta_obs = self._calc_delta_obs()
        scaled_delta_par = self._calc_delta_par()

        u,s,v = scaled_delta_obs.pseudo_inv_components()

        obs_diff = self.obsensemble.as_pyemu_matrix() -\
                   self.obsensemble_0.as_pyemu_matrix()

        scaled_ident = Cov.identity_like(s) * (self.current_lambda+1.0)
        scaled_ident += s**2
        scaled_ident = scaled_ident.inv
        #scaled_ident.autoalign = False

        # build up this matrix as a single element so we can apply
        # localization
        upgrade_1 = -1.0 * (self.half_parcov_diag * scaled_delta_par) *\
                    v * s * scaled_ident * u.T

        # apply localization
        print(upgrade_1.shape)

        # apply residual information
        upgrade_1 *= (self.obscov.inv.sqrt * obs_diff.T)

        upgrade_1 = upgrade_1.to_dataframe()
        upgrade_1.index.name = "parnme"
        upgrade_1 = upgrade_1.T
        upgrade_1.to_csv(self.pst.filename+".upgrade_1.{0:04d}.csv".\
                           format(self.iter_num))
        self.parensemble += upgrade_1

        if not self.use_approx and self.iter_num > 1:
            par_diff = (self.parensemble - self.parensemble_0).\
                as_pyemu_matrix().T
            x4 = self.Am.T * self.half_parcov_diag * par_diff
            x5 = self.Am * x4
            x6 = scaled_delta_par.T * x5
            x7 = v * scaled_ident * v.T * x6
            upgrade_2 = -1.0 * (self.half_parcov_diag *
                               scaled_delta_par * x7).to_dataframe()
            # upgrade_2 = -1.0 * self.half_parcov_diag * scaled_delta_par
            # upgrade_2.autoalign = False
            # upgrade_2 *= v
            # upgrade_2.autoalign = False
            # upgrade_2 *= scaled_ident
            # upgrade_2.autoalign = False
            # upgrade_2 *= v.T * scaled_delta_par.T
            # upgrade_2.autoalign = False
            # upgrade_2 *= self.Am * self.Am.T
            # upgrade_2.autoalign = False
            # upgrade_2 *= self.half_parcov_diag * par_diff
            #upgrade_2 = upgrade_2.to_dataframe()

            upgrade_2.index.name = "parnme"
            upgrade_2.T.to_csv(self.pst.filename+".upgrade_2.{0:04d}.csv".\
                               format(self.iter_num))
            self.parensemble += upgrade_2.T
        self.parensemble.enforce()
        self.parensemble.to_csv(self.pst.filename+self.paren_prefix.\
                                format(self.iter_num))





