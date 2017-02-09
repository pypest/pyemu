from __future__ import print_function, division
import os
from datetime import datetime
import threading
import time
import numpy as np
import pandas as pd
import pyemu
from pyemu.en import ParameterEnsemble,ObservationEnsemble
from pyemu.mat import Cov,Matrix

from pyemu.pst import Pst

"""this is a prototype ensemble smoother based on the LM-EnRML
algorithm of Chen and Oliver 2013.  It requires the pest++ "sweep" utility
 to propagate the ensemble forward.

 TODO:
 handle fixed and tied pars
 handle "bunk" mod-sim equivs, like dry cells
 handle failed runs

"""

class EnsembleSmoother():

    def __init__(self,pst,parcov=None,obscov=None,num_slaves=0,use_approx=True,
                 restart_iter=0,submit_file=None):
        self.num_slaves = int(num_slaves)
        self.submit_file = submit_file
        self.use_approx = bool(use_approx)
        self.paren_prefix = ".parensemble.{0:04d}.csv"
        self.obsen_prefix = ".obsensemble.{0:04d}.csv"

        if isinstance(pst,str):
            pst = Pst(pst)
        assert isinstance(pst,Pst)
        self.pst = pst
        self.sweep_in_csv = pst.pestpp_options.get("sweep_parameter_csv_file","sweep_in.csv")
        self.sweep_out_csv = pst.pestpp_options.get("sweep_output_csv_file","sweep_out.csv")
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

    def initialize(self,num_reals,init_lambda=None):
        '''
        (re)initialize the process
        '''
        assert num_reals > 1
        # initialize the phi report csv
        self.phi_csv = open(self.pst.filename+".iobj.csv",'w')
        self.phi_csv.write("iter_num,total_runs,lambda,min,max,mean,median,std,")
        self.phi_csv.write(','.join(["{0:010d}".\
                                    format(i+1) for i in range(num_reals)]))
        self.phi_csv.write('\n')
        self.total_runs = 0
        # this matrix gets used a lot, so only calc once and store
        self.obscov_inv_sqrt = self.obscov.get(self.pst.nnz_obs_names).inv.sqrt
        if self.restart:
            print("restarting...ignoring num_reals")
            raise NotImplementedError()
            df = pd.read_csv(self.pst.filename+self.paren_prefix.format(self.restart_iter))
            self.parensemble_0 = ParameterEnsemble.from_dataframe(df=df,pst=self.pst)
            self.parensemble = self.parensemble_0.copy()
            df = pd.read_csv(self.pst.filename+self.obsen_prefix.format(0))
            self.obsensemble_0 = ObservationEnsemble.from_dataframe(df=df.loc[:,self.pst.nnz_obs_names],
                                                                    pst=self.pst)
            # this matrix gets used a lot, so only calc once
            self.obs0_matrix = self.obsensemble_0.as_pyemu_matrix()
            df = pd.read_csv(self.pst.filename+self.obsen_prefix.format(self.restart_iter))
            self.obsensemble = ObservationEnsemble.from_dataframe(df=df.loc[:,self.pst.nnz_obs_names],
                                                                  pst=self.pst)
            assert self.parensemble.shape[0] == self.obsensemble.shape[0]
            self.num_reals = self.parensemble.shape[0]

        else:
            self.num_reals = int(num_reals)
            self.parensemble_0 = ParameterEnsemble(self.pst)
            self.parensemble_0.draw(cov=self.parcov,num_reals=num_reals)
            self.parensemble_0.enforce()
            self.parensemble = self.parensemble_0.copy()
            self.parensemble_0.to_csv(self.pst.filename +\
                                      self.paren_prefix.format(0))
            self.obsensemble_0 = ObservationEnsemble(self.pst)
            self.obsensemble_0.draw(cov=self.obscov,num_reals=num_reals)
            #self.obsensemble = self.obsensemble_0.copy()

            # save the base obsensemble
            self.obsensemble_0.to_csv(self.pst.filename +\
                                      self.obsen_prefix.format(-1))
            self.obs0_matrix = self.obsensemble_0.nonzero.as_pyemu_matrix()

            # run the initial parameter ensemble
            self.obsensemble = self._calc_obs(self.parensemble)
            self.obsensemble.to_csv(self.pst.filename +\
                                      self.obsen_prefix.format(0))
        self.current_phi_vec = self._calc_phi_vec(self.obsensemble)
        self._phi_report(self.current_phi_vec,0.0)
        self.last_best_mean = self.current_phi_vec.mean()
        self.last_best_std = self.current_phi_vec.std()
        if init_lambda is not None:
            self.current_lambda = float(init_lambda)
        else:
            #following chen and oliver
            x = self.last_best_mean / (2.0 * float(self.obsensemble.shape[1]))
            self.current_lambda = 10.0**(np.floor(np.log10(x)))

        # if using the approximate form of the algorithm, let
        # the parameter scaling matrix be the identity matrix
        # jwhite - dec 5 2016 - using the actual parcov inv
        # for upgrades seems to be pushing parameters around
        # too much.  for now, just not using it, maybe
        # better choices of lambda will tame it
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
            self.delta_par_prior = self._calc_delta_par(self.parensemble_0)
            u,s,v = self.delta_par_prior.pseudo_inv_components()
            self.Am = u * s.inv
        self.__initialized = True

    def get_localizer(self):
        onames = self.pst.nnz_obs_names
        pnames = self.pst.adj_par_names
        localizer = Matrix(x=np.ones((len(onames),len(pnames))),row_names=onames,col_names=pnames)
        return localizer

    def _calc_delta_par(self,parensemble):
        '''
        calc the scaled parameter ensemble differences from the mean
        '''
        return self._calc_delta(parensemble, self.half_parcov_diag)

    def _calc_delta_obs(self,obsensemble):
        '''
        calc the scaled observation ensemble differences from the mean
        '''
        return self._calc_delta(obsensemble.nonzero, self.obscov.inv.sqrt)

    def _calc_delta(self,ensemble,scaling_matrix):
        '''
        calc the scaled  ensemble differences from the mean
        '''
        mean = np.array(ensemble.mean(axis=0))
        delta = ensemble.as_pyemu_matrix()
        for i in range(self.num_reals):
            delta.x[i,:] -= mean
        delta = scaling_matrix * delta.T
        delta *= (1.0 / np.sqrt(float(self.num_reals - 1.0)))
        return delta

    def _calc_obs(self,parensemble):
        if self.submit_file is None:
            return self._calc_obs_local(parensemble)
        else:
            return self._calc_obs_condor(parensemble)


    def _calc_obs_condor(self,parensemble):
        parensemble.to_csv(self.sweep_in_csv)
        os.system("condor_rm -all")
        port = 4004
        def master():
            os.system("sweep {0} /h :{1} >nul".format(self.pst.filename,port))
        master_thread = threading.Thread(target=master)
        master_thread.start()
        time.sleep(1.5) #just some time for the master to get up and running to take slaves
        pyemu.utils.start_slaves("template","sweep",self.pst.filename,
                                 self.num_slaves,slave_root='.',port=port)
        os.system("condor_submit {0}".format(self.submit_file))
        master_thread.join()

    def _calc_obs_local(self,parensemble):
        '''
        propagate the ensemble forward using sweep.
        '''
        parensemble.to_csv(self.sweep_in_csv)
        if self.num_slaves > 0:
            port = 4004
            def master():
                os.system("sweep {0} /h :{1} >nul".format(self.pst.filename,port))
            master_thread = threading.Thread(target=master)
            master_thread.start()
            time.sleep(1.5) #just some time for the master to get up and running to take slaves
            pyemu.utils.start_slaves("template","sweep",self.pst.filename,
                                     self.num_slaves,slave_root='.',port=port)
            master_thread.join()
        else:
            os.system("sweep {0}".format(self.pst.filename))

        obs = pd.read_csv(self.sweep_out_csv)
        obs.columns = [item.lower() for item in obs.columns]
        self.total_runs += obs.shape[0]
        return ObservationEnsemble.from_dataframe(df=obs.loc[:,self.obscov.row_names],
                                                  pst=self.pst)

    def _calc_phi_vec(self,obsensemble):
        obs_diff = self._get_residual_matrix(obsensemble)
        phi_vec = np.diagonal((obs_diff * self.obscov_inv_sqrt.get(row_names=obs_diff.col_names,
                                                                   col_names=obs_diff.col_names) * obs_diff.T).x)
        return phi_vec

    def _phi_report(self,phi_vec,cur_lam):
        assert phi_vec.shape[0] == self.num_reals
        self.phi_csv.write("{0},{1},{2},{3},{4},{5},{6}".format(self.iter_num,
                                                             self.total_runs,
                                                             cur_lam,
                                                             phi_vec.min(),
                                                             phi_vec.max(),
                                                             phi_vec.mean(),
                                                             np.median(phi_vec),
                                                             phi_vec.std()))
        self.phi_csv.write(",".join(["{0:20.8}".format(phi) for phi in phi_vec]))
        self.phi_csv.write("\n")
        self.phi_csv.flush()

    def _get_residual_matrix(self, obsensemble):
        obs_matrix = obsensemble.nonzero.as_pyemu_matrix()
        return  obs_matrix - self.obs0_matrix.get(col_names=obs_matrix.col_names,row_names=obs_matrix.row_names)

    def update(self,lambda_mults=[1.0],localizer=None,run_subset=None):

        self.iter_num += 1
        if not self.__initialized:
            raise Exception("must call initialize() before update()")

        scaled_delta_obs = self._calc_delta_obs(self.obsensemble)
        scaled_delta_par = self._calc_delta_par(self.parensemble)

        u,s,v = scaled_delta_obs.pseudo_inv_components()

        obs_diff = self._get_residual_matrix(self.obsensemble)

        if run_subset is not None:
            subset_idx = ["{0:d}".format(i) for i in np.random.randint(0,self.num_reals-1,run_subset)]
            print("subset idxs: " + ','.join(subset_idx))

        mean_lam,std_lam,paren_lam,obsen_lam = [],[],[],[]
        for ilam,cur_lam_mult in enumerate(lambda_mults):

            parensemble_cur_lam = self.parensemble.copy()

            cur_lam = self.current_lambda * cur_lam_mult

            scaled_ident = Cov.identity_like(s) * (cur_lam+1.0)
            scaled_ident += s**2
            scaled_ident = scaled_ident.inv

            # build up this matrix as a single element so we can apply
            # localization
            upgrade_1 = -1.0 * (self.half_parcov_diag * scaled_delta_par) *\
                        v * s * scaled_ident * u.T

            # apply localization
            #print(cur_lam,upgrade_1)
            if localizer is not None:
                upgrade_1.hadamard_product(localizer)

            # apply residual information
            upgrade_1 *= (self.obscov_inv_sqrt * obs_diff.T)

            upgrade_1 = upgrade_1.to_dataframe()
            upgrade_1.index.name = "parnme"
            upgrade_1 = upgrade_1.T
            upgrade_1.to_csv(self.pst.filename+".upgrade_1.{0:04d}.csv".\
                               format(self.iter_num))
            parensemble_cur_lam += upgrade_1

            # parameter-based upgrade portion
            if not self.use_approx and self.iter_num > 1:
                par_diff = (self.parensemble - self.parensemble_0).\
                    as_pyemu_matrix().T
                x4 = self.Am.T * self.half_parcov_diag * par_diff
                x5 = self.Am * x4
                x6 = scaled_delta_par.T * x5
                x7 = v * scaled_ident * v.T * x6
                upgrade_2 = -1.0 * (self.half_parcov_diag *
                                   scaled_delta_par * x7).to_dataframe()

                upgrade_2.index.name = "parnme"
                upgrade_2.T.to_csv(self.pst.filename+".upgrade_2.{0:04d}.csv".\
                                   format(self.iter_num))
                parensemble_cur_lam += upgrade_2.T
            parensemble_cur_lam.enforce()
            paren_lam.append(parensemble_cur_lam)
            if run_subset is not None:
                #phi_series = pd.Series(data=self.current_phi_vec)
                #phi_series.sort_values(inplace=True,ascending=False)
                #subset_idx = ["{0:d}".format(i) for i in phi_series.index.values[:run_subset]]

                parensemble_subset = parensemble_cur_lam.loc[subset_idx,:]
                obsensemble_cur_lam = self._calc_obs(parensemble_subset)
            else:
                obsensemble_cur_lam = self._calc_obs(parensemble_cur_lam)
            #print(obsensemble_cur_lam.head())
            obsen_lam.append(obsensemble_cur_lam)


        # here is where we need to select out the "best" lambda par and obs
        # ensembles
        print("\n**************************")
        print(str(datetime.now()))
        print("total runs:{0}".format(self.total_runs))
        print("iteration: {0}".format(self.iter_num))
        print("current lambda:{0:15.6G}, mean:{1:15.6G}, std:{2:15.6G}".\
                  format(self.current_lambda,
                         self.last_best_mean,self.last_best_std))
        phi_vecs = [self._calc_phi_vec(obsen) for obsen in obsen_lam]
        mean_std = [(pv.mean(),pv.std()) for pv in phi_vecs]
        update_pars = False
        update_lambda = False
        # accept a new best if its within 10%
        best_mean = self.last_best_mean * 1.1
        best_std = self.last_best_std * 1.1
        best_i = 0
        for i,(m,s) in enumerate(mean_std):
            print(" tested lambda:{0:15.6G}, mean:{1:15.6G}, std:{2:15.6G}".\
                  format(self.current_lambda * lambda_mults[i],m,s))
            if m < best_mean:
                update_pars = True
                best_mean = m
                best_i = i
                if s < best_std:
                    update_lambda = True
                    best_std = s

        if not update_pars:
            self.current_lambda *= max(lambda_mults) * 3.0
            self.current_lambda = min(self.current_lambda,100000)
            print("not accepting iteration, increased lambda:{0}".\
                  format(self.current_lambda))

        else:

            self.parensemble = paren_lam[best_i]
            if run_subset is not None:
                self.obsensemble = self._calc_obs(self.parensemble)
                self.current_phi_vec = self._calc_phi_vec(self.obsensemble)
                self._phi_report(self.current_phi_vec,self.current_lambda * lambda_mults[best_i])
                best_mean = self.current_phi_vec.mean()
                best_std = self.current_phi_vec.std()
            else:
                self.obsensemble = obsen_lam[best_i]
                self._phi_report(phi_vecs[best_i],self.current_lambda * lambda_mults[best_i])
                self.current_phi_vec = phi_vecs[best_i]

            print("\n" + "   best lambda:{0:15.6G}, mean:{1:15.6G}, std:{2:15.6G}".\
                  format(self.current_lambda*lambda_mults[best_i],
                         best_mean,best_std))
            self.last_best_mean = best_mean
            self.last_best_std = best_std

        if update_lambda:
            # be aggressive - cut best lambda in half
            self.current_lambda *= (lambda_mults[best_i] * 0.75)
            # but don't let lambda get too small
            self.current_lambda = max(self.current_lambda,0.001)
            print("updating lambda: {0:15.6G}".\
                  format(self.current_lambda ))


        print("**************************\n")

        self.parensemble.to_csv(self.pst.filename+self.paren_prefix.\
                                    format(self.iter_num))

        self.obsensemble.to_csv(self.pst.filename+self.obsen_prefix.\
                                    format(self.iter_num))