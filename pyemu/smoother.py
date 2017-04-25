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
from .logger import Logger

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
                 restart_iter=0,submit_file=None,verbose=False):
        self.logger = Logger(verbose)
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
        self.enforce_bounds = None

    def initialize(self,num_reals,init_lambda=None,enforce_bounds="reset"):
        '''
        (re)initialize the process
        '''
        self.logger.log("initializing smoother with {0} realizations".format(num_reals))
        assert num_reals > 1
        self.enforce_bounds = enforce_bounds
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
            self.logger.statement("restarting smoother from existing csv files...ignoring num_reals")
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
            self.logger.log("initializing parensemble")
            self.parensemble_0 = ParameterEnsemble(self.pst)
            self.parensemble_0.draw(cov=self.parcov,num_reals=num_reals)
            self.parensemble_0.enforce(enforce_bounds=self.enforce_bounds)
            self.logger.log("initializing parensemble")
            self.parensemble = self.parensemble_0.copy()
            self.parensemble_0.to_csv(self.pst.filename +\
                                      self.paren_prefix.format(0))
            self.logger.log("initializing parensemble")
            self.logger.log("initializing obsensemble")
            self.obsensemble_0 = ObservationEnsemble(self.pst)
            self.obsensemble_0.draw(cov=self.obscov,num_reals=num_reals)
            #self.obsensemble = self.obsensemble_0.copy()

            # save the base obsensemble
            self.obsensemble_0.to_csv(self.pst.filename +\
                                      self.obsen_prefix.format(-1))
            self.logger.log("initializing obsensemble")
            self.obs0_matrix = self.obsensemble_0.nonzero.as_pyemu_matrix()

            # run the initial parameter ensemble
            self.logger.log("evaluating initial ensembles")
            self.obsensemble = self._calc_obs(self.parensemble)
            self.obsensemble.to_csv(self.pst.filename +\
                                      self.obsen_prefix.format(0))
            self.logger.log("evaluating initial ensembles")
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
        self.logger.statement("current lambda:{0:15.6g}".format(self.current_lambda))

        if self.use_approx:
            self.logger.statement("using approximate parcov in solution")
            self.half_parcov_diag = 1.0
        else:
            self.logger.statement("using full parcov in solution")
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
        self.logger.log("initializing smoother with {0} realizations".format(num_reals))

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
        self.logger.log("removing existing sweep in/out files")
        try:
            os.remove(self.sweep_in_csv)
        except Exception as e:
            self.logger.warn("error removing existing sweep in file:{0}".format(str(e)))
        try:
            os.remove(self.sweep_out_csv)
        except:
            self.logger.warn("error removing existing sweep out file:{0}".format(str(e)))

        if self.submit_file is None:
            return self._calc_obs_local(parensemble)
        else:
            return self._calc_obs_condor(parensemble)


    def _calc_obs_condor(self,parensemble):
        self.logger.log("evaluating ensemble of size {0} locally with htcondor".\
                        format(parensemble.shape[0]))
        parensemble.to_csv(self.sweep_in_csv)
        #os.system("condor_rm -all")
        port = 4004
        def master():
            try:
                #os.system("sweep {0} /h :{1} >_condor_master_stdout.dat".format(self.pst.filename,port))
                os.system("sweep {0} /h :{1}".format(self.pst.filename,port))
            except Exception as e:
                self.logger.lraise("error starting condor master: {0}".format(str(e)))
        master_thread = threading.Thread(target=master)
        master_thread.start()
        time.sleep(2.0) #just some time for the master to get up and running to take slaves
        #pyemu.utils.start_slaves("template","sweep",self.pst.filename,
        #                         self.num_slaves,slave_root='.',port=port)
        condor_temp_file = "_condor_submit_stdout.dat"
        self.logger.log("calling condor_submit with submit file {0}".format(self.submit_file))
        try:
            os.system("condor_submit {0} >{1}".format(self.submit_file,condor_temp_file))
        except Exception as e:
            self.logger.lraise("error in condor_submit: {0}".format(str(e)))
        self.logger.log("calling condor_submit with submit file {0}".format(self.submit_file))
        time.sleep(2.0) #some time for condor to submit the job and echo to stdout
        condor_submit_string = "submitted to cluster"
        with open(condor_temp_file,'r') as f:
            lines = f.readlines()
        self.logger.statement("condor_submit stdout: {0}".format(','.join([line.strip() for line in lines])))
        for line in lines:
            if condor_submit_string in line.lower():
                cluster_number = int(float(line.split(condor_submit_string)[-1]))
        self.logger.statement("condor cluster: {0}".format(cluster_number))
        master_thread.join()
        self.logger.statement("condor master thread exited")
        self.logger.log("calling condor_rm on cluster {0}".format(cluster_number))
        os.system("condor_rm cluster {0}".format(cluster_number))
        self.logger.log("calling condor_rm on cluster {0}".format(cluster_number))

        self.logger.log("reading sweep out csv {0}".format(self.sweep_out_csv))
        obs = pd.read_csv(self.sweep_out_csv)
        self.logger.log("reading sweep out csv {0}".format(self.sweep_out_csv))
        obs.columns = [item.lower() for item in obs.columns]
        self.total_runs += obs.shape[0]
        self.logger.statement("total runs:{0}".format(self.total_runs))
        self.logger.log("evaluating ensemble of size {0} locally with htcondor".\
                        format(parensemble.shape[0]))
        return ObservationEnsemble.from_dataframe(df=obs.loc[:,self.obscov.row_names],
                                                  pst=self.pst)

    def _calc_obs_local(self,parensemble):
        '''
        propagate the ensemble forward using sweep.
        '''
        self.logger.log("evaluating ensemble of size {0} locally with sweep".\
                        format(parensemble.shape[0]))
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
        self.logger.statement("total runs so far :{0}".format(self.total_runs))
        self.logger.log("evaluating ensemble of size {0} locally with sweep".\
                        format(parensemble.shape[0]))
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

        if run_subset is not None:
            assert run_subset < self.num_reals

        self.iter_num += 1
        self.logger.log("iteration {0}".format(self.iter_num))
        if not self.__initialized:
            #raise Exception("must call initialize() before update()")
            self.logger.lraise("must call initialize() before update()")

        self.logger.log("calculate scaled delta obs")
        scaled_delta_obs = self._calc_delta_obs(self.obsensemble)
        self.logger.log("calculate scaled delta obs")
        self.logger.log("calculate scaled delta par")
        scaled_delta_par = self._calc_delta_par(self.parensemble)
        self.logger.log("calculate scaled delta par")

        self.logger.log("calculate pseudo inv comps")
        u,s,v = scaled_delta_obs.pseudo_inv_components()
        self.logger.log("calculate pseudo inv comps")

        self.logger.log("calculate obs diff matrix")
        obs_diff = self._get_residual_matrix(self.obsensemble)
        self.logger.log("calculate obs diff matrix")


        # here is the math part...calculate upgrade matrices
        mean_lam,std_lam,paren_lam,obsen_lam = [],[],[],[]
        lam_vals = []
        for ilam,cur_lam_mult in enumerate(lambda_mults):

            parensemble_cur_lam = self.parensemble.copy()

            cur_lam = self.current_lambda * cur_lam_mult
            lam_vals.append(cur_lam)
            self.logger.log("calcs for  lambda {0}".format(cur_lam_mult))
            scaled_ident = Cov.identity_like(s) * (cur_lam+1.0)
            scaled_ident += s**2
            scaled_ident = scaled_ident.inv

            # build up this matrix as a single element so we can apply
            # localization
            self.logger.log("building upgrade matrix")
            upgrade_1 = -1.0 * (self.half_parcov_diag * scaled_delta_par) *\
                        v * s * scaled_ident * u.T
            self.logger.log("building upgrade matrix")
            # apply localization
            #print(cur_lam,upgrade_1)
            if localizer is not None:
                self.logger.log("applying localization")
                upgrade_1.hadamard_product(localizer)
                self.logger.log("applying localization")

            # apply residual information
            self.logger.log("applying residuals")
            upgrade_1 *= (self.obscov_inv_sqrt * obs_diff.T)
            self.logger.log("applying residuals")

            upgrade_1 = upgrade_1.to_dataframe()
            upgrade_1.index.name = "parnme"
            upgrade_1 = upgrade_1.T
            upgrade_1.to_csv(self.pst.filename+".upgrade_1.{0:04d}.csv".\
                               format(self.iter_num))
            parensemble_cur_lam += upgrade_1

            # parameter-based upgrade portion
            if not self.use_approx and self.iter_num > 1:
                self.logger.log("applying parameter prior information")
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
                self.logger.log("applying parameter prior information")

            parensemble_cur_lam.enforce(self.enforce_bounds)
            paren_lam.append(pd.DataFrame(parensemble_cur_lam.loc[:,:]))
            self.logger.log("calcs for  lambda {0}".format(cur_lam_mult))


        # subset if needed
        # and combine lambda par ensembles into one par ensemble for evaluation
        if run_subset is not None:
            subset_idx = ["{0:d}".format(i) for i in np.random.randint(0,self.num_reals-1,run_subset)]
            self.logger.statement("subset idxs: " + ','.join(subset_idx))
            paren_lam_subset = [pe.loc[subset_idx,:] for pe in paren_lam]
            paren_combine = pd.concat(paren_lam_subset)
            paren_lam_subset = None
        else:
            paren_combine = pd.concat(paren_lam)


        self.logger.log("evaluating ensembles for lambdas : {0}".\
                        format(','.join(["{0:8.3E}".format(l) for l in lam_vals])))
        obsen_combine = self._calc_obs(paren_combine)
        self.logger.log("evaluating ensembles for lambdas : {0}".\
                        format(','.join(["{0:8.3E}".format(l) for l in lam_vals])))
        paren_combine = None

        # unpack lambda obs ensembles from combined obs ensemble
        nrun_per_lam = self.num_reals
        if run_subset is not None:
            nrun_per_lam = run_subset
        obsen_lam = []
        for i in range(len(lam_vals)):
            sidx = i * nrun_per_lam
            eidx = sidx + nrun_per_lam
            oe = ObservationEnsemble.from_dataframe(df=obsen_combine.iloc[sidx:eidx,:].copy(),
                                                    pst=self.pst)
            #print(oe.shape)
            obsen_lam.append(oe)
        obsen_combine = None


        # here is where we need to select out the "best" lambda par and obs
        # ensembles
        self.logger.statement("\n**************************")
        self.logger.statement(str(datetime.now()))
        self.logger.statement("total runs:{0}".format(self.total_runs))
        self.logger.statement("iteration: {0}".format(self.iter_num))
        self.logger.statement("current lambda:{0:15.6G}, mean:{1:15.6G}, std:{2:15.6G}".\
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
            self.logger.statement(" tested lambda:{0:15.6G}, mean:{1:15.6G}, std:{2:15.6G}".\
                                 format(self.current_lambda * lambda_mults[i],m,s))
            if m < best_mean:
                update_pars = True
                best_mean = m
                best_i = i
                if s < best_std:
                    update_lambda = True
                    best_std = s

        if not update_pars:
            self.current_lambda *= max(lambda_mults) * 10.0
            self.current_lambda = min(self.current_lambda,100000)
            self.logger.statement("not accepting iteration, increased lambda:{0}".\
                  format(self.current_lambda))
            #print("not accepting iteration, increased lambda:{0}".\
            #      format(self.current_lambda))

        else:

            self.parensemble = ParameterEnsemble.from_dataframe(df=paren_lam[best_i],pst=self.pst)
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

            self.logger.statement("   best lambda:{0:15.6G}, mean:{1:15.6G}, std:{2:15.6G}".\
                  format(self.current_lambda*lambda_mults[best_i],
                         best_mean,best_std))
            self.last_best_mean = best_mean
            self.last_best_std = best_std

        if update_lambda:
            # be aggressive
            self.current_lambda *= (lambda_mults[best_i] * 0.75)
            # but don't let lambda get too small
            self.current_lambda = max(self.current_lambda,0.001)
            self.logger.statement("updating lambda: {0:15.6G}".\
                  format(self.current_lambda ))


        self.logger.statement("**************************\n")

        self.parensemble.to_csv(self.pst.filename+self.paren_prefix.\
                                    format(self.iter_num))

        self.obsensemble.to_csv(self.pst.filename+self.obsen_prefix.\
                                    format(self.iter_num))
        self.logger.log("iteration {0}".format(self.iter_num))