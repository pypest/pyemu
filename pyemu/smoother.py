"""this is a prototype ensemble smoother based on the LM-EnRML
algorithm of Chen and Oliver 2013.  It requires the pest++ "sweep" utility
to propagate the ensemble forward.
"""
from __future__ import print_function, division
import os
from datetime import datetime
import shutil
import threading
import time
import warnings
import numpy as np
import pandas as pd
import pyemu
from pyemu.en import ParameterEnsemble,ObservationEnsemble
from pyemu.mat import Cov,Matrix

from pyemu.pst import Pst
from .logger import Logger


class Phi(object):
    """Container class for dealing with ensemble residuals and phi vectors

    Parameters
    ----------
        em : EnsembleMethod
    """
    def __init__(self,em):
        assert isinstance(em,EnsembleMethod)
        self.em = em
        self.phi_files = {}
        self._tags = ["composite","meas","actual","reg"]
        self._prepare_output_files()
        self.obs0_matrix = self.em.obsensemble_0.nonzero.as_pyemu_matrix()
        if self.em.parensemble.istransformed:
            self.par0_matrix = self.em.parensemble_0.as_pyemu_matrix()
        else:
            self.par0_matrix = self.em.parensemble_0._transform(inplace=False).as_pyemu_matrix()
        self.par0_matrix = self.par0_matrix.get(col_names=self.em.parcov.row_names)
        self.em.logger.log("inverting parcov for regul phi calcs")
        self.inv_parcov = self.em.parcov.pseudo_inv(eigthresh=self.em.pst.svd_data.eigthresh)
        self.em.logger.log("inverting parcov for regul phi calcs")
        self.update()

    def _prepare_output_files(self):
        num_reals = self.em.obsensemble.shape[0]
        for tag in self._tags:
            f = open(self.em.pst.filename + ".iobj.{0}.csv".format(tag), 'w')
            f.write("iter_num,total_runs,lambda,min,max,mean,median,std,")
            f.write(','.join(["{0:010d}".format(i + 1) for i in range(num_reals)]))
            f.write('\n')
            self.phi_files[tag] = f

    def write(self,cur_lam=0.0):
        for t,pv in self.phi_vec_dict.items():
            self._phi_report(self.phi_files[t],pv,cur_lam)

    @property
    def phi_vec_dict(self):
        d = {t:pv for t,pv in zip(self._tags,[self.comp_phi,self.meas_phi,self.meas_phi_actual,self.reg_phi])}
        return d

    def _phi_report(self,phi_csv,phi_vec,cur_lam=0.0):
        phi_csv.write("{0},{1},{2},{3},{4},{5},{6},{7},".format(self.em.iter_num,
                                                             self.em.total_runs,
                                                             cur_lam,
                                                             phi_vec.min(),
                                                             phi_vec.max(),
                                                             phi_vec.mean(),
                                                             np.median(phi_vec),
                                                             phi_vec.std()))
        #[print(phi) for phi in phi_vec]
        phi_csv.write(",".join(["{0:20.8}".format(phi) for phi in phi_vec]))
        phi_csv.write("\n")
        phi_csv.flush()

    def get_meas_and_regul_phi(self,obsensemble,parensemble):
        assert obsensemble.shape[0] == parensemble.shape[0]
        meas_phi = self._calc_meas_phi(obsensemble)
        reg_phi = self._calc_regul_phi(parensemble)
        return meas_phi,reg_phi



    def update(self):
        #assert obsensemble.shape[0] == parensemble.shape[0]
        #self.cur_lam = cur_lam
        self.meas_phi = self._calc_meas_phi(self.em.obsensemble)
        self.meas_phi_actual = self._calc_meas_phi_actual(self.em.obsensemble)
        self.reg_phi = self._calc_regul_phi(self.em.parensemble)
        self.comp_phi = self.meas_phi + (self.reg_phi * self.em.regul_factor)


    # dynamic updating is too costly when lots of pars are being used...
    # @property
    # def meas_phi(self):
    #     #print('calc meas phi')
    #     return self._calc_meas_phi(self.em.obsensemble)
    #
    # @property
    # def meas_phi_actual(self):
    #     #print('calc actual phi')
    #     return self._calc_meas_phi_actual(self.em.obsensemble)
    #
    # @property
    # def reg_phi(self):
    #     #print('calc regul phi')
    #     return self._calc_regul_phi(self.em.parensemble)
    #
    # @property
    # def comp_phi(self):
    #     #print('calc comp phi')
    #     return self.meas_phi + (self.reg_phi * self.em.regul_factor)

    def report(self,cur_lam=0.0):
        self.write(cur_lam)
        ls = self.em.logger.statement
        pvd = self.phi_vec_dict
        ls("**** phi summary ****")
        ls("{0:12s}: {1:>15.6G}".format("regul factor",self.em.regul_factor))
        for t in self._tags:
            ls("{0:9s} mean: {1:>15.6G}, std: {2:>15.6G}".\
               format(t,pvd[t].mean(),pvd[t].std()))
        ls("*********************")

    def _calc_meas_phi(self,obsensemble):
        obs_diff = self.get_residual_obs_matrix(obsensemble)

        q = np.diagonal(self.em.obscov_inv_sqrt.get(row_names=obs_diff.col_names,col_names=obs_diff.col_names).x)
        phi_vec = []
        for i in range(obs_diff.shape[0]):
            o = obs_diff.x[i,:]
            phi_vec.append(((obs_diff.x[i,:] * q)**2).sum())

        return np.array(phi_vec)


    def _calc_regul_phi(self,parensemble):
        if isinstance(parensemble,pd.DataFrame):
            parensemble = pyemu.ParameterEnsemble.from_dataframe(pst=self.em.pst,df=parensemble)
        par_diff = self.get_residual_par_matrix(parensemble)
        #reg_phi_vec = np.diag(np.dot(np.dot(par_diff.x,self.inv_parcov.x),par_diff.x.transpose()))
        reg_phi_vec = np.diag((par_diff * self.inv_parcov * par_diff.T).x)
        return reg_phi_vec

    def _calc_meas_phi_actual(self,obsensemble):
        return obsensemble.phi_vector

    def get_residual_obs_matrix(self, obsensemble):
        obs_matrix = obsensemble.nonzero.as_pyemu_matrix()

        res_mat = obs_matrix - self.obs0_matrix.get(col_names=obs_matrix.col_names,
                                                    row_names=obs_matrix.row_names)
        self._apply_inequality_constraints(res_mat)
        return  res_mat

    def get_residual_par_matrix(self, parensemble):
        # these should already be transformed
        # if parensemble.istransformed:
        #     par_matrix = parensemble.as_pyemu_matrix()
        # else:
        #     par_matrix = parensemble._transform(inplace=False).as_pyemu_matrix()
        par_matrix = parensemble.as_pyemu_matrix()
        #res_mat = par_matrix.get(col_names=self.em.pst.adj_par_names) - \
        #          self.par0_matrix.get(col_names=self.em.pst.adj_par_names)
        res_mat = par_matrix.get(col_names=self.par0_matrix.col_names) - self.par0_matrix
        return  res_mat

    def _apply_inequality_constraints(self,res_mat):
        obs = self.em.pst.observation_data.loc[res_mat.col_names]
        gt_names = obs.loc[obs.obgnme.apply(lambda x: x.startswith("g_") or x.startswith("less")), "obsnme"]
        lt_names = obs.loc[obs.obgnme.apply(lambda x: x.startswith("l_") or x.startswith("greater")), "obsnme"]
        if gt_names.shape[0] == 0 and lt_names.shape[0] == 0:
            return res_mat
        res_df = res_mat.to_dataframe()
        if gt_names.shape[0] > 0:
            for gt_name in gt_names:
                #print(res_df.loc[:,gt_name])
                #if the residual is greater than zero, this means the ineq is satisified
                res_df.loc[res_df.loc[:,gt_name] > 0,gt_name] = 0.0
                #print(res_df.loc[:,gt_name])
                #print()


        if lt_names.shape[0] > 0:
            for lt_name in lt_names:
                #print(res_df.loc[:,lt_name])
                #f the residual is less than zero, this means the ineq is satisfied
                res_df.loc[res_df.loc[:,lt_name] < 0,lt_name] = 0.0
                #print(res_df.loc[:,lt_name])
                #print()


class EnsembleMethod(object):
    """Base class for ensemble-type methods.  Should not be instantiated directly

    Parameters
    ----------
        pst : pyemu.Pst or str
            a control file instance or filename
        parcov : pyemu.Cov or str
            a prior parameter covariance matrix or filename. If None,
            parcov is constructed from parameter bounds (diagonal)
        obscov : pyemu.Cov or str
            a measurement noise covariance matrix or filename. If None,
            obscov is constructed from observation weights.
        num_slaves : int
            number of slaves to use in (local machine) parallel evaluation of the parmaeter
            ensemble.  If 0, serial evaluation is used.  Ignored if submit_file is not None
        submit_file : str
            the name of a HTCondor submit file.  If not None, HTCondor is used to
            evaluate the parameter ensemble in parallel by issuing condor_submit
            as a system command
        port : int
            the TCP port number to communicate on for parallel run management
        slave_dir : str
            path to a directory with a complete set of model files and PEST
            interface files

    """

    def __init__(self,pst,parcov=None,obscov=None,num_slaves=0,use_approx_prior=True,
                 submit_file=None,verbose=False,port=4004,slave_dir="template"):
        self.logger = Logger(verbose)
        if verbose is not False:
            self.logger.echo = True
        self.num_slaves = int(num_slaves)
        if submit_file is not None:
            if not os.path.exists(submit_file):
                self.logger.lraise("submit_file {0} not found".format(submit_file))
        elif num_slaves > 0:
            if not os.path.exists(slave_dir):
                self.logger.lraise("template dir {0} not found".format(slave_dir))

        self.slave_dir = slave_dir
        self.submit_file = submit_file
        self.port = int(port)
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

        self.__initialized = False
        self.iter_num = 0
        self.raw_sweep_out = None

    def initialize(self,*args,**kwargs):
        raise Exception("EnsembleMethod.initialize() must be implemented by the derived types")

    def _calc_delta(self,ensemble,scaling_matrix=None):
        '''
        calc the scaled  ensemble differences from the mean
        '''
        mean = np.array(ensemble.mean(axis=0))
        delta = ensemble.as_pyemu_matrix()
        for i in range(ensemble.shape[0]):
            delta.x[i,:] -= mean
        if scaling_matrix is not None:
            delta = scaling_matrix * delta.T
        delta *= (1.0 / np.sqrt(float(ensemble.shape[0] - 1.0)))
        return delta

    def _calc_obs(self,parensemble):
        self.logger.log("removing existing sweep in/out files")
        try:
            os.remove(self.sweep_in_csv)
        except Exception as e:
            self.logger.warn("error removing existing sweep in file:{0}".format(str(e)))
        try:
            os.remove(self.sweep_out_csv)
        except Exception as e:
            self.logger.warn("error removing existing sweep out file:{0}".format(str(e)))
        self.logger.log("removing existing sweep in/out files")

        if parensemble.isnull().values.any():
            parensemble.to_csv("_nan.csv")
            self.logger.lraise("_calc_obs() error: NaNs in parensemble (written to '_nan.csv')")

        if self.submit_file is None:
            self._calc_obs_local(parensemble)
        else:
            self._calc_obs_condor(parensemble)

        # make a copy of sweep out for restart purposes
        # sweep_out = str(self.iter_num)+"_raw_"+self.sweep_out_csv
        # if os.path.exists(sweep_out):
        #     os.remove(sweep_out)
        # shutil.copy2(self.sweep_out_csv,sweep_out)

        self.logger.log("reading sweep out csv {0}".format(self.sweep_out_csv))
        failed_runs,obs = self._load_obs_ensemble(self.sweep_out_csv)
        self.logger.log("reading sweep out csv {0}".format(self.sweep_out_csv))
        self.total_runs += obs.shape[0]
        self.logger.statement("total runs:{0}".format(self.total_runs))
        return failed_runs,obs

    def _load_obs_ensemble(self,filename):
        if not os.path.exists(filename):
            self.logger.lraise("obsensemble file {0} does not exists".format(filename))
        obs = pd.read_csv(filename)
        obs.columns = [item.lower() for item in obs.columns]
        self.raw_sweep_out = obs.copy() # save this for later to support restart
        assert "input_run_id" in obs.columns,\
            "'input_run_id' col missing...need newer version of sweep"
        obs.index = obs.input_run_id
        failed_runs = None
        if 1 in obs.failed_flag.values:
            failed_runs = obs.loc[obs.failed_flag == 1].index.values
            self.logger.warn("{0} runs failed (indices: {1})".\
                             format(len(failed_runs),','.join([str(f) for f in failed_runs])))
        obs = ObservationEnsemble.from_dataframe(df=obs.loc[:,self.obscov.row_names],
                                                               pst=self.pst)
        if obs.isnull().values.any():
            self.logger.lraise("_calc_obs() error: NaNs in obsensemble")
        return failed_runs, obs

    def _get_master_thread(self):
        master_stdout = "_master_stdout.dat"
        master_stderr = "_master_stderr.dat"
        def master():
            try:
                #os.system("sweep {0} /h :{1} 1>{2} 2>{3}". \
                #          format(self.pst.filename, self.port, master_stdout, master_stderr))
                pyemu.helpers.run("sweep {0} /h :{1} 1>{2} 2>{3}". \
                          format(self.pst.filename, self.port, master_stdout, master_stderr))

            except Exception as e:
                self.logger.lraise("error starting condor master: {0}".format(str(e)))
            with open(master_stderr, 'r') as f:
                err_lines = f.readlines()
            if len(err_lines) > 0:
                self.logger.warn("master stderr lines: {0}".
                                 format(','.join([l.strip() for l in err_lines])))

        master_thread = threading.Thread(target=master)
        master_thread.start()
        time.sleep(2.0)
        return master_thread

    def _calc_obs_condor(self,parensemble):
        self.logger.log("evaluating ensemble of size {0} with htcondor".\
                        format(parensemble.shape[0]))

        parensemble.to_csv(self.sweep_in_csv)
        master_thread = self._get_master_thread()
        condor_temp_file = "_condor_submit_stdout.dat"
        condor_err_file = "_condor_submit_stderr.dat"
        self.logger.log("calling condor_submit with submit file {0}".format(self.submit_file))
        try:
            os.system("condor_submit {0} 1>{1} 2>{2}".\
                      format(self.submit_file,condor_temp_file,condor_err_file))
        except Exception as e:
            self.logger.lraise("error in condor_submit: {0}".format(str(e)))
        self.logger.log("calling condor_submit with submit file {0}".format(self.submit_file))
        time.sleep(2.0) #some time for condor to submit the job and echo to stdout
        condor_submit_string = "submitted to cluster"
        with open(condor_temp_file,'r') as f:
            lines = f.readlines()
        self.logger.statement("condor_submit stdout: {0}".\
                              format(','.join([l.strip() for l in lines])))
        with open(condor_err_file,'r') as f:
            err_lines = f.readlines()
        if len(err_lines) > 0:
            self.logger.warn("stderr from condor_submit:{0}".\
                             format([l.strip() for l in err_lines]))
        cluster_number = None
        for line in lines:
            if condor_submit_string in line.lower():
                cluster_number = int(float(line.split(condor_submit_string)[-1]))
        if cluster_number is None:
            self.logger.lraise("couldn't find cluster number...")
        self.logger.statement("condor cluster: {0}".format(cluster_number))
        master_thread.join()
        self.logger.statement("condor master thread exited")
        self.logger.log("calling condor_rm on cluster {0}".format(cluster_number))
        os.system("condor_rm cluster {0}".format(cluster_number))
        self.logger.log("calling condor_rm on cluster {0}".format(cluster_number))
        self.logger.log("evaluating ensemble of size {0} with htcondor".\
                        format(parensemble.shape[0]))


    def _calc_obs_local(self,parensemble):
        '''
        propagate the ensemble forward using sweep.
        '''
        self.logger.log("evaluating ensemble of size {0} locally with sweep".\
                        format(parensemble.shape[0]))
        parensemble.to_csv(self.sweep_in_csv)
        if self.num_slaves > 0:
            master_thread = self._get_master_thread()
            pyemu.utils.start_slaves(self.slave_dir,"sweep",self.pst.filename,
                                     self.num_slaves,slave_root='..',port=self.port)
            master_thread.join()
        else:
            os.system("sweep {0}".format(self.pst.filename))

        self.logger.log("evaluating ensemble of size {0} locally with sweep".\
                        format(parensemble.shape[0]))


    def update(self,lambda_mults=[1.0],localizer=None,run_subset=None,use_approx=True):
        raise Exception("EnsembleMethod.update() must be implemented by the derived types")


class EnsembleSmoother(EnsembleMethod):
    """an implementation of the GLM iterative ensemble smoother

    Parameters
    ----------
        pst : pyemu.Pst or str
            a control file instance or filename
        parcov : pyemu.Cov or str
            a prior parameter covariance matrix or filename. If None,
            parcov is constructed from parameter bounds (diagonal)
        obscov : pyemu.Cov or str
            a measurement noise covariance matrix or filename. If None,
            obscov is constructed from observation weights.
        num_slaves : int
            number of slaves to use in (local machine) parallel evaluation of the parmaeter
            ensemble.  If 0, serial evaluation is used.  Ignored if submit_file is not None
        use_approx_prior : bool
             a flag to use the MLE (approx) upgrade solution.  If True, a MAP
             solution upgrade is used
        submit_file : str
            the name of a HTCondor submit file.  If not None, HTCondor is used to
            evaluate the parameter ensemble in parallel by issuing condor_submit
            as a system command
        port : int
            the TCP port number to communicate on for parallel run management
        slave_dir : str
            path to a directory with a complete set of model files and PEST
            interface files
        drop_bad_reals : float
                drop realizations with phi greater than drop_bad_reals. If None, all
                realizations are kept. Default is None

    Example
    -------
    ``>>>import pyemu``

    ``>>>es = pyemu.EnsembleSmoother(pst="pest.pst")``
    """

    def __init__(self,pst,parcov=None,obscov=None,num_slaves=0,submit_file=None,verbose=False,
                 port=4004,slave_dir="template",drop_bad_reals=None,save_mats=False):



        super(EnsembleSmoother,self).__init__(pst=pst,parcov=parcov,obscov=obscov,num_slaves=num_slaves,
                                              submit_file=submit_file,verbose=verbose,port=port,
                                              slave_dir=slave_dir)
        warnings.warn("pyemu's EnsembleSmoother is for prototyping only.  Use PESTPP-IES for a production " +
                         "implementation of iterative ensemble smoother")
        self.logger.warn("pyemu's EnsembleSmoother is for prototyping only.  Use PESTPP-IES for a production " +
                      "implementation of iterative ensemble smoother")
        #self.use_approx_prior = bool(use_approx_prior)
        self.parcov_inv_sqrt = None
        self.half_obscov_diag = None
        self.delta_par_prior = None
        self.drop_bad_reals = drop_bad_reals
        self.save_mats = save_mats
    # @classmethod
    # def from_pestpp_args(cls,pst):
    #     if isinstance(pst,str):
    #         pst = pyemu.Pst(pst)
    #     pp = pst.pestpp_options
    #     parcov = pp.pop("parcov_filename",pyemu.Cov.from_parameter_data(pst,sigma_range=6))
    #     obscov = pyemu.Cov.from_observation_data(pst)
    #     par_csv = pp.pop("ies_par_csv")
    #     obs_csv = pp.pop("ies_obs_csv")
    #     restart_csv = pp.pop("ies_obs_restart_csv",None)
    #
    #     use_approx = False
    #     tags = ["ies_use_approx","ies_use_approximate_solution"]
    #     for tag in tags:
    #         if tag in pp.keys() and pp[tag].startswith('t'):
    #
    #             use_approx = True
    #             pp.pop(tag)
    #
    #     lambda_mults = pp.pop("lambda",[1.0])
    #     init_lambda = pp.pop("ies_init_lam",pp.pop("ies_initial_lambda",None))
    #     subset = pp.pop("ies_subset_size",None)




    def initialize(self,num_reals=1,init_lambda=None,enforce_bounds="reset",
                   parensemble=None,obsensemble=None,restart_obsensemble=None,
                   regul_factor=0.0,use_approx_prior=True,build_empirical_prior=False):
        """Initialize the iES process.  Depending on arguments, draws or loads
        initial parameter observations ensembles and runs the initial parameter
        ensemble

        Parameters
        ----------
            num_reals : int
                the number of realizations to draw.  Ignored if parensemble/obsensemble
                are not None
            init_lambda : float
                the initial lambda to use.  During subsequent updates, the lambda is
                updated according to upgrade success
            enforce_bounds : str
                how to enfore parameter bound transgression.  options are
                reset, drop, or None
            parensemble : pyemu.ParameterEnsemble or str
                a parameter ensemble or filename to use as the initial
                parameter ensemble.  If not None, then obsenemble must not be
                None
            obsensemble : pyemu.ObservationEnsemble or str
                an observation ensemble or filename to use as the initial
                observation ensemble.  If not None, then parensemble must
                not be None
            restart_obsensemble : pyemu.ObservationEnsemble or str
                an observation ensemble or filename to use as an
                evaluated observation ensemble.  If not None, this will skip the initial
                parameter ensemble evaluation - user beware!
            regul_factor : float
                the regularization penalty fraction of the composite objective.  The
                Prurist, MAP solution would be regul_factor = 1.0, yielding equal
                parts measurement and regularization to the composite objective function.
                Default is 0.0, which means only seek to minimize the measurement objective
                function
            use_approx_prior : bool
                a flag to use the inverse, square root of the prior ccovariance matrix
                for scaling the upgrade calculation.  If True, this matrix is not used.
                Default is True
            build_empirical_prior : bool
                flag to build the prior parameter covariance matrix from an existing parensemble.
                If True and parensemble is None, an exception is raised


        Example
        -------
        ``>>>import pyemu``

        ``>>>es = pyemu.EnsembleSmoother(pst="pest.pst")``

        ``>>>es.initialize(num_reals=100)``

        """
        '''
        (re)initialize the process
        '''
        # initialize the phi report csv
        self.enforce_bounds = enforce_bounds

        self.regul_factor = float(regul_factor)

        self.total_runs = 0
        # this matrix gets used a lot, so only calc once and store
        self.obscov_inv_sqrt = self.obscov.get(self.pst.nnz_obs_names).inv.sqrt

        if use_approx_prior:
            self.logger.statement("using approximate parcov in solution")
            self.parcov_inv_sqrt = 1.0
        else:
            self.logger.statement("using full parcov in solution")
            # Chen and Oliver use a low rank approx here, but so far,
            # I haven't needed it - not using enough parameters yet
            self.logger.log("forming inverse sqrt parcov matrix")
            self.parcov_inv_sqrt = self.parcov.inv.sqrt
            self.logger.log("forming inverse sqrt parcov matrix")

        if parensemble is not None and obsensemble is not None:
            self.logger.log("initializing with existing ensembles")
            if isinstance(parensemble,str):
                self.logger.log("loading parensemble from file")
                if not os.path.exists(obsensemble):
                    self.logger.lraise("can not find parensemble file: {0}".\
                                       format(parensemble))
                df = pd.read_csv(parensemble,index_col=0)
                #df.index = [str(i) for i in df.index]
                self.parensemble_0 = ParameterEnsemble.from_dataframe(df=df,pst=self.pst)
                self.logger.log("loading parensemble from file")

            elif isinstance(parensemble,ParameterEnsemble):
                self.parensemble_0 = parensemble.copy()
            else:
                raise Exception("unrecognized arg type for parensemble, " +\
                                "should be filename or ParameterEnsemble" +\
                                ", not {0}".format(type(parensemble)))
            self.parensemble = self.parensemble_0.copy()
            if isinstance(obsensemble,str):
                self.logger.log("loading obsensemble from file")
                if not os.path.exists(obsensemble):
                    self.logger.lraise("can not find obsensemble file: {0}".\
                                       format(obsensemble))
                df = pd.read_csv(obsensemble,index_col=0).loc[:,self.pst.nnz_obs_names]
                #df.index = [str(i) for i in df.index]
                self.obsensemble_0 = ObservationEnsemble.from_dataframe(df=df,pst=self.pst)
                self.logger.log("loading obsensemble from file")

            elif isinstance(obsensemble,ObservationEnsemble):
                self.obsensemble_0 = obsensemble.copy()
            else:
                raise Exception("unrecognized arg type for obsensemble, " +\
                                "should be filename or ObservationEnsemble" +\
                                ", not {0}".format(type(obsensemble)))

            assert self.parensemble_0.shape[0] == self.obsensemble_0.shape[0]
            #self.num_reals = self.parensemble_0.shape[0]
            num_reals = self.parensemble.shape[0]
            self.logger.log("initializing with existing ensembles")

            if build_empirical_prior:

                self.reset_parcov(self.parensemble.covariance_matrix())
                if self.save_mats:
                    self.parcov.to_binary(self.pst.filename+".empcov.jcb")

        else:
            if build_empirical_prior:
                self.logger.lraise("can't use build_emprirical_prior without parensemble...")
            self.logger.log("initializing smoother with {0} realizations".format(num_reals))
            self.logger.log("initializing parensemble")
            self.parensemble_0 = pyemu.ParameterEnsemble.from_gaussian_draw(self.pst,
                                                                            self.parcov,num_reals=num_reals)
            self.parensemble_0.enforce(enforce_bounds=enforce_bounds)
            self.logger.log("initializing parensemble")
            self.parensemble = self.parensemble_0.copy()
            self.parensemble_0.to_csv(self.pst.filename +\
                                      self.paren_prefix.format(0))
            self.logger.log("initializing parensemble")
            self.logger.log("initializing obsensemble")
            self.obsensemble_0 = pyemu.ObservationEnsemble.from_id_gaussian_draw(self.pst,
                                                                                 num_reals=num_reals)
            #self.obsensemble = self.obsensemble_0.copy()

            # save the base obsensemble
            self.obsensemble_0.to_csv(self.pst.filename +\
                                      self.obsen_prefix.format(-1))
            self.logger.log("initializing obsensemble")
            self.logger.log("initializing smoother with {0} realizations".format(num_reals))


        if use_approx_prior:
            self.logger.statement("using approximate parcov in solution")
            self.parcov_inv_sqrt = 1.0
        else:
            self.logger.statement("using full parcov in solution")
            # Chen and Oliver use a low rank approx here, but so far,
            # I haven't needed it - not using enough parameters yet
            self.logger.log("forming inverse sqrt parcov matrix")
            self.parcov_inv_sqrt = self.parcov.inv.sqrt
            self.logger.log("forming inverse sqrt parcov matrix")

        # self.obs0_matrix = self.obsensemble_0.nonzero.as_pyemu_matrix()
        # self.par0_matrix = self.parensemble_0.as_pyemu_matrix()
        self.enforce_bounds = enforce_bounds

        if restart_obsensemble is not None:
            self.logger.log("loading restart_obsensemble {0}".format(restart_obsensemble))
            failed_runs,self.obsensemble = self._load_obs_ensemble(restart_obsensemble)
            assert self.obsensemble.shape[0] == self.obsensemble_0.shape[0]
            assert list(self.obsensemble.columns) == list(self.obsensemble_0.columns)
            self.logger.log("loading restart_obsensemble {0}".format(restart_obsensemble))

        else:
            # run the initial parameter ensemble
            self.logger.log("evaluating initial ensembles")
            failed_runs, self.obsensemble = self._calc_obs(self.parensemble)
            self.obsensemble.to_csv(self.pst.filename +\
                                      self.obsen_prefix.format(0))
            if self.raw_sweep_out is not None:
                self.raw_sweep_out.to_csv(self.pst.filename + "_sweepraw0.csv")
            self.logger.log("evaluating initial ensembles")

        if failed_runs is not None:
            self.logger.warn("dropping failed realizations")
            #failed_runs_str = [str(f) for f in failed_runs]
            #self.parensemble = self.parensemble.drop(failed_runs)
            #self.obsensemble = self.obsensemble.drop(failed_runs)
            self.parensemble.loc[failed_runs,:] = np.NaN
            self.parensemble = self.parensemble.dropna()
            self.obsensemble.loc[failed_runs,:] = np.NaN
            self.obsensemble = self.obsensemble.dropna()

        if not self.parensemble.istransformed:
            self.parensemble._transform(inplace=True)
        if not self.parensemble_0.istransformed:
            self.parensemble_0._transform(inplace=True)

        self.phi = Phi(self)

        if self.drop_bad_reals is not None:
            #drop_idx = np.argwhere(self.current_phi_vec > self.drop_bad_reals).flatten()
            #comp_phi = self.phi.comp_phi
            #drop_idx = np.argwhere(self.phi.comp_phi > self.drop_bad_reals).flatten()
            #meas_phi = self.phi.meas_phi
            drop_idx = np.argwhere(self.phi.meas_phi > self.drop_bad_reals).flatten()
            run_ids = self.obsensemble.index.values
            drop_idx = run_ids[drop_idx]
            if len(drop_idx) == self.obsensemble.shape[0]:
                raise Exception("dropped all realizations as 'bad'")
            if len(drop_idx) > 0:
                self.logger.warn("{0} realizations dropped as 'bad' (indices :{1})".\
                                 format(len(drop_idx),','.join([str(d) for d in drop_idx])))
                self.parensemble.loc[drop_idx,:] = np.NaN
                self.parensemble = self.parensemble.dropna()
                self.obsensemble.loc[drop_idx,:] = np.NaN
                self.obsensemble = self.obsensemble.dropna()

                self.phi.update()

        self.phi.report(cur_lam=0.0)

        self.last_best_mean = self.phi.comp_phi.mean()
        self.last_best_std = self.phi.comp_phi.std()

        #self.logger.statement("initial phi (mean, std): {0:15.6G},{1:15.6G}".\
        #                      format(self.last_best_mean,self.last_best_std))
        if init_lambda is not None:
            self.current_lambda = float(init_lambda)
        else:
            #following chen and oliver
            x = self.last_best_mean / (2.0 * float(self.obsensemble.shape[1]))
            self.current_lambda = 10.0**(np.floor(np.log10(x)))

        self.logger.statement("current lambda:{0:15.6g}".format(self.current_lambda))

        self.delta_par_prior = self._calc_delta_par(self.parensemble_0)
        u,s,v = self.delta_par_prior.pseudo_inv_components(eigthresh=self.pst.svd_data.eigthresh)
        self.Am = u * s.inv
        if self.save_mats:
            np.savetxt(self.pst.filename.replace(".pst",'.') + "0.prior_par_diff.dat", self.delta_par_prior.x, fmt="%15.6e")
            np.savetxt(self.pst.filename.replace(".pst",'.') + "0.am_u.dat",u.x,fmt="%15.6e")
            np.savetxt(self.pst.filename.replace(".pst", '.') + "0.am_v.dat", v.x, fmt="%15.6e")
            np.savetxt(self.pst.filename.replace(".pst",'.') + "0.am_s_inv.dat", s.inv.as_2d, fmt="%15.6e")
            np.savetxt(self.pst.filename.replace(".pst",'.') + "0.am.dat", self.Am.x, fmt="%15.6e")

        self.__initialized = True

    def get_localizer(self):
        """ get an empty/generic localizer matrix that can be filled

        Returns
        -------
            localizer : pyemu.Matrix
                matrix with nnz obs names for rows and adj par names for columns

        """
        onames = self.pst.nnz_obs_names
        pnames = self.pst.adj_par_names
        localizer = Matrix(x=np.ones((len(onames),len(pnames))),row_names=onames,col_names=pnames)
        return localizer

    def _calc_delta_par(self,parensemble):
        '''
        calc the scaled parameter ensemble differences from the mean
        '''
        return self._calc_delta(parensemble, self.parcov_inv_sqrt)

    def _calc_delta_obs(self,obsensemble):
        '''
        calc the scaled observation ensemble differences from the mean
        '''
        return self._calc_delta(obsensemble.nonzero, self.obscov_inv_sqrt)


    def update(self,lambda_mults=[1.0],localizer=None,run_subset=None,use_approx=True,
               calc_only=False):
        """update the iES one GLM cycle

        Parameters
        ----------
            lambda_mults : list
                a list of lambda multipliers to test.  Each lambda mult value will require
                evaluating (a subset of) the parameter ensemble.
            localizer : pyemu.Matrix
                a jacobian localizing matrix
            run_subset : int
                the number of realizations to test for each lambda_mult value.  For example,
                if run_subset = 30 and num_reals=100, the first 30 realizations will be run (in
                parallel) for each lambda_mult value.  Then the best lambda_mult is selected and the
                remaining 70 realizations for that lambda_mult value are run (in parallel).
            use_approx : bool
                 a flag to use the MLE or MAP upgrade solution.  True indicates use MLE solution
            calc_only : bool
                a flag to calculate the upgrade matrix only (not run the ensemble). This is mostly for
                debugging and testing on travis. Default is False

        Example
        -------

        ``>>>import pyemu``

        ``>>>es = pyemu.EnsembleSmoother(pst="pest.pst")``

        ``>>>es.initialize(num_reals=100)``

        ``>>>es.update(lambda_mults=[0.1,1.0,10.0],run_subset=30)``

         """

        #if not self.parensemble.istransformed:
        #    self.parensemble._transform(inplace=False)

        if run_subset is not None:
            if run_subset >= self.obsensemble.shape[0]:
                self.logger.warn("run_subset ({0}) >= num of active reals ({1})...ignoring ".\
                                 format(run_subset,self.obsensemble.shape[0]))
                run_subset = None

        self.iter_num += 1
        mat_prefix =  self.pst.filename.replace('.pst','')+".{0}".format(self.iter_num)
        self.logger.log("iteration {0}".format(self.iter_num))
        self.logger.statement("{0} active realizations".format(self.obsensemble.shape[0]))
        if self.obsensemble.shape[0] < 2:
            self.logger.lraise("at least active 2 realizations (really like 300) are needed to update")
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
        u,s,v = scaled_delta_obs.pseudo_inv_components(eigthresh=self.pst.svd_data.eigthresh)
        self.logger.log("calculate pseudo inv comps")

        self.logger.log("calculate obs diff matrix")
        #obs_diff = self.obscov_inv_sqrt * self._get_residual_obs_matrix(self.obsensemble).T
        obs_diff = self.obscov_inv_sqrt * self.phi.get_residual_obs_matrix(self.obsensemble).T
        self.logger.log("calculate obs diff matrix")

        if self.save_mats:
            np.savetxt(mat_prefix + ".obs_diff.dat", scaled_delta_obs.x, fmt="%15.6e")
            np.savetxt(mat_prefix + ".par_diff.dat", scaled_delta_par.x, fmt="%15.6e")
            np.savetxt(mat_prefix + ".u.dat", u.x, fmt="%15.6e")
            np.savetxt(mat_prefix + ".s.dat", s.x, fmt="%15.6e")
            np.savetxt(mat_prefix + ".v.dat", v.x, fmt="%15.6e")
        # here is the math part...calculate upgrade matrices
        mean_lam,std_lam,paren_lam,obsen_lam = [],[],[],[]
        lam_vals = []
        for ilam,cur_lam_mult in enumerate(lambda_mults):

            parensemble_cur_lam = self.parensemble.copy()
            #print(parensemble_cur_lam.isnull().values.any())

            cur_lam = self.current_lambda * cur_lam_mult
            lam_vals.append(cur_lam)
            self.logger.log("calcs for  lambda {0}".format(cur_lam_mult))
            scaled_ident = Cov.identity_like(s) * (cur_lam+1.0)
            scaled_ident += s**2
            scaled_ident = scaled_ident.inv

            # build up this matrix as a single element so we can apply
            # localization
            self.logger.log("building upgrade_1 matrix")
            upgrade_1 = -1.0 * (self.parcov_inv_sqrt * scaled_delta_par) *\
                        v * s * scaled_ident * u.T
            if self.save_mats:
                np.savetxt(mat_prefix+".ivec.dat".format(self.iter_num), scaled_ident.x, fmt="%15.6e")
            self.logger.log("building upgrade_1 matrix")

            # apply localization
            if localizer is not None:
                self.logger.log("applying localization")
                upgrade_1.hadamard_product(localizer)
                self.logger.log("applying localization")

            # apply residual information
            self.logger.log("applying residuals")
            upgrade_1 *= obs_diff
            self.logger.log("applying residuals")

            self.logger.log("processing upgrade_1")
            if self.save_mats:
                np.savetxt(mat_prefix + ".upgrade_1.dat", upgrade_1.T.x, fmt="%15.6e")
            upgrade_1 = upgrade_1.to_dataframe()
            upgrade_1.index.name = "parnme"
            upgrade_1 = upgrade_1.T
            upgrade_1.index = [int(i) for i in upgrade_1.index]
            upgrade_1.to_csv(self.pst.filename+".upgrade_1.{0:04d}.csv".\
                               format(self.iter_num))
            if upgrade_1.isnull().values.any():
                    self.logger.lraise("NaNs in upgrade_1")
            self.logger.log("processing upgrade_1")

            #print(upgrade_1.isnull().values.any())
            #print(parensemble_cur_lam.index)
            #print(upgrade_1.index)
            parensemble_cur_lam += upgrade_1

            # parameter-based upgrade portion
            if not use_approx and self.iter_num > 1:
            #if True:
                self.logger.log("building upgrade_2 matrix")
                par_diff = (self.parensemble - self.parensemble_0.loc[self.parensemble.index,:]).\
                    as_pyemu_matrix().T
                x4 = self.Am.T * self.parcov_inv_sqrt * par_diff
                x5 = self.Am * x4
                x6 = scaled_delta_par.T * x5
                x7 = v * scaled_ident * v.T * x6
                ug2_mat = -1.0 * (self.parcov_inv_sqrt *
                                   scaled_delta_par * x7)
                upgrade_2 = ug2_mat.to_dataframe()
                upgrade_2.index.name = "parnme"
                upgrade_2 = upgrade_2.T
                upgrade_2.to_csv(self.pst.filename+".upgrade_2.{0:04d}.csv".\
                                   format(self.iter_num))
                upgrade_2.index = [int(i) for i in upgrade_2.index]

                if self.save_mats:
                    np.savetxt(mat_prefix + ".scaled_par_resid.dat", par_diff.x, fmt="%15.6e")
                    np.savetxt(mat_prefix + ".x4.dat", x4.x, fmt="%15.6e")
                    np.savetxt(mat_prefix + ".x5.dat", x5.x, fmt="%15.6e")
                    np.savetxt(mat_prefix + ".x6.dat", x6.x, fmt="%15.6e")
                    np.savetxt(mat_prefix + ".x7.dat", x7.x, fmt="%15.6e")
                    np.savetxt(mat_prefix + ".upgrade_2.dat", ug2_mat.T.x, fmt="%15.6e")

                if upgrade_2.isnull().values.any():
                    self.logger.lraise("NaNs in upgrade_2")

                parensemble_cur_lam += upgrade_2
                self.logger.log("building upgrade_2 matrix")
            self.logger.log("enforcing bounds")
            parensemble_cur_lam.enforce(self.enforce_bounds)
            self.logger.log("enforcing bounds")

            self.logger.log("filling fixed parameters")
            #fill in fixed pars with initial values
            fi = parensemble_cur_lam.fixed_indexer
            li = parensemble_cur_lam.log_indexer
            log_values = self.pst.parameter_data.loc[:,"parval1"].copy()
            log_values.loc[li] = log_values.loc[li].apply(np.log10)
            fixed_vals = log_values.loc[fi]

            for fname, fval in zip(fixed_vals.index, fixed_vals.values):
                # if fname not in df.columns:
                #    continue
                # print(fname)
                parensemble_cur_lam.loc[:, fname] = fval
            self.logger.log("filling fixed parameters")
            # this is for testing failed runs on upgrade testing
            # works with the 10par_xsec smoother test
            #parensemble_cur_lam.iloc[:,:] = -1000000.0

            # some hackery - we lose track of the transform flag here, but just
            # know it is transformed.  Need to create dataframe here because
            # pd.concat doesn't like par ensembles later
            paren_lam.append(pd.DataFrame(parensemble_cur_lam.loc[:,:]))
            self.logger.log("calcs for  lambda {0}".format(cur_lam_mult))

        if calc_only:
            return


        # subset if needed
        # and combine lambda par ensembles into one par ensemble for evaluation
        if run_subset is not None and run_subset < self.parensemble.shape[0]:
            #subset_idx = ["{0:d}".format(i) for i in np.random.randint(0,self.parensemble.shape[0]-1,run_subset)]
            subset_idx = self.parensemble.iloc[:run_subset,:].index.values
            self.logger.statement("subset idxs: " + ','.join([str(s) for s in subset_idx]))

            # more tracking of transformed - just know it! Creating dataframes...
            paren_lam_subset = [pe.loc[subset_idx,:] for pe in paren_lam]
            paren_combine = pd.concat(paren_lam_subset,ignore_index=True)
            paren_lam_subset = None
        else:
            subset_idx = self.parensemble.index.values
            paren_combine = pd.concat(paren_lam,ignore_index=True)


        self.logger.log("evaluating ensembles for lambdas : {0}".\
                        format(','.join(["{0:8.3E}".format(l) for l in lam_vals])))
        # back to par ensemble and know it is transformed
        paren_combine = ParameterEnsemble.from_dataframe(df=paren_combine,pst=self.pst,istransformed=True)
        failed_runs, obsen_combine = self._calc_obs(paren_combine)
        self.logger.log("evaluating ensembles for lambdas : {0}".\
                        format(','.join(["{0:8.3E}".format(l) for l in lam_vals])))
        paren_combine = None

        if failed_runs is not None and len(failed_runs) == obsen_combine.shape[0]:
                self.logger.lraise("all runs failed - cannot continue")


        # unpack lambda obs ensembles from combined obs ensemble
        nrun_per_lam = self.obsensemble.shape[0]
        if run_subset is not None:
            nrun_per_lam = run_subset
        obsen_lam = []

        for i in range(len(lam_vals)):
            sidx = i * nrun_per_lam
            eidx = sidx + nrun_per_lam
            oe = ObservationEnsemble.from_dataframe(df=obsen_combine.iloc[sidx:eidx,:].copy(),
                                                    pst=self.pst)
            oe.index = subset_idx
            # check for failed runs in this set - drop failed runs from obs ensembles
            if failed_runs is not None:
                failed_runs_this = np.array([f for f in failed_runs if f >= sidx and f < eidx]) - sidx
                if len(failed_runs_this) > 0:
                    if len(failed_runs_this) == oe.shape[0]:
                        self.logger.warn("all runs failed for lambda {0}".format(lam_vals[i]))
                    else:
                        self.logger.warn("{0} run failed for lambda {1}".\
                                         format(len(failed_runs_this),lam_vals[i]))
                    oe.iloc[failed_runs_this,:] = np.NaN
                    oe = oe.dropna()
                    paren_lam[i].iloc[failed_runs_this,:] = np.NaN
                    paren_lam[i] = ParameterEnsemble.from_dataframe(df=paren_lam[i].dropna(),pst=self.pst)
                    paren_lam[i].__instransformed = True

            # don't drop bad reals here, instead, mask bad reals in the lambda
            # selection and drop later
            # if self.drop_bad_reals is not None:
            #     assert isinstance(drop_bad_reals, float)
            #     drop_idx = np.argwhere(self.current_phi_vec > self.drop_bad_reals).flatten()
            #     run_ids = self.obsensemble.index.values
            #     drop_idx = run_ids[drop_idx]
            #     if len(drop_idx) == self.obsensemble.shape[0]:
            #         raise Exception("dropped all realizations as 'bad'")
            #     if len(drop_idx) > 0:
            #         self.logger.warn("{0} realizations dropped as 'bad' (indices :{1})". \
            #                          format(len(drop_idx), ','.join([str(d) for d in drop_idx])))
            #         self.parensemble.loc[drop_idx, :] = np.NaN
            #         self.parensemble = self.parensemble.dropna()
            #         self.obsensemble.loc[drop_idx, :] = np.NaN
            #         self.obsensemble = self.obsensemble.dropna()
            #
            #         self.current_phi_vec = self._calc_phi_vec(self.obsensemble)

            obsen_lam.append(oe)
        obsen_combine = None

        # here is where we need to select out the "best" lambda par and obs
        # ensembles

        #phi_vecs = [self._calc_phi_vec(obsen) for obsen in obsen_lam]
        #phi_vecs_reg = [self._calc_regul_phi_vec(paren) for paren in paren_lam]
        #if self.regul_factor > 0.0:
        #    for i,(pv,prv) in enumerate(zip(phi_vecs,phi_vecs_reg)):
        #        phi_vecs[i] = pv + (prv * self.regul_factor)
        self.logger.log("calc lambda phi vectors")
        phi_vecs = [self.phi.get_meas_and_regul_phi(oe,pe.loc[oe.index,:]) for oe,pe in zip(obsen_lam,paren_lam)]
        self.logger.log("calc lambda phi vectors")
        if self.drop_bad_reals is not None:
            for i,(meas_pv,regul_pv) in enumerate(phi_vecs):
                #for testing the drop_bad_reals functionality
                #pv[[0,3,7]] = self.drop_bad_reals + 1.0
                regul_pv = regul_pv.copy()
                regul_pv[meas_pv>self.drop_bad_reals] = np.NaN
                regul_pv = regul_pv[~np.isnan(regul_pv)]
                meas_pv[meas_pv>self.drop_bad_reals] = np.NaN
                meas_pv = meas_pv[~np.isnan(meas_pv)]
                if len(meas_pv) == 0:
                    #raise Exception("all realization for lambda {0} dropped as 'bad'".\
                    #                format(lam_vals[i]))
                    self.logger.warn("all realizations for lambda {0} marked as 'bad'")
                    meas_pv = np.zeros_like(obsen_lam[0].shape[0]) + 1.0e+30
                    regul_pv = np.zeros_like(obsen_lam[0].shape[0]) + 1.0e+30
                phi_vecs[i] = (meas_pv,regul_pv)
        mean_std_meas = [(pv[0].mean(),pv[0].std()) for pv in phi_vecs]
        mean_std_regul = [(pv[1].mean(), pv[1].std()) for pv in phi_vecs]
        update_pars = False
        update_lambda = False
        self.logger.statement("**************************")
        # self.logger.statement(str(datetime.now()))
        self.logger.statement("lambda testing summary")
        self.logger.statement("total runs:{0}".format(self.total_runs))
        self.logger.statement("iteration: {0}".format(self.iter_num))
        self.logger.statement("current lambda:{0:15.6G}, mean:{1:15.6G}, std:{2:15.6G}". \
                              format(self.current_lambda,
                                     self.last_best_mean, self.last_best_std))

        # accept a new best if its within 10%
        best_mean = self.last_best_mean * 1.1
        best_std = self.last_best_std * 1.1
        best_i = 0
        for i,((mm,ms),(rm,rs)) in enumerate(zip(mean_std_meas,mean_std_regul)):
            self.logger.statement(" tested lambda:{0:15.6G}, meas mean:{1:15.6G}, meas std:{2:15.6G}".
                                  format(self.current_lambda * lambda_mults[i],mm,ms))
            self.logger.statement("{0:30s}regul mean:{1:15.6G}, regul std:{2:15.6G}".\
                                  format(' ',rm,rs))
            m = mm + (self.regul_factor * rm)
            s = ms + (self.regul_factor * rs)
            if m < best_mean:
                update_pars = True
                best_mean = m
                best_i = i
                if s < best_std:
                    update_lambda = True
                    best_std = s
        if np.isnan(best_mean):
            self.logger.lraise("best mean = NaN")
        if np.isnan(best_std):
            self.logger.lraise("best std = NaN")

        if not update_pars:
            self.current_lambda *= max(lambda_mults) * 10.0
            self.current_lambda = min(self.current_lambda,100000)
            self.logger.statement("not accepting iteration, increased lambda:{0}".\
                  format(self.current_lambda))
        else:
            #more transformation status hard coding - ugly
            self.parensemble = ParameterEnsemble.from_dataframe(df=paren_lam[best_i],pst=self.pst,istransformed=True)
            if run_subset is not None:
                failed_runs, self.obsensemble = self._calc_obs(self.parensemble)
                if failed_runs is not None:
                    self.logger.warn("dropping failed realizations")
                    self.parensemble.loc[failed_runs, :] = np.NaN
                    self.parensemble = self.parensemble.dropna()
                    self.obsensemble.loc[failed_runs, :] = np.NaN
                    self.obsensemble = self.obsensemble.dropna()

                self.phi.update()
                best_mean = self.phi.comp_phi.mean()
                best_std = self.phi.comp_phi.std()
            else:
                self.obsensemble = obsen_lam[best_i]
                # reindex parensemble in case failed runs
                self.parensemble = ParameterEnsemble.from_dataframe(df=self.parensemble.loc[self.obsensemble.index],
                                                                    pst=self.pst,
                                                                    istransformed=self.parensemble.istransformed)
                self.phi.update()
            if self.drop_bad_reals is not None:
                # for testing drop_bad_reals functionality
                # self.current_phi_vec[::2] = self.drop_bad_reals + 1.0
                #drop_idx = np.argwhere(self.current_phi_vec > self.drop_bad_reals).flatten()
                drop_idx = np.argwhere(self.phi.comp_phi > self.drop_bad_reals).flatten()
                run_ids = self.obsensemble.index.values
                drop_idx = run_ids[drop_idx]
                if len(drop_idx) > self.obsensemble.shape[0] - 3:
                    raise Exception("dropped too many realizations as 'bad'")
                if len(drop_idx) > 0:
                    self.logger.warn("{0} realizations dropped as 'bad' (indices :{1})". \
                                     format(len(drop_idx), ','.join([str(d) for d in drop_idx])))
                    self.parensemble.loc[drop_idx, :] = np.NaN
                    self.parensemble = self.parensemble.dropna()
                    self.obsensemble.loc[drop_idx, :] = np.NaN
                    self.obsensemble = self.obsensemble.dropna()

                    self.phi.update()
                    best_mean = self.phi.comp_phi.mean()
                    best_std = self.phi.comp_phi.std()

            self.phi.report(cur_lam=self.current_lambda * lambda_mults[best_i])

            self.logger.statement("   best lambda:{0:15.6G}, mean:{1:15.6G}, std:{2:15.6G}".\
                  format(self.current_lambda*lambda_mults[best_i],
                         best_mean,best_std))
            #self.logger.statement("   actual mean phi: {0:15.6G}".format(float(self.current_actual_phi.mean())))
            self.last_best_mean = best_mean
            self.last_best_std = best_std


        if update_lambda:
            # be aggressive
            self.current_lambda *= (lambda_mults[best_i] * 0.75)
            # but don't let lambda get too small
            self.current_lambda = max(self.current_lambda,0.00001)
            self.logger.statement("updating lambda: {0:15.6G}".\
                  format(self.current_lambda ))

        self.logger.statement("**************************\n")
        self.parensemble.to_csv(self.pst.filename+self.paren_prefix.\
                                    format(self.iter_num))
        self.obsensemble.to_csv(self.pst.filename+self.obsen_prefix.\
                                    format(self.iter_num))
        if self.raw_sweep_out is not None:
            self.raw_sweep_out.to_csv(self.pst.filename+"_sweepraw{0}.csv".\
                                        format(self.iter_num))
        self.logger.log("iteration {0}".format(self.iter_num))
