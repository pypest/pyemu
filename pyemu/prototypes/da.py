import os
import sys
from shutil import copyfile
import multiprocessing as mp
import copy
import numpy as np
import pandas as pd
import pyemu
# from pyemu.prototypes.sampler import Parstat
from .ensemble_method import EnsembleMethod
from pyemu.en import ParameterEnsemble, ObservationEnsemble
from pyemu.mat import Cov
from ..logger import Logger
from pyemu.pst import Pst
from pyemu.mat import Matrix
from pyemu.pst.pst_utils import write_to_template, parse_tpl_file, parse_ins_file
from scipy.linalg import blas as blas
from scipy.linalg import lapack as lap
from scipy.stats import ortho_group


class Assimilator(EnsembleMethod):
    def __init__(self, type='Smoother', iterate=False, pst=None, mode='stochastic',
                 parens=None, obs_ens=None, num_slaves=0, use_approx_prior=True,
                 submit_file=None, verbose=False, port=4004, slave_dir="template", num_real=100,
                 enforce_bounds=True, parcov=None, obscov=None):
        """
        A clase to implement data assimilation including EnKF, ES, and EnKS and More
        """
        self.__init2__(pst=pst, num_slaves=num_slaves, submit_file=submit_file,
                       verbose=verbose, port=port, slave_dir=slave_dir)
        self.obscov = obscov
        self.parcov = parcov

        self.mode_options = ['stochastic', 'deterministic']
        self.type_options = ['Smoother', 'Kalman Filter', 'Kalman Smoother']

        if not isinstance(iterate, bool):
            raise ValueError("Error, 'iterate' must be boolian")  # I think warning would better
        self.iterate = iterate  # If true Chen-Oliver scheme will be used, otherwise classical data assimilation will used.

        if type not in self.type_options:
            raise ValueError("Assimilation type [{}] specified is not supported".format(type))
        self.type = type

        if mode not in self.mode_options:
            raise ValueError("Assimilation mode [{}] specified is not supported".format(mode))
        self.mode = mode

        # Obtain problem setting from the pst object.
        if isinstance(pst, pyemu.pst.Pst):
            self.pst = pst
        else:  # TODO: it is better if we check if this is a file, throw exception otherwise
            self.pst = pyemu.Pst(pst)
        self.obs_ens = None
        self.par_ens = None
        if not (parens is None):
            self.par_ens = ParameterEnsemble.from_dataframe(pst=self.pst, df=parens)
        if not (obs_ens is None):
            # sys.setrecursionlimit(10000)
            self.obs_ens = ObservationEnsemble.from_dataframe(pst=self.pst, df=obs_ens)
        self.num_real = num_real
        self.enforce_bounds = enforce_bounds



    def __init2__(self, pst, num_slaves=0, use_approx_prior=True,
                  submit_file=None, verbose=False, port=4004, slave_dir="template"):
        """
        (This might need to be rewritten)
        The goal of this function is to bypass the __init__ function in esembble method
        so that a covariance matrix is not passed since it can be huge and there is no need to computed it
        explicitly
        :return:
        """
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

        if isinstance(pst, str):
            pst = Pst(pst)
        assert isinstance(pst, Pst)
        self.pst = pst
        self.sweep_in_csv = pst.pestpp_options.get("sweep_parameter_csv_file", "sweep_in.csv")
        self.sweep_out_csv = pst.pestpp_options.get("sweep_output_csv_file", "sweep_out.csv")

        self.parcov = None
        self.obscov = None

        self._initialized = False
        self.iter_num = 0
        self.total_runs = 0
        self.raw_sweep_out = None

    def initialize(self):
        ### The inialization is modefied from the Smoother

        """Initialize.  Depending on arguments, draws or loads
        initial parameter observations ensembles and runs the initial parameter
        ensemble

        Parameters
        ----------
            num_reals : int
                the number of realizations to draw.  Ignored if parensemble/obsensemble
                are not None
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

        """
        num_reals = self.num_real
        enforce_bounds = "reset"
        parensemble = self.par_ens
        obsensemble = self.obs_ens
        restart_obsensemble = None

        build_empirical_prior = False

        # initialize the phi report csv
        self.enforce_bounds = enforce_bounds

        self.total_runs = 0
        # this matrix gets used a lot, so only calc once and store
        if False:  # todo: not sure what is that...
            self.obscov_inv_sqrt = self.obscov.get(self.pst.nnz_obs_names).inv.sqrt

            self.logger.log("forming inverse sqrt parcov matrix")
            self.parcov_inv_sqrt = self.parcov.inv.sqrt
            self.logger.log("forming inverse sqrt parcov matrix")

        # load or generate parameter ensmble
        if parensemble is not None:
            self.logger.log("initializing with existing ensembles")
            if isinstance(parensemble, str):
                self.logger.log("loading parensemble from file")
                if not os.path.exists(obsensemble):
                    self.logger.lraise("can not find parensemble file: {0}". \
                                       format(parensemble))
                df = pd.read_csv(parensemble, index_col=0)
                # df.index = [str(i) for i in df.index]
                self.parensemble_0 = ParameterEnsemble.from_dataframe(df=df, pst=self.pst)
                self.logger.log("loading parensemble from file")

            elif isinstance(parensemble, ParameterEnsemble):
                self.parensemble_0 = parensemble.copy()
            else:
                raise Exception("unrecognized arg type for parensemble, " + \
                                "should be filename or ParameterEnsemble" + \
                                ", not {0}".format(type(parensemble)))
            self.parensemble = self.parensemble_0.copy()


        else:
            if build_empirical_prior:
                self.logger.lraise("can't use build_emprirical_prior without parensemble...")
            self.logger.log("initializing with {0} realizations".format(num_reals))
            self.logger.log("initializing parensemble")
            self.parensemble_0 = pyemu.ParameterEnsemble.from_gaussian_draw(self.pst,
                                                                            self.parcov, num_reals=num_reals)
            self.parensemble_0.enforce(enforce_bounds=enforce_bounds)
            self.logger.log("initializing parensemble")
            self.parensemble = self.parensemble_0.copy()
            self.parensemble_0.to_csv(self.pst.filename + \
                                      self.paren_prefix.format(0))
            self.logger.log("initializing parensemble")
            self.logger.log("initializing obsensemble")

        # load or generate obs error ensemble
        if obsensemble is not None:
            self.logger.log("initializing with existing ensembles")

            if isinstance(obsensemble, str):
                self.logger.log("loading obsensemble from file")
                if not os.path.exists(obsensemble):
                    self.logger.lraise("can not find obsensemble file: {0}". \
                                       format(obsensemble))
                df = pd.read_csv(obsensemble, index_col=0).loc[:, self.pst.nnz_obs_names]
                # df.index = [str(i) for i in df.index]
                self.obsensemble_0 = ObservationEnsemble.from_dataframe(df=df, pst=self.pst)
                self.logger.log("loading obsensemble from file")

            elif isinstance(obsensemble, ObservationEnsemble):
                self.obsensemble_0 = obsensemble.copy()
            else:
                raise Exception("unrecognized arg type for obsensemble, " + \
                                "should be filename or ObservationEnsemble" + \
                                ", not {0}".format(type(obsensemble)))

            assert self.parensemble_0.shape[0] == self.obsensemble_0.shape[0]
            num_reals = self.parensemble.shape[0]
            self.logger.log("initializing with existing ensembles")

        else:
            if build_empirical_prior:
                self.logger.lraise("can't use build_emprirical_prior without parensemble...")
            self.obsensemble_0 = pyemu.ObservationEnsemble.from_id_gaussian_draw(self.pst,
                                                                                 num_reals=num_reals)

            self.obsensemble_0.to_csv(self.pst.filename + \
                                      self.obsen_prefix.format(-1))
            self.logger.log("initializing obsensemble")
            self.logger.log("initializing with {0} realizations".format(num_reals))

        self.enforce_bounds = enforce_bounds

        if False:  # deactiavet for now
            # todo: Jeremy, what is this?
            if restart_obsensemble is not None:
                self.logger.log("loading restart_obsensemble {0}".format(restart_obsensemble))
                failed_runs, self.obsensemble = self._load_obs_ensemble(restart_obsensemble)
                assert self.obsensemble.shape[0] == self.obsensemble_0.shape[0]
                assert list(self.obsensemble.columns) == list(self.obsensemble_0.columns)
                self.logger.log("loading restart_obsensemble {0}".format(restart_obsensemble))

            else:
                # run the initial parameter ensemble
                self.logger.log("evaluating initial ensembles")
                self.obsensemble = self.forecast()
                self.logger.log("evaluating initial ensembles")

        # Transformation
        if not self.parensemble.istransformed:
            self.parensemble._transform(inplace=True)
        if not self.parensemble_0.istransformed:
            self.parensemble_0._transform(inplace=True)
        self._initialized = True

    def forcast(self):
        """ This simply moves the ensemble forward by running the model
               once for each realization"""

        if not (self.parensemble is None):
            parensemble = self.parensemble
        self.logger.log("evaluating ensemble")
        failed_runs, obsensemble = self._calc_obs(self.parensemble)
        obsensemble_0 = self.obsensemble_0
        if failed_runs is not None:
            self.logger.warn("dropping failed realizations")
            parensemble.loc[failed_runs, :] = np.NaN
            parensemble = parensemble.dropna()
            obsensemble.loc[failed_runs, :] = np.NaN
            obsensemble = obsensemble.dropna()
            obsensemble_0.loc[failed_runs, :] = np.NaN
            obsensemble_0 = obsensemble_0.dropna()
            self.obsensemble = obsensemble
            self.parensemble = parensemble
            self.obsensemble_0 = obsensemble_0
        else:
            self.obsensemble = obsensemble
        self.failed_runs = failed_runs
        self.logger.log("evaluating ensemble")

    def _calc_delta_obs(self, obsensemble, scaled=False):
        """
        compute the ensemble of deviation of model observable predictions from the meam
        :return:
        """
        if scaled:
            return self._calc_delta(obsensemble.nonzero, self.obscov_inv_sqrt)
        else:
            return self._calc_delta(obsensemble.nonzero, None)

    def _calc_delta_par(self, parensemble, scaled=False):
        '''
        calc the scaled parameter ensemble differences from the mean
        '''
        if scaled:
            return self._calc_delta(parensemble, self.parcov_inv_sqrt)
        else:
            return self._calc_delta(parensemble, None)


    def update(self, truncation = 1e-3):

        self.truncation = truncation

        if self.mode == 'stochastic':
            self.stochastic_update()

        else:
            """
            Determistic update (or least square filtering)
            """
            # Xa' = Xf' + C'H(H'CH + R) (do' - df')

            self.deterministic_update()


    def stochastic_update(self):

        """
        Solve the anlaysis step
        Xa = Xf + Cpo(Coo+ R) (do - df)
        where Xa is the updated (posterior) ensemble of system parameters and (or) states
              Xf is the forecast (prior) ensemble
              Cpo is the cross covariance matrix of parameters and observation
              Coo is the covariance matrix of observable predictions

        :return:
        """

        self.iter_num += 1
        mat_prefix = self.pst.filename.replace('.pst', '') + ".{0}".format(self.iter_num)
        self.logger.log("iteration {0}".format(self.iter_num))
        self.logger.statement("{0} active realizations".format(self.obsensemble.shape[0]))
        if self.obsensemble.shape[0] < 2:
            self.logger.lraise("at least active 2 realizations (really like 300) are needed to update")
        if not self._initialized:
            self.logger.lraise("must call initialize() before update()")

        self.logger.log("calculate scaled delta obs")

        delta_obs = self._calc_delta_obs(obsensemble=self.obsensemble, scaled=False)
        self.logger.log("calculate scaled delta obs")
        self.logger.log("calculate scaled delta par")
        delta_par = self._calc_delta_par(parensemble=self.parensemble, scaled=False)
        self.logger.log("calculate scaled delta par")

        # error ensemble
        obs_nms = self.pst.nnz_obs_names
        if self.type == 'Kalman Filter':
            stat_nm = self.dynamic_states['parnme'].values
            for snm in stat_nm:
                obs_nms.remove(snm)
            pass

        err_ens = self.obsensemble_0.loc[:, obs_nms] - \
                  self.pst.observation_data.loc[obs_nms, :]['obsval']
        err_ens = err_ens / np.sqrt(err_ens.shape[0] - 1)
        err_ens = Matrix(x=err_ens.values, row_names=err_ens.index, col_names=err_ens.columns)

        # Innovation matrix (deviation arround the mean)
        Ddash = self.obsensemble_0.loc[:, obs_nms] - self.obsensemble.loc[:, obs_nms]
        Ddash = Matrix(x=Ddash.values, row_names=Ddash.index, col_names=Ddash.columns)
        Ddash = Ddash.T

        C = delta_obs.T + err_ens.T
        m, N = C.shape
        try:
            u, s, v = np.linalg.svd(C.as_2d)
        except:
            xxccc = 1
            pass
        ns = len(s)
        s_perc = 100.0 * s / np.sum(s)
        s_ = np.power(s, -2.0)
        s_[s_perc < self.truncation] = 0.0
        ss_ = np.zeros((m, m))
        np.fill_diagonal(ss_, s_)
        if m <= N:
            u = u[:, :ns]
            ss_ = ss_[:ns, :ns]

        X1 = np.dot(ss_, u.T)
        X1 = np.dot(X1, Ddash.as_2d)
        X1 = np.dot(u, X1)
        delobs = delta_obs.df().loc[:, obs_nms]
        #X1 = np.dot(delta_obs.as_2d, X1)
        X1 = np.dot(delobs, X1)

        # This is the change in parameter values
        del_par = np.dot(delta_par.as_2d.T, X1)
        self.parensemble_a = (self.parensemble.T + del_par).T
        self.parensemble_a = ParameterEnsemble.from_dataframe(df=self.parensemble_a,
                                                              pst=self.pst,
                                                              istransformed=True)

    def deterministic_update(self):

        """
        Solve the anlaysis step
        Xa = Xf + Cpo(Coo+ R) (do - df)
        where Xa is the updated (posterior) ensemble of system parameters and (or) states
              Xf is the forecast (prior) ensemble
              Cpo is the cross covariance matrix of parameters and observation
              Coo is the covariance matrix of observable predictions

        :return:
        """

        self.iter_num += 1
        mat_prefix = self.pst.filename.replace('.pst', '') + ".{0}".format(self.iter_num)
        self.logger.log("iteration {0}".format(self.iter_num))
        self.logger.statement("{0} active realizations".format(self.obsensemble.shape[0]))
        if self.obsensemble.shape[0] < 2:
            self.logger.lraise("at least active 2 realizations (really like 300) are needed to update")
        if not self._initialized:
            self.logger.lraise("must call initialize() before update()")

        self.logger.log("calculate scaled delta obs")

        delta_obs = self._calc_delta_obs(obsensemble=self.obsensemble, scaled=False)
        self.logger.log("calculate scaled delta obs")
        self.logger.log("calculate scaled delta par")
        delta_par = self._calc_delta_par(parensemble=self.parensemble, scaled=False)
        self.logger.log("calculate scaled delta par")

        # error ensemble
        obs_nms = self.pst.nnz_obs_names
        if self.type == 'Kalman Filter':
            stat_nm = self.dynamic_states['parnme'].values
            for snm in stat_nm:
                obs_nms.remove(snm)
            pass

        err_ens = self.obsensemble_0.loc[:, obs_nms] - \
                  self.pst.observation_data.loc[obs_nms, :]['obsval']
        err_ens = err_ens / np.sqrt(err_ens.shape[0] - 1)
        err_ens = Matrix(x=err_ens.values, row_names=err_ens.index, col_names=err_ens.columns)

        # Innovation matrix (deviation arround the mean)
        Ddash = self.obsensemble_0.loc[:, obs_nms] - self.obsensemble.loc[:, obs_nms]
        Ddash = Matrix(x=Ddash.values, row_names=Ddash.index, col_names=Ddash.columns)
        Ddash = Ddash.T

        C = delta_obs.T + err_ens.T
        m, N = C.shape

        #------------------------------
        prior_k_mean = self.parensemble.mean().values
        prior_h_mean = self.obsensemble.mean().values
        H_dash = delta_obs.as_2d.T
        K_dash = delta_par.as_2d.T
        innov = self.pst.observation_data['obsval'].values - prior_h_mean

        # SVD of matrix C
        u, s, vt, ierr = lap.dgesvd(C.as_2d)
        if ierr != 0: ValueError('Sqrt_KF: ierr from call dgesvd = {}'.format(ierr))

        sig = s.copy()

        s_ = np.power(s, 2.0)
        sums_ = np.sum(s_)
        if self.truncation is None:
            s_perc = s_ / np.sum(s_)
            truncation = self.truncation_percent / 100.0
            s_ = s_[s_perc >= truncation]
        else:
            truncation = self.truncation
            s_ = s_[s_ >= truncation]

        p = len(s_)
        print('      analysis: dominant sing. values and'
              ' share {}, {}'.format(p, 100.0 * (np.sum(s_) / sums_)))

        s_ = 1.0 / s_

        u_ = u[:, 0:p]

        x1 = s_[:, np.newaxis] * u_.T
        x2 = blas.dgemv(alpha=1, a=x1, x=innov)
        x3 = blas.dgemv(alpha=1, a=u_, x=x2)
        x4 = blas.dgemv(alpha=1, a=H_dash.T, x=x3)

        Ka = prior_k_mean + blas.dgemv(alpha=1, a=K_dash, x=x4)

        # Compute perturbation
        # Xa = Xf*Z*(sig^0.5)
        # Z.Sig.Zt =  I - Y(C^-1)Y

        # compute C^-1
        sig = np.power(sig[0:p], -2.0)
        x2 = sig[:, np.newaxis] * u_.T
        c_1 = blas.dgemm(alpha=1, a=u_, b=x2)

        # I - Y(C^-1)Y
        c_1 = blas.dgemm(alpha=1, a=c_1, b=H_dash)
        c_1 = blas.dgemm(alpha=1, a=H_dash.T, b=c_1)
        diag = 1 - np.diag(c_1)
        np.fill_diagonal(c_1, diag)

        # decompose I - Y(C^-1)Y
        u2, sig2, vt2, ierr = lap.dgesvd(c_1)
        sig2[sig2 < 0] = 0
        sig2 = np.power(sig2, 0.5)
        p2 = len(sig2)
        if p2 < N:
            sig2 = np.append(sig2, np.zeros(N - p2))
        x2 = u2 * sig2[np.newaxis, :]
        x2 = blas.dgemm(alpha=1.0, a=x2, b=vt2)

        x2 = blas.dgemm(alpha=1, a=K_dash, b=x2)
        theta = ortho_group.rvs(N)
        x2 = blas.dgemm(alpha=1, a=x2, b=theta.T)
        Aa = Ka[:, np.newaxis] + x2
        Aa = pd.DataFrame(Aa.T, columns = self.parensemble.columns)

        self.parensemble_a = ParameterEnsemble.from_dataframe(df=Aa,
                                                              pst=self.pst,
                                                              istransformed=True)

    def analysis(self):
        """

        :return:
        """
        # TODO: add exceptions and warnings
        if self.type == "Smoother":
            self.smoother()

        elif self.type == 'Kalman Filter':
            self.enkf()

        elif self.type == "Kalman_smoother":
            self.enks()
        else:
            print("You should'nt be here....")  # TODO: ???

    def smoother(self):
        if self.iterate:
            # TODO: Chen_oliver algorithim
            pass
        else:
            # prepare prior ensembles ...
            self.initialize()

            # run forward models ...
            self.forcast()

            #  compute the posteriors...
            self.update()

    def enkf(self):
        """
        Ensemble Kalman Filter (EnKF)
        Successively update parameters and states
        """

        # Global pst is to hold data about all cycles, while pst will hold data about current cycle.
        self.global_pst = self.pst
        self.pst = []
        cycles = self.global_pst.observation_data['cycle'].unique()
        cycles = np.sort(cycles).tolist()
        self.all_ensembles = []
        if self.iterate:
            # TODO: Chen_oliver algorithm
            pass
        else:
            for icycle in cycles:
                if icycle < 0:
                    continue
                self.logger.log("data assimilation for cycle = {}".format(icycle))

                # before assimilation, misc parameters are updated.
                self.update_misc_pars(icycle)

                # generate child pst object for current cycle
                self.generate_child_pst(icycle)

                if icycle == 0:
                    # first cycle initialization
                    temp_pst = self.pst # todo: this is ugly
                    self.pst = self.global_pst
                    self.initialize()
                    self.pst = temp_pst
                else:

                    self.parensemble.index = np.arange(len(self.parensemble))
                    self.obsensemble_0.index = np.arange(len(self.obsensemble_0))
                    self.obsensemble.index = np.arange(len(self.obsensemble))

                # run forward models ...
                self.forcast()

                # Before updating, we need to setup the forecast matrix, which consists of
                # prior static parameters and prior dynamic states. Notice that prior dynamic
                # states are model output results from
                stat_nm_L = self.dynamic_states['parnme'].values.tolist()
                stat_nm_U = [nm.upper() for nm in stat_nm_L]
                sweep_out = pd.read_csv(self.pst.pestpp_options.get("sweep_output_csv_file", "sweep_out.csv"))
                if not(self.failed_runs is None):
                    sweep_out.loc[self.failed_runs, :] = np.NaN
                    sweep_out = sweep_out.dropna()
                    sweep_out.index = self.parensemble.index
                self.parensemble[stat_nm_L] = sweep_out[stat_nm_U]
                if True:
                    self.parensemble.to_csv("cycle_{}_prior.csv".format(icycle))

                #  compute the posteriors...
                # todo: force min/max values of parameters after updated
                self.update()
                # todo: give the user the option to write output
                self.parensemble_a.to_csv("cycle_{}_posterior.csv".format(icycle))
                self.parensemble = self.parensemble_a


    def generate_child_pst(self, icycle):
        """

        :param icycle:
        :return:
        """
        # get current static parameters and remove misc parameters as they will not participate in DA
        mask = (self.global_pst.parameter_data['cycle'] < 0) | (self.global_pst.parameter_data['cycle'] == icycle)
        param = self.global_pst.parameter_data[mask]
        param = param[~param['parnme'].isin(self.local_misc_par)]

        # get curret obs
        mask = (self.global_pst.observation_data['cycle'] < 0) | (self.global_pst.observation_data['cycle'] == icycle)
        obs = self.global_pst.observation_data[mask] # this includes both states and obs

        # get in/out files
        mask = (self.global_pst.io_files['cycle'] < 0) | (self.global_pst.io_files['cycle'] == icycle)
        curr_files = self.global_pst.io_files[mask]
        curr_files = curr_files[curr_files['Type'] != 'misc']

        # collect in/out files and ins/tpl
        infiles = [];
        outfiles = [];
        insfiles = [];
        tplfiles = []
        stats = []
        for iline, rec in curr_files.iterrows():
            if rec['tpl/ins'].split('.')[-1] == 'tpl':
                infiles.append(rec['in/out'])
                tplfiles.append(rec['tpl/ins'])
            elif rec['tpl/ins'].split('.')[-1] == 'ins':
                outfiles.append(rec['in/out'])
                insfiles.append(rec['tpl/ins'])
                if rec['Type'] == 'stat':
                    stat_ = parse_ins_file(os.path.join(self.slave_dir, rec['tpl/ins']))
                    stats = stats + stat_
            else:
                raise ValueError("Unrecognized file type. Only tpl and ins are allowed ")

        # dynamic state should be in both obs and param sections
        obs_states = obs[obs['obsnme'].isin(stats)]
        dynamic_states = pd.DataFrame(columns=param.columns)
        dynamic_states['parnme'] = obs_states['obsnme'].values
        dynamic_states['partrans'] = 'none' # todo: we need to expose this to users
        dynamic_states['parval1'] = obs_states['obsval'].values
        dynamic_states['pargp'] = obs_states['obgnme'].values
        dynamic_states.index = obs_states['obsnme'].values
        self.dynamic_states = dynamic_states

        # generate pst
        stat_par_names = param['parnme'].values.tolist() + dynamic_states['parnme'].values.tolist()
        pst = pyemu.Pst.from_par_obs_names(par_names=stat_par_names,
                                           obs_names=obs['obsnme'].values)
        pst.control_data.noptmax = 0
        pst.pestpp_options["sweep_parameter_csv_file"] = self.global_pst.pestpp_options["sweep_parameter_csv_file"]
        pst.model_command = self.global_pst.model_command

        pst.input_files = infiles
        pst.output_files = outfiles
        pst.template_files = tplfiles
        pst.instruction_files = insfiles

        param_stat = pd.concat([param, dynamic_states])
        for field in pst.parameter_data.columns:
            if field in param_stat.columns:
                pst.parameter_data[field] = param_stat[field].values
        for field in pst.observation_data.columns:
            if field in obs.columns:
                pst.observation_data[field] = obs[field].values

        fname = os.path.splitext(self.global_pst.filename)[0] + "_{}".format(icycle) + ".pst"
        pst_file = os.path.join(self.slave_dir, fname)
        pst.filename = fname
        pst.write(new_filename=pst_file)

        # Todo: This is bad ...!!! we need to allow the user to specify the master folder in a better way
        copyfile(pst_file, fname)
        self.pst = pst

        # update pst obj associtaed with par_ens and obs_env
        if isinstance(self.par_ens, pyemu.en.ParameterEnsemble):
            self.par_ens.pst = pst
        if isinstance(self.obs_ens, pyemu.en.ParameterEnsemble):
            self.obs_ens.pst = pst

    def update_misc_pars(self, icycle):
        """
        Update model input files that are related to miscellaneous parameters.
        Miscellaneous parameters are any input parameter that will not be updated in DA
        :param icycle:
        :return:
        """
        self.logger.log("updating miscellaneous model input files for cycle number {} ...".format(icycle))
        io_files = self.global_pst.io_files[
            (self.global_pst.io_files["cycle"] < 0) | (self.global_pst.io_files["cycle"] == icycle)]

        # update misc files
        misc_io_files = io_files[io_files['Type'] == 'misc']
        unique_iofiles = misc_io_files['in/out'].unique()
        self.local_misc_par = []  # clear record in case it is populated by previous cycle.
        for par_file in unique_iofiles:
            tpl_file = misc_io_files.loc[misc_io_files['in/out'] == par_file, 'tpl/ins'].values[0]
            tpl_file = os.path.join(self.slave_dir, tpl_file)
            parnames = parse_tpl_file(tpl_file)
            self.local_misc_par = self.local_misc_par + parnames
            parvals = {}
            for par in parnames:
                val = self.global_pst.parameter_data[self.global_pst.parameter_data['parnme']
                                                     == par]['parval1'].values[0]
                parvals[par] = val

            write_to_template(parvals, tpl_file, os.path.join(self.slave_dir, par_file))



if __name__ == "__main__":
    sm = Assimilator(type='Smoother', iterate=False, mode='Stochastic', pst='pst.control')
    sm.analysis()

    # Ensemble Kalman Filter
    kf = Assimilator(type='Kalman_filter', pst='pst.control')
    kf.analysis()

    # Kalman Smoother
    ks = Assimilator(type='Kalman_smoother', pst='pst.control')
    ks.analysis()
