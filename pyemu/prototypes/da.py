import os
import sys
import multiprocessing as mp
import copy
import numpy as np

import pyemu
# from pyemu.prototypes.sampler import Parstat
from .ensemble_method import EnsembleMethod
from pyemu.en import ParameterEnsemble, ObservationEnsemble
from pyemu.mat import Cov
from ..logger import Logger
from pyemu.pst import Pst
from pyemu.mat import Matrix


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

        self.threshold_percent = 0.1  # 0.1% of eigen will be removed

        # Time Cycle info. example
        # {[ ['file_1.tpl', 'file_1.dat'],
        #    ['file_2.tpl', 'file_2.dat'],
        #    ['out_file.ins', 'out_file.out']}

        self.cycle_update_files = {}

    def __init2__(self, pst, num_slaves=0, use_approx_prior=True,
                  submit_file=None, verbose=False, port=4004, slave_dir="template"):
        """
        (This might need to be rewritten)
        The goal of this function is to bypass the __init__ function in esembble method
        so that a covariance matrix is not passed as it can be huge and there is no need to computed
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

        if False:  # it is better
            # todo: Jeremy, what us this?
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

        # todo: check this ... what is that?
        if False:
            if not self.parensemble.istransformed:
                self.parensemble._transform(inplace=True)
            if not self.parensemble_0.istransformed:
                self.parensemble_0._transform(inplace=True)
        self._initialized = True

    def forcast(self):
        """ This simply moves the ensemble forward by running the model
               once for each realization"""

        if self.parensemble is None:
            parensemble = self.parensemble
        self.logger.log("evaluating ensemble")
        failed_runs, obsensemble = self._calc_obs(self.parensemble)

        if failed_runs is not None:
            self.logger.warn("dropping failed realizations")
            parensemble.loc[failed_runs, :] = np.NaN
            parensemble = parensemble.dropna()
            obsensemble.loc[failed_runs, :] = np.NaN
            obsensemble = obsensemble.dropna()
            self.obsensemble = obsensemble
            self.parensemble = parensemble
        else:
            self.obsensemble = obsensemble

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

    def generate_priors(self):
        """
        Use parameter groups dataframe from Pst to generate prior realizations
        :return:
        """
        # ToDo: Any plans to deal with nonGaussian distributions?
        # Todo: it's critical to avoid generating the covariance matrix explicitlty since we do not need it.
        self.parensemble_0 = pyemu.ParameterEnsemble.from_gaussian_draw(self.pst,
                                                                        self.parcov, num_reals=self.num_reals)

        pass

    def model_evalutions(self):
        pass

    def update(self):

        if self.mode == 'stochastic':
            self.stochastic_update()

        else:
            """
            Determistic update (or least square filtering)
            """
            # Xa' = Xf' + C'H(H'CH + R) (do' - df')
            self.deterministic_update()
            pass

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
        err_ens = self.obsensemble_0.loc[:, self.pst.nnz_obs_names] - \
                  self.pst.observation_data.loc[self.pst.nnz_obs_names, :]['obsval']
        err_ens = err_ens / np.sqrt(err_ens.shape[0] - 1)
        err_ens = Matrix(x=err_ens.values, row_names=err_ens.index, col_names=err_ens.columns)

        # Innovation matrix (deviation arround the mean)
        Ddash = self.obsensemble_0.loc[:, self.pst.nnz_obs_names] - self.obsensemble.loc[:, self.pst.nnz_obs_names]
        Ddash = Matrix(x=Ddash.values, row_names=Ddash.index, col_names=Ddash.columns)
        Ddash = Ddash.T

        C = delta_obs.T + err_ens.T
        m, N = C.shape
        u, s, v = np.linalg.svd(C.as_2d)
        ns = len(s)
        s_perc = 100.0 * s / np.sum(s)
        s_ = np.power(s, -2.0)
        s_[s_perc < self.threshold_percent] = 0.0
        ss_ = np.zeros((m, m))
        np.fill_diagonal(ss_, s_)
        if m <= N:
            u = u[:, :ns]
            ss_ = ss_[:ns, :ns]

        X1 = np.dot(ss_, u.T)
        X1 = np.dot(X1, Ddash.as_2d)
        X1 = np.dot(u, X1)
        X1 = np.dot(delta_obs.as_2d, X1)

        # This is the change in parameter values
        del_par = np.dot(delta_par.as_2d.T, X1)
        self.parensemble_a = (self.parensemble.T + del_par).T
        self.parensemble_a = ParameterEnsemble.from_dataframe(df=self.parensemble_a,
                                                              pst=self.pst,
                                                              istransformed=True)

    def analysis(self):
        """

        :return:
        """
        # TODO: add exceptions and warnings
        if self.type == "Smoother":
            self.smoother()
        elif self.type == "Kalman_filter":
            self.enkf()
        elif self.type == "Kalman_smoother":
            self.enks()
        else:
            print("We should'nt be here....")  # TODO: ???

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
        
        Loop over time windows and apply da
        :return:
        """

        for cycle_index, time_point in enumerate(self.timeline):
            if cycle_index >= len(self.timeline) - 1:
                # Logging : Last Update cycle has finished
                break

            print("Print information about this assimilation Cycle ???")  # should be handeled in Logger

            # each cycle should have a dictionary of template files and instruction files to update the model inout
            # files
            # get current cycle update information
            current_cycle_files = self.cycle_update_files[cycle_index]

            #  (1)  update model input files for this cycle
            self.model_temporal_evolotion(cycle_index, current_cycle_files)

            # (2) generate new Pst object for the current time cycle
            current_pst = copy.deepcopy(self.pst)
            # update observation dataframe
            # update parameter dataframe
            # update in/out files if needed

            # At this stage the problem is equivalent to smoother problem
            self.smoother(current_pst)

            # how to save results for this cycle???



    def model_temporal_evolotion(self, time_index, cycle_files):
        """
         - The function prepares the model for this time cycle
         - Any time-dependant forcing should be handled here. This includes temporal stresses, boundary conditions, and
         initial conditions.
         - Two options are available (1) template files to update input parameters
                                     (2) use results from previous cycle to update input files using instruction
                                     files.
                                     (3) the user must prepare a pair of files : .tpl(or .ins) and corresponding file to change
        :param time_index:
        :return:
        """
        for file in cycle_files:
            # generate log here about files being updated
            self.update_inputfile(file[0], file[1])  # Jeremy: do we have something like this in python?


if __name__ == "__main__":
    sm = Assimilator(type='Smoother', iterate=False, mode='Stochastic', pst='pst.control')
    sm.analysis()

    # Ensemble Kalman Filter
    kf = Assimilator(type='Kalman_filter', pst='pst.control')
    kf.analysis()

    # Kalman Smoother
    ks = Assimilator(type='Kalman_smoother', pst='pst.control')
    ks.analysis()
