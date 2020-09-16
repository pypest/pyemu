import os
import sys
import multiprocessing as mp
import copy
import numpy as np
import pandas as pd
import pyemu
from .ensemble_method import EnsembleMethod


class EnsembleKalmanFilter(EnsembleMethod):
    def __init__(
        self,
        pst,
        parcov=None,
        obscov=None,
        num_workers=0,
        submit_file=None,
        verbose=False,
        port=4004,
        worker_dir="template",
    ):
        super(EnsembleKalmanFilter, self).__init__(
            pst=pst,
            parcov=parcov,
            obscov=obscov,
            num_workers=num_workers,
            submit_file=submit_file,
            verbose=verbose,
            port=port,
            worker_dir=worker_dir,
        )

    def initialize(
        self,
        num_reals=1,
        enforce_bounds="reset",
        parensemble=None,
        obsensemble=None,
        restart_obsensemble=None,
    ):
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

        build_empirical_prior = False

        # initialize the phi report csv
        self.enforce_bounds = enforce_bounds

        self.total_runs = 0
        # this matrix gets used a lot, so only calc once and store
        self.obscov_inv_sqrt = self.obscov.get(self.pst.nnz_obs_names).inv.sqrt

        self.logger.log("forming inverse sqrt parcov matrix")
        self.parcov_inv_sqrt = self.parcov.inv.sqrt
        self.logger.log("forming inverse sqrt parcov matrix")

        if parensemble is not None and obsensemble is not None:
            self.logger.log("initializing with existing ensembles")
            if isinstance(parensemble, str):
                self.logger.log("loading parensemble from file")
                if not os.path.exists(obsensemble):
                    self.logger.lraise(
                        "can not find parensemble file: {0}".format(parensemble)
                    )
                df = pd.read_csv(parensemble, index_col=0)
                df.columns = df.columns.str.lower()
                # df.index = [str(i) for i in df.index]
                self.parensemble_0 = pyemu.ParameterEnsemble.from_dataframe(
                    df=df, pst=self.pst
                )
                self.logger.log("loading parensemble from file")

            elif isinstance(parensemble, ParameterEnsemble):
                self.parensemble_0 = parensemble.copy()
            else:
                raise Exception(
                    "unrecognized arg type for parensemble, "
                    + "should be filename or ParameterEnsemble"
                    + ", not {0}".format(type(parensemble))
                )
            self.parensemble = self.parensemble_0.copy()
            if isinstance(obsensemble, str):
                self.logger.log("loading obsensemble from file")
                if not os.path.exists(obsensemble):
                    self.logger.lraise(
                        "can not find obsensemble file: {0}".format(obsensemble)
                    )
                df = pd.read_csv(obsensemble, index_col=0)
                df.columns = df.columns.str.lower()
                df = df.loc[:, self.pst.nnz_obs_names]
                # df.index = [str(i) for i in df.index]
                self.obsensemble_0 = pyemu.ObservationEnsemble.from_dataframe(
                    df=df, pst=self.pst
                )
                self.logger.log("loading obsensemble from file")

            elif isinstance(obsensemble, ObservationEnsemble):
                self.obsensemble_0 = obsensemble.copy()
            else:
                raise Exception(
                    "unrecognized arg type for obsensemble, "
                    + "should be filename or ObservationEnsemble"
                    + ", not {0}".format(type(obsensemble))
                )

            assert self.parensemble_0.shape[0] == self.obsensemble_0.shape[0]
            # self.num_reals = self.parensemble_0.shape[0]
            num_reals = self.parensemble.shape[0]
            self.logger.log("initializing with existing ensembles")

            if build_empirical_prior:

                self.reset_parcov(self.parensemble.covariance_matrix())
                if self.save_mats:
                    self.parcov.to_binary(self.pst.filename + ".empcov.jcb")

        else:
            if build_empirical_prior:
                self.logger.lraise(
                    "can't use build_emprirical_prior without parensemble..."
                )
            self.logger.log("initializing with {0} realizations".format(num_reals))
            self.logger.log("initializing parensemble")
            self.parensemble_0 = pyemu.ParameterEnsemble.from_gaussian_draw(
                self.pst, self.parcov, num_reals=num_reals
            )
            self.parensemble_0.enforce(enforce_bounds=enforce_bounds)
            self.logger.log("initializing parensemble")
            self.parensemble = self.parensemble_0.copy()
            self.parensemble_0.to_csv(self.pst.filename + self.paren_prefix.format(0))
            self.logger.log("initializing parensemble")
            self.logger.log("initializing obsensemble")
            self.obsensemble_0 = pyemu.ObservationEnsemble.from_id_gaussian_draw(
                self.pst, num_reals=num_reals
            )
            # self.obsensemble = self.obsensemble_0.copy()

            # save the base obsensemble
            self.obsensemble_0.to_csv(self.pst.filename + self.obsen_prefix.format(-1))
            self.logger.log("initializing obsensemble")
            self.logger.log("initializing with {0} realizations".format(num_reals))

        self.enforce_bounds = enforce_bounds

        if restart_obsensemble is not None:
            self.logger.log(
                "loading restart_obsensemble {0}".format(restart_obsensemble)
            )
            # failed_runs,self.obsensemble = self._load_obs_ensemble(restart_obsensemble)
            df = pd.read_csv(restart_obsensemble, index_col=0)
            df.columns = df.columns.str.lower()
            # df = df.loc[:, self.pst.nnz_obs_names]
            # df.index = [str(i) for i in df.index]
            self.obsensemble = pyemu.ObservationEnsemble.from_dataframe(
                df=df, pst=self.pst
            )
            assert self.obsensemble.shape[0] == self.obsensemble_0.shape[0]
            assert list(self.obsensemble.columns) == list(self.obsensemble_0.columns)
            self.logger.log(
                "loading restart_obsensemble {0}".format(restart_obsensemble)
            )

        else:
            # run the initial parameter ensemble
            self.logger.log("evaluating initial ensembles")
            self.obsensemble = self.forecast()
            self.logger.log("evaluating initial ensembles")

        # if not self.parensemble.istransformed:
        self.parensemble._transform(inplace=True)
        # if not self.parensemble_0.istransformed:
        self.parensemble_0._transform(inplace=True)
        self._initialized = True

    def forecast(self, parensemble=None):
        """for the enkf formulation, this simply moves the ensemble forward by running the model
        once for each realization"""
        if parensemble is None:
            parensemble = self.parensemble
        self.logger.log("evaluating ensemble")
        failed_runs, obsensemble = self._calc_obs(parensemble)

        if failed_runs is not None:
            self.logger.warn("dropping failed realizations")
            parensemble.loc[failed_runs, :] = np.NaN
            parensemble = parensemble.dropna()
            obsensemble.loc[failed_runs, :] = np.NaN
            obsensemble = obsensemble.dropna()
        self.logger.log("evaluating ensemble")

        return obsensemble

    def analysis(self):

        nz_names = self.pst.nnz_obs_names
        nreals = self.obsensemble.shape[0]

        h_dash = pyemu.Matrix.from_dataframe(
            self.obsensemble.get_deviations().loc[:, nz_names].T
        )

        R = self.obscov

        Chh = ((h_dash * h_dash.T) * (1.0 / nreals - 1)) + R

        Cinv = Chh.pseudo_inv(maxsing=1, eigthresh=self.pst.svd_data.eigthresh)

        # Chh = None

        d_dash = pyemu.Matrix.from_dataframe(
            self.obsensemble_0.loc[self.obsensemble.index, nz_names]
            - self.obsensemble.loc[:, nz_names]
        ).T

        k_dash = pyemu.Matrix.from_dataframe(
            self.parensemble.get_deviations().loc[:, self.pst.adj_par_names]
        ).T

        Chk = (k_dash * h_dash.T) * (1.0 / nreals - 1)

        Chk_dot_Cinv = Chk * Cinv
        # Chk = None
        upgrade = Chk_dot_Cinv * d_dash
        parensemble = self.parensemble.copy()
        upgrade = upgrade.to_dataframe().T

        upgrade.index = parensemble.index
        parensemble += upgrade
        parensemble = pyemu.ParameterEnsemble.from_dataframe(
            df=parensemble, pst=self.pst, istransformed=True
        )
        parensemble.enforce()
        return parensemble

    def analysis_evensen(self):
        """Ayman here!!!.  some peices that may be useful:
        self.parcov = parameter covariance matrix
        self.obscov = obseravtion noise covariance matrix
        self.parensmeble = current parameter ensemble
        self.obsensemble = current observation (model output) ensemble

        the ensemble instances have two useful methods:
        Ensemble.covariance_matrix() - get an empirical covariance matrix
        Ensemble.get_deviations() - get an ensemble of deviations around the mean vector

        If you use pyemu.Matrix (and pyemu.Cov) for the linear algebra parts, you don't
        have to worry about the alignment of the matrices (pyemu will dynamically reorder/align
        based in row and/or col names).  The Matrix instance also has a .inv for the inverse
        as well as .s, .u and .v for the SVD components (gets dynamically evaluated if you try to
        access these attributes)



        Following Evensen 2003..
        """

        nz_names = self.pst.nnz_obs_names
        nreals = self.obsensemble.shape[0]
        self.parensemble._transform()
        # nonzero weighted state deviations
        HA_prime = self.obsensemble.get_deviations().loc[:, nz_names].T

        # obs noise pertubations - move to constuctor
        E = (
            self.obsensemble_0.loc[:, nz_names]
            - self.pst.observation_data.obsval.loc[nz_names]
        ).T
        # print(E)

        # innovations:  account for any failed runs (use obsensemble index)
        D_prime = (
            self.obsensemble_0.loc[self.obsensemble.index, nz_names]
            - self.obsensemble.loc[:, nz_names]
        ).T

        ES = HA_prime.loc[nz_names, E.columns] + E.loc[nz_names, :]
        assert ES.shape == ES.dropna().shape

        ES = pyemu.Matrix.from_dataframe(ES)

        nrmin = min(self.pst.nnz_obs, self.obsensemble.shape[0])
        U, s, v = ES.pseudo_inv_components(
            maxsing=nrmin,
            eigthresh=ES.get_maxsing(self.pst.svd_data.eigthresh),
            truncate=True,
        )

        # half assed inverse
        s_inv = s.T
        for i, sval in enumerate(np.diag(s.x)):
            if sval == 0.0:
                break
            s_inv.x[i, i] = 1.0 / (sval * sval)

        X1 = s_inv * U.T
        X1.autoalign = False  # since the row/col names don't mean anything for singular components and everything is aligned

        X2 = X1 * D_prime  # these are aligned through the use of nz_names

        X3 = U * X2  # also aligned

        X4 = pyemu.Matrix.from_dataframe(HA_prime.T) * X3

        I = np.identity(X4.shape[0])
        X5 = X4 + I
        # print(X5.x.sum(axis=1))

        # deviations of adj pars
        A_prime = pyemu.Matrix.from_dataframe(
            self.parensemble.get_deviations().loc[:, self.pst.adj_par_names]
        ).T

        upgrade = (A_prime * X4).to_dataframe().T

        assert upgrade.shape == upgrade.dropna().shape

        upgrade.index = self.parensemble.index
        # print(upgrade)
        parensemble = self.parensemble + upgrade

        parensemble = pyemu.ParameterEnsemble.from_dataframe(
            df=parensemble, pst=self.pst, istransformed=True
        )

        assert parensemble.shape == parensemble.dropna().shape
        return parensemble

    def update(self):
        """update performs the analysis, then runs the forecast using the updated self.parensemble.
        This can be called repeatedly to iterate..."""
        parensemble = self.analysis_evensen()
        obsensemble = self.forecast(parensemble=parensemble)
        # todo: check for phi improvement
        if True:
            self.obsensemble = obsensemble
            self.parensemble = parensemble

        self.iter_num += 1


class Assimilator:
    def __init__(
        self, type="Smoother", iterate=False, pst=None, mode="stochastic", options={}
    ):
        """
        A clase to implement one or multiple update cycle. For the Ensemble smoother (ES), the update cycle includes all available
        observations. Ror Ensemble Kalman Filter (EnKF), the update is acheived on multiple cycles (time windows); and finally the nsemble Kalman Smoother (EnKS)
        updat parameters given all observations available up to a certain time
        """
        self.mode_options = ["Stochastic", "Deterministic"]
        self.type_options = ["Smoother", "Kalman Filter", "Kalman Smoother"]

        if not isinstance(iterate, bool):
            raise ValueError(
                "Error, 'iterate' must be boolian"
            )  # I think warning would better
        self.iterate = iterate  # If true Chen-Oliver scheme will be used, otherwise classical data assimilation will used.

        if type not in self.type_options:
            raise ValueError(
                "Assimilation type [{}] specified is not supported".format(type)
            )
        self.type = type

        if mode not in self.mode_options:
            raise ValueError(
                "Assimilation mode [{}] specified is not supported".format(mode)
            )
        self.mode = mode

        # Obtain problem setting from the pst object.
        if isinstance(pst, pyemu):
            self.pst = pst
        else:  # TODO: it is better if we check if this is a file, throw exception otherwise
            self.pst = pyemu.Pst(pst)

        # Time Cycle info. example
        # {[ ['file_1.tpl', 'file_1.dat'],
        #    ['file_2.tpl', 'file_2.dat'],
        #    ['out_file.ins', 'out_file.out']}

        self.cycle_update_files = {}

    def forcast(self):
        """
        This function implements (1) generation of random ensemble for parameters, (2) forward model runs.

        :return:

        """
        self.generate_priors()
        self.model_evalutions()

    def generate_priors(self):
        """
        Use parameters dataframe from Pst to generate prior realizations
        :return:
        """
        # ToDo: Any plans to deal with nonGaussian distributions?

        pass

    def model_evalutions(self):
        pass

    def update(self, Pst):

        """
        Solve the anlaysis step
        Xa = Xf + C'H(H'CH + R) (do - df)
        """
        if self.mode == "stochastic":
            pass
        else:
            # deterministic
            # Xa' = Xf' + C'H(H'CH + R) (do' - df')
            pass

    def run(self):
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

    def smoother(self, pst):
        if self.iterate:
            # Chen_oliver algorithim
            pass
        else:
            self.forcast(pst)
            self.update(pst)

    def enkf(self):
        """
        Loop over time windows and apply da
        :return:
        """

        for cycle_index, time_point in enumerate(self.timeline):
            if cycle_index >= len(self.timeline) - 1:
                # Logging : Last Update cycle has finished
                break

            print(
                "Print information about this assimilation Cycle ???"
            )  # should be handeled in Logger

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

    def enks(self, pst):
        """ Similiar to EnkF ---  wait???"""

        pass

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
            self.update_inputfile(
                file[0], file[1]
            )  # Jeremy: do we have something like this in python?


if __name__ == "__main__":
    sm = Assimilator(
        type="Smoother", iterate=False, mode="stochastic", pst="pst.control"
    )
    sm.run()

    # Ensemble Kalman Filter
    kf = Assimilator(type="Kalman_filter", pst="pst.control")
    kf.run()

    # Kalman Smoother
    ks = Assimilator(type="Kalman_smoother", pst="pst.control")
    ks.run()
