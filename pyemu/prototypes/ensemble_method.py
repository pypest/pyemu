"""A base class for developing prototype ensemble methods
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
from pyemu.en import ParameterEnsemble, ObservationEnsemble
from pyemu.mat import Cov, Matrix

from pyemu.pst import Pst
from ..logger import Logger


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
        num_workers : int
            number of workers to use in (local machine) parallel evaluation of the parmaeter
            ensemble.  If 0, serial evaluation is used.  Ignored if submit_file is not None
        submit_file : str
            the name of a HTCondor submit file.  If not None, HTCondor is used to
            evaluate the parameter ensemble in parallel by issuing condor_submit
            as a system command
        port : int
            the TCP port number to communicate on for parallel run management
        worker_dir : str
            path to a directory with a complete set of model files and PEST
            interface files

    """

    def __init__(
        self,
        pst,
        parcov=None,
        obscov=None,
        num_workers=0,
        use_approx_prior=True,
        submit_file=None,
        verbose=False,
        port=4004,
        worker_dir="template",
    ):
        self.logger = Logger(verbose)
        if verbose is not False:
            self.logger.echo = True
        self.num_workers = int(num_workers)
        if submit_file is not None:
            if not os.path.exists(submit_file):
                self.logger.lraise("submit_file {0} not found".format(submit_file))
        elif num_workers > 0:
            if not os.path.exists(worker_dir):
                self.logger.lraise("template dir {0} not found".format(worker_dir))

        self.worker_dir = worker_dir
        self.submit_file = submit_file
        self.port = int(port)
        self.paren_prefix = ".parensemble.{0:04d}.csv"
        self.obsen_prefix = ".obsensemble.{0:04d}.csv"

        if isinstance(pst, str):
            pst = Pst(pst)
        assert isinstance(pst, Pst)
        self.pst = pst
        self.sweep_in_csv = pst.pestpp_options.get(
            "sweep_parameter_csv_file", "sweep_in.csv"
        )
        self.sweep_out_csv = pst.pestpp_options.get(
            "sweep_output_csv_file", "sweep_out.csv"
        )
        if parcov is not None:
            assert isinstance(parcov, Cov)
        else:
            parcov = Cov.from_parameter_data(self.pst)
        if obscov is not None:
            assert isinstance(obscov, Cov)
        else:
            obscov = Cov.from_observation_data(pst)

        self.parcov = parcov
        self.obscov = obscov

        self._initialized = False
        self.iter_num = 0
        self.total_runs = 0
        self.raw_sweep_out = None

    def initialize(self, *args, **kwargs):
        raise Exception(
            "EnsembleMethod.initialize() must be implemented by the derived types"
        )

    def _calc_delta(self, ensemble, scaling_matrix=None):
        """
        calc the scaled  ensemble differences from the mean
        """
        mean = np.array(ensemble.mean(axis=0))
        delta = ensemble.as_pyemu_matrix()
        for i in range(ensemble.shape[0]):
            delta.x[i, :] -= mean
        if scaling_matrix is not None:
            delta = scaling_matrix * delta.T
        delta *= 1.0 / np.sqrt(float(ensemble.shape[0] - 1.0))
        return delta

    def _calc_obs(self, parensemble):
        self.logger.log("removing existing sweep in/out files")
        try:
            os.remove(self.sweep_in_csv)
        except Exception as e:
            self.logger.warn("error removing existing sweep in file:{0}".format(str(e)))
        try:
            os.remove(self.sweep_out_csv)
        except Exception as e:
            self.logger.warn(
                "error removing existing sweep out file:{0}".format(str(e))
            )
        self.logger.log("removing existing sweep in/out files")

        if parensemble.isnull().values.any():
            parensemble.to_csv("_nan.csv")
            self.logger.lraise(
                "_calc_obs() error: NaNs in parensemble (written to '_nan.csv')"
            )

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
        failed_runs, obs = self._load_obs_ensemble(self.sweep_out_csv)
        self.logger.log("reading sweep out csv {0}".format(self.sweep_out_csv))
        self.total_runs += obs.shape[0]
        self.logger.statement("total runs:{0}".format(self.total_runs))
        return failed_runs, obs

    def _load_obs_ensemble(self, filename):
        if not os.path.exists(filename):
            self.logger.lraise("obsensemble file {0} does not exists".format(filename))
        obs = pd.read_csv(filename)
        obs.columns = [item.lower() for item in obs.columns]
        self.raw_sweep_out = obs.copy()  # save this for later to support restart
        assert (
            "input_run_id" in obs.columns
        ), "'input_run_id' col missing...need newer version of sweep"
        obs.index = obs.input_run_id
        failed_runs = None
        if 1 in obs.failed_flag.values:
            failed_runs = obs.loc[obs.failed_flag == 1].index.values
            self.logger.warn(
                "{0} runs failed (indices: {1})".format(
                    len(failed_runs), ",".join([str(f) for f in failed_runs])
                )
            )
        obs = ObservationEnsemble.from_dataframe(
            df=obs.loc[:, self.obscov.row_names], pst=self.pst
        )
        if obs.isnull().values.any():
            self.logger.lraise("_calc_obs() error: NaNs in obsensemble")
        return failed_runs, obs

    def _get_master_thread(self):
        master_stdout = "_master_stdout.dat"
        master_stderr = "_master_stderr.dat"

        def master():
            try:
                # os.system("sweep {0} /h :{1} 1>{2} 2>{3}". \
                #          format(self.pst.filename, self.port, master_stdout, master_stderr))
                pyemu.os_utils.run(
                    "pestpp-swp {0} /h :{1} 1>{2} 2>{3}".format(
                        self.pst.filename, self.port, master_stdout, master_stderr
                    )
                )

            except Exception as e:
                self.logger.lraise("error starting condor master: {0}".format(str(e)))
            with open(master_stderr, "r") as f:
                err_lines = f.readlines()
            if len(err_lines) > 0:
                self.logger.warn(
                    "master stderr lines: {0}".format(
                        ",".join([l.strip() for l in err_lines])
                    )
                )

        master_thread = threading.Thread(target=master)
        master_thread.start()
        time.sleep(2.0)
        return master_thread

    def _calc_obs_condor(self, parensemble):
        self.logger.log(
            "evaluating ensemble of size {0} with htcondor".format(parensemble.shape[0])
        )

        parensemble.to_csv(self.sweep_in_csv)
        master_thread = self._get_master_thread()
        condor_temp_file = "_condor_submit_stdout.dat"
        condor_err_file = "_condor_submit_stderr.dat"
        self.logger.log(
            "calling condor_submit with submit file {0}".format(self.submit_file)
        )
        try:
            os.system(
                "condor_submit {0} 1>{1} 2>{2}".format(
                    self.submit_file, condor_temp_file, condor_err_file
                )
            )
        except Exception as e:
            self.logger.lraise("error in condor_submit: {0}".format(str(e)))
        self.logger.log(
            "calling condor_submit with submit file {0}".format(self.submit_file)
        )
        time.sleep(2.0)  # some time for condor to submit the job and echo to stdout
        condor_submit_string = "submitted to cluster"
        with open(condor_temp_file, "r") as f:
            lines = f.readlines()
        self.logger.statement(
            "condor_submit stdout: {0}".format(",".join([l.strip() for l in lines]))
        )
        with open(condor_err_file, "r") as f:
            err_lines = f.readlines()
        if len(err_lines) > 0:
            self.logger.warn(
                "stderr from condor_submit:{0}".format([l.strip() for l in err_lines])
            )
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
        self.logger.log(
            "evaluating ensemble of size {0} with htcondor".format(parensemble.shape[0])
        )

    def _calc_obs_local(self, parensemble):
        """
        propagate the ensemble forward using sweep.
        """
        self.logger.log(
            "evaluating ensemble of size {0} locally with sweep".format(
                parensemble.shape[0]
            )
        )
        parensemble.to_csv(self.sweep_in_csv)
        if self.num_workers > 0:
            master_thread = self._get_master_thread()
            pyemu.utils.start_workers(
                self.worker_dir,
                "pestpp-swp",
                self.pst.filename,
                self.num_workers,
                worker_root="..",
                port=self.port,
            )
            master_thread.join()
        else:
            os.system("pestpp-swp {0}".format(self.pst.filename))

        self.logger.log(
            "evaluating ensemble of size {0} locally with sweep".format(
                parensemble.shape[0]
            )
        )

    def update(
        self, lambda_mults=[1.0], localizer=None, run_subset=None, use_approx=True
    ):
        raise Exception(
            "EnsembleMethod.update() must be implemented by the derived types"
        )
