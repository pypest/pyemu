from __future__ import print_function, division
import os
import copy
from datetime import datetime
import numpy as np
import pandas as pd
from pyemu.en import ObservationEnsemble
from pyemu.mat.mat_handler import Matrix, Jco, Cov
from pyemu.pst.pst_handler import Pst
from pyemu.utils.os_utils import _istextfile
from .logger import Logger


class EnDS(object):
    """Ensemle Data Space Analysis.

    Args:
        pst (varies): something that can be cast into a `pyemu.Pst`.  Can be an `str` for a
            filename or an existing `pyemu.Pst`.  If `None`, a pst filename is sought
            with the same base name as the jco argument (if passed)
        ensemble (varies): something that can be cast into a `pyemu.ObservationEnsemble`.  Can be
            an `str` fora  filename or `pd.DataFrame` or an existing `pyemu.ObservationEnsemble`.
        obscov (varies, optional): observation noise covariance matrix.  If `str`, a filename is assumed and
            the noise covariance matrix is loaded from a file using
            the file extension (".jcb"/".jco" for binary, ".cov"/".mat" for PEST-style ASCII matrix,
            or ".unc" for uncertainty files).  If `None`, the noise covariance matrix is
            constructed from the obsevation weights (and optionally "standard_deviation")
            .  Can also be a `pyemu.Cov` instance
        forecasts (enumerable of `str`): the names of the entries in `pst.osbervation_data` (and in `ensemble`).
        verbose (`bool`): controls screen output.  If `str`, a filename is assumed and
                and log file is written.
        
    Note:

    Example::

        #assumes "my.pst" exists
        ends = pyemu.EnDS(ensemble="my.0.obs.jcb",forecasts=["fore1","fore2"])


    """

    def __init__(
        self,
        pst=None,
        ensemble=None,
        obscov=None,
        predictions=None,
        verbose=False,
        forecasts=None,
    ):
        self.logger = Logger(verbose)
        self.log = self.logger.log
        self.en_arg = ensemble
        # if jco is None:
        self.__en = ensemble

        self.pst_arg = pst
        if obscov is None and pst is not None:
            obscov = pst
        if forecasts is not None and predictions is not None:
            raise Exception("can't pass both forecasts and predictions")

        # private attributes - access is through @decorated functions
        self.__pst = None
        self.__obscov = None
        self.__predictions = None

        self.log("pre-loading base components")
        if ensemble is not None:
            self.__load_ensemble()
        if pst is not None:
            self.__load_pst()
        if obscov is not None:
            self.__load_obscov()

        self.prediction_arg = None
        if predictions is not None:
            self.prediction_arg = predictions
        elif forecasts is not None:
            self.prediction_arg = forecasts
        elif self.pst is not None:
            if self.pst.forecast_names is not None:
                self.prediction_arg = self.pst.forecast_names
        if self.prediction_arg:
            self.__load_predictions()

        self.log("pre-loading base components")

        # automatically do some things that should be done
        assert type(self.obscov) == Cov

    def __fromfile(self, filename, astype=None):
        """a private method to deduce and load a filename into a matrix object.
        Uses extension: 'jco' or 'jcb': binary, 'mat','vec' or 'cov': ASCII,
        'unc': pest uncertainty file.

        """
        assert os.path.exists(filename), (
            "LinearAnalysis.__fromfile(): " + "file not found:" + filename
        )
        ext = filename.split(".")[-1].lower()
        if ext in ["jco", "jcb"]:
            self.log("loading jcb format: " + filename)
            if astype is None:
                astype = Jco
            m = astype.from_binary(filename)
            self.log("loading jcb format: " + filename)
        elif ext in ["mat", "vec"]:
            self.log("loading ascii format: " + filename)
            if astype is None:
                astype = Matrix
            m = astype.from_ascii(filename)
            self.log("loading ascii format: " + filename)
        elif ext in ["cov"]:
            self.log("loading cov format: " + filename)
            if astype is None:
                astype = Cov
            if _istextfile(filename):
                m = astype.from_ascii(filename)
            else:
                m = astype.from_binary(filename)
            self.log("loading cov format: " + filename)
        elif ext in ["unc"]:
            self.log("loading unc file format: " + filename)
            if astype is None:
                astype = Cov
            m = astype.from_uncfile(filename)
            self.log("loading unc file format: " + filename)
        elif ext in [".csv"]:
            self.log("loading csv format: " + filename)
            if astype is None:
                astype = ObservationEnsemble
            m = astype.from_csv(self.pst,filename=filename)
            self.log("loading csv format: " + filename)
        else:
            raise Exception(
                "EnDS.__fromfile(): unrecognized"
                + " filename extension:"
                + str(ext)
            )
        return m

    def __load_pst(self):
        """private method set the pst attribute"""
        if self.pst_arg is None:
            return None
        if isinstance(self.pst_arg, Pst):
            self.__pst = self.pst_arg
            return self.pst
        else:
            try:
                self.log("loading pst: " + str(self.pst_arg))
                self.__pst = Pst(self.pst_arg)
                self.log("loading pst: " + str(self.pst_arg))
                return self.pst
            except Exception as e:
                raise Exception(
                    "EnDS.__load_pst(): error loading"
                    + " pest control from argument: "
                    + str(self.pst_arg)
                    + "\n->"
                    + str(e)
                )

    def __load_ensemble(self):
        """private method to set the ensemble attribute from a file or a dataframe object"""
        if self.ensemble_arg is None:
            return None
            # raise Exception("linear_analysis.__load_jco(): jco_arg is None")
        if isinstance(self.ensemble_arg, Matrix):
            self.__ensemble = ObservationEnsemble(pst=self.pst,df=self.ensemble_arg.to_dataframe())
        elif isinstance(self.ensemble_arg, str):
            self.__ensemble = self.__fromfile(self.ensemble_arg, astype=ObservationEnsemble)
        else:
            raise Exception(
                "EnDS.__load_ensemble(): ensemble_arg must "
                + "be a matrix object, dataframe, or a file name: "
                + str(self.ensemble_arg)
            )



    def __load_obscov(self):
        """private method to set the obscov attribute from:
        a pest control file (observation weights)
        a pst object
        a matrix object
        an uncert file
        an ascii matrix file
        """
        # if the obscov arg is None, but the pst arg is not None,
        # reset and load from obs weights
        if not self.obscov_arg:
            if self.pst_arg:
                self.obscov_arg = self.pst_arg
            else:
                raise Exception(
                    "linear_analysis.__load_obscov(): " + "obscov_arg is None"
                )
        if isinstance(self.obscov_arg, Matrix):
            self.__obscov = self.obscov_arg
            return
        self.log("loading obscov")
        if isinstance(self.obscov_arg, str):
            if self.obscov_arg.lower().endswith(".pst"):
                self.__obscov = Cov.from_obsweights(self.obscov_arg)
            else:
                self.__obscov = self.__fromfile(self.obscov_arg, astype=Cov)
        elif isinstance(self.obscov_arg, Pst):
            self.__obscov = Cov.from_observation_data(self.obscov_arg)
        else:
            raise Exception(
                "EnDS.__load_obscov(): "
                + "obscov_arg must be a "
                + "matrix object or a file name: "
                + str(self.obscov_arg)
            )
        self.log("loading obscov")



    # these property decorators help keep from loading potentially
    # unneeded items until they are called
    # returns a reference - cheap, but can be dangerous

    @property
    def forecast_names(self):
        """get the forecast (aka prediction) names

        Returns:
            ([`str`]): list of forecast names

        """
        if self.forecasts is None:
            return []
        return list(self.predictions)


    @property
    def obscov(self):
        """get the observation noise covariance matrix attribute

        Returns:
            `pyemu.Cov`: a reference to the `LinearAnalysis.obscov` attribute

        """
        if not self.__obscov:
            self.__load_obscov()
        return self.__obscov


    @property
    def pst(self):
        """the pst attribute

        Returns:
            `pyemu.Pst`: the pst attribute

        """
        if self.__pst is None and self.pst_arg is None:
            raise Exception(
                "linear_analysis.pst: can't access self.pst:"
                + "no pest control argument passed"
            )
        elif self.__pst:
            return self.__pst
        else:
            self.__load_pst()
            return self.__pst

    def reset_pst(self, arg):
        """reset the EnDS.pst attribute

        Args:
            arg (`str` or `pyemu.Pst`): the value to assign to the pst attribute

        """
        self.logger.statement("resetting pst")
        self.__pst = None
        self.pst_arg = arg

    def reset_obscov(self, arg=None):
        """reset the obscov attribute to None

        Args:
            arg (`str` or `pyemu.Matrix`): the value to assign to the obscov
                attribute.  If None, the private __obscov attribute is cleared
                but not reset
        """
        self.logger.statement("resetting obscov")
        self.__obscov = None
        if arg is not None:
            self.obscov_arg = arg


    def __prep_ensemble_schur_components(self):
        pass
    


    def get_added_obs_importance(
            self, obslist_dict=None, base_obslist=None, reset_zero_weight=1.0
    ):
        """A dataworth method to analyze the posterior uncertainty as a result of gathering
         some additional observations

        Args:
            obslist_dict (`dict`, optional): a nested dictionary-list of groups of observations
                that are to be treated as gained/collected.  key values become
                row labels in returned dataframe. If `None`, then every zero-weighted
                observation is tested sequentially. Default is `None`
            base_obslist ([`str`], optional): observation names to treat as the "existing" observations.
                The values of `obslist_dict` will be added to this list during
                each test.  If `None`, then the values in each `obslist_dict` entry will
                be treated as the entire calibration dataset.  That is, there
                are no existing observations. Default is `None`.  Standard practice would
                be to pass this argument as `pyemu.Schur.pst.nnz_obs_names` so that existing,
                non-zero-weighted observations are accounted for in evaluating the worth of
                new yet-to-be-collected observations.
            reset_zero_weight (`float`, optional)
                a flag to reset observations with zero weight in `obslist_dict`
                If `reset_zero_weights` passed as 0.0, no weights adjustments are made.
                Default is 1.0.

        Returns:
            `pandas.DataFrame`: a dataframe with row labels (index) of `obslist_dict.keys()` and
            columns of forecast names.  The values in the dataframe are the
            posterior variance of the forecasts resulting from notional inversion
            using the observations in `obslist_dict[key value]` plus the observations
            in `base_obslist` (if any).  One row in the dataframe is labeled "base" - this is
            posterior forecast variance resulting from the notional calibration with the
            observations in `base_obslist` (if `base_obslist` is `None`, then the "base" row is the
            prior forecast variance).  Conceptually, the forecast variance should either not change or
            decrease as a result of gaining additional observations.  The magnitude of the decrease
            represents the worth of the potential new observation(s) being tested.

        Note:
            Observations listed in `base_obslist` is required to only include observations
            with weight not equal to zero. If zero-weighted observations are in `base_obslist` an exception will
            be thrown.  In most cases, users will want to reset zero-weighted observations as part
            dataworth testing process. If `reset_zero_weights` == 0, no weights adjustments will be made - this is
            most appropriate if different weights are assigned to the added observation values in `Schur.pst`

        Example::

            sc = pyemu.Schur("my.jco")
            obslist_dict = {"hds":["head1","head2"],"flux":["flux1","flux2"]}
            df = sc.get_added_obs_importance(obslist_dict=obslist_dict,
                                             base_obslist=sc.pst.nnz_obs_names,
                                             reset_zero_weight=1.0)

        """

        if obslist_dict is not None:
            if type(obslist_dict) == list:
                obslist_dict = dict(zip(obslist_dict, obslist_dict))
            base_obslist = []
            for key, names in obslist_dict.items():
                if isinstance(names, str):
                    names = [names]
                base_obslist.extend(names)
            # dedup
            base_obslist = list(set(base_obslist))
            zero_basenames = []
            try:
                base_obslist.extend(self.pst.nnz_obs_names)
                # dedup again
                base_obslist = list(set(base_obslist))
                sbase_obslist = set(base_obslist)
                zero_basenames = [
                    name
                    for name in self.pst.zero_weight_obs_names
                    if name in sbase_obslist
                ]
            except:
                pass
            if len(zero_basenames) > 0:
                raise Exception(
                    "Observations in baseobs_list must have "
                    + "nonzero weight. The following observations "
                    + "violate that condition: "
                    + ",".join(zero_basenames)
                )

        else:
            try:
                self.pst
            except Exception as e:
                raise Exception(
                    "'obslist_dict' not passed and self.pst is not available"
                )

            if self.pst.nnz_obs == 0:
                raise Exception(
                    "not resetting weights and there are no non-zero weight obs to remove"
                )
            obslist_dict = dict(zip(self.pst.nnz_obs_names, self.pst.nnz_obs_names))
            base_obslist = self.pst.nnz_obs_names

    def get_removed_obs_importance(self, obslist_dict=None, reset_zero_weight=None):
        """A dataworth method to analyze the posterior uncertainty as a result of losing
         some existing observations

        Args:
            obslist_dict (`dict`, optional): a nested dictionary-list of groups of observations
                that are to be treated as lost.  key values become
                row labels in returned dataframe. If `None`, then every zero-weighted
                observation is tested sequentially. Default is `None`
            reset_zero_weight DEPRECATED

        Returns:
            `pandas.DataFrame`: A dataframe with index of obslist_dict.keys() and columns
            of forecast names.  The values in the dataframe are the posterior
            variances of the forecasts resulting from losing the information
            contained in obslist_dict[key value]. One row in the dataframe is labeled "base" - this is
            posterior forecast variance resulting from the notional calibration with the
            non-zero-weighed observations in `Schur.pst`.  Conceptually, the forecast variance should
            either not change or increase as a result of losing existing observations.  The magnitude
            of the increase represents the worth of the existing observation(s) being tested.

            Note:
            All observations that may be evaluated as removed must have non-zero weight


        Example::

            sc = pyemu.Schur("my.jco")
            df = sc.get_removed_obs_importance()

        """

        if obslist_dict is not None:
            if type(obslist_dict) == list:
                obslist_dict = dict(zip(obslist_dict, obslist_dict))
            base_obslist = []
            for key, names in obslist_dict.items():
                if isinstance(names, str):
                    names = [names]
                base_obslist.extend(names)
            # dedup
            base_obslist = list(set(base_obslist))
            zero_basenames = []
            try:
                base_obslist.extend(self.pst.nnz_obs_names)
                # dedup again
                base_obslist = list(set(base_obslist))
                sbase_obslist = set(base_obslist)
                zero_basenames = [
                    name
                    for name in self.pst.zero_weight_obs_names
                    if name in sbase_obslist
                ]
            except:
                pass
            if len(zero_basenames) > 0:
                raise Exception(
                    "Observations in baseobs_list must have "
                    + "nonzero weight. The following observations "
                    + "violate that condition: "
                    + ",".join(zero_basenames)
                )

        else:
            try:
                self.pst
            except Exception as e:
                raise Exception(
                    "'obslist_dict' not passed and self.pst is not available"
                )

            if self.pst.nnz_obs == 0:
                raise Exception(
                    "not resetting weights and there are no non-zero weight obs to remove"
                )
            obslist_dict = dict(zip(self.pst.nnz_obs_names, self.pst.nnz_obs_names))
            base_obslist = self.pst.nnz_obs_names



