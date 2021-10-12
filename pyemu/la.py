"""LinearAnalysis object, which is the base class for other
 pyemu analysis objects (Schur, ErrVar, MonteCarlo, and EnsembleSmoother).
 This class is usually not used directly.
"""
from __future__ import print_function, division
import os
import copy
from datetime import datetime
import numpy as np
import pandas as pd
from pyemu.mat.mat_handler import Matrix, Jco, Cov
from pyemu.pst.pst_handler import Pst
from pyemu.utils.os_utils import _istextfile
from .logger import Logger


class LinearAnalysis(object):
    """The base/parent class for linear analysis.

    Args:
        jco (varies, optional): something that can be cast or loaded into a `pyemu.Jco`.  Can be a
            str for a filename or `pyemu.Matrix`/`pyemu.Jco` object.
        pst (varies, optional): something that can be cast into a `pyemu.Pst`.  Can be an `str` for a
            filename or an existing `pyemu.Pst`.  If `None`, a pst filename is sought
            with the same base name as the jco argument (if passed)
        parcov (varies, optional): prior parameter covariance matrix.  If `str`, a filename is assumed and
            the prior parameter covariance matrix is loaded from a file using
            the file extension (".jcb"/".jco" for binary, ".cov"/".mat" for PEST-style ASCII matrix,
            or ".unc" for uncertainty files).  If `None`, the prior parameter covariance matrix is
            constructed from the parameter bounds in `LinearAnalysis.pst`.  Can also be a `pyemu.Cov` instance
        obscov (varies, optional): observation noise covariance matrix.  If `str`, a filename is assumed and
            the noise covariance matrix is loaded from a file using
            the file extension (".jcb"/".jco" for binary, ".cov"/".mat" for PEST-style ASCII matrix,
            or ".unc" for uncertainty files).  If `None`, the noise covariance matrix is
            constructed from the obsevation weights in `LinearAnalysis.pst`.  Can also be a `pyemu.Cov` instance
        forecasts (varies, optional): forecast sensitivity vectors.  If `str`, first an observation name is assumed (a row
            in `LinearAnalysis.jco`).  If that is not found, a filename is assumed and predictions are
            loaded from a file using the file extension.  If [`str`], a list of observation names is assumed.
            Can also be a `pyemu.Matrix` instance, a `numpy.ndarray` or a collection
            of `pyemu.Matrix` or `numpy.ndarray`.
        ref_var (float, optional): reference variance.  Default is 1.0
        verbose (`bool`): controls screen output.  If `str`, a filename is assumed and
                and log file is written.
        sigma_range (`float`, optional): defines range of upper bound - lower bound in terms of standard
            deviation (sigma). For example, if sigma_range = 4, the bounds represent 4 * sigma.
            Default is 4.0, representing approximately 95% confidence of implied normal distribution.
            This arg is only used if constructing parcov from parameter bounds.
        scale_offset (`bool`, optional): flag to apply parameter scale and offset to parameter bounds
            when calculating prior parameter covariance matrix from bounds.  This arg is onlyused if
            constructing parcov from parameter bounds.Default is True.

    Note:

        Can be used directly, but for prior uncertainty analyses only.

        The derived types (`pyemu.Schur`, `pyemu.ErrVar`) are for different
        forms of FOMS-based posterior uncertainty analyses.

        This class tries hard to not load items until they are needed; all arguments are optional.

        The class makes heavy use of property decorator to encapsulated private attributes

    Example::

        #assumes "my.pst" exists
        la = pyemu.LinearAnalysis(jco="my.jco",forecasts=["fore1","fore2"])
        print(la.prior_forecast)


    """

    def __init__(
        self,
        jco=None,
        pst=None,
        parcov=None,
        obscov=None,
        predictions=None,
        ref_var=1.0,
        verbose=False,
        resfile=False,
        forecasts=None,
        sigma_range=4.0,
        scale_offset=True,
        **kwargs
    ):
        self.logger = Logger(verbose)
        self.log = self.logger.log
        self.jco_arg = jco
        # if jco is None:
        self.__jco = jco
        if pst is None:
            if isinstance(jco, str):
                pst_case = jco.replace(".jco", ".pst").replace(".jcb", ".pst")
                if os.path.exists(pst_case):
                    pst = pst_case
        self.pst_arg = pst
        if parcov is None and pst is not None:
            parcov = pst
        self.parcov_arg = parcov
        if obscov is None and pst is not None:
            obscov = pst
        self.obscov_arg = obscov
        self.ref_var = ref_var
        if forecasts is not None and predictions is not None:
            raise Exception("can't pass both forecasts and predictions")

        self.sigma_range = sigma_range
        self.scale_offset = scale_offset

        # private attributes - access is through @decorated functions
        self.__pst = None
        self.__parcov = None
        self.__obscov = None
        self.__predictions = None
        self.__qhalf = None
        self.__qhalfx = None
        self.__xtqx = None
        self.__fehalf = None
        self.__prior_prediction = None
        self.prediction_extract = None

        self.log("pre-loading base components")
        if jco is not None:
            self.__load_jco()
        if pst is not None:
            self.__load_pst()
        if parcov is not None:
            self.__load_parcov()
        if obscov is not None:
            self.__load_obscov()

        self.prediction_arg = None
        if predictions is not None:
            self.prediction_arg = predictions
        elif forecasts is not None:
            self.prediction_arg = forecasts
        elif self.pst is not None and self.jco is not None:
            if self.pst.forecast_names is not None:
                self.prediction_arg = self.pst.forecast_names
        if self.prediction_arg:
            self.__load_predictions()

        self.log("pre-loading base components")
        if len(kwargs.keys()) > 0:
            self.logger.warn(
                "unused kwargs in type "
                + str(self.__class__.__name__)
                + " : "
                + str(kwargs)
            )
            raise Exception("unused kwargs" + " : " + str(kwargs))
        # automatically do some things that should be done
        self.log("dropping prior information")
        pi = None
        try:
            pi = self.pst.prior_information
        except:
            self.logger.statement(
                "unable to access self.pst: can't tell if "
                + " any prior information needs to be dropped."
            )
        if pi is not None:
            self.drop_prior_information()
        self.log("dropping prior information")

        if resfile != False:
            self.log("scaling obscov by residual phi components")
            try:
                self.adjust_obscov_resfile(resfile=resfile)
            except:
                self.logger.statement(
                    "unable to a find a residuals file for " + " scaling obscov"
                )
                self.resfile = None
                self.res = None
            self.log("scaling obscov by residual phi components")
        assert type(self.parcov) == Cov
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
            self.log("loading jco: " + filename)
            if astype is None:
                astype = Jco
            m = astype.from_binary(filename)
            self.log("loading jco: " + filename)
        elif ext in ["mat", "vec"]:
            self.log("loading ascii: " + filename)
            if astype is None:
                astype = Matrix
            m = astype.from_ascii(filename)
            self.log("loading ascii: " + filename)
        elif ext in ["cov"]:
            self.log("loading cov: " + filename)
            if astype is None:
                astype = Cov
            if _istextfile(filename):
                m = astype.from_ascii(filename)
            else:
                m = astype.from_binary(filename)
            self.log("loading cov: " + filename)
        elif ext in ["unc"]:
            self.log("loading unc: " + filename)
            if astype is None:
                astype = Cov
            m = astype.from_uncfile(filename)
            self.log("loading unc: " + filename)
        else:
            raise Exception(
                "linear_analysis.__fromfile(): unrecognized"
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
                    "linear_analysis.__load_pst(): error loading"
                    + " pest control from argument: "
                    + str(self.pst_arg)
                    + "\n->"
                    + str(e)
                )

    def __load_jco(self):
        """private method to set the jco attribute from a file or a matrix object"""
        if self.jco_arg is None:
            return None
            # raise Exception("linear_analysis.__load_jco(): jco_arg is None")
        if isinstance(self.jco_arg, Matrix):
            self.__jco = self.jco_arg
        elif isinstance(self.jco_arg, str):
            self.__jco = self.__fromfile(self.jco_arg, astype=Jco)
        else:
            raise Exception(
                "linear_analysis.__load_jco(): jco_arg must "
                + "be a matrix object or a file name: "
                + str(self.jco_arg)
            )

    def __load_parcov(self):
        """private method to set the parcov attribute from:
        a pest control file (parameter bounds)
        a pst object
        a matrix object
        an uncert file
        an ascii matrix file
        """
        # if the parcov arg was not passed but the pst arg was,
        # reset and use parbounds to build parcov
        if not self.parcov_arg:
            if self.pst_arg:
                self.parcov_arg = self.pst_arg
            else:
                raise Exception(
                    "linear_analysis.__load_parcov(): " + "parcov_arg is None"
                )
        if isinstance(self.parcov_arg, Matrix):
            self.__parcov = self.parcov_arg
            return
        if isinstance(self.parcov_arg, np.ndarray):
            # if the passed array is a vector,
            # then assume it is the diagonal of the parcov matrix
            if len(self.parcov_arg.shape) == 1:
                assert self.parcov_arg.shape[0] == self.jco.shape[1]
                isdiagonal = True
            else:
                assert self.parcov_arg.shape[0] == self.jco.shape[1]
                assert self.parcov_arg.shape[1] == self.jco.shape[1]
                isdiagonal = False
            self.logger.warn(
                "linear_analysis.__load_parcov(): "
                + "instantiating parcov from ndarray, can't "
                + "verify parameters alignment with jco"
            )
            self.__parcov = Matrix(
                x=self.parcov_arg,
                isdiagonal=isdiagonal,
                row_names=self.jco.col_names,
                col_names=self.jco.col_names,
            )
        self.log("loading parcov")
        if isinstance(self.parcov_arg, str):
            # if the arg is a string ending with "pst"
            # then load parcov from parbounds
            if self.parcov_arg.lower().endswith(".pst"):
                self.__parcov = Cov.from_parbounds(
                    self.parcov_arg,
                    sigma_range=self.sigma_range,
                    scale_offset=self.scale_offset,
                )
            else:
                self.__parcov = self.__fromfile(self.parcov_arg, astype=Cov)
        # if the arg is a pst object
        elif isinstance(self.parcov_arg, Pst):
            self.__parcov = Cov.from_parameter_data(
                self.parcov_arg,
                sigma_range=self.sigma_range,
                scale_offset=self.scale_offset,
            )
        else:
            raise Exception(
                "linear_analysis.__load_parcov(): "
                + "parcov_arg must be a "
                + "matrix object or a file name: "
                + str(self.parcov_arg)
            )
        self.log("loading parcov")

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
        if isinstance(self.obscov_arg, np.ndarray):
            # if the ndarray arg is a vector,
            # assume it is the diagonal of the obscov matrix
            if len(self.obscov_arg.shape) == 1:
                assert self.obscov_arg.shape[0] == self.jco.shape[1]
                isdiagonal = True
            else:
                assert self.obscov_arg.shape[0] == self.jco.shape[0]
                assert self.obscov_arg.shape[1] == self.jco.shape[0]
                isdiagonal = False
            self.logger.warn(
                "linear_analysis.__load_obscov(): "
                + "instantiating obscov from ndarray,  "
                + "can't verify observation alignment with jco"
            )
            self.__obscov = Matrix(
                x=self.obscov_arg,
                isdiagonal=isdiagonal,
                row_names=self.jco.row_names,
                col_names=self.jco.row_names,
            )
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
                "linear_analysis.__load_obscov(): "
                + "obscov_arg must be a "
                + "matrix object or a file name: "
                + str(self.obscov_arg)
            )
        self.log("loading obscov")

    def __load_predictions(self):
        """private method set the predictions attribute from:
            mixed list of row names, matrix files and ndarrays
            a single row name
            an ascii file
        can be none if only interested in parameters.

        """
        if self.prediction_arg is None:
            self.__predictions = None
            return
        self.log("loading forecasts")
        if not isinstance(self.prediction_arg, list):
            self.prediction_arg = [self.prediction_arg]

        row_names = []
        vecs = []
        mat = None
        for arg in self.prediction_arg:
            if isinstance(arg, Matrix):
                # a vector
                if arg.shape[1] == 1:
                    vecs.append(arg)
                else:
                    if self.jco is not None:
                        assert arg.shape[0] == self.jco.shape[1], (
                            "linear_analysis.__load_predictions(): "
                            + "multi-prediction matrix(npar,npred) not aligned "
                            + "with jco(nobs,npar): "
                            + str(arg.shape)
                            + " "
                            + str(self.jco.shape)
                        )
                        # for pred_name in arg.row_names:
                        #    vecs.append(arg.extract(row_names=pred_name).T)
                    mat = arg
            elif isinstance(arg, str):
                if arg.lower() in self.jco.row_names:
                    row_names.append(arg.lower())
                else:
                    try:
                        pred_mat = self.__fromfile(arg, astype=Matrix)
                    except Exception as e:
                        raise Exception(
                            "forecast argument: "
                            + arg
                            + " not found in "
                            + "jco row names and could not be "
                            + "loaded from a file."
                        )
                    # vector
                    if pred_mat.shape[1] == 1:
                        vecs.append(pred_mat)
                    else:
                        # for pred_name in pred_mat.row_names:
                        #    vecs.append(pred_mat.get(row_names=pred_name))
                        if mat is None:
                            mat = pred_mat
                        else:
                            mat = mat.extend((pred_mat))
            elif isinstance(arg, np.ndarray):
                self.logger.warn(
                    "linear_analysis.__load_predictions(): "
                    + "instantiating prediction matrix from "
                    + "ndarray, can't verify alignment"
                )
                self.logger.warn(
                    "linear_analysis.__load_predictions(): "
                    + "instantiating prediction matrix from "
                    + "ndarray, generating generic prediction names"
                )

                pred_names = ["pred_{0}".format(i + 1) for i in range(arg.shape[0])]

                if self.jco:
                    names = self.jco.col_names
                elif self.parcov:
                    names = self.parcov.col_names
                else:
                    raise Exception(
                        "linear_analysis.__load_predictions(): "
                        + "ndarray passed for predicitons "
                        + "requires jco or parcov to get "
                        + "parameter names"
                    )
                if mat is None:
                    mat = Matrix(x=arg, row_names=pred_names, col_names=names).T
                else:
                    mat = mat.extend(
                        Matrix(x=arg, row_names=pred_names, col_names=names).T
                    )
                # for pred_name in pred_names:
                #    vecs.append(pred_matrix.get(row_names=pred_name).T)
            else:
                raise Exception("unrecognized predictions argument: " + str(arg))
        # turn vecs into a pyemu.Matrix

        if len(vecs) > 0:
            xs = vecs[0].x
            for vec in vecs[1:]:
                xs = xs.extend(vec.x)
            names = [vec.col_names[0] for vec in vecs]
            if mat is None:
                mat = Matrix(x=xs, row_names=vecs[0].row_names, col_names=names)
            else:
                mat = mat.extend(
                    Matrix(x=np.array(xs), row_names=vecs[0].row_names, col_names=names)
                )

        if len(row_names) > 0:
            extract = self.jco.extract(row_names=row_names).T
            if mat is None:
                mat = extract
            else:
                mat = mat.extend(extract)
            # for row_name in row_names:
            #    vecs.append(extract.get(row_names=row_name).T)
            # call obscov to load __obscov so that __obscov
            # (priavte) can be manipulated
            # check if any forecasts are in the obscov
            so_names = set(self.__obscov.row_names)
            drop_names = [r for r in row_names if r in so_names]
            if len(drop_names) > 0:
                self.__obscov.drop(drop_names, axis=0)
        self.__predictions = mat
        try:
            nz_names = set(self.pst.nnz_obs_names)
            fnames = [fname for fname in self.forecast_names if fname in nz_names]
            # if len(row_names) > 0:
            #    srow_names = set(row_names)
            #    fnames = [fname for fname in self.forecast_names if fname in srow_names]
            # else:
            #    fnames = []
        except:
            fnames = []
        if len(fnames) > 0:
            self.logger.warn(
                "forecasts with non-zero weight in pst: {0}...".format(",".join(fnames))
                + "\n -> re-setting these forecast weights to zero..."
            )
            self.pst.observation_data.loc[fnames, "weight"] = 0.0
        self.log("loading forecasts")
        self.logger.statement("forecast names: {0}".format(",".join(mat.col_names)))
        return self.__predictions

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
        # return [fore.col_names[0] for fore in self.forecasts]
        return list(self.predictions.col_names)

    @property
    def parcov(self):
        """get the prior parameter covariance matrix attribute

        Returns:
            `pyemu.Cov`: a reference to the `LinearAnalysis.parcov` attribute

        """
        if not self.__parcov:
            self.__load_parcov()
        return self.__parcov

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
    def nnz_obs_names(self):
        """non-zero-weighted observation names

        Returns:
            ['str`]: list of non-zero-weighted observation names

        Note:
            if `LinearAnalysis.pst` is `None`, returns `LinearAnalysis.jco.row_names`

        """
        if self.__pst is not None:
            return self.pst.nnz_obs_names
        else:
            return self.jco.obs_names

    @property
    def adj_par_names(self):
        """adjustable parameter names

        Returns:
            ['str`]: list of adjustable parameter names

        Note:
            if `LinearAnalysis.pst` is `None`, returns `LinearAnalysis.jco.col_names`

        """
        if self.__pst is not None:
            return self.pst.adj_par_names
        else:
            return self.jco.par_names

    @property
    def jco(self):
        """the jacobian matrix attribute

        Returns:
            `pyemu.Jco`: the jacobian matrix attribute

        """
        if not self.__jco:
            self.__load_jco()
        return self.__jco

    @property
    def predictions(self):
        """the prediction (aka forecast) sentivity vectors attribute

        Returns:
            `pyemu.Matrix`: a matrix of prediction sensitivity vectors (column wise)

        """
        if not self.__predictions:
            self.__load_predictions()
        return self.__predictions

    @property
    def predictions_iter(self):
        """prediction sensitivity vectors iterator

        Returns:
            `iterator`: iterator on prediction sensitivity vectors (matrix)

        Note:
            this is used for processing huge numbers of predictions
        """
        for fname in self.forecast_names:
            yield self.predictions.get(col_names=fname)

    @property
    def forecasts_iter(self):
        """forecast (e.g. prediction) sensitivity vectors iterator

        Returns:
            `iterator`: iterator on forecasts (e.g. predictions) sensitivity vectors (matrix)

        Note:
            This is used for processing huge numbers of predictions

            This is a synonym for LinearAnalysis.predictions_iter()
        """
        return self.predictions_iter

    @property
    def forecasts(self):
        """the forecast sentivity vectors attribute

        Returns:
            `pyemu.Matrix`: a matrix of forecast (prediction) sensitivity vectors (column wise)

        """
        return self.predictions

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

    @property
    def fehalf(self):
        """Karhunen-Loeve scaling matrix attribute.

        Returns:
            `pyemu.Matrix`: the Karhunen-Loeve scaling matrix based on the prior
            parameter covariance matrix

        """
        if self.__fehalf != None:
            return self.__fehalf
        self.log("fehalf")
        self.__fehalf = self.parcov.u * (self.parcov.s ** (0.5))
        self.log("fehalf")
        return self.__fehalf

    @property
    def qhalf(self):
        """square root of the cofactor matrix attribute. Create the attribute if
        it has not yet been created

        Returns:
            `pyemu.Matrix`: square root of the cofactor matrix

        """
        if self.__qhalf != None:
            return self.__qhalf
        self.log("qhalf")
        self.__qhalf = self.obscov ** (-0.5)
        self.log("qhalf")
        return self.__qhalf

    @property
    def qhalfx(self):
        """half normal matrix attribute.

        Returns:
            `pyemu.Matrix`: half normal matrix attribute

        """
        if self.__qhalfx is None:
            self.log("qhalfx")
            self.__qhalfx = self.qhalf * self.jco
            self.log("qhalfx")
        return self.__qhalfx

    @property
    def xtqx(self):
        """normal matrix attribute.

        Returns:
            `pyemu.Matrix`: normal matrix attribute

        """
        if self.__xtqx is None:
            self.log("xtqx")
            self.__xtqx = self.jco.T * (self.obscov ** -1) * self.jco
            self.log("xtqx")
        return self.__xtqx

    @property
    def mle_covariance(self):
        """maximum likelihood parameter covariance matrix.

        Returns:
            `pyemu.Matrix`: maximum likelihood parameter covariance matrix

        """
        return self.xtqx.inv

    @property
    def prior_parameter(self):
        """prior parameter covariance matrix

        Returns:
            `pyemu.Cov`: prior parameter covariance matrix

        """
        return self.parcov

    @property
    def prior_forecast(self):
        """prior forecast (e.g. prediction) variances

        Returns:
            `dict`: a dictionary of forecast name, prior variance pairs

        """
        return self.prior_prediction

    @property
    def mle_parameter_estimate(self):
        """maximum likelihood parameter estimate.

        Returns:
            `pandas.Series`: the maximum likelihood parameter estimates

        """
        res = self.pst.res
        assert res is not None
        # build the prior expectation parameter vector
        prior_expt = self.pst.parameter_data.loc[:, ["parval1"]].copy()
        islog = self.pst.parameter_data.partrans == "log"
        prior_expt.loc[islog] = prior_expt.loc[islog].apply(np.log10)
        prior_expt = Matrix.from_dataframe(prior_expt)
        prior_expt.col_names = ["prior_expt"]
        # build the residual vector
        res_vec = Matrix.from_dataframe(res.loc[:, ["residual"]])

        # calc posterior expectation
        upgrade = self.mle_covariance * self.jco.T * res_vec
        upgrade.col_names = ["prior_expt"]
        post_expt = prior_expt + upgrade

        # post processing - back log transform
        post_expt = pd.DataFrame(
            data=post_expt.x, index=post_expt.row_names, columns=["post_expt"]
        )
        post_expt.loc[islog, :] = 10.0 ** post_expt.loc[islog, :]
        return post_expt

    @property
    def prior_prediction(self):
        """prior prediction (e.g. forecast) variances

        Returns:
            `dict`: a dictionary of prediction name, prior variance pairs

        """
        if self.__prior_prediction is not None:
            return self.__prior_prediction
        else:
            if self.predictions is not None:
                self.log("propagating prior to predictions")
                prior_cov = self.predictions.T * self.parcov * self.predictions
                self.__prior_prediction = {
                    n: v for n, v in zip(prior_cov.row_names, np.diag(prior_cov.x))
                }
                self.log("propagating prior to predictions")
            else:
                self.__prior_prediction = {}
            return self.__prior_prediction

    def apply_karhunen_loeve_scaling(self):
        """apply karhuene-loeve scaling to the jacobian matrix.

        Note:

            This scaling is not necessary for analyses using Schur's
            complement, but can be very important for error variance
            analyses.  This operation effectively transfers prior knowledge
            specified in the parcov to the jacobian and reset parcov to the
            identity matrix.

        """
        cnames = copy.deepcopy(self.jco.col_names)
        self.__jco *= self.fehalf
        self.__jco.col_names = cnames
        self.__parcov = self.parcov.identity

    def clean(self):
        """drop regularization and prior information observation from the jco"""
        if self.pst_arg is None:
            self.logger.statement("linear_analysis.clean(): not pst object")
            return
        if not self.pst.estimation and self.pst.nprior > 0:
            self.drop_prior_information()

    def reset_pst(self, arg):
        """reset the LinearAnalysis.pst attribute

        Args:
            arg (`str` or `pyemu.Pst`): the value to assign to the pst attribute

        """
        self.logger.statement("resetting pst")
        self.__pst = None
        self.pst_arg = arg

    def reset_parcov(self, arg=None):
        """reset the parcov attribute to None

        Args:
            arg (`str` or `pyemu.Matrix`): the value to assign to the parcov
                attribute.  If None, the private __parcov attribute is cleared
                but not reset
        """
        self.logger.statement("resetting parcov")
        self.__parcov = None
        if arg is not None:
            self.parcov_arg = arg

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

    def drop_prior_information(self):
        """drop the prior information from the jco and pst attributes"""
        if self.jco is None:
            self.logger.statement("can't drop prior info, LinearAnalysis.jco is None")
            return
        nprior_str = str(self.pst.nprior)
        self.log(
            "removing " + nprior_str + " prior info from jco, pst, and " + "obs cov"
        )
        # pi_names = list(self.pst.prior_information.pilbl.values)
        pi_names = list(self.pst.prior_names)
        missing = [name for name in pi_names if name not in self.jco.obs_names]
        if len(missing) > 0:
            raise Exception(
                "LinearAnalysis.drop_prior_information(): "
                + " prior info not found: {0}".format(missing)
            )
        if self.jco is not None:
            self.__jco.drop(pi_names, axis=0)
        self.__pst.prior_information = self.pst.null_prior
        self.__pst.control_data.pestmode = "estimation"
        # self.__obscov.drop(pi_names,axis=0)
        self.log(
            "removing " + nprior_str + " prior info from jco, pst, and " + "obs cov"
        )

    def get(self, par_names=None, obs_names=None, astype=None):
        """method to get a new LinearAnalysis class using a
        subset of parameters and/or observations

        Args:
            par_names ([`'str`]): par names for new object
            obs_names ([`'str`]): obs names for new object
            astype (`pyemu.Schur` or `pyemu.ErrVar`): type to
                cast the new object.  If None, return type is
                same as self

        Returns:
            `LinearAnalysis`: new instance

        """
        # make sure we aren't fooling with unwanted prior information
        self.clean()
        # if there is nothing to do but copy
        if par_names is None and obs_names is None:
            if astype is not None:
                self.logger.warn(
                    "LinearAnalysis.get(): astype is not None, "
                    + "but par_names and obs_names are None so"
                    + "\n  ->Omitted attributes will not be "
                    + "propagated to new instance"
                )
            else:
                return copy.deepcopy(self)
        # make sure the args are lists
        if par_names is not None and not isinstance(par_names, list):
            par_names = [par_names]
        if obs_names is not None and not isinstance(obs_names, list):
            obs_names = [obs_names]

        if par_names is None:
            par_names = self.jco.col_names
        if obs_names is None:
            obs_names = self.jco.row_names
        # if possible, get a new parcov
        if self.parcov:
            new_parcov = self.parcov.get(
                col_names=[
                    pname for pname in par_names if pname in self.parcov.col_names
                ]
            )
        else:
            new_parcov = None
        # if possible, get a new obscov
        if self.obscov_arg is not None:
            new_obscov = self.obscov.get(row_names=obs_names)
        else:
            new_obscov = None
        # if possible, get a new pst
        if self.pst_arg is not None:
            new_pst = self.pst.get(par_names=par_names, obs_names=obs_names)
        else:
            new_pst = None
        new_extract = None
        if self.predictions:
            # new_preds = []
            # for prediction in self.predictions:
            #     new_preds.append(prediction.get(row_names=par_names))
            new_preds = self.predictions.get(row_names=par_names)

        else:
            new_preds = None
        if self.jco_arg is not None:
            new_jco = self.jco.get(row_names=obs_names, col_names=par_names)
        else:
            new_jco = None
        if astype is not None:
            new = astype(
                jco=new_jco,
                pst=new_pst,
                parcov=new_parcov,
                obscov=new_obscov,
                predictions=new_preds,
                verbose=False,
            )
        else:
            # return a new object of the same type
            new = type(self)(
                jco=new_jco,
                pst=new_pst,
                parcov=new_parcov,
                obscov=new_obscov,
                predictions=new_preds,
                verbose=False,
            )
        return new

    def adjust_obscov_resfile(self, resfile=None):
        """reset the elements of obscov by scaling the implied weights
        based on the phi components in res_file so that the total phi
        is equal to the number of non-zero weights.

        Args:
            resfile (`str`): residual file to use.  If None, residual
                file with case name is sought. default is None

        Note:
            calls `pyemu.Pst.adjust_weights_resfile()`

        """
        self.pst.adjust_weights_resfile(resfile)
        self.__obscov.from_observation_data(self.pst)

    def get_par_css_dataframe(self):
        """get a dataframe of composite scaled sensitivities.  Includes both
        PEST-style and Hill-style.

        Returns:
            `pandas.DataFrame`: a dataframe of parameter names, PEST-style and
            Hill-style composite scaled sensitivity

        """

        if self.jco is None:
            raise Exception("jco is None")
        if self.pst is None:
            raise Exception("pst is None")
        jco = self.jco.to_dataframe()
        weights = self.pst.observation_data.loc[jco.index, "weight"].copy().values
        jco = (jco.T * weights).T

        dss_sum = jco.apply(np.linalg.norm)
        css = (dss_sum / float(self.pst.nnz_obs)).to_frame()
        css.columns = ["pest_css"]
        # log transform stuff
        self.pst.add_transform_columns()
        parval1 = self.pst.parameter_data.loc[dss_sum.index, "parval1_trans"].values
        css.loc[:, "hill_css"] = (dss_sum * parval1) / (float(self.pst.nnz_obs) ** 2)
        return css

    def get_cso_dataframe(self):
        """get a dataframe of composite observation sensitivity, as returned by PEST in the
        seo file.

        Returns:
            `pandas.DataFrame`: dataframe of observation names and composite observation
            sensitivity

        Note:
             That this formulation deviates slightly from the PEST documentation in that the
             values are divided by (npar-1) rather than by (npar).

             The equation is cso_j = ((Q^1/2*J*J^T*Q^1/2)^1/2)_jj/(NPAR-1)


        """
        if self.jco is None:
            raise Exception("jco is None")
        if self.pst is None:
            raise Exception("pst is None")
        weights = (
            self.pst.observation_data.loc[self.jco.to_dataframe().index, "weight"]
            .copy()
            .values
        )
        cso = np.diag(np.sqrt((self.qhalfx.x.dot(self.qhalfx.x.T)))) / (
            float(self.pst.npar - 1)
        )
        cso_df = pd.DataFrame.from_dict(
            {"obnme": self.jco.to_dataframe().index, "cso": cso}
        )
        cso_df.index = cso_df["obnme"]
        cso_df.drop("obnme", axis=1, inplace=True)
        return cso_df

    def get_obs_competition_dataframe(self):
        """get the observation competition stat a la PEST utility

        Returns:
            `pandas.DataFrame`: a dataframe of observation names by
            observation names with values equal to the PEST
            competition statistic

        """
        if self.jco is None:
            raise Exception("jco is None")
        if self.pst is None:
            raise Exception("pst is None")
        if self.pst.res is None:
            raise Exception("res is None")
        onames = self.pst.nnz_obs_names
        weights = self.pst.observation_data.loc[onames, "weight"].to_dict()
        residuals = self.pst.res.loc[onames, "residual"].to_dict()
        jco = self.jco.to_dataframe()
        df = pd.DataFrame(columns=onames, index=onames)
        for i, oname in enumerate(onames):
            df.loc[oname, oname] = 0.0
            for ooname in onames[i + 1 :]:
                oc = (
                    weights[oname]
                    * weights[ooname]
                    * np.dot(
                        jco.loc[oname, :].values, jco.loc[ooname, :].values.transpose()
                    )
                )
                df.loc[oname, ooname] = oc
                df.loc[ooname, oname] = oc
        return df
