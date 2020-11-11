from __future__ import print_function, division
import numpy as np
import pandas as pd
from pyemu.la import LinearAnalysis
from pyemu.mat.mat_handler import Matrix, Jco, Cov


class ErrVar(LinearAnalysis):
    """FOSM-based error variance analysis

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
            Can also be a `pyemu.Matrix` instance, a `numpy.ndarray` or a collection.  Note if the PEST++ option
            "++forecasts()" is set in the pest control file (under the `pyemu.Pst.pestpp_options` dictionary),
            then there is no need to pass this argument (unless you want to analyze different forecasts)
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
        omitted_parameters ([`str`]): list of parameters to treat as "omitted".  Passing this argument
            activates 3-term error variance analysis.
        omitted_parcov (varies): an argument that can be cast to a parcov for the omitted parameters.
            If None, omitted_parcov will be formed by extracting a sub-matrix from the `LinearAnalsis.parcov`
            attribute.
        omitted_predictions (varies): an argument that can be cast to a "predictions" (e.g. "forecasts")
            attribute to form prediction sensitivity vectors with respec to the omitted parameters.  If None,
            these vectors will be extracted from the `pyemu.LinearAnalysis.predictions` attribute
        kl (`bool`, optional): flag to perform Karhunen-Loeve scaling on the jacobian before error variance
            calculations. If `True`, the `pyemu.ErrVar.jco` and `pyemu.ErrVar.parcov` are altered in place.
            Default is `False`.

    Example::

        ev = pyemu.ErrVar(jco="my.jco",omitted_parameters=["wel1","wel2"])
        df = ev.get_errvar_dataframe()

    """

    def __init__(self, jco, **kwargs):

        self.__need_omitted = False
        if "omitted_parameters" in kwargs.keys():
            self.omitted_par_arg = kwargs["omitted_parameters"]
            kwargs.pop("omitted_parameters")
            self.__need_omitted = True
        else:
            self.omitted_par_arg = None
        if "omitted_parcov" in kwargs.keys():
            self.omitted_parcov_arg = kwargs["omitted_parcov"]
            kwargs.pop("omitted_parcov")
            self.__need_omitted = True
        else:
            self.omitted_parcov_arg = None

        if "omitted_forecasts" in kwargs.keys():
            self.omitted_predictions_arg = kwargs["omitted_forecasts"]
            kwargs.pop("omitted_forecasts")
            self.__need_omitted = True
        elif "omitted_predictions" in kwargs.keys():
            self.omitted_predictions_arg = kwargs["omitted_predictions"]
            kwargs.pop("omitted_predictions")
            self.__need_omitted = True
        else:
            self.omitted_predictions_arg = None

        kl = False
        if "kl" in kwargs.keys():
            kl = bool(kwargs["kl"])
            kwargs.pop("kl")

        self.__qhalfx = None
        self.__R = None
        self.__R_sv = None
        self.__G = None
        self.__G_sv = None
        self.__I_R = None
        self.__I_R_sv = None
        self.__omitted_jco = None
        self.__omitted_parcov = None
        self.__omitted_predictions = None

        # instantiate the parent class
        super(ErrVar, self).__init__(jco, **kwargs)
        if self.__need_omitted:
            self.log("pre-loading omitted components")
            # self._LinearAnalysis__load_jco()
            # self._LinearAnalysis__load_parcov()
            # self._LinearAnalysis__load_obscov()
            # if self.prediction_arg is not None:
            #    self._LinearAnalysis__load_predictions()
            self.__load_omitted_jco()
            self.__load_omitted_parcov()
            if self.prediction_arg is not None:
                self.__load_omitted_predictions()
            self.log("pre-loading omitted components")
        if kl:
            self.log("applying KL scaling")
            self.apply_karhunen_loeve_scaling()
            self.log("applying KL scaling")

        self.valid_terms = ["null", "solution", "omitted", "all"]
        self.valid_return_types = ["parameters", "predictions"]

    def __load_omitted_predictions(self):
        """private: set the omitted_predictions attribute"""
        # if there are no base predictions
        if self.predictions is None:
            raise Exception(
                "ErrVar.__load_omitted_predictions(): "
                + "no 'included' predictions is None"
            )
        if self.omitted_predictions_arg is None and self.omitted_par_arg is None:
            raise Exception(
                "ErrVar.__load_omitted_predictions: " + "both omitted args are None"
            )
        # try to set omitted_predictions by
        # extracting from existing predictions
        if self.omitted_predictions_arg is None and self.omitted_par_arg is not None:
            # check to see if omitted par names are in each predictions
            found = True
            missing_par, missing_pred = None, None
            for par_name in self.omitted_jco.col_names:
                for prediction in self.predictions_iter:
                    if par_name not in prediction.row_names:
                        found = False
                        missing_par = par_name
                        missing_pred = prediction.col_names[0]
                        break
            if found:
                opreds = []
                # need to access the attribute directly,
                # not a view of attribute
                opred_mat = self._LinearAnalysis__predictions.extract(
                    row_names=self.omitted_jco.col_names
                )
                opreds = [opred_mat.get(col_names=name) for name in self.forecast_names]

                # for prediction in self._LinearAnalysis__predictions:
                #    opred = prediction.extract(self.omitted_jco.col_names)
                #    opreds.append(opred)
                self.__omitted_predictions = opreds
            else:
                raise Exception(
                    "ErrVar.__load_omitted_predictions(): "
                    + " omitted parameter "
                    + str(missing_par)
                    + " not found in prediction vector "
                    + str(missing_pred)
                )
        elif self.omitted_parcov_arg is not None:
            raise NotImplementedError()

    def __load_omitted_parcov(self):
        """private: set the omitted_parcov attribute"""
        if self.omitted_parcov_arg is None and self.omitted_par_arg is None:
            raise Exception(
                "ErrVar.__load_omitted_parcov: " + "both omitted args are None"
            )
        # try to set omitted_parcov by extracting from base parcov
        if self.omitted_parcov_arg is None and self.omitted_par_arg is not None:
            # check to see if omitted par names are in parcov
            found = True
            for par_name in self.omitted_jco.col_names:
                if par_name not in self.parcov.col_names:
                    found = False
                    break
            if found:
                # need to access attribute directly, not view of attribute
                self.__omitted_parcov = self._LinearAnalysis__parcov.extract(
                    row_names=self.omitted_jco.col_names
                )
            else:
                self.logger.warn(
                    "ErrVar.__load_omitted_parun: "
                    + "no omitted parcov arg passed: "
                    + "setting omitted parcov as identity Matrix"
                )
                self.__omitted_parcov = Cov(
                    x=np.ones(self.omitted_jco.shape[1]),
                    names=self.omitted_jco.col_names,
                    isdiagonal=True,
                )
        elif self.omitted_parcov_arg is not None:
            raise NotImplementedError()

    def __load_omitted_jco(self):
        """private: set the omitted jco attribute"""
        if self.omitted_par_arg is None:
            raise Exception("ErrVar.__load_omitted: omitted_arg is None")
        if isinstance(self.omitted_par_arg, str):
            if self.omitted_par_arg in self.jco.col_names:
                # need to access attribute directly, not view of attribute
                self.__omitted_jco = self._LinearAnalysis__jco.extract(
                    col_names=self.omitted_par_arg
                )
            else:
                # must be a filename
                self.__omitted_jco = self.__fromfile(self.omitted_par_arg)
        # if the arg is an already instantiated Matrix (or jco) object
        elif isinstance(self.omitted_par_arg, Jco) or isinstance(
            self.omitted_par_arg, Matrix
        ):
            self.__omitted_jco = Jco(
                x=self.omitted_par_arg.newx(),
                row_names=self.omitted_par_arg.row_names,
                col_names=self.omitted_par_arg.col_names,
            )
        # if it is a list, then it must be a list
        # of parameter names in self.jco
        elif isinstance(self.omitted_par_arg, list):
            for arg in self.omitted_par_arg:
                if isinstance(arg, str):
                    assert arg in self.jco.col_names, (
                        "ErrVar.__load_omitted_jco: omitted_jco "
                        + "arg str not in jco par_names: "
                        + str(arg)
                    )
            self.__omitted_jco = self._LinearAnalysis__jco.extract(
                col_names=self.omitted_par_arg
            )

    # these property decorators help keep from loading potentially
    # unneeded items until they are called
    # returns a view only - cheap, but can be dangerous

    @property
    def omitted_predictions(self):
        """omitted prediction sensitivity vectors

        Returns:
            `pyemu.Matrix`: a matrix of prediction sensitivity vectors (column wise) to
            omitted parameters

        """
        if self.__omitted_predictions is None:
            self.log("loading omitted_predictions")
            self.__load_omitted_predictions()
            self.log("loading omitted_predictions")
        return self.__omitted_predictions

    @property
    def omitted_jco(self):
        """the omitted-parameters jacobian matrix

        Returns:
            `pyemu.Jco`: the jacobian matrix instance of non-zero-weighted observations and
            omitted parameters

        """
        if self.__omitted_jco is None:
            self.log("loading omitted_jco")
            self.__load_omitted_jco()
            self.log("loading omitted_jco")
        return self.__omitted_jco

    @property
    def omitted_parcov(self):
        """the prior omitted-parameter covariance matrix

        Returns:
            `pyemu.Cov`: the prior parameter covariance matrix of the
            omitted parameters

        """
        if self.__omitted_parcov is None:
            self.log("loading omitted_parcov")
            self.__load_omitted_parcov()
            self.log("loading omitted_parcov")
        return self.__omitted_parcov

    def get_errvar_dataframe(self, singular_values=None):
        """primary entry point for error variance analysis.

        Args:
            singular_values ([`int`], optional): a list singular values to test. If `None`,
                defaults to `range(0,min(nnz_obs,nadj_par) + 1)`.

        Returns:
            `pandas.DataFrame`: a multi-indexed pandas dataframe summarizing each of the
            error variance terms for each nominated forecast. Rows are the singluar values
            tested, columns are a multi-index of forecast name and error variance term number
            (e.g. 1,2 or (optionally) 3).

        Example::

            ev = pyemu.ErrVar(jco="my.jco",omitted_parameters=["wel1","wel2"])
            df = ev.get_errvar_dataframe()

        """
        if singular_values is None:
            singular_values = np.arange(0, min(self.pst.nnz_obs, self.pst.npar_adj) + 1)
        if not isinstance(singular_values, list) and not isinstance(
            singular_values, np.ndarray
        ):
            singular_values = [singular_values]
        results = {}
        for singular_value in singular_values:
            sv_results = self.variance_at(singular_value)
            for key, val in sv_results.items():
                if key not in results.keys():
                    results[key] = []
                results[key].append(val)
        return pd.DataFrame(results, index=singular_values)

    def get_identifiability_dataframe(self, singular_value=None, precondition=False):
        """primary entry point for identifiability analysis

        Args:
            singular_value (`int`): the singular spectrum truncation point. Defaults
                to minimum of non-zero-weighted observations and adjustable parameters
            precondition (`bool`): flag to use the preconditioned hessian with the prior
                parameter covariance matrix (xtqt + sigma_theta^-1).  This should be used
                KL scaling. Default is `False`.

        Returns:
            `pandas.DataFrame`: A pandas dataframe of the right solution-space singular
            vectors and identifiability (identifiabiity is in the column labeled "ident")

        Examples::

            ev = pyemu.ErrVar(jco="my.jco")
            df = ev.get_identifiability_dataframe(singular_value=20)
            df.ident.plot(kind="bar")

        """
        if singular_value is None:
            singular_value = int(min(self.pst.nnz_obs, self.pst.npar_adj))
        # v1_df = self.qhalfx.v[:, :singular_value].to_dataframe() ** 2
        xtqx = self.xtqx
        if precondition:
            xtqx = xtqx + self.parcov.inv
        # v1_df = self.xtqx.v[:, :singular_value].to_dataframe() ** 2
        v1_df = xtqx.v[:, :singular_value].to_dataframe() ** 2
        v1_df["ident"] = v1_df.sum(axis=1)
        return v1_df

    def variance_at(self, singular_value):
        """get the error variance of all three error variance terms at a
         given singluar value

        Args:
            singular_value (`int`): singular value to test

        Returns:
            `dict`: dictionary of (err var term,prediction_name), variance pairs

        """
        results = {}
        results.update(self.first_prediction(singular_value))
        results.update(self.second_prediction(singular_value))
        results.update(self.third_prediction(singular_value))
        return results

    def R(self, singular_value):
        """get resolution Matrix (V_1 * V_1^T) at a given singular value

        Args:
        singular_value (`int`): singular value to calculate `R` at

        Returns:
            `pyemu.Matrix`: resolution matrix at `singular_value`

        """
        if self.__R is not None and singular_value == self.__R_sv:
            return self.__R

        elif singular_value > self.jco.ncol:
            self.__R_sv = self.jco.ncol
            return self.parcov.identity
        else:
            self.log("calc R @" + str(singular_value))
            # v1 = self.qhalfx.v[:, :singular_value]
            v1 = self.xtqx.v[:, :singular_value]
            self.__R = v1 * v1.T
            self.__R_sv = singular_value
            self.log("calc R @" + str(singular_value))
            return self.__R

    def I_minus_R(self, singular_value):
        """get I - R at a given singular value

        Args:
            singular_value (`int`): singular value to calculate I - R at

        Returns:
            `pyemu.Matrix`: identity matrix minus resolution matrix at `singular_value`

        """
        if self.__I_R is not None and singular_value == self.__I_R_sv:
            return self.__I_R
        else:
            if singular_value > self.jco.ncol:
                return self.parcov.zero
            else:
                # v2 = self.qhalfx.v[:, singular_value:]
                v2 = self.xtqx.v[:, singular_value:]
                self.__I_R = v2 * v2.T
                self.__I_R_sv = singular_value
                return self.__I_R

    def G(self, singular_value):
        """get the parameter solution Matrix at a given singular value

        Args:
            singular_value (`int`): singular value to calc G at

        Returns:
            `pyemu.Matrix`: parameter solution matrix  (V_1 * S_1^(_1) * U_1^T) at `singular_value`

        """
        if self.__G is not None and singular_value == self.__G_sv:
            return self.__G

        if singular_value == 0:
            self.__G_sv = 0
            self.__G = Matrix(
                x=np.zeros((self.jco.ncol, self.jco.nrow)),
                row_names=self.jco.col_names,
                col_names=self.jco.row_names,
            )
            return self.__G
        mn = min(self.jco.shape)
        try:
            mn = min(self.pst.npar_adj, self.pst.nnz_obs)
        except:
            pass
        if singular_value > mn:
            self.logger.warn(
                "ErrVar.G(): singular_value > min(npar,nobs):"
                + "resetting to min(npar,nobs): "
                + str(min(self.pst.npar_adj, self.pst.nnz_obs))
            )
            singular_value = min(self.pst.npar_adj, self.pst.nnz_obs)
        self.log("calc G @" + str(singular_value))
        # v1 = self.qhalfx.v[:, :singular_value]
        v1 = self.xtqx.v[:, :singular_value]
        # s1 = ((self.qhalfx.s[:singular_value]) ** 2).inv
        s1 = (self.xtqx.s[:singular_value]).inv
        self.__G = v1 * s1 * v1.T * self.jco.T * self.obscov.inv
        self.__G_sv = singular_value
        self.__G.row_names = self.jco.col_names
        self.__G.col_names = self.jco.row_names
        self.__G.autoalign = True
        self.log("calc G @" + str(singular_value))
        return self.__G

    def first_forecast(self, singular_value):
        """get the null space term (first term) contribution to forecast (e.g. prediction)
         error variance at a given singular value.

        Args:
            singular_value (`int`): singular value to calc first term at

        Note:
             This method is used to construct the error variance dataframe

             Just a wrapper around `ErrVar.first_forecast`

        Returns:
            `dict`: dictionary of ("first",prediction_names),error variance pairs at `singular_value`

        """
        return self.first_prediction(singular_value)

    def first_prediction(self, singular_value):
        """get the null space term (first term) contribution to prediction error variance
            at a given singular value.

        Args:
            singular_value (`int`): singular value to calc first term at

        Note:
             This method is used to construct the error variance dataframe

        Returns:
            `dict`: dictionary of ("first",prediction_names),error variance pairs at `singular_value`

        """
        if not self.predictions:
            raise Exception("ErrVar.first(): no predictions are set")
        if singular_value > self.jco.ncol:
            zero_preds = {}
            for pred in self.predictions_iter:
                zero_preds[("first", pred.col_names[0])] = 0.0
            return zero_preds
        self.log("calc first term parameter @" + str(singular_value))
        first_term = (
            self.I_minus_R(singular_value).T
            * self.parcov
            * self.I_minus_R(singular_value)
        )
        if self.predictions:
            results = {}
            for prediction in self.predictions_iter:
                results[("first", prediction.col_names[0])] = float(
                    (prediction.T * first_term * prediction).x
                )
            self.log("calc first term parameter @" + str(singular_value))
            return results

    def first_parameter(self, singular_value):
        """get the null space term (first term) contribution to parameter error variance
            at a given singular value

        Args:
            singular_value (`int`): singular value to calc first term at

        Returns:
            `pyemu.Cov`: first term contribution to parameter error variance

        """
        self.log("calc first term parameter @" + str(singular_value))
        first_term = (
            self.I_minus_R(singular_value)
            * self.parcov
            * self.I_minus_R(singular_value)
        )
        self.log("calc first term parameter @" + str(singular_value))
        return first_term

    def second_forecast(self, singular_value):
        """get the solution space contribution to forecast (e.g. "prediction") error variance
        at a given singular value

        Args:
            singular_value (`int`): singular value to calc second term at

        Note:
            This method is used to construct error variance dataframe

            Just a thin wrapper around `ErrVar.second_prediction`

        Returns:
            `dict`:  dictionary of ("second",prediction_names), error variance
            arising from the solution space contribution (y^t * G * obscov * G^T * y)

        """
        return self.second_prediction(singular_value)

    def second_prediction(self, singular_value):
        """get the solution space contribution to predictive error variance
        at a given singular value

        Args:
            singular_value (`int`): singular value to calc second term at

        Note:
            This method is used to construct error variance dataframe

        Returns:\
            `dict`:  dictionary of ("second",prediction_names), error variance
            arising from the solution space contribution (y^t * G * obscov * G^T * y)

        """

        if not self.predictions:
            raise Exception("ErrVar.second(): not predictions are set")
        self.log("calc second term prediction @" + str(singular_value))

        mn = min(self.jco.shape)
        try:
            mn = min(self.pst.npar_adj, self.pst.nnz_obs)
        except:
            pass
        if singular_value > mn:
            inf_pred = {}
            for pred in self.predictions_iter:
                inf_pred[("second", pred.col_names[0])] = 1.0e35
            return inf_pred
        else:
            second_term = (
                self.G(singular_value) * self.obscov * self.G(singular_value).T
            )
            results = {}
            for prediction in self.predictions_iter:
                results[("second", prediction.col_names[0])] = float(
                    (prediction.T * second_term * prediction).x
                )
            self.log("calc second term prediction @" + str(singular_value))
            return results

    def second_parameter(self, singular_value):
        """get the solution space contribution to parameter error variance
             at a given singular value (G * obscov * G^T)

        Args:
            singular_value (`int`): singular value to calc second term at

        Returns:
            `pyemu.Cov`: the second term contribution to parameter error variance
            (G * obscov * G^T)

        """
        self.log("calc second term parameter @" + str(singular_value))
        result = self.G(singular_value) * self.obscov * self.G(singular_value).T
        self.log("calc second term parameter @" + str(singular_value))
        return result

    def third_forecast(self, singular_value):
        """get the omitted parameter contribution to forecast (`prediction`) error variance
         at a given singular value.

        Args:
            singular_value (`int`): singular value to calc third term at

        Note:
             used to construct error variance dataframe
             just a thin wrapper around `ErrVar.third_prediction()`

        Returns:
            `dict`: a dictionary of ("third",prediction_names),error variance

        """
        return self.third_prediction(singular_value)

    def third_prediction(self, singular_value):
        """get the omitted parameter contribution to prediction error variance
         at a given singular value.

        Args:
            singular_value (`int`): singular value to calc third term at

        Note:
             used to construct error variance dataframe

        Returns:
            `dict`: a dictionary of ("third",prediction_names),error variance
        """
        if not self.predictions:
            raise Exception("ErrVar.third(): not predictions are set")
        if self.__need_omitted is False:
            zero_preds = {}
            for pred in self.predictions_iter:
                zero_preds[("third", pred.col_names[0])] = 0.0
            return zero_preds
        self.log("calc third term prediction @" + str(singular_value))
        mn = min(self.jco.shape)
        try:
            mn = min(self.pst.npar_adj, self.pst.nnz_obs)
        except:
            pass
        if singular_value > mn:
            inf_pred = {}
            for pred in self.predictions_iter:
                inf_pred[("third", pred.col_names[0])] = 1.0e35
            return inf_pred
        else:
            results = {}
            for prediction, omitted_prediction in zip(
                self.predictions_iter, self.omitted_predictions
            ):
                # comes out as row vector, but needs to be a column vector
                p = (
                    (prediction.T * self.G(singular_value) * self.omitted_jco)
                    - omitted_prediction.T
                ).T
                result = float((p.T * self.omitted_parcov * p).x)
                results[("third", prediction.col_names[0])] = result
            self.log("calc third term prediction @" + str(singular_value))
            return results

    def third_parameter(self, singular_value):
        """get the omitted parameter contribution to parameter error variance
             at a given singular value

        Args:
            singular_value (`int`): singular value to calc third term at

        Returns:
            `pyemu.Cov`: the third term contribution to parameter error variance
            calculated at `singular_value` (G * omitted_jco * Sigma_(omitted_pars) *
            omitted_jco^T * G^T).  Returns 0.0 if third term calculations are not
            being used.

        """
        if self.__need_omitted is False:
            return 0.0
        self.log("calc third term parameter @" + str(singular_value))
        GZo = self.G(singular_value) * self.omitted_jco
        result = GZo * self.omitted_parcov * GZo.T
        self.log("calc third term parameter @" + str(singular_value))
        return result

    def get_null_proj(self, maxsing=None, eigthresh=1.0e-6):
        """get a null-space projection matrix of XTQX

        Args:
            maxsing (`int`, optional): number of singular components
                to use (the truncation point).  If None, `pyemu.Matrx.get_maxsing()
                is used to determine the truncation point with `eigthresh`. Default
                is None
            eigthresh (`float`, optional): the ratio of smallest to largest singular
                value to keep in the range (solution) space of XtQX.  Not used if
                `maxsing` is not `None`.  Default is 1.0e-6

        Note:
            used for null-space monte carlo operations.

        Returns:
            `pyemu.Matrix` the null-space projection matrix (V2V2^T)

        """
        if maxsing is None:
            maxsing = self.xtqx.get_maxsing(eigthresh=eigthresh)
        print("using {0} singular components".format(maxsing))
        self.log(
            "forming null space projection matrix with "
            + "{0} of {1} singular components".format(maxsing, self.jco.shape[1])
        )

        v2_proj = self.xtqx.v[:, maxsing:] * self.xtqx.v[:, maxsing:].T
        self.log(
            "forming null space projection matrix with "
            + "{0} of {1} singular components".format(maxsing, self.jco.shape[1])
        )

        return v2_proj

    # def get_nsing(self, epsilon=1.0e-4):
    #     """ get the number of solution space dimensions given
    #     a ratio between the largest and smallest singular values
    #
    #     Parameters
    #     ----------
    #     epsilon: float
    #         singular value ratio
    #
    #     Returns
    #     -------
    #     nsing : float
    #         number of singular components above the epsilon ratio threshold
    #
    #     Note
    #     -----
    #         If nsing == nadj_par, then None is returned
    #
    #     """
    #     mx = self.xtqx.shape[0]
    #     nsing = mx - np.searchsorted(
    #         np.sort((self.xtqx.s.x / self.xtqx.s.x.max())[:, 0]), epsilon)
    #     if nsing == mx:
    #         self.logger.warn("optimal nsing=npar")
    #         nsing = None
    #     return nsing
