"""module for error variance analysis, using FOSM
assumptions.
"""
from __future__ import print_function, division
import numpy as np
import pandas as pd
from pyemu.la import LinearAnalysis
from pyemu.mat.mat_handler import Matrix, Jco, Cov

class ErrVar(LinearAnalysis):
    """Derived class for error variance analysis.  Supports 2-term and
    3-term error variance analysis.  Inherits from pyemu.LinearAnalysis.

    
    Parameters
    ----------
    **kwargs : dict
        keyword args to pass to the LinearAnalysis base constructor.
    omitted_parameters :  list
        list of parameters to treat as "omitted".  Passing this argument
        activates 3-term error variance analysis.

    omitted_parcov : varies
        argument that can be cast to a parcov for the omitted parameters.
        If None, omitted_parcov will be formed by extracting from the
        LinearAnalsis.parcov attribute.
    omitted_predictions : varies
        argument that can be cast to a "predictions" attribute for
        prediction sensitivity vectors WRT omitted parameters.  If None,
        these vectors will be extracted from the LinearAnalysis.predictions
        attribute
    kl : bool
        flag to perform KL scaling on the jacobian before error variance
        calculations

    Note
    ----
    There are some additional keyword args that can be passed to active
    the 3-term error variance calculation


    """

    def __init__(self,jco,**kwargs):

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
            #self._LinearAnalysis__load_jco()
            #self._LinearAnalysis__load_parcov()
            #self._LinearAnalysis__load_obscov()
            #if self.prediction_arg is not None:
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

        self.valid_terms = ["null","solution", "omitted", "all"]
        self.valid_return_types = ["parameters", "predictions"]

    def __load_omitted_predictions(self):
        """private: set the omitted_predictions attribute
        """
        # if there are no base predictions
        if self.predictions is None:
            raise Exception("ErrVar.__load_omitted_predictions(): " +
                            "no 'included' predictions is None")
        if self.omitted_predictions_arg is None and \
                        self.omitted_par_arg is None:
            raise Exception("ErrVar.__load_omitted_predictions: " +
                            "both omitted args are None")
        # try to set omitted_predictions by
        # extracting from existing predictions
        if self.omitted_predictions_arg is None and \
                        self.omitted_par_arg is not None:
            # check to see if omitted par names are in each predictions
            found = True
            missing_par,missing_pred = None, None
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
                            row_names=self.omitted_jco.col_names)
                opreds = [opred_mat.get(col_names=name) for name in self.forecast_names]

                #for prediction in self._LinearAnalysis__predictions:
                #    opred = prediction.extract(self.omitted_jco.col_names)
                #    opreds.append(opred)
                self.__omitted_predictions = opreds
            else:
                raise Exception("ErrVar.__load_omitted_predictions(): " +
                                " omitted parameter " + str(missing_par) +\
                                " not found in prediction vector " +
                                str(missing_pred))
        elif self.omitted_parcov_arg is not None:
            raise NotImplementedError()

    def __load_omitted_parcov(self):
        """private: set the omitted_parcov attribute
        """
        if self.omitted_parcov_arg is None and self.omitted_par_arg is None:
            raise Exception("ErrVar.__load_omitted_parcov: " +
                            "both omitted args are None")
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
                self.__omitted_parcov = \
                    self._LinearAnalysis__parcov.extract(
                        row_names=self.omitted_jco.col_names)
            else:
                self.logger.warn("ErrVar.__load_omitted_parun: " +
                                 "no omitted parcov arg passed: " +
                        "setting omitted parcov as identity Matrix")
                self.__omitted_parcov = Cov(
                    x=np.ones(self.omitted_jco.shape[1]),
                    names=self.omitted_jco.col_names, isdiagonal=True)
        elif self.omitted_parcov_arg is not None:
            raise NotImplementedError()

    def __load_omitted_jco(self):
        """private: set the omitted jco attribute
        """
        if self.omitted_par_arg is None:
            raise Exception("ErrVar.__load_omitted: omitted_arg is None")
        if isinstance(self.omitted_par_arg,str):
            if self.omitted_par_arg in self.jco.col_names:
                # need to access attribute directly, not view of attribute
                self.__omitted_jco = \
                    self._LinearAnalysis__jco.extract(
                        col_names=self.omitted_par_arg)
            else:
                # must be a filename
                self.__omitted_jco = self.__fromfile(self.omitted_par_arg)
        # if the arg is an already instantiated Matrix (or jco) object
        elif isinstance(self.omitted_par_arg,Jco) or \
                isinstance(self.omitted_par_arg,Matrix):
            self.__omitted_jco = \
                Jco(x=self.omitted_par_arg.newx(),
                          row_names=self.omitted_par_arg.row_names,
                          col_names=self.omitted_par_arg.col_names)
        # if it is a list, then it must be a list
        # of parameter names in self.jco
        elif isinstance(self.omitted_par_arg,list):
            for arg in self.omitted_par_arg:
                if isinstance(arg,str):
                    assert arg in self.jco.col_names,\
                        "ErrVar.__load_omitted_jco: omitted_jco " +\
                        "arg str not in jco par_names: " + str(arg)
            self.__omitted_jco = \
                self._LinearAnalysis__jco.extract(col_names=self.omitted_par_arg)

    # these property decorators help keep from loading potentially
    # unneeded items until they are called
    # returns a view only - cheap, but can be dangerous

    @property
    def omitted_predictions(self):
        """ get the omitted prediction sensitivity vectors (stored as
        a pyemu.Matrix)

        Returns
        -------
        omitted_predictions : pyemu.Matrix
            a matrix of prediction sensitivity vectors (column wise) to
            omitted parameters

        Note
        ----
        returns a reference
        
        if ErrorVariance.__omitted_predictions is not set, then dynamically load the
        attribute before returning

        """
        if self.__omitted_predictions is None:
            self.log("loading omitted_predictions")
            self.__load_omitted_predictions()
            self.log("loading omitted_predictions")
        return self.__omitted_predictions

    @property
    def omitted_jco(self):
        """get the omitted jco

        Returns
        -------
        omitted_jco : pyemu.Jco

        Note
        ----
        returns a reference
        
        if ErrorVariance.__omitted_jco is None,
        then dynamically load the attribute before returning
        
        """
        if self.__omitted_jco is None:
            self.log("loading omitted_jco")
            self.__load_omitted_jco()
            self.log("loading omitted_jco")
        return self.__omitted_jco

    @property
    def omitted_parcov(self):
        """get the omitted prior parameter covariance matrix

        Returns
        -------
        omitted_parcov : pyemu.Cov

        Note
        ----
        returns a reference
        
        If ErrorVariance.__omitted_parcov is None,
        attribute is dynamically loaded
        
        """
        if self.__omitted_parcov is None:
            self.log("loading omitted_parcov")
            self.__load_omitted_parcov()
            self.log("loading omitted_parcov")
        return self.__omitted_parcov

    def get_errvar_dataframe(self, singular_values=None):
        """get a pandas dataframe of error variance results indexed
            on singular value and (prediction name,<errvar term>)

        Parameters
        ----------
        singular_values : list
            singular values to test.  defaults to
            range(0,min(nnz_obs,nadj_par) + 1)

        Returns
        -------
        pandas.DataFrame : pandas.DataFrame
            multi-indexed pandas dataframe
        
        """
        if singular_values is None:
            singular_values = \
                np.arange(0, min(self.pst.nnz_obs, self.pst.npar_adj) + 1)
        if not isinstance(singular_values, list) and \
                not isinstance(singular_values, np.ndarray):
            singular_values = [singular_values]
        results = {}
        for singular_value in singular_values:
            sv_results = self.variance_at(singular_value)
            for key, val in sv_results.items():
                if key not in results.keys():
                    results[key] = []
                results[key].append(val)
        return pd.DataFrame(results, index=singular_values)


    def get_identifiability_dataframe(self,singular_value=None,precondition=False):
        """get the parameter identifiability as a pandas dataframe

        Parameters
        ----------
        singular_value : int
            the singular spectrum truncation point. Defaults to minimum of
            non-zero-weighted observations and adjustable parameters
        precondition : bool
            flag to use the preconditioned hessian (xtqt + sigma_theta^-1).
            Default is False

        Returns
        -------
        pandas.DataFrame : pandas.DataFrame
            A pandas dataframe of the V_1**2 Matrix with the
            identifiability in the column labeled "ident"

        """
        if singular_value is None:
            singular_value = int(min(self.pst.nnz_obs, self.pst.npar_adj))
        #v1_df = self.qhalfx.v[:, :singular_value].to_dataframe() ** 2
        xtqx = self.xtqx
        if precondition:
            xtqx = xtqx + self.parcov.inv
        #v1_df = self.xtqx.v[:, :singular_value].to_dataframe() ** 2
        v1_df = xtqx.v[:, :singular_value].to_dataframe() ** 2
        v1_df["ident"] = v1_df.sum(axis=1)
        return v1_df

    def variance_at(self, singular_value):
        """get the error variance of all three terms at a singluar value

        Parameters
        ----------
        singular_value : int
            singular value to test

        Returns
        -------
        dict : dict
            dictionary of (err var term,prediction_name), standard_deviation pairs

        """
        results = {}
        results.update(self.first_prediction(singular_value))
        results.update(self.second_prediction(singular_value))
        results.update(self.third_prediction(singular_value))
        return results

    def R(self, singular_value):
        """get resolution Matrix (V_1 * V_1^T) at a singular value

        Parameters
        ----------
        singular_value : int
            singular value to calc R at

        Returns
        -------
        R : pyemu.Matrix
            resolution matrix at singular_value
        """
        if self.__R is not None and singular_value == self.__R_sv:
            return self.__R

        elif singular_value > self.jco.ncol:
            self.__R_sv = self.jco.ncol
            return self.parcov.identity
        else:
            self.log("calc R @" + str(singular_value))
            #v1 = self.qhalfx.v[:, :singular_value]
            v1 = self.xtqx.v[:, :singular_value]
            self.__R = v1 * v1.T
            self.__R_sv = singular_value
            self.log("calc R @" + str(singular_value))
            return self.__R

    def I_minus_R(self,singular_value):
        """get I - R at singular value

        Parameters
        ----------
        singular_value : int
            singular value to calc R at

        Returns
        -------
        I - R : pyemu.Matrix
            identity matrix minus resolution matrix at singular_value

        """
        if self.__I_R is not None and singular_value == self.__I_R_sv:
            return self.__I_R
        else:
            if singular_value > self.jco.ncol:
                return self.parcov.zero
            else:
                #v2 = self.qhalfx.v[:, singular_value:]
                v2 = self.xtqx.v[:, singular_value:]
                self.__I_R = v2 * v2.T
                self.__I_R_sv = singular_value
                return self.__I_R

    def G(self, singular_value):
        """get the parameter solution Matrix at a singular value
            V_1 * S_1^(_1) * U_1^T

        Parameters
        ----------
        singular_value : int
            singular value to calc R at

        Returns
        -------
        G : pyemu.Matrix
            parameter solution matrix at singular value

        """
        if self.__G is not None and singular_value == self.__G_sv:
            return self.__G

        if singular_value == 0:
            self.__G_sv = 0
            self.__G = Matrix(
                x=np.zeros((self.jco.ncol,self.jco.nrow)),
                row_names=self.jco.col_names, col_names=self.jco.row_names)
            return self.__G
        mn = min(self.jco.shape)
        try:
            mn = min(self.pst.npar_adj, self.pst.nnz_obs)
        except:
            pass
        if singular_value > mn:
            self.logger.warn(
                "ErrVar.G(): singular_value > min(npar,nobs):" +
                "resetting to min(npar,nobs): " +
                str(min(self.pst.npar_adj, self.pst.nnz_obs)))
            singular_value = min(self.pst.npar_adj, self.pst.nnz_obs)
        self.log("calc G @" + str(singular_value))
        #v1 = self.qhalfx.v[:, :singular_value]
        v1 = self.xtqx.v[:, :singular_value]
        #s1 = ((self.qhalfx.s[:singular_value]) ** 2).inv
        s1 = (self.xtqx.s[:singular_value]).inv
        self.__G = v1 * s1 * v1.T * self.jco.T * self.obscov.inv
        self.__G_sv = singular_value
        self.__G.row_names = self.jco.col_names
        self.__G.col_names = self.jco.row_names
        self.__G.autoalign = True
        self.log("calc G @" + str(singular_value))
        return self.__G

    def first_forecast(self,singular_value):
        """wrapper around ErrVar.first_forecast
        """
        return self.first_prediction(singular_value)

    def first_prediction(self, singular_value):
        """get the null space term (first term) contribution to prediction error variance
            at a singular value.  used to construct error variance dataframe

        Parameters
        ----------
        singular_value : int
            singular value to calc first term at

        Returns
        -------
        dict : dict
            dictionary of ("first",prediction_names),error variance pairs at singular_value

        """
        if not self.predictions:
            raise Exception("ErrVar.first(): no predictions are set")
        if singular_value > self.jco.ncol:
            zero_preds = {}
            for pred in self.predictions_iter:
                zero_preds[("first", pred.col_names[0])] = 0.0
            return zero_preds
        self.log("calc first term parameter @" + str(singular_value))
        first_term = self.I_minus_R(singular_value).T * self.parcov *\
                     self.I_minus_R(singular_value)
        if self.predictions:
            results = {}
            for prediction in self.predictions_iter:
                results[("first",prediction.col_names[0])] = \
                    float((prediction.T * first_term * prediction).x)
            self.log("calc first term parameter @" + str(singular_value))
            return results

    def first_parameter(self, singular_value):
        """get the null space term (first term) contribution to parameter error variance
            at a singular value

        Parameters
        ----------
        singular_value : int
            singular value to calc first term at

        Returns
        -------
        first_term : pyemu.Cov
            first term contribution to parameter error variance

        """
        self.log("calc first term parameter @" + str(singular_value))
        first_term = self.I_minus_R(singular_value) * self.parcov * \
                     self.I_minus_R(singular_value)
        self.log("calc first term parameter @" + str(singular_value))
        return first_term

    def second_forecast(self,singular_value):
        """wrapper around ErrVar.second_prediction
        """
        return self.second_prediction(singular_value)

    def second_prediction(self, singular_value):
        """get the solution space contribution to predictive error variance
            at a singular value (y^t * G * obscov * G^T * y).  Used to construct
            error variance dataframe

        Parameters
        ----------
        singular_value : int
            singular value to calc second term at

        Returns
        -------
        dict : dict
            dictionary of ("second",prediction_names), error variance

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
                inf_pred[("second",pred.col_names[0])] = 1.0E+35
            return inf_pred
        else:
            second_term = self.G(singular_value) * self.obscov * \
                          self.G(singular_value).T
            results = {}
            for prediction in self.predictions_iter:
                results[("second",prediction.col_names[0])] = \
                    float((prediction.T * second_term * prediction).x)
            self.log("calc second term prediction @" + str(singular_value))
            return results

    def second_parameter(self, singular_value):
        """get the solution space contribution to parameter error variance
             at a singular value (G * obscov * G^T)

        Parameters
        ----------
        singular_value : int
            singular value to calc second term at

        Returns
        -------
        second_parameter : pyemu.Cov
            second term contribution to parameter error variance

        """
        self.log("calc second term parameter @" + str(singular_value))
        result = self.G(singular_value) * self.obscov * self.G(singular_value).T
        self.log("calc second term parameter @" + str(singular_value))
        return result

    def third_forecast(self,singular_value):
        """wrapper around ErrVar.third_prediction
        """
        return self.third_prediction(singular_value)

    def third_prediction(self,singular_value):
        """get the omitted parameter contribution to prediction error variance
         at a singular value. used to construct error variance dataframe

        Parameters
        ----------
        singular_value : int
            singular value to calc third term at

        Returns
        -------
        dict : dict
            dictionary of ("third",prediction_names),error variance
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
                inf_pred[("third",pred.col_names[0])] = 1.0E+35
            return inf_pred
        else:
            results = {}
            for prediction,omitted_prediction in \
                    zip(self.predictions_iter, self.omitted_predictions):
                # comes out as row vector, but needs to be a column vector
                p = ((prediction.T * self.G(singular_value) * self.omitted_jco)
                     - omitted_prediction.T).T
                result = float((p.T * self.omitted_parcov * p).x)
                results[("third", prediction.col_names[0])] = result
            self.log("calc third term prediction @" + str(singular_value))
            return results

    def third_parameter(self, singular_value):
        """get the omitted parameter contribution to parameter error variance
             at a singular value (G * omitted_jco * Sigma_(omitted_pars) * omitted_jco^T * G^T)

        Parameters
        ----------
        singular_value : int
            singular value to calc third term at

        Returns
        -------
        third_parameter : pyemu.Cov
            0.0 if need_omitted is False

        """
        if self.__need_omitted is False:
            return 0.0
        self.log("calc third term parameter @" + str(singular_value))
        GZo = self.G(singular_value) * self.omitted_jco
        result = GZo * self.omitted_parcov * GZo.T
        self.log("calc third term parameter @" + str(singular_value))
        return result

