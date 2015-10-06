from __future__ import print_function, division
import numpy as np
import pandas as pd
from pyemu.la import linear_analysis
from pyemu.mat.mat_handler import matrix,jco,cov

class errvar(linear_analysis):
    """child class for error variance analysis
        todo: add KL parameter scaling with parcov -> identity reset
    """
    def __init__(self,jco,**kwargs):
        """there are some additional keyword args that can be passed to active
            the 3-term error variance calculation
        Args:
            omitted_parameters (list of str): argument that identifies
                parameters that will be treated as omitted
            omitted_parcov (matrix or str): argument that identifies
                omitted parameter parcov
            omitted_predictions (matrix or str): argument that identifies
            omitted prediction vectors

        Note: if only omitted_parameters is passed, then the omitted_parameter
            argument must be a string or list of strings that identifies
            parameters that are in the linear_analysis attributes that will
             extracted
        """
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

        #--instantiate the parent class
        super(errvar, self).__init__(jco, **kwargs)
        if self.__need_omitted:
            self.log("pre-loading omitted components")
            #self._linear_analysis__load_jco()
            #self._linear_analysis__load_parcov()
            #self._linear_analysis__load_obscov()
            #if self.prediction_arg is not None:
            #    self._linear_analysis__load_predictions()
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
        #--if there are no base predictions
        if self.predictions is None:
            raise Exception("errvar.__load_omitted_predictions(): " +
                            "no 'included' predictions is None")
        if self.omitted_predictions_arg is None and \
                        self.omitted_par_arg is None:
            raise Exception("errvar.__load_omitted_predictions: " +
                            "both omitted args are None")
        # try to set omitted_predictions by
        # extracting from existing predictions
        if self.omitted_predictions_arg is None and \
                        self.omitted_par_arg is not None:
            #--check to see if omitted par names are in each predictions
            found = True
            missing_par,missing_pred = None, None
            for par_name in self.omitted_jco.col_names:
                for prediction in self.predictions:
                    if par_name not in prediction.row_names:
                        found = False
                        missing_par = par_name
                        missing_pred = prediction.col_names[0]
                        break
            if found:
                opreds = []
                # need to access the attribute directly,
                # not a view of attribute
                for prediction in self._linear_analysis__predictions:
                    opred = prediction.extract(self.omitted_jco.col_names)
                    opreds.append(opred)
                self.__omitted_predictions = opreds
            else:
                raise Exception("errvar.__load_omitted_predictions(): " +
                                " omitted parameter " + str(missing_par) +\
                                " not found in prediction vector " +
                                str(missing_pred))
        elif self.omitted_parcov_arg is not None:
            raise NotImplementedError()


    def __load_omitted_parcov(self):
        """private: set the omitted_parcov attribute
        """
        if self.omitted_parcov_arg is None and self.omitted_par_arg is None:
            raise Exception("errvar.__load_omitted_parcov: " +
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
                #--need to access attribute directly, not view of attribute
                self.__omitted_parcov = \
                    self._linear_analysis__parcov.extract(
                        row_names=self.omitted_jco.col_names)
            else:
                self.logger.warn("errvar.__load_omitted_parun: " +
                                 "no omitted parcov arg passed: " +
                        "setting omitted parcov as identity matrix")
                self.__omitted_parcov = cov(
                    x=np.ones(self.omitted_jco.shape[1]),
                    names=self.omitted_jco.col_names, isdiagonal=True)
        elif self.omitted_parcov_arg is not None:
            raise NotImplementedError()


    def __load_omitted_jco(self):
        """private: set the omitted jco attribute
        """
        if self.omitted_par_arg is None:
            raise Exception("errvar.__load_omitted: omitted_arg is None")
        if isinstance(self.omitted_par_arg,str):
            if self.omitted_par_arg in self.jco.col_names:
                #--need to access attribute directly, not view of attribute
                self.__omitted_jco = \
                    self._linear_analysis__jco.extract(
                        col_names=self.omitted_par_arg)
            else:
                # must be a filename
                self.__omitted_jco = self.__fromfile(self.omitted_par_arg)
        # if the arg is an already instantiated matrix (or jco) object
        elif isinstance(self.omitted_par_arg,jco) or \
                isinstance(self.omitted_par_arg,matrix):
            self.__omitted_jco = \
                jco(x=self.omitted_par_arg.newx(),
                          row_names=self.omitted_par_arg.row_names,
                          col_names=self.omitted_par_arg.col_names)
        # if it is a list, then it must be a list
        # of parameter names in self.jco
        elif isinstance(self.omitted_par_arg,list):
            for arg in self.omitted_par_arg:
                if isinstance(arg,str):
                    assert arg in self.jco.col_names,\
                        "errvar.__load_omitted_jco: omitted_jco " +\
                        "arg str not in jco par_names: " + str(arg)
            self.__omitted_jco = \
                self._linear_analysis__jco.extract(col_names=self.omitted_par_arg)


    # these property decorators help keep from loading potentially
    # unneeded items until they are called
    # returns a view only - cheap, but can be dangerous


    @property
    def omitted_predictions(self):
        if self.__omitted_predictions is None:
            self.log("loading omitted_predictions")
            self.__load_omitted_predictions()
            self.log("loading omitted_predictions")
        return self.__omitted_predictions


    @property
    def omitted_jco(self):
        if self.__omitted_jco is None:
            self.log("loading omitted_jco")
            self.__load_omitted_jco()
            self.log("loading omitted_jco")
        return self.__omitted_jco


    @property
    def omitted_parcov(self):
        if self.__omitted_parcov is None:
            self.log("loading omitted_parcov")
            self.__load_omitted_parcov()
            self.log("loading omitted_parcov")
        return self.__omitted_parcov


    def get_errvar_dataframe(self, singular_values):
        """get a pandas dataframe of error variance results indexed
            on singular value and (prediction name,<term>)
        Args:
            singular_values (list of int) : singular values to test
        Returns:
            multi-indexed pandas dataframe
        Raises:
            None
        """
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


    def get_identifiability_dataframe(self,singular_value):
        """get the parameter identifiability as a pandas dataframe
        Aergs:
            singular_value (int) : the truncation point
        Returns:
            A pandas dataframe of the V_1**2 matrix with the
             identifiability in the column labeled "ident"
         Raises:
            None
        """
        #v1_df = self.qhalfx.v[:, :singular_value].to_dataframe() ** 2
        v1_df = self.xtqx.v[:, :singular_value].to_dataframe() ** 2
        v1_df["ident"] = v1_df.sum(axis=1)
        return v1_df


    def variance_at(self, singular_value):
        """get the error variance of all three terms
        Args:
            singular_value (int) : singular value to test
        Returns:
            dict{[<term>,prediction_name]:standard_deviation}
        Raises:
            None
        """
        results = {}
        results.update(self.first_prediction(singular_value))
        results.update(self.second_prediction(singular_value))
        results.update(self.third_prediction(singular_value))
        return results


    def R(self, singular_value):
        """get resolution matrix at a singular value
             V_1 * V_1^T
        Args:
            singular_value (int) : singular value to calc R at
        Returns:
            R at singular_value
        Raises:
            None
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
             V_2 * V_2^T
         Args:
            singular_value (int) : singular value to calc I - R at
        Returns:
            I - R at singular_value
        Raises:
            None
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
        """get the parameter solution matrix at a singular value
            V_1 * S_1^(_1) * U_1^T
        Args:
            singular_value (int) : singular value to calc G at
        Returns:
            G at singular_value
        Raises:
            None
        """
        if self.__G is not None and singular_value == self.__G_sv:
            return self.__G

        if singular_value == 0:
            self.__G_sv = 0
            self.__G = matrix(
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
                "errvar.G(): singular_value > min(npar,nobs):" +
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
        return self.first_prediction(singular_value)

    def first_prediction(self, singular_value):
        """get the null space term contribution to prediction error variance
            at a singular value
        Args:
            singular_value (int) : singular value to calc first term at
        Returns:
            dict{["first",prediction_names]:error variance} at singular_value
        Raises:
            Exception if no predictions are set
        """
        if not self.predictions:
            raise Exception("errvar.first(): no predictions are set")
        if singular_value > self.jco.ncol:
            zero_preds = {}
            for pred in self.predictions:
                zero_preds[("first", pred.col_names[0])] = 0.0
            return zero_preds
        self.log("calc first term parameter @" + str(singular_value))
        first_term = self.I_minus_R(singular_value).T * self.parcov *\
                     self.I_minus_R(singular_value)
        if self.predictions:
            results = {}
            for prediction in self.predictions:
                results[("first",prediction.col_names[0])] = \
                    float((prediction.T * first_term * prediction).x)
            self.log("calc first term parameter @" + str(singular_value))
            return results


    def first_parameter(self, singular_value):
        """get the null space term contribution to parameter error variance
            at a singular value
        Args:
            singular_value (int) : singular value to calc first term at
        Returns:
            Cov object of first term error variance
        Raises:
            None
        """
        self.log("calc first term parameter @" + str(singular_value))
        first_term = self.I_minus_R(singular_value) * self.parcov * \
                     self.I_minus_R(singular_value)
        self.log("calc first term parameter @" + str(singular_value))
        return first_term


    def second_forecast(self,singular_value):
        return self.second_prediction(singular_value)


    def second_prediction(self, singular_value):
        """get the solution space contribution to predictive error variance
            at a singular value
            y^t * G * obscov * G^T * y
        Args:
            singular_value (int) : singular value to calc second term at
        Returns:
             dict{["second",prediction_names]:error variance} at singular_value
        Raises:
            Exception if no predictions are set
        """
        if not self.predictions:
            raise Exception("errvar.second(): not predictions are set")
        self.log("calc second term prediction @" + str(singular_value))

        mn = min(self.jco.shape)
        try:
            mn = min(self.pst.npar_adj, self.pst.nnz_obs)
        except:
            pass
        if singular_value > mn:
            inf_pred = {}
            for pred in self.predictions:
                inf_pred[("second",pred.col_names[0])] = 1.0E+35
            return inf_pred
        # elif singular_value == 0:
        #     zero_preds = {}
        #     for pred in self.predictions:
        #         zero_preds[("second", pred.col_names[0])] = 0.0
        #     return zero_preds
        else:
            second_term = self.G(singular_value) * self.obscov * \
                          self.G(singular_value).T
            results = {}
            for prediction in self.predictions:
                results[("second",prediction.col_names[0])] = \
                    float((prediction.T * second_term * prediction).x)
            self.log("calc second term prediction @" + str(singular_value))
            return results


    def second_parameter(self, singular_value):
        """get the solution space contribution to parameter error variance
             at a singular value
            G * obscov * G^T
        Args:
            singular_value (int) : singular value to calc second term at
        Returns:
            Cov object of second term error variance
        Raises:
            None
        """
        self.log("calc second term parameter @" + str(singular_value))
        result = self.G(singular_value) * self.obscov * self.G(singular_value).T
        self.log("calc second term parameter @" + str(singular_value))
        return result


    def third_forecast(self,singular_value):
        return self.third_prediction(singular_value)

    def third_prediction(self,singular_value):
        """get the omitted parameter contribution to error variance at a singular value
            predictions:
                p * Simga_(omitted_pars) * p^T
                p = prediction^T * G * omitted_jco - omitted_prediction^T
        Args:
            singular_value (int) : singular value to calc third term at
        Returns:
            dict{["third",prediction_names]:error variance} at singular_value
        Raises:
            Exception if no predictions are set
        """
        if not self.predictions:
            raise Exception("errvar.third(): not predictions are set")
        if self.__need_omitted is False:
            zero_preds = {}
            for pred in self.predictions:
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
            for pred in self.predictions:
                inf_pred[("third",pred.col_names[0])] = 1.0E+35
            return inf_pred
        else:
            results = {}
            for prediction,omitted_prediction in \
                    zip(self.predictions, self.omitted_predictions):
                #--comes out as row vector, but needs to be a column vector
                p = ((prediction.T * self.G(singular_value) * self.omitted_jco)
                     - omitted_prediction.T).T
                result = float((p.T * self.omitted_parcov * p).x)
                results[("third", prediction.col_names[0])] = result
            self.log("calc third term prediction @" + str(singular_value))
            return results


    def third_parameter(self, singular_value):
        """get the omitted parameter contribution to parameter error variance
             at a singular value
                G * omitted_jco * Sigma_(omitted_pars) * omitted_jco^T * G^T
        Args:
            singular_value (int) : singular value to calc third term at
        Returns:
            0.0 if need_omitted is False
            Cov object of third term error variance
        Raises:
            None
        """
        if self.__need_omitted is False:
            return 0.0
        self.log("calc third term parameter @" + str(singular_value))
        GZo = self.G(singular_value) * self.omitted_jco
        result = GZo * self.omitted_parcov * GZo.T
        self.log("calc third term parameter @" + str(singular_value))
        return result

    @staticmethod
    def test():
        #non-pest
        pnames = ["p1","p2","p3"]
        onames = ["o1","o2","o3","o4"]
        npar = len(pnames)
        nobs = len(onames)
        j_arr = np.random.random((nobs,npar))
        jco = matrix(x=j_arr,row_names=onames,col_names=pnames)
        parcov = cov(x=np.eye(npar),names=pnames)
        obscov = cov(x=np.eye(nobs),names=onames)
        forecasts = "o2"

        omitted = "p3"

        e = errvar(jco=jco,parcov=parcov,obscov=obscov,forecasts=forecasts,
                   omitted_parameters=omitted)
        svs = [0,1,2,3,4,5]
        print(e.get_errvar_dataframe(svs))