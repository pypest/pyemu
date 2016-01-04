from __future__ import print_function, division
import numpy as np
import pandas as pd
from pyemu.la import LinearAnalysis
from pyemu.mat import Cov

class Schur(LinearAnalysis):
    """derived type for posterior covariance analysis using Schur's complement

    Note:
        Same call signature as the base LinearAnalysis class
    """
    def __init__(self,jco,**kwargs):
        self.__posterior_prediction = None
        self.__posterior_parameter = None
        super(Schur,self).__init__(jco,**kwargs)


    @property
    def pandas(self):
        """get a pandas dataframe of prior and posterior for all predictions
        """
        names,prior,posterior = [],[],[]
        for iname,name in enumerate(self.posterior_parameter.row_names):
            names.append(name)
            posterior.append(np.sqrt(float(
                self.posterior_parameter[iname, iname]. x)))
            iprior = self.parcov.row_names.index(name)
            prior.append(np.sqrt(float(self.parcov[iprior, iprior].x)))
        for pred_name, pred_var in self.posterior_prediction.items():
            names.append(pred_name)
            posterior.append(np.sqrt(pred_var))
            prior.append(self.prior_prediction[pred_name])
        return pd.DataFrame({"posterior": posterior, "prior": prior},
                                index=names)


    @property
    def posterior_parameter(self):
        """get the posterior parameter covariance matrix
        """
        if self.__posterior_parameter is not None:
            return self.__posterior_parameter
        else:
            self.clean()
            self.log("Schur's complement")
            r = (self.xtqx + self.parcov.inv).inv
            assert r.row_names == r.col_names
            self.__posterior_parameter = Cov(r.x,row_names=r.row_names,col_names=r.col_names)
            self.log("Schur's complement")
            return self.__posterior_parameter


    @property
    def posterior_forecast(self):
        """thin wrapper around posterior_prediction
        """
        return self.posterior_prediction

    @property
    def posterior_prediction(self):
        """get a dict of posterior prediction variances
        """
        if self.__posterior_prediction is not None:
            return self.__posterior_prediction
        else:
            if self.predictions is not None:
                self.log("propagating posterior to predictions")
                pred_dict = {}
                for prediction in self.predictions:
                    var = (prediction.T * self.posterior_parameter
                           * prediction).x[0, 0]
                    pred_dict[prediction.col_names[0]] = var
                self.__posterior_prediction = pred_dict
                self.log("propagating posterior to predictions")
            else:
                self.__posterior_prediction = {}
            return self.__posterior_prediction


    def get_parameter_summary(self):
        """get a summary of the parameter uncertainty
        Parameters:
        ----------
            None
        Returns:
        -------
            pd.DataFrame() of prior,posterior variances and percent
            uncertainty reduction of each parameter
        Raises:
            None
        """
        prior = self.parcov.get(self.posterior_parameter.col_names)
        if prior.isdiagonal:
            prior = prior.x.flatten()
        else:
            prior = np.diag(prior.x)
        post = np.diag(self.posterior_parameter.x)
        ureduce = 100.0 * (1.0 - (post / prior))
        return pd.DataFrame({"prior_var":prior,"post_var":post,
                                 "percent_reduction":ureduce},
                                index=self.posterior_parameter.col_names)


    def get_forecast_summary(self):
        """get a summary of the forecast uncertainty
        Parameters:
        ----------
            None
        Returns:
        -------
            pd.DataFrame() of prior,posterior variances and percent
            uncertainty reduction of each forecast
        """
        sum = {"prior_var":[], "post_var":[], "percent_reduction":[]}
        for forecast in self.prior_forecast.keys():
            pr = self.prior_forecast[forecast]
            pt = self.posterior_forecast[forecast]
            ur = 100.0 * (1.0 - (pt/pr))
            sum["prior_var"].append(pr)
            sum["post_var"].append(pt)
            sum["percent_reduction"].append(ur)
        return pd.DataFrame(sum,index=self.prior_forecast.keys())

    def __contribution_from_parameters(self, parameter_names):
        """get the prior and posterior uncertainty reduction as a result of
        some parameter becoming perfectly known
        Parameters:
        ----------
            parameter_names (list of str) : parameter that are perfectly known
        Returns:
        -------
            dict{prediction name : [prior uncertainty w/o parameter_names,
                % posterior uncertainty w/o parameter names]}
        """
        if not isinstance(parameter_names, list):
            parameter_names = [parameter_names]

        for iname, name in enumerate(parameter_names):
            parameter_names[iname] = name.lower()
            assert name.lower() in self.jco.col_names,\
                "contribution parameter " + name + " not found jco"
        keep_names = []
        for name in self.jco.col_names:
            if name not in keep_names:
                keep_names.append(name)
        if len(keep_names) == 0:
            raise Exception("Schur.contribution_from_parameters: " +
                            "atleast one parameter must remain uncertain")
        #get the reduced predictions
        if self.predictions is None:
            raise Exception("Schur.contribution_from_parameters: " +
                            "no predictions have been set")
        cond_preds = []
        for pred in self.predictions:
            cond_preds.append(pred.get(keep_names, pred.col_names))
        la_cond = Schur(jco=self.jco.get(self.jco.row_names, keep_names),
                        parcov=self.parcov.condition_on(parameter_names),
                        obscov=self.obscov, predictions=cond_preds,verbose=False)

        #get the prior and posterior for the base case
        bprior,bpost = self.prior_prediction, self.posterior_prediction
        #get the prior and posterior for the conditioned case
        cprior,cpost = la_cond.prior_prediction, la_cond.posterior_prediction
        return cprior,cpost


    def get_contribution_dataframe(self,parlist_dict=None):
        """get a dataframe the prior and posterior uncertainty
        reduction as a result of
        some parameter becoming perfectly known
        Parameters:
        ----------
            parlist_dict (dict of list of str) : groups of parameters
                that are to be treated as perfectly known.  key values become
                row labels in dataframe
        Returns:
        -------
            dataframe[parlist_dict.keys(),(forecast_name,<prior,post>)
                multiindex dataframe of Schur's complement results for each
                group of parameters in parlist_dict values.
        """
        self.log("calculating contribution from parameters")
        if parlist_dict is None:
            parlist_dict = dict(zip(self.pst.parameter_data.parnme,self.pst.parameter_data.parnme))

        results = {}
        names = ["base"]
        for forecast in self.prior_forecast.keys():
            pr = self.prior_forecast[forecast]
            pt = self.posterior_forecast[forecast]
            reduce = 100.0 * ((pr - pt) / pr)
            results[(forecast,"prior")] = [pr]
            results[(forecast,"post")] = [pt]
            results[(forecast,"percent_reduce")] = [reduce]
        for case_name,par_list in parlist_dict.items():
            names.append(case_name)
            self.log("calculating contribution from: " + str(par_list) + '\n')
            case_prior,case_post = self.__contribution_from_parameters(par_list)
            self.log("calculating contribution from: " + str(par_list) + '\n')
            for forecast in case_prior.keys():
                pr = case_prior[forecast]
                pt = case_post[forecast]
                reduce = 100.0 * ((pr - pt) / pr)
                results[(forecast, "prior")].append(pr)
                results[(forecast, "post")].append(pt)
                results[(forecast, "percent_reduce")].append(reduce)

        df = pd.DataFrame(results,index=names)
        self.log("calculating contribution from parameters")
        return df


    def get_contribution_dataframe_groups(self):
        """get the forecast uncertainty contribution from each parameter
        group.  Just some sugar for get_contribution_dataframe()
        """
        pargrp_dict = {}
        par = self.pst.parameter_data
        groups = par.groupby("pargp").groups
        for grp,idxs in groups.items():
            pargrp_dict[grp] = list(par.loc[idxs,"parnme"])
        return self.get_contribution_dataframe(pargrp_dict)


    def __importance_of_observations(self,observation_names):
        """get the importance of some observations for reducing the
        posterior uncertainty
        Parameters:
        ----------
            observation_names (list of str) : observations to analyze
        Returns:
        -------
            dict{prediction_name:% posterior reduction}
        """
        if not isinstance(observation_names, list):
            observation_names = [observation_names]
        for iname, name in enumerate(observation_names):
            observation_names[iname] = name.lower()
            if name.lower() not in self.jco.row_names:
                raise Exception("Schur.importance_of_observations: " +
                                "obs name not found in jco: " + name)
        keep_names = []
        for name in self.jco.row_names:
            if name not in observation_names:
                keep_names.append(name)
        if len(keep_names) == 0:
            raise Exception("Schur.importance_of_observations: " +
                            " atleast one observation must remain")
        if self.predictions is None:
            raise Exception("Schur.importance_of_observations: " +
                            "no predictions have been set")

        la_reduced = self.get(par_names=self.jco.col_names,
                              obs_names=keep_names)
        return la_reduced.posterior_prediction


    def get_importance_dataframe(self,obslist_dict=None):
        """get a dataframe the posterior uncertainty
        as a result of losing some observations
        Parameters:
        ----------
            obslist_dict (dict of list of str) : groups of observations
                that are to be treated as lost.  key values become
                row labels in dataframe. If None, then test every obs
        Returns:
        -------
            dataframe[obslist_dict.keys(),(forecast_name,post)
                multiindex dataframe of Schur's complement results for each
                group of observations in obslist_dict values.
        """
        self.log("calculating importance of observations")
        if obslist_dict is None:
            obs = self.pst.observation_data.loc[:,["obsnme","weight"]]
            obslist_dict = {}
            for o, w in zip(obs.obsnme,obs.weight):
                if w > 0:
                    obslist_dict[o] = [o]

        results = {}
        names = ["base"]
        for forecast,pt in self.posterior_forecast.items():
            results[forecast] = [pt]
        for case_name,obs_list in obslist_dict.items():
            names.append(case_name)
            self.log("calculating contribution from: " + str(obs_list) + '\n')
            case_post = self.__importance_of_observations(obs_list)
            self.log("calculating contribution from: " + str(obs_list) + '\n')
            for forecast,pt in case_post.items():
                results[forecast].append(pt)
        df = pd.DataFrame(results,index=names)
        self.log("calculating importance of observations")
        return df


    def get_importance_dataframe_groups(self):
        obsgrp_dict = {}
        obs = self.pst.observation_data
        obs.index = obs.obsnme
        obs = obs.loc[self.jco.row_names,:]
        groups = obs.groupby("obgnme").groups
        for grp, idxs in groups.items():
            obsgrp_dict[grp] = list(obs.loc[idxs,"obsnme"])
        return self.get_importance_dataframe(obsgrp_dict)

