"""module for FOSM-based uncertainty analysis using a
linearized form of Bayes equation known as the Schur compliment
"""

from __future__ import print_function, division
from os import name
import numpy as np
import pandas as pd
from pyemu.la import LinearAnalysis
from pyemu.mat import Cov, Matrix


class Schur(LinearAnalysis):
    """FOSM-based uncertainty and data-worth analysis

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

    Note:
        This class is the primary entry point for FOSM-based uncertainty and
        dataworth analyses

        This class replicates and extends the behavior of the PEST PREDUNC utilities.

    Example::

        #assumes "my.pst" exists
        sc = pyemu.Schur(jco="my.jco",forecasts=["fore1","fore2"])
        print(sc.get_forecast_summary())
        print(sc.get_parameter_contribution())

    """

    def __init__(self, jco, **kwargs):
        self.__posterior_prediction = None
        self.__posterior_parameter = None
        super(Schur, self).__init__(jco, **kwargs)

    # @property
    # def pandas(self):
    #     """get a pandas dataframe of prior and posterior for all predictions
    #
    #     Returns:
    #         pandas.DataFrame : pandas.DataFrame
    #             a dataframe with prior and posterior uncertainty estimates
    #             for all forecasts (predictions)
    #     """
    #     names,prior,posterior = [],[],[]
    #     for iname,name in enumerate(self.posterior_parameter.row_names):
    #         names.append(name)
    #         posterior.append(np.sqrt(float(
    #             self.posterior_parameter[iname, iname]. x)))
    #         iprior = self.parcov.row_names.index(name)
    #         prior.append(np.sqrt(float(self.parcov[iprior, iprior].x)))
    #     for pred_name, pred_var in self.posterior_prediction.items():
    #         names.append(pred_name)
    #         posterior.append(np.sqrt(pred_var))
    #         prior.append(self.prior_prediction[pred_name])
    #     return pd.DataFrame({"posterior": posterior, "prior": prior},
    #                             index=names)

    @property
    def posterior_parameter(self):
        """posterior parameter covariance matrix.

        Returns:
            `pyemu.Cov`: the posterior parameter covariance matrix

        Example::

            sc = pyemu.Schur(jco="my.jcb")
            post_cov = sc.posterior_parameter
            post_cov.to_ascii("post.cov")

        """
        if self.__posterior_parameter is not None:
            return self.__posterior_parameter
        else:
            self.clean()
            self.log("Schur's complement")
            try:
                pinv = self.parcov.inv
                r = self.xtqx + pinv
                r = r.inv
            except Exception as e:

                pinv.to_ascii("parcov_inv.err.cov")
                self.logger.warn("error forming schur's complement: {0}".format(str(e)))
                self.xtqx.to_binary("xtqx.err.jcb")
                self.logger.warn("problemtic xtqx saved to xtqx.err.jcb")
                self.logger.warn(
                    "problematic inverse parcov saved to parcov_inv.err.cov"
                )
                raise Exception("error forming schur's complement: {0}".format(str(e)))
            assert r.row_names == r.col_names
            self.__posterior_parameter = Cov(
                r.x, row_names=r.row_names, col_names=r.col_names
            )
            self.log("Schur's complement")
            return self.__posterior_parameter

    # @property
    # def map_parameter_estimate(self):
    #     """ get the posterior expectation for parameters using Bayes linear
    #     estimation
    #
    #     Returns:
    #         `pandas.DataFrame`: a dataframe with prior and posterior parameter expectations
    #
    #     """
    #     res = self.pst.res
    #     assert res is not None
    #     # build the prior expectation parameter vector
    #     prior_expt = self.pst.parameter_data.loc[:,["parval1"]].copy()
    #     islog = self.pst.parameter_data.partrans == "log"
    #     prior_expt.loc[islog] = prior_expt.loc[islog].apply(np.log10)
    #     prior_expt = Matrix.from_dataframe(prior_expt)
    #     prior_expt.col_names = ["prior_expt"]
    #     # build the residual vector
    #     res_vec = Matrix.from_dataframe(res.loc[:,["residual"]])
    #
    #     # form the terms of Schur's complement
    #     b = self.parcov * self.jco.T
    #     c = ((self.jco * self.parcov * self.jco.T) + self.obscov).inv
    #     bc = Matrix((b * c).x, row_names=b.row_names, col_names=c.col_names)
    #
    #     # calc posterior expectation
    #     upgrade = bc * res_vec
    #     upgrade.col_names = ["prior_expt"]
    #     post_expt = prior_expt + upgrade
    #
    #     # post processing - back log transform
    #     post_expt = pd.DataFrame(data=post_expt.x,index=post_expt.row_names,
    #                              columns=["post_expt"])
    #     post_expt.loc[:,"prior_expt"] = prior_expt.x.flatten()
    #     post_expt.loc[islog,:] = 10.0**post_expt.loc[islog,:]
    #     # unc_sum = self.get_parameter_summary()
    #     # post_expt.loc[:,"standard_deviation"] = unc_sum.post_var.apply(np.sqrt)
    #     post_expt.sort_index(inplace=True)
    #     return post_expt
    #
    # @property
    # def map_forecast_estimate(self):
    #     """ get the prior and posterior forecast (prediction) expectations.
    #
    #     Returns
    #     -------
    #     pandas.DataFrame : pandas.DataFrame
    #         dataframe with prior and posterior forecast expected values
    #
    #     """
    #     assert self.forecasts is not None
    #     islog = self.pst.parameter_data.partrans == "log"
    #     par_map = self.map_parameter_estimate
    #     par_map.loc[islog,:] = np.log10(par_map.loc[islog,:])
    #     par_map = Matrix.from_dataframe(par_map.loc[:,["post_expt"]])
    #     posts,priors = [],[]
    #     post_expt = (self.predictions.T * par_map).to_dataframe()
    #     for fname in self.forecast_names:
    #         #fname = forecast.col_names[0]
    #         pr = self.pst.res.loc[fname,"modelled"]
    #         priors.append(pr)
    #         posts.append(pr + post_expt.loc[fname,"post_expt"])
    #     return pd.DataFrame(data=np.array([priors,posts]).transpose(),
    #                         columns=["prior_expt","post_expt"],
    #                         index=self.forecast_names)

    @property
    def posterior_forecast(self):
        """posterior forecast (e.g. prediction) variance(s)

        Returns:
            `dict`: dictionary of forecast names and FOSM-estimated posterior
            variances

        Note:
            Sames as `LinearAnalysis.posterior_prediction`

            See `Schur.get_forecast_summary()` for a dataframe-based container of prior and posterior
            variances

        """
        return self.posterior_prediction

    @property
    def posterior_prediction(self):
        """posterior prediction (e.g. forecast) variance estimate(s)

        Returns:
             `dict`: dictionary of forecast names and FOSM-estimated posterior
             variances

         Note:
             sames as `LinearAnalysis.posterior_forecast`

             See `Schur.get_forecast_summary()` for a dataframe-based container of prior and posterior
             variances

        """
        if self.__posterior_prediction is not None:
            return self.__posterior_prediction
        else:
            if self.predictions is not None:
                try:
                    if self.pst.nnz_obs == 0:
                        self.log("no non-zero obs, posterior equals prior")
                        return self.prior_prediction
                    self.log("propagating posterior to predictions")
                except:
                    pass
                post_cov = (
                    self.predictions.T * self.posterior_parameter * self.predictions
                )
                self.__posterior_prediction = {
                    n: v for n, v in zip(post_cov.row_names, np.diag(post_cov.x))
                }
                self.log("propagating posterior to predictions")
            else:
                self.__posterior_prediction = {}
            return self.__posterior_prediction

    def get_parameter_summary(self):
        """summary of the FOSM-based parameter uncertainty (variance) estimate(s)

        Returns:
            `pandas.DataFrame`: dataframe of prior,posterior variances and percent
            uncertainty reduction of each parameter

        Note:
            This is the primary entry point for accessing parameter uncertainty estimates

            The "Prior" column in dataframe is the diagonal of `LinearAnalysis.parcov`
            "precent_reduction" column in dataframe is calculated as 100.0 * (1.0 -
            (posterior variance / prior variance)

        Example::

            sc = pyemu.Schur(jco="my.jcb",forecasts=["fore1","fore2"])
            df = sc.get_parameter_summary()
            df.loc[:,["prior","posterior"]].plot(kind="bar")
            plt.show()
            df.percent_reduction.plot(kind="bar")
            plt.show()

        """
        prior_mat = self.parcov.get(self.posterior_parameter.col_names)
        if prior_mat.isdiagonal:
            prior = prior_mat.x.flatten()
        else:
            prior = np.diag(prior_mat.x)
        post = np.diag(self.posterior_parameter.x)

        ureduce = 100.0 * (1.0 - (post / prior))

        return pd.DataFrame(
            {"prior_var": prior, "post_var": post, "percent_reduction": ureduce},
            index=self.posterior_parameter.col_names,
        )

    def get_forecast_summary(self):
        """summary of the FOSM-based forecast uncertainty (variance) estimate(s)

        Returns:
            `pandas.DataFrame`: dataframe of prior,posterior variances and percent
            uncertainty reduction of each forecast (e.g. prediction)

        Note:
            This is the primary entry point for accessing forecast uncertainty estimates
            "precent_reduction" column in dataframe is calculated as
            100.0 * (1.0 - (posterior variance / prior variance)

        Example::

            sc = pyemu.Schur(jco="my.jcb",forecasts=["fore1","fore2"])
            df = sc.get_parameter_summary()
            df.loc[:,["prior","posterior"]].plot(kind="bar")
            plt.show()
            df.percent_reduction.plot(kind="bar")
            plt.show()

        """
        sum = {"prior_var": [], "post_var": [], "percent_reduction": []}
        for forecast in self.prior_forecast.keys():
            pr = self.prior_forecast[forecast]
            pt = self.posterior_forecast[forecast]
            ur = 100.0 * (1.0 - (pt / pr))
            sum["prior_var"].append(pr)
            sum["post_var"].append(pt)
            sum["percent_reduction"].append(ur)
        return pd.DataFrame(sum, index=self.prior_forecast.keys())

    def __contribution_from_parameters(self, parameter_names):
        """private method get the prior and posterior uncertainty reduction as a result of
        some parameter becoming perfectly known

        """

        # get the prior and posterior for the base case
        bprior, bpost = self.prior_prediction, self.posterior_prediction
        # get the prior and posterior for the conditioned case
        la_cond = self.get_conditional_instance(parameter_names)
        cprior, cpost = la_cond.prior_prediction, la_cond.posterior_prediction
        return cprior, cpost

    def get_conditional_instance(self, parameter_names):
        """get a new `pyemu.Schur` instance that includes conditional update from
        some parameters becoming known perfectly

        Args:
            parameter_names ([`str`]): list of parameters that are to be treated as
                notionally perfectly known

        Returns:
            `pyemu.Schur`: a new Schur instance conditional on perfect knowledge
            of some parameters. The new instance has an updated `parcov` that is less
            the names listed in `parameter_names`.

        Note:
            This method is primarily for use by the `LinearAnalysis.get_parameter_contribution()`
            dataworth method.

        """
        if not isinstance(parameter_names, list):
            parameter_names = [parameter_names]

        for iname, name in enumerate(parameter_names):
            name = str(name).lower()
            parameter_names[iname] = name
            assert name in self.jco.col_names, (
                "contribution parameter " + name + " not found jco"
            )
        keep_names = []
        for name in self.jco.col_names:
            if name not in parameter_names:
                keep_names.append(name)
        if len(keep_names) == 0:
            raise Exception(
                "Schur.contribution_from_Parameters "
                + "atleast one parameter must remain uncertain"
            )
        # get the reduced predictions
        if self.predictions is None:
            raise Exception(
                "Schur.contribution_from_Parameters " + "no predictions have been set"
            )
        # cond_preds = []
        # for pred in self.predictions:
        #     cond_preds.append(pred.get(keep_names, pred.col_names))
        cond_preds = self.predictions.get(row_names=keep_names)
        try:
            pst = self.pst
        except:
            pst = None
        la_cond = Schur(
            jco=self.jco.get(self.jco.row_names, keep_names),
            pst=pst,
            parcov=self.parcov.condition_on(parameter_names),
            obscov=self.obscov,
            predictions=cond_preds,
            verbose=False,
        )
        return la_cond

    def get_par_contribution(self, parlist_dict=None, include_prior_results=False):
        """A dataworth method to get a dataframe the prior and posterior uncertainty
        reduction as a result of some parameter becoming perfectly known

        Args:
            parlist_dict : (`dict`, optional): a nested dictionary-list of groups of parameters
                that are to be treated as perfectly known.  key values become
                row labels in returned dataframe.  If `None`, each adjustable parameter
                is sequentially treated as known and the returned dataframe
                has row labels for each adjustable parameter
            include_prior_results (`bool`, optional):  flag to return a multi-indexed dataframe with both conditional
                prior and posterior forecast uncertainty estimates.  This is because
                the notional learning about parameters potentially effects both the prior
                and posterior forecast uncertainty estimates. If `False`, only posterior
                results are returned.  Default is `False`

        Returns:
            `pandas.DataFrame`: a dataframe that summarizes the parameter contribution
            dataworth analysis. The dataframe has index (row labels) of the keys in parlist_dict
            and a column labels of forecast names.  The values in the dataframe
            are the posterior variance of the forecast conditional on perfect
            knowledge of the parameters in the values of parlist_dict.  One row in the
            dataframe will be labeled `base` - this is the forecast uncertainty estimates
            that include the effects of all adjustable parameters.  Percent decreases in
            forecast uncertainty can be calculated by differencing all rows against the
            "base" row.  Varies depending on `include_prior_results`.

        Note:
            This is the primary dataworth method for assessing the contribution of one or more
            parameters to forecast uncertainty.

        Example::

            sc = pyemu.Schur(jco="my.jco")
            parlist_dict = {"hk":["hk1","hk2"],"rech"["rech1","rech2"]}
            df = sc.get_par_contribution(parlist_dict=parlist_dict)


        """
        self.log("calculating contribution from parameters")
        if parlist_dict is None:
            parlist_dict = (
                {}
            )  # dict(zip(self.pst.adj_par_names,self.pst.adj_par_names))
            # make sure all of the adjustable pars are in the jco
            for pname in self.pst.adj_par_names:
                if pname in self.jco.col_names:
                    parlist_dict[pname] = pname
        else:
            if type(parlist_dict) == list:
                parlist_dict = dict(zip(parlist_dict, parlist_dict))

        results = {}
        names = ["base"]
        for forecast in self.prior_forecast.keys():
            pr = self.prior_forecast[forecast]
            pt = self.posterior_forecast[forecast]
            # reduce = 100.0 * ((pr - pt) / pr)
            results[(forecast, "prior")] = [pr]
            results[(forecast, "post")] = [pt]
            # results[(forecast,"percent_reduce")] = [reduce]
        for case_name, par_list in parlist_dict.items():
            if len(par_list) == 0:
                continue
            names.append(case_name)
            self.log("calculating contribution from: " + str(par_list))
            case_prior, case_post = self.__contribution_from_parameters(par_list)
            self.log("calculating contribution from: " + str(par_list))
            for forecast in case_prior.keys():
                pr = case_prior[forecast]
                pt = case_post[forecast]
                # reduce = 100.0 * ((pr - pt) / pr)
                results[(forecast, "prior")].append(pr)
                results[(forecast, "post")].append(pt)
                # results[(forecast, "percent_reduce")].append(reduce)

        df = pd.DataFrame(results, index=names)
        # base = df.loc["base",df.columns.get_level_values(1)=="post"]
        # df = 1.0 - (df.loc[:,df.columns.get_level_values(1)=="post"] / base)

        self.log("calculating contribution from parameters")
        if include_prior_results:
            return df
        else:
            df = df.xs("post", level=1, drop_level=True, axis=1)
            return df

    def get_par_group_contribution(self, include_prior_results=False):
        """A dataworth method to get the forecast uncertainty contribution from each parameter
        group

        Args:
            include_prior_results (`bool`, optional):  flag to return a multi-indexed dataframe with both conditional
                prior and posterior forecast uncertainty estimates.  This is because
                the notional learning about parameters potentially effects both the prior
                and posterior forecast uncertainty estimates. If `False`, only posterior
                results are returned.  Default is `False`


        Returns:

            `pandas.DataFrame`: a dataframe that summarizes the parameter contribution analysis.
            The dataframe has index (row labels) that are the parameter group names
            and a column labels of forecast names.  The values in the dataframe
            are the posterior variance of the forecast conditional on perfect
            knowledge of the adjustable parameters in each parameter group.  One
            row is labelled "base" - this is the variance of the forecasts that includes
            the effects of all adjustable parameters. Varies depending on `include_prior_results`.

        Note:
            This method is just a thin wrapper around get_contribution_dataframe() - this method
            automatically constructs the parlist_dict argument where the keys are the
            group names and the values are the adjustable parameters in the groups

        Example::

            sc = pyemu.Schur(jco="my.jco")
            df = sc.get_par_group_contribution()



        """
        pargrp_dict = {}
        par = self.pst.parameter_data
        groups = par.groupby("pargp").groups
        for grp, idxs in groups.items():
            # pargrp_dict[grp] = list(par.loc[idxs,"parnme"])
            pargrp_dict[grp] = [
                pname
                for pname in list(par.loc[idxs, "parnme"])
                if pname in self.jco.col_names and pname in self.parcov.row_names
            ]
        return self.get_par_contribution(
            pargrp_dict, include_prior_results=include_prior_results
        )

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

        reset = False
        if reset_zero_weight > 0:

            if not self.obscov.isdiagonal:
                raise NotImplementedError(
                    "cannot reset weights for non-" + "diagonal obscov"
                )
            reset = True
            weight = reset_zero_weight
            org_obscov = self.obscov.copy()
            org_pst = None
            try:
                org_pst = self.pst.get()
            except:
                raise Exception(
                    "'reset_zero_weight' > 0 only supported when pst is available"
                )
            self.logger.statement(
                "resetting zero weights in obslist_dict to {0}".format(weight)
            )
        else:
            self.logger.statement("not resetting zero weights in obslist_dict")

        # obs = self.pst.observation_data
        # obs.index = obs.obsnme

        # if we don't care about grouping obs, then just reset all weights at once
        if base_obslist is None and obslist_dict is None and reset:
            onames = [
                name
                for name in self.pst.zero_weight_obs_names
                if name in self.jco.row_names and name in self.obscov.row_names
            ]

            self.pst.observation_data.loc[onames, "weight"] = weight

        if base_obslist is None:
            base_obslist = []
        else:
            if type(base_obslist) != list:
                self.logger.lraise(
                    "Schur.get_added_obs)_importance: base_obslist must be"
                    + " type 'list', not {0}".format(str(type(base_obslist)))
                )

        # ensure that no base observations have zero weight
        if base_obslist is not None:
            try:
                self.pst
            except:
                pass
            else:
                zero_basenames = [
                    name
                    for name in self.pst.zero_weight_obs_names
                    if name in base_obslist
                ]
                if len(zero_basenames) > 0:
                    raise Exception(
                        "Observations in baseobs_list must have "
                        + "nonzero weight. The following observations "
                        + "violate that condition: "
                        + ",".join(zero_basenames)
                    )

        # if needed reset the zero-weight obs in obslist_dict
        if obslist_dict is not None and reset:
            z_obs = []
            for case, obslist in obslist_dict.items():
                if not isinstance(obslist, list):
                    obslist_dict[case] = [obslist]
                    obslist = [obslist]
                inboth = set(base_obslist).intersection(set(obslist))
                if len(inboth) > 0:
                    raise Exception(
                        "observation(s) listed in both "
                        + "base_obslist and obslist_dict: "
                        + ",".join(inboth)
                    )
                z_obs.extend(obslist)
            self.log("resetting zero weight obs in obslist_dict")
            self.pst._adjust_weights_by_list(z_obs, weight)
            self.log("resetting zero weight obs in obslist_dict")

        # for a comprehensive obslist_dict
        if obslist_dict is None and reset:
            obs = self.pst.observation_data
            obs.index = obs.obsnme
            onames = [
                name
                for name in self.pst.zero_weight_obs_names
                if name in self.jco.row_names and name in self.obscov.row_names
            ]
            obs.loc[onames, "weight"] = weight

        if obslist_dict is None:
            obslist_dict = {
                name: name
                for name in self.pst.nnz_obs_names
                if name in self.jco.row_names and name in self.obscov.row_names
            }

        # reset the obs cov from the newly adjusted weights
        if reset:
            self.log("resetting self.obscov")
            self.reset_obscov(self.pst)
            self.log("resetting self.obscov")

        results = {}
        names = ["base"]

        if base_obslist is None or len(base_obslist) == 0:
            self.logger.statement(
                "no base observation passed, 'base' case"
                + " is just the prior of the forecasts"
            )
            for forecast, pr in self.prior_forecast.items():
                results[forecast] = [pr]
            # reset base obslist for use later
            base_obslist = []

        else:
            base_posterior = self.get(
                par_names=self.jco.par_names, obs_names=base_obslist
            ).posterior_forecast
            for forecast, pt in base_posterior.items():
                results[forecast] = [pt]

        for case_name, obslist in obslist_dict.items():
            names.append(case_name)
            if not isinstance(obslist, list):
                obslist = [obslist]
            self.log(
                "calculating importance of observations by adding: "
                + str(obslist)
                + "\n"
            )
            # this case is the combination of the base obs plus whatever unique
            # obs names in obslist
            case_obslist = list(base_obslist)
            dedup_obslist = [oname for oname in obslist if oname not in case_obslist]
            case_obslist.extend(dedup_obslist)
            # print(self.pst.observation_data.loc[case_obslist,:])
            case_post = self.get(
                par_names=self.jco.col_names, obs_names=case_obslist
            ).posterior_forecast
            for forecast, pt in case_post.items():
                results[forecast].append(pt)
            self.log(
                "calculating importance of observations by adding: "
                + str(obslist)
                + "\n"
            )
        df = pd.DataFrame(results, index=names)

        if reset:
            self.reset_obscov(org_obscov)
            if org_pst is not None:
                self.reset_pst(org_pst)

        return df

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

        if reset_zero_weight is not None:
            self.log(
                "Deprecation Warning: reset_zero_weight supplied to get_removed_obs_importance. "
                + "This value is ignored"
            )
            print(
                "Deprecation Warning: reset_zero_weight supplied to get_removed_obs_importance. "
                + "This value is ignored"
            )
        # obs = self.pst.observation_data
        # obs.index = obs.obsnme

        self.log("calculating importance of observations")
        org_obscov = self.obscov.copy()
        org_pst = None
        try:
            org_pst = self.pst.get()
        except:
            pass

        cases = list(obslist_dict.keys())
        cases.sort()
        # for case, obslist in obslist_dict.items():
        for case in cases:
            obslist = obslist_dict[case]

            if not isinstance(obslist, list):
                obslist = [obslist]
            obslist_dict[case] = obslist

        results = {}
        names = ["base"]
        for forecast, pt in self.posterior_forecast.items():
            results[forecast] = [pt]
        base_obslist.sort()
        # for case_name, obslist in obslist_dict.items():
        for case_name in cases:
            obslist = obslist_dict[case_name]
            if not isinstance(obslist, list):
                obslist = [obslist]
            names.append(case_name)
            self.log(
                "calculating importance of observations by removing: "
                + str(obslist)
                + "\n"
            )
            # check for missing names
            missing_onames = [
                oname for oname in obslist if oname not in self.jco.row_names
            ]
            if len(missing_onames) > 0:
                raise Exception(
                    "case {0} has observation names ".format(case_name)
                    + "not found: "
                    + ",".join(missing_onames)
                )
            # find the set difference between obslist and jco obs names
            # diff_onames = [oname for oname in self.jco.obs_names if oname not in obslist]
            diff_onames = [
                oname
                for oname in base_obslist
                if oname not in obslist and oname not in self.forecast_names
            ]

            # calculate the increase in forecast variance by not using the obs
            # in obslist
            case_post = self.get(
                par_names=self.jco.col_names, obs_names=diff_onames
            ).posterior_forecast

            for forecast, pt in case_post.items():
                results[forecast].append(pt)
        df = pd.DataFrame(results, index=names)
        self.log(
            "calculating importance of observations by removing: " + str(obslist) + "\n"
        )

        # if reset:
        self.reset_obscov(org_obscov)
        if org_pst is not None:
            self.reset_pst(org_pst)
        return df

    def get_obs_group_dict(self):
        """get a dictionary of observations grouped by observation group name

        Returns:
            `dict`: a dictionary of observations grouped by observation group name.
            Useful for dataworth processing in `pyemu.Schur`

        Note:
            only includes observations that are listed in `Schur.jco.row_names`

        Example::

            sc = pyemu.Schur("my.jco")
            obsgrp_dict = sc.get_obs_group_dict()
            df = sc.get_removed_obs_importance(obsgrp_dict=obsgrp_dict)

        """
        obsgrp_dict = {}
        obs = self.pst.observation_data
        obs.index = obs.obsnme
        obs = obs.loc[self.jco.row_names, :]
        groups = obs.groupby("obgnme").groups
        for grp, idxs in groups.items():
            obsgrp_dict[grp] = list(obs.loc[idxs, "obsnme"])
        return obsgrp_dict

    def get_removed_obs_group_importance(self, reset_zero_weight=None):
        """A dataworth method to analyze the posterior uncertainty as a result of losing
         existing observations, tested by observation groups

        Args:
            reset_zero_weight DEPRECATED


        Returns:
            `pandas.DataFrame`: A dataframe with index of observation group names and columns
            of forecast names.  The values in the dataframe are the posterior
            variances of the forecasts resulting from losing the information
            contained in each observation group. One row in the dataframe is labeled "base" - this is
            posterior forecast variance resulting from the notional calibration with the
            non-zero-weighed observations in `Schur.pst`.  Conceptually, the forecast variance should
            either not change or increase as a result of losing existing observations.  The magnitude
            of the increase represents the worth of the existing observation(s) in each group being tested.

        Example::

            sc = pyemu.Schur("my.jco")
            df = sc.get_removed_obs_group_importance()

        """
        if reset_zero_weight is not None:
            self.log(
                "Deprecation Warning: reset_zero_weight supplied to get_removed_obs_importance. "
                + "This value is ignored"
            )
            print(
                "Deprecation Warning: reset_zero_weight supplied to get_removed_obs_importance. "
                + "This value is ignored"
            )
        return self.get_removed_obs_importance(self.get_obs_group_dict())

    def get_added_obs_group_importance(self, reset_zero_weight=1.0):
        """A dataworth method to analyze the posterior uncertainty as a result of gaining
         existing observations, tested by observation groups

        Args:
            reset_zero_weight (`float`, optional)
                a flag to reset observations with zero weight in `obslist_dict`
                If `reset_zero_weights` passed as 0.0, no weights adjustments are made.
                Default is 1.0.

        Returns:
            `pandas.DataFrame`: A dataframe with index of observation group names and columns
            of forecast names.  The values in the dataframe are the posterior
            variances of the forecasts resulting from gaining the information
            contained in each observation group. One row in the dataframe is labeled "base" - this is
            posterior forecast variance resulting from the notional calibration with the
            non-zero-weighed observations in `Schur.pst`.  Conceptually, the forecast variance should
            either not change or decrease as a result of gaining new observations.  The magnitude
            of the decrease represents the worth of the potential new observation(s) in each group
            being tested.

        Note:
            Observations in `Schur.pst` with zero weights are not included in the analysis unless
            `reset_zero_weight` is a float greater than zero.  In most cases, users
            will want to reset zero-weighted observations as part dataworth testing process.

        Example::

            sc = pyemu.Schur("my.jco")
            df = sc.get_added_obs_group_importance(reset_zero_weight=1.0)

        """
        return self.get_added_obs_importance(
            self.get_obs_group_dict(), reset_zero_weight=reset_zero_weight
        )

    def next_most_important_added_obs(
        self,
        forecast=None,
        niter=3,
        obslist_dict=None,
        base_obslist=None,
        reset_zero_weight=1.0,
    ):
        """find the most important observation(s) by sequentially evaluating
        the importance of the observations in `obslist_dict`.

        Args:
            forecast (`str`, optional): name of the forecast to use in the ranking process.  If
                more than one forecast has been listed, this argument is required.  This is because
                the data worth must be ranked with respect to the variance reduction for a single
                forecast
            niter (`int`, optional):  number of sequential dataworth testing iterations.  Default is 3
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
            `pandas.DataFrame`: a dataFrame with columns of `obslist_dict` key for each iteration
            the yields the largest variance reduction for the named `forecast`. Columns are forecast
            variance percent reduction for each iteration (percent reduction compared to initial "base"
            case with all non-zero weighted observations included in the notional calibration)


        Note:
            The most important observations from each iteration is added to `base_obslist`
            and removed `obslist_dict` for the next iteration.  In this way, the added
            observation importance values include the conditional information from
            the last iteration.


        Example::

            sc = pyemu.Schur(jco="my.jco")
            df = sc.next_most_important_added_obs(forecast="fore1",base_obslist=sc.pst.nnz_obs_names)

        """

        if forecast is None:
            assert self.forecasts.shape[1] == 1, (
                "forecast arg list one and only one" + " forecast"
            )
            forecast = self.forecasts[0].col_names[0]
        # elif forecast not in self.prediction_arg:
        #    raise Exception("forecast {0} not found".format(forecast))

        else:
            forecast = forecast.lower()
            found = False
            for fore in self.forecasts.col_names:
                if fore == forecast:
                    found = True
                    break
            if not found:
                raise Exception("forecast {0} not found".format(forecast))

        if base_obslist:
            obs_being_used = list(base_obslist)
        else:
            obs_being_used = []

        best_case, best_results = [], []
        for iiter in range(niter):
            self.log("next most important added obs iteration {0}".format(iiter + 1))
            df = self.get_added_obs_importance(
                obslist_dict=obslist_dict,
                base_obslist=obs_being_used,
                reset_zero_weight=reset_zero_weight,
            )

            if iiter == 0:
                init_base = df.loc["base", forecast].copy()
            fore_df = df.loc[:, forecast]
            fore_diff_df = fore_df - fore_df.loc["base"]
            fore_diff_df.sort_values(inplace=True)
            iter_best_name = fore_diff_df.index[0]
            iter_best_result = df.loc[iter_best_name, forecast]
            iter_base_result = df.loc["base", forecast]
            diff_percent_init = 100.0 * (init_base - iter_best_result) / init_base
            diff_percent_iter = (
                100.0 * (iter_base_result - iter_best_result) / iter_base_result
            )
            self.log("next most important added obs iteration {0}".format(iiter + 1))

            best_results.append(
                [iter_best_name, iter_best_result, diff_percent_iter, diff_percent_init]
            )
            best_case.append(iter_best_name)

            if iter_best_name.lower() == "base":
                break

            if obslist_dict is None:
                onames = [iter_best_name]
            else:
                onames = obslist_dict.pop(iter_best_name)
            if not isinstance(onames, list):
                onames = [onames]
            obs_being_used.extend(onames)
            if reset_zero_weight > 0.0:
                snames = set(self.pst.nnz_obs_names)
                reset_names = [o for o in onames if o not in snames]
                self.pst.observation_data.loc[reset_names, "weight"] = reset_zero_weight
        columns = [
            "best_obs",
            forecast + "_variance",
            "unc_reduce_iter_base",
            "unc_reduce_initial_base",
        ]
        return pd.DataFrame(best_results, index=best_case, columns=columns)

    def next_most_par_contribution(self, niter=3, forecast=None, parlist_dict=None):
        """find the parameter(s) contributing most to posterior
        forecast  by sequentially evaluating the contribution of parameters in
        `parlist_dict`.

        Args:
            forecast (`str`, optional): name of the forecast to use in the ranking process.  If
                more than one forecast has been listed, this argument is required.  This is because
                the data worth must be ranked with respect to the variance reduction for a single
                forecast
            niter (`int`, optional):  number of sequential dataworth testing iterations.  Default is 3
            parlist_dict : dict
                a nested dictionary-list of groups of parameters
                that are to be treated as perfectly known.  key values become
                row labels in dataframe
            parlist_dict (`dict`, optional): a nested dictionary-list of groups of parameters
                that are to be treated as perfectly known (zero uncertainty).  key values become
                row labels in returned dataframe. If `None`, then every adustable parameter is tested
                sequentially. Default is `None`. Conceptually, the forecast variance should
                either not change or decrease as a result of knowing parameter perfectly.  The magnitude
                of the decrease represents the worth of gathering information about the parameter(s) being
                tested.

        Note:
            The largest contributing parameters from each iteration are
            treated as known perfectly for the remaining iterations.  In this way, the
            next iteration seeks the next most influential group of parameters.

        Returns:
            `pandas.DataFrame`: a dataframe with index of iteration number and columns
            of `parlist_dict.keys()`.  The values are the results of the knowing
            each parlist_dict entry expressed as posterior variance reduction

        """
        if forecast is None:
            assert len(self.forecasts) == 1, (
                "forecast arg list one and only one" + " forecast"
            )
        elif forecast not in self.prediction_arg:
            raise Exception("forecast {0} not found".format(forecast))
        org_parcov = self.parcov.get(row_names=self.parcov.row_names)
        if parlist_dict is None:
            parlist_dict = dict(zip(self.pst.adj_par_names, self.pst.adj_par_names))

        base_prior, base_post = self.prior_forecast, self.posterior_forecast
        iter_results = [base_post[forecast].copy()]
        iter_names = ["base"]
        for iiter in range(niter):
            iter_contrib = {forecast: [base_post[forecast]]}
            iter_case_names = ["base"]
            self.log("next most par iteration {0}".format(iiter + 1))

            for case, parlist in parlist_dict.items():
                iter_case_names.append(case)
                la_cond = self.get_conditional_instance(parlist)
                iter_contrib[forecast].append(la_cond.posterior_forecast[forecast])
            df = pd.DataFrame(iter_contrib, index=iter_case_names)
            df.sort_values(by=forecast, inplace=True)
            iter_best = df.index[0]
            self.logger.statement(
                "next best iter {0}: {1}".format(iiter + 1, iter_best)
            )
            self.log("next most par iteration {0}".format(iiter + 1))
            if iter_best.lower() == "base":
                break
            iter_results.append(df.loc[iter_best, forecast])
            iter_names.append(iter_best)
            self.reset_parcov(self.parcov.condition_on(parlist_dict.pop(iter_best)))

        self.reset_parcov(org_parcov)
        return pd.DataFrame(iter_results, index=iter_names)
