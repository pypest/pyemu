import pandas as pd
import numpy as np
import pyemu

# lowest level functions
# see https://pypi.org/project/hydroeval/#files for example implementation which were consulted
def _NSE(obs, mod):
    """
    Calculate the Nash-Sutcliffe Efficiency
    (https://www.sciencedirect.com/science/article/pii/0022169470902556?via%3Dihub)
    Args:
        obs: numpy array of observed values
        mod: numpy array of modeled values

    Returns:
        NSE: Nash-Sutcliffe Efficiency


    """
    return 1 - (np.sum((obs - mod) ** 2) / np.sum((obs - np.mean(obs)) ** 2))


def _NNSE(obs, mod):
    """
    Calculate the Normalized Nash-Sutcliffe Efficiency
    (https://meetingorganizer.copernicus.org/EGU2012/EGU2012-237.pdf)
    Args:
        obs: numpy array of observed values
        mod: numpy array of modeled values

    Returns:
        NSE: Nash-Sutcliffe Efficiency


    """
    return 1 / (2 - _NSE(obs, mod))


def _MAE(mod, obs):
    """
    Calculate the Mean Absolute Error
    Args:
        obs: numpy array of observed values
        mod: numpy array of modeled values

    Returns:
        MAE: Mean Absolute Error
    """
    return np.mean(np.abs(obs - mod))

def _STANDARD_ERROR(mod, obs):
    """
    Calculate Standard Error as defined in TSPROC manual
    https://pubs.usgs.gov/tm/tm7c7/pdf/TM7_C7_112712.pdf

    Args:
        obs: numpy array of observed values
        mod: numpy array of modeled values
    """
    return np.sqrt(np.sum((mod-obs)**2)/(len(obs)-1))

def _RELATIVE_STANDARD_ERROR(mod, obs):
    """
    Calculate Relative Standard Error as defined in TSPROC manual
    https://pubs.usgs.gov/tm/tm7c7/pdf/TM7_C7_112712.pdf

    Args:
        obs: numpy array of observed values
        mod: numpy array of modeled values
    """
    return _STANDARD_ERROR(mod, obs) / np.sqrt(np.sum((obs-np.nanmean(obs))**2)/(len(obs)-1))



    

def _VOLUMETRIC_EFFICIENCY(mod, obs):
    """
    Calculate Volumetric Efficiency as defined in TSPROC manual
    https://pubs.usgs.gov/tm/tm7c7/pdf/TM7_C7_112712.pdf

    Args:
        obs: numpy array of observed values
        mod: numpy array of modeled values
    """
    return 1 - ((np.sum(np.abs(mod-obs)))/(np.sum(obs)))

def _MSE(mod, obs):
    """
    Calculate the Mean Squared Error
    Args:
        obs: numpy array of observed values
        mod: numpy array of modeled values

    Returns:
        MSE: Mean Squared Error
    """
    return np.mean((obs - mod) ** 2)


def _RMSE(mod, obs):
    """
    Calculate the Root Mean Squared Error
    Args:
        obs: numpy array of observed values
        mod: numpy array of modeled values

    Returns:
        RMSE: Root Mean Squared Error
    """
    return np.sqrt(_MSE(obs, mod))


def _NRMSE_SD(mod, obs):
    """
    Calculate the Normalized Root Mean Squared Error
    normalized by observation standard deviation
    https://www.marinedatascience.co/blog/2019/01/07/normalizing-the-rmse/
    Args:
        obs: numpy array of observed values
        mod: numpy array of modeled values

    Returns:
        NRMSE_SD: Root Mean Squared Error normalized by observation standard deviation
    """
    return _RMSE(mod, obs) / np.std(obs)


def _NRMSE_MEAN(mod, obs):
    """
    Calculate the Normalized Root Mean Squared Error
    normalized by observation mean
    https://www.marinedatascience.co/blog/2019/01/07/normalizing-the-rmse/
    Args:
        obs: numpy array of observed values
        mod: numpy array of modeled values

    Returns:
        NRMSE_SD: Root Mean Squared Error normalized by observation mean
    """
    return _RMSE(mod, obs) / np.mean(obs)


def _NRMSE_IQ(mod, obs):
    """
    Calculate the Normalized Root Mean Squared Error
    normalized by observation interquartile range
    https://www.marinedatascience.co/blog/2019/01/07/normalizing-the-rmse/
    Args:
        obs: numpy array of observed values
        mod: numpy array of modeled values

    Returns:
        NRMSE_SD: Root Mean Squared Error normalized by observation interquartile range
    """
    return _RMSE(mod, obs) / (np.percentile(obs, 75) - np.percentile(obs, 25))


def _NRMSE_MAXMIN(mod, obs):
    """
    Calculate the Normalized Root Mean Squared Error
    normalized by observation max - min
    https://www.marinedatascience.co/blog/2019/01/07/normalizing-the-rmse/
    Args:
        obs: numpy array of observed values
        mod: numpy array of modeled values

    Returns:
        NRMSE_SD: Root Mean Squared Error normalized by observation max - min
    """
    return _RMSE(mod, obs) / (np.max(obs) - np.min(obs))


def _PBIAS(mod, obs):
    """
    Calculate the percent bias
    Args:
        obs: numpy array of observed values
        mod: numpy array of modeled values

    Returns:
        PBIAS: Percent Bias
    """
    return 100 * ((np.sum(mod - obs)) / (np.sum(obs)))

def _BIAS(mod, obs):
    """
    Calculate Bias as defined in TSPROC manual
    https://pubs.usgs.gov/tm/tm7c7/pdf/TM7_C7_112712.pdf

    Args:
        obs: numpy array of observed values
        mod: numpy array of modeled values
    """
    return ((np.sum(mod - obs)) / (np.sum(obs)))/len(obs)

def _RELATIVE_BIAS(mod, obs):
    """
    Calculate Relative Bias as defined in TSPROC manual
    https://pubs.usgs.gov/tm/tm7c7/pdf/TM7_C7_112712.pdf

    Args:
        obs: numpy array of observed values
        mod: numpy array of modeled values
    """
    return _BIAS(mod, obs) / np.nanmean(obs)

def _KGE(mod, obs):
    """
    Calculate the Kling-Gupta Efficiency (KGE)
    (https://www.sciencedirect.com/science/article/pii/S0022169409004843)
    Args:
        obs: numpy array of observed values
        mod: numpy array of modeled values

    Returns:
        KGE: Kling-Gupta Efficiency
    """
    obsmean = np.mean(obs)
    modmean = np.mean(mod)
    d_obs = obs - obsmean
    d_mod = mod - modmean
    r = np.sum(d_obs * d_mod) / np.sqrt(np.sum(d_mod ** 2) * np.sum(d_obs ** 2))
    alpha = np.std(mod) / np.std(obs)
    beta = np.sum(mod) / np.sum(obs)

    ED = np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)
    return 1 - ED


# available metrics to calculate
ALLMETRICS = {
    "pbias": _PBIAS,
    "bias": _BIAS,
    "relative_bias": _RELATIVE_BIAS,
    "rmse": _RMSE,
    "mse": _MSE,
    "nse": _NSE,
    "nnse": _NNSE,
    "mae": _MAE,
    "kge": _KGE,
    "nrmse_sd": _NRMSE_SD,
    "nrmse_mean": _NRMSE_MEAN,
    "nrmse_iq": _NRMSE_IQ,
    "nrmse_maxmin": _NRMSE_MAXMIN,
    "standard_error": _STANDARD_ERROR,
    "volumetric_efficiency": _VOLUMETRIC_EFFICIENCY,
    "relative_standard_error": _RELATIVE_STANDARD_ERROR

}


def calc_metric_res(res, metric="all", bygroups=True, drop_zero_weight=True):
    """Calculates unweighted metrics to quantify fit to observations for residuals

    Args:
        res (pandas DataFrame or filename): DataFrame read from a residuals file or filename
        metric (list of str): metric to calculate (PBIAS, RMSE, MSE, NSE, MAE, NRMSE_SD,
            NRMSE_MEAN, NRMSE_IQ, NRMSE_MAXMIN) case insensitive
            Defaults to 'all' which calculates all available metrics
        bygroups (Bool): Flag to summarize by groups or not. Defaults to True.
        drop_zero_weight (Bool): flag to exclude zero-weighted observations

    Returns:
        **pandas.DataFrame**: single row. Columns are groups. Content is requested metrics
    """
    # check that metrics are a list - if 'all' calculate for them all
    if isinstance(metric, str):
        if metric.lower() == "all":
            metric = list(ALLMETRICS.keys())
        else:
            metric = list(metric)
    missing_metrics = ()
    for cm in metric:
        if cm.lower() not in ALLMETRICS:
            missing_metrics.append(cm)
    if len(missing_metrics) > 0:
        raise Exception(
            "Requested metrics: {} not implemented".format(" ".join(missing_metrics))
        )

    # sort out the res arg to be a file (to read) or already a dataframe
    if isinstance(res, str):
        try:
            res = pyemu.pst_utils.read_resfile(res)
        except:
            raise Exception("{} is not a valid file path".format(res))
    elif not isinstance(res, pd.DataFrame):
        raise Exception(
            "{} must be either a valid file path or a residuals dataframe".format(res)
        )

    if drop_zero_weight:
        res = res.loc[res.weight != 0]

    ret_df = pd.DataFrame(index=["single_realization"])

    # calculate the matrics
    for cm in metric:
        f = ALLMETRICS[cm.lower()]
        ret_df["{}_total".format(cm.upper())] = [
            f(res.modelled, res.measured) for i in ret_df.index
        ]

        # if bygroups, do the groups as columns
        if bygroups is True:
            for cn, cg in res.groupby("group"):
                ret_df["{}_{}".format(cm.upper(), cn)] = f(cg.modelled, cg.measured)
    return ret_df


def calc_metric_ensemble(
    ens,
    pst,
    metric="all",
    bygroups=True,
    subset_realizations=None,
    drop_zero_weight=True,
):
    """Calculates unweighted metrics to quantify fit to observations for ensemble members

    Args:
        ens (pandas DataFrame): DataFrame read from an observation
        pst (pyemu.Pst object):  needed to obtain observation values and weights
        metric (list of str): metric to calculate (PBIAS, RMSE, MSE, NSE, MAE, NRMSE_SD,
            NRMSE_MEAN, NRMSE_IQ, NRMSE_MAXMIN) case insensitive
            Defaults to 'all' which calculates all available metrics
        bygroups (Bool): Flag to summarize by groups or not. Defaults to True.
        subset_realizations (iterable, optional): Subset of realizations for which
                to report metric. Defaults to None which returns all realizations.
        drop_zero_weight (Bool): flag to exclude zero-weighted observations
    Returns:
        **pandas.DataFrame**: rows are realizations. Columns are groups. Content is requested metrics
    """

    # TODO: handle zero weights due to PDC
    # check that metrics are a list - if 'all' calculate for them all
    if isinstance(metric, str):
        if metric.lower() == "all":
            metric = list(ALLMETRICS.keys())
        else:
            metric = list(metric)
    missing_metrics = ()
    for cm in metric:
        if cm.lower() not in ALLMETRICS:
            missing_metrics.append(cm)
    if len(missing_metrics) > 0:
        raise Exception(
            "Requested metrics: {} not implemented".format(" ".join(missing_metrics))
        )

    # make sure subset_realizations is a list
    if not isinstance(subset_realizations, list) and subset_realizations is not None:
        subset_realizations = list(subset_realizations)

    if "real_name" in ens.columns:
        ens.set_index("real_name", inplace=True)

    if not isinstance(pst, pyemu.Pst):
        raise Exception("pst object must be of type pyemu.Pst")

    # get the observation data
    obs = pst.observation_data.copy()

    # confirm that the indices and observations line up
    if False in np.unique(ens.columns == obs.index):
        raise Exception("ens and pst observation names do not align")

    if drop_zero_weight:
        # subset to only include non-zero-weighted obs
        ens = ens[pst.nnz_obs_names]
        obs = obs.loc[pst.nnz_obs_names]

    ret_df = pd.DataFrame(index=ens.index)
    if subset_realizations is not None:
        ret_df = ret_df.loc[subset_realizations]

    # calculate the rmse total first
    for cm in metric:
        f = ALLMETRICS[cm.lower()]
        ret_df["{}_total".format(cm.upper())] = [
            f(ens.loc[i], obs.obsval) for i in ret_df.index
        ]

        # if bygroups, do the groups as columns
        if bygroups is True:
            for cg in obs.obgnme.unique():
                cnames = obs.loc[obs.obgnme == cg].obsnme
                ret_df["{}_{}".format(cm.upper(), cg)] = [
                    f(ens.loc[i][cnames], obs.loc[cnames].obsval) for i in ret_df.index
                ]
        ret_df = ret_df.copy()

    return ret_df
