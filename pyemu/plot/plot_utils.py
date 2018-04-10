import os
import shutil
import numpy as np
import pandas as pd
from datetime import datetime
import string
from pyemu.logger import Logger
font = {'size'   : 6}
import matplotlib
matplotlib.rc("font",**font)

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec


import pyemu
figsize=(8,10.5)
nr,nc = 4,2
#page_gs = GridSpec(nr,nc)

abet = string.ascii_uppercase

def plot_summary_distributions(df,ax=None,label_post=False,label_prior=False,
                               subplots=False,figsize=(11,8.5),pt_color='b'):
    """ helper function to plot gaussian distrbutions from prior and posterior
    means and standard deviations

    Parameters
    ----------
    df : pandas.DataFrame
        a dataframe and csv file.  Must have columns named:
        'prior_mean','prior_stdev','post_mean','post_stdev'.  If loaded
        from a csv file, column 0 is assumed to tbe the index
    ax: matplotlib.pyplot.axis
        If None, and not subplots, then one is created
        and all distributions are plotted on a single plot
    label_post: bool
        flag to add text labels to the peak of the posterior
    label_prior: bool
        flag to add text labels to the peak of the prior
    subplots: (boolean)
        flag to use subplots.  If True, then 6 axes per page
        are used and a single prior and posterior is plotted on each
    figsize: tuple
        matplotlib figure size

    Returns
    -------
    figs : list
        list of figures
    axes : list
        list of axes

    Note
    ----
    This is useful for demystifying FOSM results

    if subplots is False, a single axis is returned

    Example
    -------
    ``>>>import matplotlib.pyplot as plt``

    ``>>>import pyemu``

    ``>>>pyemu.helpers.plot_summary_distributions("pest.par.usum.csv")``

    ``>>>plt.show()``
    """
    import matplotlib.pyplot as plt
    if isinstance(df,str):
        df = pd.read_csv(df,index_col=0)
    if ax is None and not subplots:
        fig = plt.figure(figsize=figsize)
        ax = plt.subplot(111)
        ax.grid()


    if "post_stdev" not in df.columns and "post_var" in df.columns:
        df.loc[:,"post_stdev"] = df.post_var.apply(np.sqrt)
    if "prior_stdev" not in df.columns and "prior_var" in df.columns:
        df.loc[:,"prior_stdev"] = df.prior_var.apply(np.sqrt)
    if "prior_expt" not in df.columns and "prior_mean" in df.columns:
        df.loc[:,"prior_expt"] = df.prior_mean
    if "post_expt" not in df.columns and "post_mean" in df.columns:
        df.loc[:,"post_expt"] = df.post_mean

    if subplots:
        fig = plt.figure(figsize=figsize)
        ax = plt.subplot(2,3,1)
        ax_per_page = 6
        ax_count = 0
        axes = []
        figs = []
    for name in df.index:
        x,y = gaussian_distribution(df.loc[name,"post_expt"],
                                    df.loc[name,"post_stdev"])
        ax.fill_between(x,0,y,facecolor=pt_color,edgecolor="none",alpha=0.25)
        if label_post:
            mx_idx = np.argmax(y)
            xtxt,ytxt = x[mx_idx],y[mx_idx] * 1.001
            ax.text(xtxt,ytxt,name,ha="center",alpha=0.5)

        x,y = gaussian_distribution(df.loc[name,"prior_expt"],
                                    df.loc[name,"prior_stdev"])
        ax.plot(x,y,color='0.5',lw=3.0,dashes=(2,1))
        if label_prior:
            mx_idx = np.argmax(y)
            xtxt,ytxt = x[mx_idx],y[mx_idx] * 1.001
            ax.text(xtxt,ytxt,name,ha="center",alpha=0.5)
        #ylim = list(ax.get_ylim())
        #ylim[1] *= 1.2
        #ylim[0] = 0.0
        #ax.set_ylim(ylim)
        if subplots:
            ax.set_title(name)
            ax_count += 1
            ax.set_yticklabels([])
            axes.append(ax)
            if name == df.index[-1]:
                break
            if ax_count >= ax_per_page:
                figs.append(fig)
                fig = plt.figure(figsize=figsize)
                ax_count = 0
            ax = plt.subplot(2,3,ax_count+1)
    if subplots:
        figs.append(fig)
        return figs, axes
    ylim = list(ax.get_ylim())
    ylim[1] *= 1.2
    ylim[0] = 0.0
    ax.set_ylim(ylim)
    ax.set_yticklabels([])
    return ax

def gaussian_distribution(mean, stdev, num_pts=50):
    """ get an x and y numpy.ndarray that spans the +/- 4
    standard deviation range of a gaussian distribution with
    a given mean and standard deviation. useful for plotting

    Parameters
    ----------
    mean : float
        the mean of the distribution
    stdev : float
        the standard deviation of the distribution
    num_pts : int
        the number of points in the returned ndarrays.
        Default is 50

    Returns
    -------
    x : numpy.ndarray
        the x-values of the distribution
    y : numpy.ndarray
        the y-values of the distribution

    """
    xstart = mean - (4.0 * stdev)
    xend = mean + (4.0 * stdev)
    x = np.linspace(xstart,xend,num_pts)
    y = (1.0/np.sqrt(2.0*np.pi*stdev*stdev)) * np.exp(-1.0 * ((x - mean)**2)/(2.0*stdev*stdev))
    return x,y

def pst_helper(pst,kind=None,**kwargs):

    echo = kwargs.get("echo",True)
    logger = pyemu.Logger("plot_pst_helper.log",echo=echo)
    logger.statement("plot_utils.pst_helper()")

    kinds = {"prior":pst_prior,"1to1":res_1to1,"obs_v_sim":res_obs_v_sim,
             "phi_pie":res_phi_pie,"weight_hist":pst_weight_hist}

    if kind is None:
        logger.statement("kind=None, nothing to do")
        return
    elif kind not in kinds:
        logger.lraise("unrecognized kind:{0}, should one of {1}"
                      .format(kind,','.join(list(kinds.keys()))))
    return kinds[kind](pst, logger, **kwargs)


def res_1to1(pst,logger=None,filename=None,plot_hexbin=False,**kwargs):
    """ make 1-to-1 plots and also observed vs residual by observation group
    Parameters
    ----------
    pst : pyemu.Pst
    logger : Logger
        if None, a generic one is created.  Default is None
    filename : str
        PDF filename to save figures to.  If None, figures are returned.  Default is None
    kwargs : dict
        optional keyword args to pass to plotting functions

    TODO: color symbols by weight


    """
    if logger is None:
        logger=Logger('Default_Loggger.log',echo=False)
    logger.log("plot res_1to1")
    if pst.res is None:
        logger.lraise("res_1to1: pst.res is None, couldn't find residuals file")
    obs = pst.observation_data
    res = pst.res

    if "grouper" in kwargs:
        raise NotImplementedError()
    else:
        grouper = obs.groupby(obs.obgnme).groups

    fig = plt.figure(figsize=figsize)
    if "fig_title" in kwargs:
        plt.figtext(0.5,0.5,kwargs["fig_title"])
    else:
        plt.figtext(0.5, 0.5, "pyemu.Pst.plot(kind='1to1')\nfrom pest control file '{0}'\n at {1}"
                    .format(pst.filename, str(datetime.now())), ha="center")
    #if plot_hexbin:
    #    pdfname = pst.filename.replace(".pst", ".1to1.hexbin.pdf")
    #else:
    #    pdfname = pst.filename.replace(".pst", ".1to1.pdf")
    figs = []
    ax_count = 0
    for g, names in grouper.items():
        logger.log("plotting 1to1 for {0}".format(g))

        obs_g = obs.loc[names, :]
        obs_g.loc[:, "sim"] = res.loc[names, "modelled"]
        logger.statement("using control file obsvals to calculate residuals")
        obs_g.loc[:,'res'] = obs_g.sim - obs_g.obsval
        if "include_zero" not in kwargs or kwargs["include_zero"] is True:
            obs_g = obs_g.loc[obs_g.weight > 0, :]
        if obs_g.shape[0] == 0:
            logger.statement("no non-zero obs for group '{0}'".format(g))
            logger.log("plotting 1to1 for {0}".format(g))
            continue

        if ax_count % (nr * nc) == 0:
            if ax_count > 0:
                plt.tight_layout()
            #pdf.savefig()
            #plt.close(fig)
            figs.append(fig)
            fig = plt.figure(figsize=figsize)
            axes = get_page_axes()
            ax_count = 0

        ax = axes[ax_count]

        #if obs_g.shape[0] == 1:
        #    ax.scatter(list(obs_g.sim),list(obs_g.obsval),marker='.',s=30,color='b')
        #else:
        mx = max(obs_g.obsval.max(), obs_g.sim.max())
        mn = min(obs_g.obsval.min(), obs_g.sim.min())

        #if obs_g.shape[0] == 1:
        mx *= 1.1
        mn *= 0.9
        ax.axis('square')
        if plot_hexbin:
            ax.hexbin(obs_g.sim.values, obs_g.obsval.values, mincnt=1, gridsize=(75, 75),
                      extent=(mn, mx, mn, mx), bins='log', edgecolors=None)
#               plt.colorbar(ax=ax)
        else:
            ax.scatter([obs_g.sim], [obs_g.obsval], marker='.', s=10, color='b')



        ax.plot([mn,mx],[mn,mx],'k--',lw=1.0)
        xlim = (mn,mx)
        ax.set_xlim(mn,mx)
        ax.set_ylim(mn,mx)
        ax.grid()

        ax.set_ylabel("observed",labelpad=0.1)
        ax.set_xlabel("simulated",labelpad=0.1)
        ax.set_title("{0}) group:{1}, {2} observations".
                                 format(abet[ax_count], g, obs_g.shape[0]), loc="left")

        ax_count += 1

        ax = axes[ax_count]
        ax.scatter(obs_g.obsval, obs_g.res, marker='.', s=10, color='b')
        ylim = ax.get_ylim()
        mx = max(np.abs(ylim[0]), np.abs(ylim[1]))
        if obs_g.shape[0] == 1:
            mx *= 1.1
        ax.set_ylim(-mx, mx)
        #show a zero residuals line
        ax.plot(xlim, [0,0], 'k--', lw=1.0)
        meanres= obs_g.res.mean()
        # show mean residuals line
        ax.plot(xlim,[meanres,meanres], 'r-', lw=1.0)
        ax.set_xlim(xlim)
        ax.set_ylabel("residual",labelpad=0.1)
        ax.set_xlabel("observed",labelpad=0.1)
        ax.set_title("{0}) group:{1}, {2} observations".
                     format(abet[ax_count], g, obs_g.shape[0]), loc="left")
        ax.grid()
        ax_count += 1

        logger.log("plotting 1to1 for {0}".format(g))

    for a in range(ax_count, nr * nc):
        axes[a].set_axis_off()
        axes[a].set_yticks([])
        axes[a].set_xticks([])

    #plt.tight_layout()
    #pdf.savefig()
    #plt.close(fig)
    figs.append(fig)
    if filename is not None:
        with PdfPages(filename) as pdf:
            for fig in figs:
                pdf.savefig(fig)
                plt.close(fig)
        logger.log("plot res_1to1")
    else:
        logger.log("plot res_1to1")
        return figs

def res_obs_v_sim(pst,logger=None, filename=None,  **kwargs):
    """
    timeseries plot helper...in progress

    """
    if logger is None:
        logger=Logger('Default_Loggger.log',echo=False)
    logger.log("plot res_obs_v_sim")
    if pst.res is None:
        logger.lraise("res_obs_v_sim: pst.res is None, couldn't find residuals file")
    obs = pst.observation_data
    res = pst.res

    if "grouper" in kwargs:
        raise NotImplementedError()
    else:
        grouper = obs.groupby(obs.obgnme).groups

    fig = plt.figure(figsize=figsize)
    if "fig_title" in kwargs:
        plt.figtext(0.5,0.5,kwargs["fig_title"])
    else:
        plt.figtext(0.5, 0.5, "pyemu.Pst.plot(kind='obs_v_sim')\nfrom pest control file '{0}'\n at {1}"
                    .format(pst.filename, str(datetime.now())), ha="center")
    figs = []
    ax_count = 0
    for g, names in grouper.items():
        logger.log("plotting obs_v_sim for {0}".format(g))

        obs_g = obs.loc[names, :]
        obs_g.loc[:, "sim"] = res.loc[names, "modelled"]
        if "include_zero" not in kwargs or kwargs["include_zero"] is False:
            obs_g = obs_g.loc[obs_g.weight > 0, :]

        if obs_g.shape[0] == 0:
            logger.statement("no non-zero obs for group '{0}'".format(g))
            logger.log("plotting obs_v_sim for {0}".format(g))
            continue

        # parse datetimes
        try:
            obs_g.loc[:, "datetime_str"] = obs_g.obsnme.apply(lambda x: x.split('_')[-1])
        except Exception as e:
            logger.warn("res_obs_v_sim error forming datetime_str:{0}".
                        format(str(e)))
            continue

        try:
            obs_g.loc[:, "datetime"] = pd.to_datetime(obs_g.datetime_str,format="%Y%m%d")
        except Exception as e:
            logger.warn("res_obs_v_sim error casting datetime: {0}".
                        format(str(e)))
            continue

        if ax_count % (nr * nc) == 0:
            plt.tight_layout()
            #pdf.savefig()
            #plt.close(fig)
            figs.append(fig)
            fig = plt.figure(figsize=figsize)
            axes = get_page_axes()
            ax_count = 0

        ax = axes[ax_count]
        obs_g.loc[:,"site"] = obs_g.obsnme.apply(lambda x: x.split('_')[0])
        for site in obs_g.site.unique():
            obs_s = obs_g.loc[obs_g.site==site,:]
            obs_s.sort_values(by="datetime")
            ax.plot(obs_s.datetime, obs_s.obsval, ls='-', marker='.', ms=10, color='b')
            ax.plot(obs_s.datetime, obs_s.sim, ls='-', marker='.', ms=10, color='0.5')
        ax.set_xlim(obs_g.datetime.min(),obs_g.datetime.max())
        ax.grid()
        ax.set_xlabel("datetime",labelpad=0.1)
        ax.set_title("{0}) group:{1}, {2} observations".
                     format(abet[ax_count], g, names.shape[0]), loc="left")
        ax_count += 1
        logger.log("plotting obs_v_sim for {0}".format(g))

    for a in range(ax_count,nr*nc):
        axes[a].set_axis_off()
        axes[a].set_yticks([])
        axes[a].set_xticks([])

    plt.tight_layout()
    #pdf.savefig()
    #plt.close(fig)
    figs.append(fig)
    if filename is not None:
        with PdfPages(pst.filename.replace(".pst", ".obs_v_sim.pdf")) as pdf:
            for fig in figs:
                pdf.savefig(fig)
                plt.close(fig)
                logger.log("plot res_obs_v_sim")
    else:
        logger.log("plot res_obs_v_sim")
        return figs

def res_phi_pie(pst,logger=None, **kwargs):
    """plot current phi components as a pie chart.

    Parameters
    ----------
    pst : pyemu.Pst
    logger : pyemu.Logger
    kwargs : dict
        accepts 'include_zero' as a flag to include phi groups with
        only zero-weight obs (not sure why anyone would do this, but
        whatevs).
    Returns
    -------
    ax : matplotlib.Axis


    """
    if logger is None:
        logger=Logger('Default_Loggger.log',echo=False)
    logger.log("plot res_phi_pie")
    if pst.res is None:
        logger.lraise("res_phi_pie: pst.res is None, couldn't find residuals file")
    obs = pst.observation_data
    res = pst.res
    phi_comps = pst.phi_components
    norm_phi_comps = pst.phi_components_normalized
    if "include_zero" not in kwargs or kwargs["include_zero"] is True:
        phi_comps = {k:v for k,v in phi_comps.items() if v > 0.0}
        norm_phi_comps = {k:norm_phi_comps[k] for k in phi_comps.keys()}
    if "ax" in kwargs:
        ax = kwargs["ax"]
    else:
        fig = plt.figure(figsize=figsize)
        ax = plt.subplot(1,1,1,aspect="equal")
    labels = ["{0}\n{1:4G}({2:3.1f}%)".format(k,v,100. * (v / pst.phi)) for k,v in phi_comps.items()]
    ax.pie([float(v) for v in norm_phi_comps.values()],labels=labels)
    logger.log("plot res_phi_pie")
    return ax



def pst_weight_hist(pst,logger, **kwargs):
    raise NotImplementedError()

def get_page_axes():
    axes = [plt.subplot(nr,nc,i+1) for i in range(nr*nc)]
    #[ax.set_yticks([]) for ax in axes]
    return axes

def pst_prior(pst,logger=None, filename=None, **kwargs):
    """ helper to plot prior parameter histograms implied by
    parameter bounds. Saves a multipage pdf named <case>.prior.pdf

    Parameters
    ----------
    pst : pyemu.Pst
    logger : pyemu.Logger
    filename : str
        PDF filename to save plots to. If None, return figs without saving.  Default is None.
    kwargs : dict
        accepts 'grouper' as dict to group parameters on to a single axis (use
        parameter groups if not passed),
        'unqiue_only' to only show unique mean-stdev combinations within a
        given group

    Returns
    -------
    None

    TODO
    ----
    external parcov, unique mean-std pairs

    """
    if logger is None:
        logger=Logger('Default_Loggger.log',echo=False)
    logger.log("plot pst_prior")
    par = pst.parameter_data

    if "parcov_filename" in pst.pestpp_options:
        logger.warn("ignoring parcov_filename, using parameter bounds for prior cov")
    logger.log("loading cov from parameter data")
    cov = pyemu.Cov.from_parameter_data(pst)
    logger.log("loading cov from parameter data")

    logger.log("building mean parameter values")
    li = par.partrans.loc[cov.names] == "log"
    mean = par.parval1.loc[cov.names]
    info = par.loc[cov.names,:].copy()
    info.loc[:,"mean"] = mean[li].apply(np.log10)
    logger.log("building mean parameter values")

    logger.log("building stdev parameter values")
    if cov.isdiagonal:
        std = cov.x.flatten()
    else:
        std = np.diag(cov.x)
    std = np.sqrt(std)
    info.loc[:,"prior_std"] = std

    logger.log("building stdev parameter values")

    if std.shape != mean.shape:
        logger.lraise("mean.shape {0} != std.shape {1}".
                      format(mean.shape,std.shape))

    if "grouper" in kwargs:
        raise NotImplementedError()
        #check for consistency here

    else:
        par_adj = par.loc[par.partrans.apply(lambda x: x in ["log","none"]),:]
        grouper = par_adj.groupby(par_adj.pargp).groups
        #grouper = par.groupby(par.pargp).groups

    if len(grouper) == 0:
        raise Exception("no adustable parameters to plot")

    fig = plt.figure(figsize=figsize)
    if "fig_title" in kwargs:
        plt.figtext(0.5,0.5,kwargs["fig_title"])
    else:
        plt.figtext(0.5,0.5,"pyemu.Pst.plot(kind='prior')\nfrom pest control file '{0}'\n at {1}"
                 .format(pst.filename,str(datetime.now())),ha="center")
    figs = []
    ax_count = 0
    grps_names = list(grouper.keys())
    grps_names.sort()
    for g in grps_names:
        names = grouper[g]
        logger.log("plotting priors for {0}".
                   format(','.join(list(names))))
        if ax_count % (nr * nc) == 0:
            plt.tight_layout()
            #pdf.savefig()
            #plt.close(fig)
            figs.append(fig)
            fig  = plt.figure(figsize=figsize)
            axes = get_page_axes()
            ax_count = 0

        islog = False
        vc = info.partrans.value_counts()
        if vc.shape[0] > 1:
            logger.warn("mixed partrans for group {0}".format(g))
        elif "log" in vc.index:
            islog = True
        ax = axes[ax_count]
        if "unique_only" in kwargs and kwargs["unique_only"]:


            ms = info.loc[names,:].apply(lambda x: (x["mean"],x["prior_std"]),axis=1).unique()
            for (m,s) in ms:
                x, y = gaussian_distribution(m, s)
                ax.fill_between(x, 0, y, facecolor='0.5', alpha=0.5,
                                edgecolor="none")


        else:
            for m,s in zip(info.loc[names,'mean'],info.loc[names,'prior_std']):
                x,y = gaussian_distribution(m,s)
                ax.fill_between(x,0,y,facecolor='0.5',alpha=0.5,
                                            edgecolor="none")
        ax.set_title("{0}) group:{1}, {2} parameters".
                                 format(abet[ax_count],g,names.shape[0]),loc="left")

        ax.set_yticks([])
        if islog:
            ax.set_xlabel("$log_{10}$ parameter value",labelpad=0.1)
        else:
            ax.set_xlabel("parameter value", labelpad=0.1)
        logger.log("plotting priors for {0}".
                   format(','.join(list(names))))

        ax_count += 1

    for a in range(ax_count,nr*nc):
        axes[a].set_axis_off()
        axes[a].set_yticks([])
        axes[a].set_xticks([])

    plt.tight_layout()
    #pdf.savefig()
    #plt.close(fig)
    figs.append(fig)
    if filename is not None:
        with PdfPages(filename) as pdf:
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)
        logger.log("plot pst_prior")
    else:
        logger.log("plot pst_prior")
        return figs


def ensemble_helper(ensemble,bins=10,facecolor='0.5',plot_cols=None,
                    filename=None,func_dict = None,
                    sync_bins=True,deter_vals=None,std_window=4.0,**kwargs):
    """helper function to plot ensemble histograms

    Parameters
    ----------
    ensemble : varies
        the ensemble argument can be a pandas.DataFrame or derived type or a str, which
        is treated as a fileanme.  Optionally, ensemble can be a list of these types or
        a dict, in which case, the keys are treated as facecolor str (e.g., 'b', 'y', etc).
    facecolor : str
        the histogram facecolor.  Only applies if ensemble is a single thing
    plot_cols : enumerable
        a collection of columns from the ensemble(s) to plot.  If None,
        (the union of) all cols are plotted. Default is None
    filename : str
        the name of the pdf to create.  If None, return figs without saving.  Default is None.
    func_dict : dict
        a dictionary of unary functions (e.g., np.log10_ to apply to columns.  Key is
        column name.  Default is None
    sync_bins : bool
        flag to use the same bin edges for all ensembles. Only applies if more than
        one ensemble is being plotted.  Default is True
    deter_vals : dict

    """
    logger = pyemu.Logger("ensemble_helper.log")
    logger.log("pyemu.plot_utils.ensemble_helper()")
    ensembles = _process_ensemble_arg(ensemble,facecolor,logger)

    #apply any functions
    if func_dict is not None:
        logger.log("applying functions")
        for col,func in func_dict.items():
            for fc,en in ensembles.items():
                if col in en.columns:
                    en.loc[:,col] = en.loc[:,col].apply(func)
        logger.log("applying functions")


    #get a list of all cols (union)
    all_cols = set()
    for fc, en in ensembles.items():
        cols = set(en.columns)
        all_cols.update(cols)
    if plot_cols is None:
        plot_cols = all_cols
    else:
        splot_cols = set(plot_cols)
        missing = splot_cols - all_cols
        if len(missing) > 0:
            logger.lraise("the following plot_cols are missing: {0}".
                          format(','.join(missing)))

    logger.statement("plotting {0} histograms".format(len(plot_cols)))

    fig = plt.figure(figsize=figsize)
    if "fig_title" in kwargs:
        plt.figtext(0.5,0.5,kwargs["fig_title"])
    else:
        plt.figtext(0.5, 0.5, "pyemu.plot_utils.ensemble_helper()\n at {0}"
                    .format(str(datetime.now())), ha="center")
    plot_cols = list(plot_cols)
    plot_cols.sort()
    logger.statement("saving pdf to {0}".format(filename))
    figs = []

    ax_count = 0

    for plot_col in plot_cols:
        logger.log("plotting reals for {0}".format(plot_col))
        if ax_count % (nr * nc) == 0:
            plt.tight_layout()
            #pdf.savefig()
            #plt.close(fig)
            figs.append(fig)
            fig = plt.figure(figsize=figsize)
            axes = get_page_axes()
            [ax.set_yticks([]) for ax in axes]
            ax_count = 0

        ax = axes[ax_count]
        ax.set_title("{0}) {1}".format(abet[ax_count],plot_col),loc="left")
        if sync_bins:
            mx,mn = -1.0e+30,1.0e+30
            for fc,en in ensembles.items():
                if plot_col in en.columns:
                    emx,emn = en.loc[:,plot_col].max(),en.loc[:,plot_col].min()
                    mx = max(mx,emx)
                    mn = min(mn,emn)
            plot_bins = np.linspace(mn,mx,num=bins)
            logger.statement("{0} min:{1:5G}, max:{2:5G}".format(plot_col,mn,mx))
        else:
            plot_bins=bins
        for fc,en in ensembles.items():

            if plot_col in en.columns:
                try:
                    en.loc[:,plot_col].hist(bins=plot_bins,facecolor=fc,
                                            edgecolor="none",alpha=0.5,
                                            normed=True,ax=ax)
                except Exception as e:
                    logger.warn("error plotting histogram for {0}:{1}".
                                format(plot_col,str(e)))


            if deter_vals is not None and plot_col in deter_vals:
                ylim = ax.get_ylim()
                v = deter_vals[plot_col]
                ax.plot([v,v],ylim,"k--",lw=1.5)
                ax.set_ylim(ylim)
            if std_window is not None:
                try:
                    ylim = ax.get_ylim()
                    mn, st = en.loc[:,plot_col].mean(), en.loc[:,plot_col].std() * (std_window / 2.0)

                    ax.plot([mn - st, mn - st], ylim, color=fc, lw=1.5,ls='--')
                    ax.plot([mn + st, mn + st], ylim, color=fc, lw=1.5,ls='--')
                    ax.set_ylim(ylim)
                except:
                    logger.warn("error plotting std window for {0}".
                                format(plot_col))
        ax.grid()

        ax_count += 1

    for a in range(ax_count, nr * nc):
        axes[a].set_axis_off()
        axes[a].set_yticks([])
        axes[a].set_xticks([])

    plt.tight_layout()
    #pdf.savefig()
    #plt.close(fig)
    figs.append(fig)
    if filename is not None:
        plt.tight_layout()
        with PdfPages(filename) as pdf:
            for fig in figs:
                pdf.savefig(fig)
                plt.close(fig)
    logger.log("pyemu.plot_utils.ensemble_helper()")


def _process_ensemble_arg(ensemble,facecolor, logger):
    ensembles = {}
    if isinstance(ensemble, pd.DataFrame):
        if not isinstance(facecolor, str):
            logger.lraise("facecolor must be str")
        ensembles[facecolor] = ensemble

    elif isinstance(ensemble, str):
        if not isinstance(facecolor, str):
            logger.lraise("facecolor must be str")

        logger.log('loading ensemble from csv file {0}'.format(ensemble))
        en = pd.read_csv(ensemble, index_col=0)
        en.columns = en.columns.map(str.lower)
        logger.statement("{0} shape: {1}".format(ensemble, en.shape))
        ensembles[facecolor] = en
        logger.log('loading ensemble from csv file {0}'.format(ensemble))

    elif isinstance(ensemble, list):
        if isinstance(facecolor, list):
            if len(ensemble) != len(facecolor):
                logger.lraise("facecolor list len != ensemble list len")
        else:
            colors = ['m', 'c', 'b', 'r', 'g', 'y']

            facecolor = [colors[i] for i in range(len(ensemble))]
        ensembles = {}
        for fc, en_arg in zip(facecolor, ensemble):
            if isinstance(en_arg, str):
                logger.log("loading ensemble from csv file {0}".format(en_arg))
                en = pd.read_csv(en_arg, index_col=0)
                en.columns = en.columns.map(str.lower)
                logger.log("loading ensemble from csv file {0}".format(en_arg))
                logger.statement("ensemble {0} gets facecolor {1}".format(en_arg, fc))

            elif isinstance(en_arg, pd.DataFrame):
                en = en_arg
            else:
                logger.lraise("unrecognized ensemble list arg:{0}".format(en_file))
            ensembles[fc] = en

    elif isinstance(ensemble, dict):
        for fc, en_arg in ensemble.items():
            if isinstance(en_arg, pd.DataFrame):
                ensembles[fc] = en_arg
            elif isinstance(en_arg, str):
                logger.log("loading ensemble from csv file {0}".format(en_arg))
                en = pd.read_csv(en_arg, index_col=0)
                en.columns = en.columns.map(str.lower)
                logger.log("loading ensemble from csv file {0}".format(en_arg))
                ensembles[fc] = en
            else:
                logger.lraise("unrecognized ensemble list arg:{0}".format(en_arg))

    else:
        raise Exception("unrecognized 'ensemble' arg")

    return ensembles

def ensemble_res_1to1(ensemble, pst,facecolor='0.5',logger=None,filename=None,**kwargs):
    """helper function to plot ensemble 1-to-1 plots sbowing the simulated range

    Parameters
    ----------
    ensemble : varies
        the ensemble argument can be a pandas.DataFrame or derived type or a str, which
        is treated as a fileanme.  Optionally, ensemble can be a list of these types or
        a dict, in which case, the keys are treated as facecolor str (e.g., 'b', 'y', etc).
    pst : pyemu.Pst
        pst instance
    facecolor : str
        the histogram facecolor.  Only applies if ensemble is a single thing
    filename : str
        the name of the pdf to create. If None, return figs without saving.  Default is None.

    """
    if logger is None:
        logger=Logger('Default_Loggger.log',echo=False)
    logger.log("plot res_1to1")
    obs = pst.observation_data
    ensembles = _process_ensemble_arg(ensemble,facecolor,logger)

    if "grouper" in kwargs:
        raise NotImplementedError()
    else:
        grouper = obs.groupby(obs.obgnme).groups

    fig = plt.figure(figsize=figsize)
    if "fig_title" in kwargs:
        plt.figtext(0.5,0.5,kwargs["fig_title"])
    else:
        plt.figtext(0.5, 0.5, "pyemu.Pst.plot(kind='1to1')\nfrom pest control file '{0}'\n at {1}"
                    .format(pst.filename, str(datetime.now())), ha="center")
    #if plot_hexbin:
    #    pdfname = pst.filename.replace(".pst", ".1to1.hexbin.pdf")
    #else:
    #    pdfname = pst.filename.replace(".pst", ".1to1.pdf")
    figs = []
    ax_count = 0
    for g, names in grouper.items():
        logger.log("plotting 1to1 for {0}".format(g))

        obs_g = obs.loc[names, :]
        logger.statement("using control file obsvals to calculate residuals")
        if "include_zero" not in kwargs or kwargs["include_zero"] is False:
            obs_g = obs_g.loc[obs_g.weight > 0, :]
        if obs_g.shape[0] == 0:
            logger.statement("no non-zero obs for group '{0}'".format(g))
            logger.log("plotting 1to1 for {0}".format(g))
            continue

        if ax_count % (nr * nc) == 0:
            if ax_count > 0:
                plt.tight_layout()
            #pdf.savefig()
            #plt.close(fig)
            figs.append(fig)
            fig = plt.figure(figsize=figsize)
            axes = get_page_axes()
            ax_count = 0

        ax = axes[ax_count]

        mx = obs_g.obsval.max()
        mn =  obs_g.obsval.min()

        #if obs_g.shape[0] == 1:
        mx *= 1.1
        mn *= 0.9
        #ax.axis('square')

        #ax.scatter([obs_g.sim], [obs_g.obsval], marker='.', s=10, color='b')
        for c,en in ensembles.items():
            en_g = en.loc[:,obs_g.obsnme]
            ex = en_g.max()
            en = en_g.min()
            [ax.plot([ov,ov],[een,eex],color=c) for ov,een,eex in zip(obs_g.obsval.values,en.values,ex.values)]


        ax.plot([mn,mx],[mn,mx],'k--',lw=1.0)
        xlim = (mn,mx)
        ax.set_xlim(mn,mx)
        ax.set_ylim(mn,mx)
        ax.grid()

        ax.set_xlabel("observed",labelpad=0.1)
        ax.set_ylabel("simulated",labelpad=0.1)
        ax.set_title("{0}) group:{1}, {2} observations".
                                 format(abet[ax_count], g, obs_g.shape[0]), loc="left")

        ax_count += 1
        ax = axes[ax_count]
        #ax.scatter(obs_g.obsval, obs_g.res, marker='.', s=10, color='b')
        for c,en in ensembles.items():
            en_g = en.loc[:,obs_g.obsnme].subtract(obs_g.obsval,axis=1)
            ex = en_g.max()
            en = en_g.min()
            [ax.plot([ov,ov],[een,eex],color=c) for ov,een,eex in zip(obs_g.obsval.values,en.values,ex.values)]
        ylim = ax.get_ylim()
        mx = max(np.abs(ylim[0]), np.abs(ylim[1]))
        if obs_g.shape[0] == 1:
            mx *= 1.1
        ax.set_ylim(-mx, mx)
        #show a zero residuals line
        ax.plot(xlim, [0,0], 'k--', lw=1.0)

        ax.set_xlim(xlim)
        ax.set_ylabel("residual",labelpad=0.1)
        ax.set_xlabel("observed",labelpad=0.1)
        ax.set_title("{0}) group:{1}, {2} observations".
                     format(abet[ax_count], g, obs_g.shape[0]), loc="left")
        ax.grid()
        ax_count += 1

        logger.log("plotting 1to1 for {0}".format(g))

    for a in range(ax_count, nr * nc):
        axes[a].set_axis_off()
        axes[a].set_yticks([])
        axes[a].set_xticks([])

    #plt.tight_layout()
    #pdf.savefig()
    #plt.close(fig)
    figs.append(fig)
    if filename is not None:
        plt.tight_layout()
        with PdfPages(filename) as pdf:
            for fig in figs:
                pdf.savefig(fig)
                plt.close(fig)
        logger.log("plot res_1to1")
    else:
        logger.log("plot res_1to1")
        return figs
