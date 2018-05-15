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

    echo = kwargs.get("echo",False)
    logger = pyemu.Logger("plot_pst_helper.log",echo=echo)
    logger.statement("plot_utils.pst_helper()")

    kinds = {"prior":pst_prior,"1to1":res_1to1,"obs_v_sim":res_obs_v_sim,
             "phi_pie":res_phi_pie,"weight_hist":pst_weight_hist,
             "phi_progress":phi_progress}

    if kind is None:
        returns = []
        base_filename = pst.filename
        if pst.new_filename is not None:
            base_filename = pst.new_filename
        base_filename = base_filename.replace(".pst",'')
        for name,func in kinds.items():
            plt_name = base_filename+"."+name+".pdf"
            returns.append(func(pst,logger=logger,filename=plt_name))

        return returns
    elif kind not in kinds:
        logger.lraise("unrecognized kind:{0}, should one of {1}"
                      .format(kind,','.join(list(kinds.keys()))))
    return kinds[kind](pst, logger, **kwargs)


def phi_progress(pst,logger=None,filename=None,**kwargs):
    """ make plot of phi vs number of model runs - requires
    available pestpp .iobj file
        Parameters
        ----------
        pst : pyemu.Pst
        logger : Logger
            if None, a generic one is created.  Default is None
        filename : str
            PDF filename to save figures to.  If None, figures are returned.  Default is None
        kwargs : dict
            optional keyword args to pass to plotting functions


        """
    if logger is None:
        logger = Logger('Default_Loggger.log', echo=False)
    logger.log("plot phi_progress")

    iobj_file = pst.filename.replace(".pst",".iobj")
    if not os.path.exists(iobj_file):
        logger.lraise("couldn't find iobj file {0}".format(iobj_file))
    df = pd.read_csv(iobj_file)
    if "ax" in kwargs:
        ax = kwargs["ax"]
    else:
        fig = plt.figure(figsize=figsize)
        ax = plt.subplot(1,1,1)
    ax.plot(df.model_runs_completed,df.total_phi,marker='.')
    ax.set_xlabel("model runs")
    ax.set_ylabel("$\phi$")
    ax.grid()
    if filename is not None:
        plt.savefig(filename)
    logger.log("plot phi_progress")
    return ax



def get_page_axes(count=nr*nc):
    axes = [plt.subplot(nr,nc,i+1) for i in range(min(count,nr*nc))]
    #[ax.set_yticks([]) for ax in axes]
    return axes

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
            ax.hexbin(obs_g.obsval.values, obs_g.sim.values, mincnt=1, gridsize=(75, 75),
                      extent=(mn, mx, mn, mx), bins='log', edgecolors=None)
#               plt.colorbar(ax=ax)
        else:
            ax.scatter([obs_g.obsval], [obs_g.sim], marker='.', s=10, color='b')



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

    plt.tight_layout()
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
    axes = None
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

    if axes is None:
        return

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

def plot_id_bar(id_df, nsv=None, logger=None, **kwargs):
    """
    Plot a stacked bar chart of identifiability based on a identifiability dataframe

    Parameters
    ----------
        id_df : pandas dataframe of identifiability
        nsv : number of singular values to consider
        logger : pyemu.Logger
        kwargs : dict of keyword arguments

    Returns
    -------
    ax : matplotlib.Axis

       Example
    -------
    ``>>> import pyemu``
    ``>>> pest_obj = pyemu.Pst(pest_control_file)``
    ``>>> ev = pyemu.ErrVar(jco='freyberg_jac.jcb'))``
    ``>>> id_df = ev.get_identifiability_dataframe(singular_value=48)``
    ``>>> pyemu.plot_id_bar(id_df, nsv=12, figsize=(12,4)``
     """
    if logger is None:
        logger=Logger('Default_Loggger.log',echo=False)
    logger.log("plot id bar")


    df = id_df.copy()

    # drop the final `ident` column
    if 'ident' in df.columns:
        df.drop('ident', inplace=True, axis=1)

    if nsv is None or nsv > len(df.columns):
        nsv = len(df.columns)
        logger.log('set number of SVs and number in the dataframe')

    df = df[df.columns[:nsv]]

    df['ident'] = df.sum(axis=1)
    df.sort_values(by='ident', inplace=True, ascending=False)
    df.drop('ident', inplace=True, axis=1)

    if 'figsize' in kwargs:
        figsize=kwargs['figsize']
    else:
        figsize = (8, 10.5)
    if "ax" in kwargs:
        ax = kwargs["ax"]
    else:
        fig = plt.figure(figsize=figsize)
        ax = plt.subplot(1,1,1)

    # plto the stacked bar chart (the easy part!)
    df.plot.bar(stacked=True, cmap='jet_r', legend=False, ax=ax)

    #
    # horrible shenanigans to make a colorbar rather than a legend
    #

    # special case colormap just dark red if one SV
    if nsv == 1:
        tcm = matplotlib.colors.LinearSegmentedColormap.from_list('one_sv', [plt.get_cmap('jet_r')(0)] * 2, N=2)
        sm = plt.cm.ScalarMappable(cmap=tcm, norm=matplotlib.colors.Normalize(vmin=0, vmax=nsv + 1))
    # or typically just rock the jet_r colormap over the range of SVs
    else:
        sm = plt.cm.ScalarMappable(cmap=plt.get_cmap('jet_r'), norm=matplotlib.colors.Normalize(vmin=1, vmax=nsv))
    sm._A = []

    # now, if too many ticks for the colorbar, summarize them
    if nsv < 20:
        ticks = range(1, nsv + 1)
    else:
        ticks = np.arange(1, nsv + 1, int((nsv + 1) / 30))

    cb = plt.colorbar(sm)
    cb.set_ticks(ticks)

    logger.log('plot id bar')

    return ax

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
    if "filename" in kwargs:
        plt.savefig(kwargs["filename"])
    return ax



def pst_weight_hist(pst,logger, **kwargs):
    #raise NotImplementedError()
    pass



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
    info.loc[:,"mean"] = mean
    info.loc[li,"mean"] = mean[li].apply(np.log10)
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
                    sync_bins=True,deter_vals=None,std_window=4.0,
                    deter_range=False,**kwargs):
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
        dict of deterministic values to plot as a vertical line. key is ensemble columnn name

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

            v = None
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
                    if deter_range and v is not None:
                        xmn = v - st
                        xmx = v + st
                        ax.set_xlim(xmn,xmx)
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
        logger.log("plot res_1to1")
    else:
        logger.log("plot res_1to1")
        return figs


def ensemble_change_summary(ensemble1, ensemble2, pst,bins=10, facecolor='0.5',logger=None,filename=None,**kwargs):
    """helper function to plot first and second moment change histograms

    Parameters
    ----------
    ensemble1 : varies
        str or pd.DataFrames
    ensemble2 : varies
        str or pd.DataFrame
    pst : pyemu.Pst
        pst instance
    facecolor : str
        the histogram facecolor.
    filename : str
        the name of the pdf to create. If None, return figs without saving.  Default is None.

    """
    if logger is None:
        logger=Logger('Default_Loggger.log',echo=False)
    logger.log("plot ensemble change")

    if isinstance(ensemble1, str):
        ensemble1 = pd.read_csv(ensemble1,index_col=0)
    ensemble1.columns = ensemble1.columns.str.lower()

    if isinstance(ensemble2, str):
        ensemble2 = pd.read_csv(ensemble2,index_col=0)
    ensemble2.columns = ensemble2.columns.str.lower()

    # better to ensure this is caught by pestpp-ies ensemble csvs
    unnamed1 = [col for col in ensemble1.columns if "unnamed:" in col]
    if len(unnamed1) != 0:
        ensemble1 = ensemble1.iloc[:,:-1] # ensure unnamed col result of poor csv read only (ie last col)
    unnamed2 = [col for col in ensemble2.columns if "unnamed:" in col]
    if len(unnamed2) != 0:
        ensemble2 = ensemble2.iloc[:,:-1] # ensure unnamed col result of poor csv read only (ie last col)

    d = set(ensemble1.columns).symmetric_difference(set(ensemble2. columns))

    if len(d) != 0:
        logger.lraise("ensemble1 does not have the same columns as ensemble2: {0}".
                      format(','.join(d)))

    en1_mn,en1_std = ensemble1.mean(axis=0),ensemble1.std(axis=0)
    en2_mn, en2_std = ensemble2.mean(axis=0), ensemble2.std(axis=0)

    mn_diff = 100.0 * ((en1_mn - en2_mn) / en1_mn)
    std_diff = 100 * (( en1_std - en2_std)/ en1_std)


    if "grouper" in kwargs:
        raise NotImplementedError()
    else:
        en_cols = set(ensemble1.columns)
        if len(en_cols.symmetric_difference(set(pst.par_names))) == 0:
            par = pst.parameter_data.loc[pst.adj_par_names,:]
            grouper = par.groupby(par.pargp).groups
            grouper["all"] = pst.adj_par_names
        elif len(en_cols.symmetric_difference(set(pst.obs_names))) == 0:
            obs = pst.observation_data.loc[pst.nnz_obs_names,:]
            grouper = obs.groupby(obs.obgnme).groups
            grouper["all"] = pst.nnz_obs_names
        else:
            logger.lraise("could not match ensemble cols with par or obs...")


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
        logger.log("plotting change for {0}".format(g))

        mn_g = mn_diff.loc[names]
        std_g = std_diff.loc[names]

        if mn_g.shape[0] == 0:
            logger.statement("no entries for group '{0}'".format(g))
            logger.log("plotting change for {0}".format(g))
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
        mn_g.hist(ax=ax,facecolor=facecolor,alpha=0.5,edgecolor=None,bins=bins)
        #std_g.hist(ax=ax,facecolor='b',alpha=0.5,edgecolor=None)



        #ax.set_xlim(xlim)
        ax.set_yticklabels([])
        ax.set_xlabel("mean percent change",labelpad=0.1)
        ax.set_title("{0}) mean change group:{1}, {2} entries".
                     format(abet[ax_count], g, mn_g.shape[0]), loc="left")
        ax.grid()
        ax_count += 1

        ax = axes[ax_count]
        std_g.hist(ax=ax, facecolor=facecolor, alpha=0.5, edgecolor=None, bins=bins)
        # std_g.hist(ax=ax,facecolor='b',alpha=0.5,edgecolor=None)



        # ax.set_xlim(xlim)
        ax.set_yticklabels([])
        ax.set_xlabel("sigma percent change", labelpad=0.1)
        ax.set_title("{0}) sigma change group:{1}, {2} entries".
                     format(abet[ax_count], g, mn_g.shape[0]), loc="left")
        ax.grid()
        ax_count += 1

        logger.log("plotting change for {0}".format(g))

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
        logger.log("plot ensemble change")
    else:
        logger.log("plot ensemble change")
        return figs



# def par_cov_helper(cov,pst,logger=None,filename=None,**kwargs):
#     assert isinstance(cov,pyemu.Cov)
#     if logger is None:
#         logger=Logger('Default_Loggger.log',echo=False)
#     logger.log("plot par cov")
#
#     par = pst.parameter_data
#     if "grouper" in kwargs:
#         raise NotImplementedError()
#     else:
#         grouper = par.groupby(par.pargp).groups
#
#     fig = plt.figure(figsize=figsize)
#     if "fig_title" in kwargs:
#         plt.figtext(0.5,0.5,kwargs["fig_title"])
#     else:
#         plt.figtext(0.5, 0.5, "pyemu.Pst.plot(kind='1to1')\nfrom pest control file '{0}'\n at {1}"
#                     .format(pst.filename, str(datetime.now())), ha="center")
#
#     figs = []
#     ax_count = 0
#     for g, names in grouper.items():
#         logger.log("plotting par cov for {0}".format(g))
#         if ax_count % (nr * nc) == 0:
#             if ax_count > 0:
#                 plt.tight_layout()
#             #pdf.savefig()
#             #plt.close(fig)
#             figs.append(fig)
#             fig = plt.figure(figsize=figsize)
#             axes = get_page_axes()
#             ax_count = 0
#
#         ax = axes[ax_count]
#         names = list(names)
#         cov_g = cov.get(row_names=names,col_names=names).x
#         ax.imshow(np.ma.masked_where(cov_g==0,cov_g))
#         ax.set_title("{0}) group:{1}, {2} elements".
#                      format(abet[ax_count], g, cov_g.shape[0]), loc="left")
#
#     plt.tight_layout()
#     # pdf.savefig()
#     # plt.close(fig)
#     figs.append(fig)
#     if filename is not None:
#         plt.tight_layout()
#         with PdfPages(filename) as pdf:
#             for fig in figs:
#                 pdf.savefig(fig)
#                 plt.close(fig)
#         logger.log("plot res_1to1")
#     else:
#         logger.log("plot res_1to1")
#         return figs

def plot_jac_test(csvin, csvout, targetobs=None, filetype=None, maxoutputpages=1, outputdirectory=None):
    """ helper function to plot results of the Jacobian test performed using the pest++ program sweep.

    Parameters
    ----------
    csvin:  string
            name of csv file used as input to sweep, typically developed with
            static method pyemu.helpers.build_jac_test_csv()
    csvout: string
            name of csv file with output generated by sweep, both input
            and output files can be specified in the pest++ control file
            with pyemu using:
            pest_object.pestpp_options["sweep_parameter_csv_file"] = jactest_in_file.csv
            pest_object.pestpp_options["sweep_output_csv_file"] = jactest_out_file.csv
    targetobs: list of strings
            list of observation file names to plot, each parameter used for jactest can
            have up to 32 observations plotted per page, throws a warning if more tha
            10 pages of output are requested per parameter. If none, all observations in
            the output csv file are used.
    filetype: string
            file type to store output, if None, plt.show() is called.
    maxoutputpages: int
            maximum number of pages of output per parameter.  Each page can
            hold up to 32 observation derivatives.  If value > 10, set it to
            10 and throw a warning.  If observations in targetobs > 32*maxoutputpages,
            then a random set is selected from the targetobs list (or all observations
            in the csv file if targetobs=None).
    outputdirectory: path, string, or None
            directory to store results, if None, current working directory is used.
            If string is passed, it is joined to the current working directory and
            created if needed. If os.path is passed, it is used directly.

    Returns
    -------
    None

    Note
    ----
    Used in conjunction with pyemu.helpers.build_jac_test_csv() and sweep to perform
    a Jacobian Test and then view the results. Can generate a lot of plots so easiest
    to put into a separate directory and view the files.

    Example
    -------
    ``>>> import pyemu``
    ``>>> pest_obj = pyemu.Pst(pest_control_file)``
    ``>>> jactest_df = pyemu.helpers.build_jac_test_csv(pst=pest_obj, num_steps=5)``
    ``>>> pest_obj.pestpp_options["sweep_parameter_csv_file"] = "jactest_in.csv"``
    ``>>> pest_obj.pestpp_options["sweep_output_csv_file"] = "jactest_out.csv"``
    ``>>> jactest_df.to_csv(os.path.join(home, "SWEEP", "jactest_in.csv")) ``
    ``>>> pest_obj.write("jactest.pst")
    ``>>> pyemu.helpers.run("{0} {1}".format(sweep_exe, "jactest.pst"), cwd='.', verbose=True)``
    ``>>> pyemu.plot.plot_jac_test("jactest_in.csv", "jactest_out.csv")
    """

    localhome = os.getcwd()
    # check if the output directory exists, if not make it
    if outputdirectory is not None and not os.path.exists(os.path.join(localhome, outputdirectory)):
        os.mkdir(os.path.join(localhome, outputdirectory))
    if outputdirectory is None:
        figures_dir = localhome
    else:
        figures_dir = os.path.join(localhome, outputdirectory)

    # read the input and output files into pandas dataframes
    jactest_in_df = pd.read_csv(csvin, engine='python', index_col=0)
    jactest_in_df.index.name = 'input_run_id'
    jactest_out_df = pd.read_csv(csvout, engine='python', index_col=1)

    # subtract the base run from every row, leaves the one parameter that
    # was perturbed in any row as only non-zero value. Set zeros to nan
    # so round-off doesn't get us and sum across rows to get a column of
    # the perturbation for each row, finally extract to a series. First
    # the input csv and then the output.
    base_par = jactest_in_df.loc['base']
    delta_par_df = jactest_in_df.subtract(base_par, axis='columns')
    delta_par_df.replace(0, np.nan, inplace=True)
    delta_par_df.drop('base', axis='index', inplace=True)
    delta_par_df['change'] = delta_par_df.sum(axis='columns')
    delta_par = pd.Series(delta_par_df['change'])

    base_obs = jactest_out_df.loc['base']
    delta_obs = jactest_out_df.subtract(base_obs)
    delta_obs.drop('base', axis='index', inplace=True)
    # if targetobs is None, then reset it to all the observations.
    if targetobs is None:
        targetobs = jactest_out_df.columns.tolist()[8:]
    delta_obs = delta_obs[targetobs]

    # get the Jacobian by dividing the change in observation by the change in parameter
    # for the perturbed parameters
    jacobian = delta_obs.divide(delta_par, axis='index')

    # use the index created by build_jac_test_csv to get a column of parameter names
    # and increments, then we can plot derivative vs. increment for each parameter
    extr_df = pd.Series(jacobian.index.values).str.extract("(.+)(_\d+$)", expand=True)
    extr_df[1] = pd.to_numeric(extr_df[1].str.replace('_', '')) + 1
    extr_df.rename(columns={0: 'parameter', 1: 'increment'}, inplace=True)
    extr_df.index = jacobian.index

    # make a dataframe for plotting the Jacobian by combining the parameter name
    # and increments frame with the Jacobian frame
    plotframe = pd.concat([extr_df, jacobian], axis=1, join='inner')

    # get a list of observations to keep based on maxoutputpages.
    if maxoutputpages > 10:
        print("WARNING, more than 10 pages of output requested per parameter")
        print("maxoutputpage reset to 10.")
        maxoutputpages=10
    num_obs_plotted = np.min(np.array([maxoutputpages*32, len(targetobs)]))
    if num_obs_plotted < len(targetobs):
        # get random sample
        obs_plotted = np.random.choice(len(targetobs), num_obs_plotted, replace=False)
        real_pages = maxoutputpages
    else:
        obs_plotted = targetobs
        real_pages = int(targetobs/32) + 1

    # make a subplot of derivative vs. increment one plot for each of the
    # observations in targetobs, and outputs grouped by parameter.
    for param, group in plotframe.groupby('parameter'):
        for page in range(0, real_pages):
            fig, axes = plt.subplots(8, 4, sharex=True, figsize=(10, 15))
            for row in range(0, 8):
                for col in range(0, 4):
                    count = 32 * page + 4 * row + col
                    if count < len(targetobs):
                        axes[row, col].scatter(group['increment'], group[obs_plotted[count]])
                        axes[row, col].plot(group['increment'], group[obs_plotted[count]], 'r')
                        axes[row, col].set_title(obs_plotted[count])
                        axes[row, col].set_xticks([1, 2, 3, 4, 5])
                        axes[row, col].tick_params(direction='in')
                        if row == 3:
                            axes[row, col].set_xlabel('Increment')
            plt.tight_layout()

            if filetype is None:
                plt.show()
            else:
                plt.savefig(os.path.join(figures_dir, "{0}_jactest_{1}.{2}".format(param, page, filetype)))
