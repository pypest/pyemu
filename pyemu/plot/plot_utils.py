"""Plotting functions for various PEST(++) and pyemu operations"""
import os
import numpy as np
import pandas as pd
import warnings
from datetime import datetime
import string
from pyemu.logger import Logger
from pyemu.pst import pst_utils
from ..pyemu_warnings import PyemuWarning

font = {"size": 6}
try:
    import matplotlib

    matplotlib.rc("font", **font)

    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.gridspec import GridSpec
except Exception as e:
    # raise Exception("error importing matplotlib: {0}".format(str(e)))
    warnings.warn("error importing matplotlib: {0}".format(str(e)), PyemuWarning)

import pyemu

figsize = (8, 10.5)
nr, nc = 4, 2
# page_gs = GridSpec(nr,nc)

abet = string.ascii_uppercase


def plot_summary_distributions(
    df,
    ax=None,
    label_post=False,
    label_prior=False,
    subplots=False,
    figsize=(11, 8.5),
    pt_color="b",
):
    """helper function to plot gaussian distrbutions from prior and posterior
    means and standard deviations

    Args:
        df (`pandas.DataFrame`): a dataframe and csv file.  Must have columns named:
            'prior_mean','prior_stdev','post_mean','post_stdev'.  If loaded
            from a csv file, column 0 is assumed to tbe the index
        ax (`atplotlib.pyplot.axis`): If None, and not subplots, then one is created
            and all distributions are plotted on a single plot
        label_post (`bool`): flag to add text labels to the peak of the posterior
        label_prior (`bool`): flag to add text labels to the peak of the prior
        subplots (`bool`): flag to use subplots.  If True, then 6 axes per page
            are used and a single prior and posterior is plotted on each
        figsize (`tuple`): matplotlib figure size

    Returns:
        tuple containing:

        - **[`matplotlib.figure`]**: list of figures
        - **[`matplotlib.axis`]**: list of axes

    Note:
        This is useful for demystifying FOSM results

        if subplots is False, a single axis is returned

    Example::

        import matplotlib.pyplot as plt
        import pyemu
        pyemu.plot_utils.plot_summary_distributions("pest.par.usum.csv")
        plt.show()

    """
    import matplotlib.pyplot as plt

    if isinstance(df, str):
        df = pd.read_csv(df, index_col=0)
    if ax is None and not subplots:
        fig = plt.figure(figsize=figsize)
        ax = plt.subplot(111)
        ax.grid()

    if "post_stdev" not in df.columns and "post_var" in df.columns:
        df.loc[:, "post_stdev"] = df.post_var.apply(np.sqrt)
    if "prior_stdev" not in df.columns and "prior_var" in df.columns:
        df.loc[:, "prior_stdev"] = df.prior_var.apply(np.sqrt)
    if "prior_expt" not in df.columns and "prior_mean" in df.columns:
        df.loc[:, "prior_expt"] = df.prior_mean
    if "post_expt" not in df.columns and "post_mean" in df.columns:
        df.loc[:, "post_expt"] = df.post_mean

    if subplots:
        fig = plt.figure(figsize=figsize)
        ax = plt.subplot(2, 3, 1)
        ax_per_page = 6
        ax_count = 0
        axes = []
        figs = []
    for name in df.index:
        x, y = gaussian_distribution(
            df.loc[name, "post_expt"], df.loc[name, "post_stdev"]
        )
        ax.fill_between(x, 0, y, facecolor=pt_color, edgecolor="none", alpha=0.25)
        if label_post:
            mx_idx = np.argmax(y)
            xtxt, ytxt = x[mx_idx], y[mx_idx] * 1.001
            ax.text(xtxt, ytxt, name, ha="center", alpha=0.5)

        x, y = gaussian_distribution(
            df.loc[name, "prior_expt"], df.loc[name, "prior_stdev"]
        )
        ax.plot(x, y, color="0.5", lw=3.0, dashes=(2, 1))
        if label_prior:
            mx_idx = np.argmax(y)
            xtxt, ytxt = x[mx_idx], y[mx_idx] * 1.001
            ax.text(xtxt, ytxt, name, ha="center", alpha=0.5)
        # ylim = list(ax.get_ylim())
        # ylim[1] *= 1.2
        # ylim[0] = 0.0
        # ax.set_ylim(ylim)
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
            ax = plt.subplot(2, 3, ax_count + 1)
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
    """get an x and y numpy.ndarray that spans the +/- 4
    standard deviation range of a gaussian distribution with
    a given mean and standard deviation. useful for plotting

    Args:
        mean (`float`): the mean of the distribution
        stdev (`float`): the standard deviation of the distribution
        num_pts (`int`): the number of points in the returned ndarrays.
            Default is 50

    Returns:
        tuple containing:

        - **numpy.ndarray**: the x-values of the distribution
        - **numpy.ndarray**: the y-values of the distribution

    Example::

        mean,std = 1.0, 2.0
        x,y = pyemu.plot.gaussian_distribution(mean,std)
        plt.fill_between(x,0,y)
        plt.show()


    """
    xstart = mean - (4.0 * stdev)
    xend = mean + (4.0 * stdev)
    x = np.linspace(xstart, xend, num_pts)
    y = (1.0 / np.sqrt(2.0 * np.pi * stdev * stdev)) * np.exp(
        -1.0 * ((x - mean) ** 2) / (2.0 * stdev * stdev)
    )
    return x, y


def pst_helper(pst, kind=None, **kwargs):
    """`pyemu.Pst` plot helper - takes the
    handoff from `pyemu.Pst.plot()`

    Args:
        kind (`str`): the kind of plot to make
        **kargs (`dict`): keyword arguments to pass to the
            plotting function and ultimately to `matplotlib`

    Returns:
        varies: usually a combination of `matplotlib.figure` (s) and/or
        `matplotlib.axis` (s)

    Example::

        pst = pyemu.Pst("pest.pst") #assumes pest.res or pest.rei is found
        pst.plot(kind="1to1")
        plt.show()
        pst.plot(kind="phipie")
        plt.show()
        pst.plot(kind="prior")
        plt.show()

    """

    echo = kwargs.get("echo", False)
    logger = pyemu.Logger("plot_pst_helper.log", echo=echo)
    logger.statement("plot_utils.pst_helper()")

    kinds = {
        "prior": pst_prior,
        "1to1": res_1to1,
        "phi_pie": res_phi_pie,
        "phi_progress": phi_progress,
    }

    if kind is None:
        returns = []
        base_filename = pst.filename
        if pst.new_filename is not None:
            base_filename = pst.new_filename
        base_filename = base_filename.replace(".pst", "")
        for name, func in kinds.items():
            plt_name = base_filename + "." + name + ".pdf"
            returns.append(func(pst, logger=logger, filename=plt_name))

        return returns
    elif kind not in kinds:
        logger.lraise(
            "unrecognized kind:{0}, should one of {1}".format(
                kind, ",".join(list(kinds.keys()))
            )
        )
    return kinds[kind](pst, logger, **kwargs)


def phi_progress(pst, logger=None, filename=None, **kwargs):
    """make plot of phi vs number of model runs - requires
    available  ".iobj" file generated by a PESTPP-GLM run.

    Args:
        pst (`pyemu.Pst`): a control file instance
        logger (`pyemu.Logger`):  if None, a generic one is created.  Default is None
        filename (`str`): PDF filename to save figures to.  If None, figures
            are returned.  Default is None
        kwargs (`dict`): optional keyword args to pass to plotting function

    Returns:
        `matplotlib.axis`: the axis the plot was made on

    Example::

        import matplotlib.pyplot as plt
        import pyemu
        pst = pyemu.Pst("my.pst")
        pyemu.plot_utils.phi_progress(pst)
        plt.show()

    """
    if logger is None:
        logger = Logger("Default_Loggger.log", echo=False)
    logger.log("plot phi_progress")

    iobj_file = pst.filename.replace(".pst", ".iobj")
    if not os.path.exists(iobj_file):
        logger.lraise("couldn't find iobj file {0}".format(iobj_file))
    df = pd.read_csv(iobj_file)
    if "ax" in kwargs:
        ax = kwargs["ax"]
    else:
        fig = plt.figure(figsize=figsize)
        ax = plt.subplot(1, 1, 1)
    ax.plot(df.model_runs_completed, df.total_phi, marker=".")
    ax.set_xlabel("model runs")
    ax.set_ylabel(r"$\phi$")
    ax.grid()
    if filename is not None:
        plt.savefig(filename)
    logger.log("plot phi_progress")
    return ax


def _get_page_axes(count=nr * nc):
    axes = [plt.subplot(nr, nc, i + 1) for i in range(min(count, nr * nc))]
    # [ax.set_yticks([]) for ax in axes]
    return axes


def res_1to1(
    pst, logger=None, filename=None, plot_hexbin=False, histogram=False, **kwargs
):
    """make 1-to-1 plots and also observed vs residual by observation group

    Args:
        pst (`pyemu.Pst`): a control file instance
        logger (`pyemu.Logger`):  if None, a generic one is created.  Default is None
        filename (`str`): PDF filename to save figures to.  If None, figures
            are returned.  Default is None
        hexbin (`bool`): flag to use the hexbinning for large numbers of residuals.
            Default is False
        histogram (`bool`): flag to plot residual histograms instead of obs vs residual.
            Default is False (use `matplotlib.pyplot.scatter` )
        kwargs (`dict`): optional keyword args to pass to plotting function

    Returns:
        `matplotlib.axis`: the axis the plot was made on

    Example::

        import matplotlib.pyplot as plt
        import pyemu
        pst = pyemu.Pst("my.pst")
        pyemu.plot_utils.phi_progress(pst)
        plt.show()

    """
    if logger is None:
        logger = Logger("Default_Loggger.log", echo=False)
    logger.log("plot res_1to1")

    if "ensemble" in kwargs:
        res = pst_utils.res_from_en(pst, kwargs["ensemble"])
        try:
            res = pst_utils.res_from_en(pst, kwargs["ensemble"])
        except Exception as e:
            logger.lraise("res_1to1: error loading ensemble file: {0}".format(str(e)))
    else:
        try:
            res = pst.res
        except:
            logger.lraise("res_phi_pie: pst.res is None, couldn't find residuals file")

    obs = pst.observation_data

    if "grouper" in kwargs:
        raise NotImplementedError()
    else:
        grouper = obs.groupby(obs.obgnme).groups

    fig = plt.figure(figsize=figsize)
    if "fig_title" in kwargs:
        plt.figtext(0.5, 0.5, kwargs["fig_title"])
    else:
        plt.figtext(
            0.5,
            0.5,
            "pyemu.Pst.plot(kind='1to1')\nfrom pest control file '{0}'\n at {1}".format(
                pst.filename, str(datetime.now())
            ),
            ha="center",
        )
    # if plot_hexbin:
    #    pdfname = pst.filename.replace(".pst", ".1to1.hexbin.pdf")
    # else:
    #    pdfname = pst.filename.replace(".pst", ".1to1.pdf")
    figs = []
    ax_count = 0
    for g, names in grouper.items():
        logger.log("plotting 1to1 for {0}".format(g))

        obs_g = obs.loc[names, :]
        obs_g.loc[:, "sim"] = res.loc[names, "modelled"]
        logger.statement("using control file obsvals to calculate residuals")
        obs_g.loc[:, "res"] = obs_g.sim - obs_g.obsval
        if "include_zero" not in kwargs or kwargs["include_zero"] is True:
            obs_g = obs_g.loc[obs_g.weight > 0, :]
        if obs_g.shape[0] == 0:
            logger.statement("no non-zero obs for group '{0}'".format(g))
            logger.log("plotting 1to1 for {0}".format(g))
            continue

        if ax_count % (nr * nc) == 0:
            if ax_count > 0:
                plt.tight_layout()
            # pdf.savefig()
            # plt.close(fig)
            figs.append(fig)
            fig = plt.figure(figsize=figsize)
            axes = _get_page_axes()
            ax_count = 0

        ax = axes[ax_count]

        # if obs_g.shape[0] == 1:
        #    ax.scatter(list(obs_g.sim),list(obs_g.obsval),marker='.',s=30,color='b')
        # else:
        mx = max(obs_g.obsval.max(), obs_g.sim.max())
        mn = min(obs_g.obsval.min(), obs_g.sim.min())

        # if obs_g.shape[0] == 1:
        mx *= 1.1
        mn *= 0.9
        ax.axis("square")
        if plot_hexbin:
            ax.hexbin(
                obs_g.obsval.values,
                obs_g.sim.values,
                mincnt=1,
                gridsize=(75, 75),
                extent=(mn, mx, mn, mx),
                bins="log",
                edgecolors=None,
            )
        #               plt.colorbar(ax=ax)
        else:
            ax.scatter([obs_g.obsval], [obs_g.sim], marker=".", s=10, color="b")

        ax.plot([mn, mx], [mn, mx], "k--", lw=1.0)
        xlim = (mn, mx)
        ax.set_xlim(mn, mx)
        ax.set_ylim(mn, mx)
        ax.grid()

        ax.set_xlabel("observed", labelpad=0.1)
        ax.set_ylabel("simulated", labelpad=0.1)
        ax.set_title(
            "{0}) group:{1}, {2} observations".format(
                abet[ax_count], g, obs_g.shape[0]
            ),
            loc="left",
        )

        ax_count += 1

        if histogram == False:
            ax = axes[ax_count]
            ax.scatter(obs_g.obsval, obs_g.res, marker=".", s=10, color="b")
            ylim = ax.get_ylim()
            mx = max(np.abs(ylim[0]), np.abs(ylim[1]))
            if obs_g.shape[0] == 1:
                mx *= 1.1
            ax.set_ylim(-mx, mx)
            # show a zero residuals line
            ax.plot(xlim, [0, 0], "k--", lw=1.0)
            meanres = obs_g.res.mean()
            # show mean residuals line
            ax.plot(xlim, [meanres, meanres], "r-", lw=1.0)
            ax.set_xlim(xlim)
            ax.set_ylabel("residual", labelpad=0.1)
            ax.set_xlabel("observed", labelpad=0.1)
            ax.set_title(
                "{0}) group:{1}, {2} observations".format(
                    abet[ax_count], g, obs_g.shape[0]
                ),
                loc="left",
            )
            ax.grid()
            ax_count += 1
        else:
            # need max and min res to set xlim, otherwise wonky figsize
            mxr = obs_g.res.max()
            mnr = obs_g.res.min()

            # if obs_g.shape[0] == 1:
            mxr *= 1.1
            mnr *= 0.9
            rlim = (mnr, mxr)

            ax = axes[ax_count]
            ax.hist(obs_g.res, bins=50, color="b")
            meanres = obs_g.res.mean()
            ax.axvline(meanres, color="r", lw=1)
            b, t = ax.get_ylim()
            ax.text(meanres + meanres / 10, t - t / 10, "Mean: {:.2f}".format(meanres))
            ax.set_xlim(rlim)
            ax.set_ylabel("count", labelpad=0.1)
            ax.set_xlabel("residual", labelpad=0.1)
            ax.set_title(
                "{0}) group:{1}, {2} observations".format(
                    abet[ax_count], g, obs_g.shape[0]
                ),
                loc="left",
            )
            ax.grid()
            ax_count += 1
        logger.log("plotting 1to1 for {0}".format(g))

    for a in range(ax_count, nr * nc):
        axes[a].set_axis_off()
        axes[a].set_yticks([])
        axes[a].set_xticks([])

    plt.tight_layout()
    # pdf.savefig()
    # plt.close(fig)
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


def plot_id_bar(id_df, nsv=None, logger=None, **kwargs):
    """Plot a stacked bar chart of identifiability based on
    a the `pyemu.ErrVar.get_identifiability()` dataframe

    Args:
        id_df (`pandas.DataFrame`) : dataframe of identifiability
        nsv (`int`): number of singular values to consider
        logger (`pyemu.Logger`, optonal): a logger.  If None, a generic
            one is created
        kwargs (`dict`): a dict of keyword arguments to pass to the
            plotting function

    Returns:
        `matplotlib.Axis`: the axis with the plot

    Example::

        import pyemu
        pest_obj = pyemu.Pst(pest_control_file)
        ev = pyemu.ErrVar(jco='freyberg_jac.jcb'))
        id_df = ev.get_identifiability_dataframe(singular_value=48)
        pyemu.plot_utils.plot_id_bar(id_df, nsv=12, figsize=(12,4)

    """
    if logger is None:
        logger = Logger("Default_Loggger.log", echo=False)
    logger.log("plot id bar")

    df = id_df.copy()

    # drop the final `ident` column
    if "ident" in df.columns:
        df.drop("ident", inplace=True, axis=1)

    if nsv is None or nsv > len(df.columns):
        nsv = len(df.columns)
        logger.log("set number of SVs and number in the dataframe")

    df = df[df.columns[:nsv]]

    df["ident"] = df.sum(axis=1)
    df.sort_values(by="ident", inplace=True, ascending=False)
    df.drop("ident", inplace=True, axis=1)

    if "figsize" in kwargs:
        figsize = kwargs["figsize"]
    else:
        figsize = (8, 10.5)
    if "ax" in kwargs:
        ax = kwargs["ax"]
    else:
        fig = plt.figure(figsize=figsize)
        ax = plt.subplot(1, 1, 1)

    # plto the stacked bar chart (the easy part!)
    df.plot.bar(stacked=True, cmap="jet_r", legend=False, ax=ax)

    #
    # horrible shenanigans to make a colorbar rather than a legend
    #

    # special case colormap just dark red if one SV
    if nsv == 1:
        tcm = matplotlib.colors.LinearSegmentedColormap.from_list(
            "one_sv", [plt.get_cmap("jet_r")(0)] * 2, N=2
        )
        sm = plt.cm.ScalarMappable(
            cmap=tcm, norm=matplotlib.colors.Normalize(vmin=0, vmax=nsv + 1)
        )
    # or typically just rock the jet_r colormap over the range of SVs
    else:
        sm = plt.cm.ScalarMappable(
            cmap=plt.get_cmap("jet_r"),
            norm=matplotlib.colors.Normalize(vmin=1, vmax=nsv),
        )
    sm._A = []

    # now, if too many ticks for the colorbar, summarize them
    if nsv < 20:
        ticks = range(1, nsv + 1)
    else:
        ticks = np.arange(1, nsv + 1, int((nsv + 1) / 30))

    cb = plt.colorbar(sm, ax=ax)
    cb.set_ticks(ticks)

    logger.log("plot id bar")

    return ax


def res_phi_pie(pst, logger=None, **kwargs):
    """plot current phi components as a pie chart.

    Args:
        pst (`pyemu.Pst`): a control file instance with the residual datafrane
            instance available.
        logger (`pyemu.Logger`): a logger.  If None, a generic one is created
        kwargs (`dict`): a dict of plotting options. Accepts 'include_zero'
            as a flag to include phi groups with only zero-weight obs (not
            sure why anyone would do this, but whatevs).

            Also accepts 'label_comps': list of components for the labels. Options are
            ['name', 'phi_comp', 'phi_percent']. Labels will use those three components
            in the order of the 'label_comps' list.

            Any additional
            args are passed to `matplotlib`.

    Returns:
        `matplotlib.Axis`: the axis with the plot.

    Example::

        import pyemu
        pst = pyemu.Pst("my.pst")
        pyemu.plot_utils.res_phi_pie(pst,figsize=(12,4))
        pyemu.plot_utils.res_phi_pie(pst,label_comps = ['name','phi_percent'], figsize=(12,4))


    """
    if logger is None:
        logger = Logger("Default_Loggger.log", echo=False)
    logger.log("plot res_phi_pie")

    if "ensemble" in kwargs:
        try:
            res = pst_utils.res_from_en(pst, kwargs["ensemble"])
        except:
            logger.statement(
                "res_1to1: could not find ensemble file {0}".format(kwargs["ensemble"])
            )
    else:
        try:
            res = pst.res
        except:
            logger.lraise("res_phi_pie: pst.res is None, couldn't find residuals file")

    obs = pst.observation_data
    phi = pst.phi
    phi_comps = pst.phi_components
    norm_phi_comps = pst.phi_components_normalized
    keys = list(phi_comps.keys())
    if "include_zero" not in kwargs or kwargs["include_zero"] is False:
        phi_comps = {k: phi_comps[k] for k in keys if phi_comps[k] > 0.0}
        keys = list(phi_comps.keys())
        norm_phi_comps = {k: norm_phi_comps[k] for k in keys}
    if "ax" in kwargs:
        ax = kwargs["ax"]
    else:
        fig = plt.figure(figsize=figsize)
        ax = plt.subplot(1, 1, 1, aspect="equal")

    if "label_comps" not in kwargs:
        labels = [
            "{0}\n{1:4G}\n({2:3.1f}%)".format(
                k, phi_comps[k], 100.0 * (phi_comps[k] / phi)
            )
            for k in keys
        ]
    else:
        # make sure the components for the labels are in a list
        if not isinstance(kwargs["label_comps"], list):
            fmtchoices = list([kwargs["label_comps"]])
        else:
            fmtchoices = kwargs["label_comps"]
        # assemble all possible label components
        labfmts = {
            "name": ["{}\n", keys],
            "phi_comp": ["{:4G}\n", [phi_comps[k] for k in keys]],
            "phi_percent": ["({:3.1f}%)", [100.0 * (phi_comps[k] / phi) for k in keys]],
        }
        if fmtchoices[0] == "phi_percent":
            labfmts["phi_percent"][0] = "{}\n".format(labfmts["phi_percent"][0])
        # make the string format
        labfmtstr = "".join([labfmts[k][0] for k in fmtchoices])
        # pull it together
        labels = [
            labfmtstr.format(*k) for k in zip(*[labfmts[j][1] for j in fmtchoices])
        ]

    ax.pie([float(norm_phi_comps[k]) for k in keys], labels=labels)
    logger.log("plot res_phi_pie")
    if "filename" in kwargs:
        plt.savefig(kwargs["filename"])
    return ax


def pst_prior(pst, logger=None, filename=None, **kwargs):
    """helper to plot prior parameter histograms implied by
    parameter bounds. Saves a multipage pdf named <case>.prior.pdf

    Args:
        pst (`pyemu.Pst`): control file
        logger (`pyemu.Logger`): a logger.  If None, a generic one is created.
        filename (`str`):  PDF filename to save plots to.
            If None, return figs without saving.  Default is None.
        kwargs (`dict`): additional plotting options. Accepts 'grouper' as
            dict to group parameters on to a single axis (use
            parameter groups if not passed),'unqiue_only' to only show unique
            mean-stdev combinations within a given group.  Any additional args
            are passed to `matplotlib`.

    Returns:
        [`matplotlib.Figure`]: a list of figures created.

    Example::

        pst = pyemu.Pst("pest.pst")
        pyemu.pst_utils.pst_prior(pst)
        plt.show()

    """
    if logger is None:
        logger = Logger("Default_Loggger.log", echo=False)
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
    info = par.loc[cov.names, :].copy()
    info.loc[:, "mean"] = mean
    info.loc[li, "mean"] = mean[li].apply(np.log10)
    logger.log("building mean parameter values")

    logger.log("building stdev parameter values")
    if cov.isdiagonal:
        std = cov.x.flatten()
    else:
        std = np.diag(cov.x)
    std = np.sqrt(std)
    info.loc[:, "prior_std"] = std

    logger.log("building stdev parameter values")

    if std.shape != mean.shape:
        logger.lraise("mean.shape {0} != std.shape {1}".format(mean.shape, std.shape))

    if "grouper" in kwargs:
        raise NotImplementedError()
        # check for consistency here

    else:
        par_adj = par.loc[par.partrans.apply(lambda x: x in ["log", "none"]), :]
        grouper = par_adj.groupby(par_adj.pargp).groups
        # grouper = par.groupby(par.pargp).groups

    if len(grouper) == 0:
        raise Exception("no adustable parameters to plot")

    fig = plt.figure(figsize=figsize)
    if "fig_title" in kwargs:
        plt.figtext(0.5, 0.5, kwargs["fig_title"])
    else:
        plt.figtext(
            0.5,
            0.5,
            "pyemu.Pst.plot(kind='prior')\nfrom pest control file '{0}'\n at {1}".format(
                pst.filename, str(datetime.now())
            ),
            ha="center",
        )
    figs = []
    ax_count = 0
    grps_names = list(grouper.keys())
    grps_names.sort()
    for g in grps_names:
        names = grouper[g]
        logger.log("plotting priors for {0}".format(",".join(list(names))))
        if ax_count % (nr * nc) == 0:
            plt.tight_layout()
            # pdf.savefig()
            # plt.close(fig)
            figs.append(fig)
            fig = plt.figure(figsize=figsize)
            axes = _get_page_axes()
            ax_count = 0

        islog = False
        vc = info.partrans.value_counts()
        if vc.shape[0] > 1:
            logger.warn("mixed partrans for group {0}".format(g))
        elif "log" in vc.index:
            islog = True
        ax = axes[ax_count]
        if "unique_only" in kwargs and kwargs["unique_only"]:

            ms = (
                info.loc[names, :]
                .apply(lambda x: (x["mean"], x["prior_std"]), axis=1)
                .unique()
            )
            for (m, s) in ms:
                x, y = gaussian_distribution(m, s)
                ax.fill_between(x, 0, y, facecolor="0.5", alpha=0.5, edgecolor="none")

        else:
            for m, s in zip(info.loc[names, "mean"], info.loc[names, "prior_std"]):
                x, y = gaussian_distribution(m, s)
                ax.fill_between(x, 0, y, facecolor="0.5", alpha=0.5, edgecolor="none")
        ax.set_title(
            "{0}) group:{1}, {2} parameters".format(abet[ax_count], g, names.shape[0]),
            loc="left",
        )

        ax.set_yticks([])
        if islog:
            ax.set_xlabel("$log_{10}$ parameter value", labelpad=0.1)
        else:
            ax.set_xlabel("parameter value", labelpad=0.1)
        logger.log("plotting priors for {0}".format(",".join(list(names))))

        ax_count += 1

    for a in range(ax_count, nr * nc):
        axes[a].set_axis_off()
        axes[a].set_yticks([])
        axes[a].set_xticks([])

    plt.tight_layout()
    # pdf.savefig()
    # plt.close(fig)
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


def ensemble_helper(
    ensemble,
    bins=10,
    facecolor="0.5",
    plot_cols=None,
    filename=None,
    func_dict=None,
    sync_bins=True,
    deter_vals=None,
    std_window=None,
    deter_range=False,
    **kwargs
):
    """helper function to plot ensemble histograms

    Args:
        ensemble : varies
            the ensemble argument can be a pandas.DataFrame or derived type or a str, which
            is treated as a filename.  Optionally, ensemble can be a list of these types or
            a dict, in which case, the keys are treated as facecolor str (e.g., 'b', 'y', etc).
        facecolor : str
            the histogram facecolor.  Only applies if ensemble is a single thing
        plot_cols : enumerable
            a collection of columns (in form of a list of parameters, or a dict with keys for
            parsing plot axes and values of parameters) from the ensemble(s) to plot.  If None,
            (the union of) all cols are plotted. Default is None
        filename : str
            the name of the pdf to create.  If None, return figs without saving.  Default is None.
        func_dict : dict
            a dictionary of unary functions (e.g., `np.log10` to apply to columns.  Key is
            column name.  Default is None
        sync_bins : bool
            flag to use the same bin edges for all ensembles. Only applies if more than
            one ensemble is being plotted.  Default is True
        deter_vals : dict
            dict of deterministic values to plot as a vertical line. key is ensemble columnn name
        std_window : float
            the number of standard deviations around the mean to mark as vertical lines.  If None,
            nothing happens.  Default is None
        deter_range : bool
            flag to set xlims to deterministic value +/- std window.  If True, std_window must not be None.
            Default is False

    Example::

        # plot prior and posterior par ensembles
        pst = pyemu.Pst("my.pst")
        prior = pyemu.ParameterEnsemble.from_binary(pst=pst, filename="prior.jcb")
        post = pyemu.ParameterEnsemble.from_binary(pst=pst, filename="my.3.par.jcb")
        pyemu.plot_utils.ensemble_helper(ensemble={"0.5":prior, "b":post},filename="ensemble.pdf")

        #plot prior and posterior simulated equivalents to observations with obs noise and obs vals
        pst = pyemu.Pst("my.pst")
        prior = pyemu.ObservationEnsemble.from_binary(pst=pst, filename="my.0.obs.jcb")
        post = pyemu.ObservationEnsemble.from_binary(pst=pst, filename="my.3.obs.jcb")
        noise = pyemu.ObservationEnsemble.from_binary(pst=pst, filename="my.obs+noise.jcb")
        pyemu.plot_utils.ensemble_helper(ensemble={"0.5":prior, "b":post,"r":noise},
                                         filename="ensemble.pdf",
                                         deter_vals=pst.observation_data.obsval.to_dict())


    """
    logger = pyemu.Logger("ensemble_helper.log")
    logger.log("pyemu.plot_utils.ensemble_helper()")
    ensembles = _process_ensemble_arg(ensemble, facecolor, logger)
    if len(ensembles) == 0:
        raise Exception("plot_uitls.ensemble_helper() error processing `ensemble` arg")
    # apply any functions
    if func_dict is not None:
        logger.log("applying functions")
        for col, func in func_dict.items():
            for fc, en in ensembles.items():
                if col in en.columns:
                    en.loc[:, col] = en.loc[:, col].apply(func)
        logger.log("applying functions")

    # get a list of all cols (union)
    all_cols = set()
    for fc, en in ensembles.items():
        cols = set(en.columns)
        all_cols.update(cols)
    if plot_cols is None:
        plot_cols = {i: [v] for i, v in (zip(all_cols, all_cols))}
    else:
        if isinstance(plot_cols, list):
            splot_cols = set(plot_cols)
            plot_cols = {i: [v] for i, v in (zip(plot_cols, plot_cols))}
        elif isinstance(plot_cols, dict):
            splot_cols = []
            for label, pcols in plot_cols.items():
                splot_cols.extend(list(pcols))
            splot_cols = set(splot_cols)
        else:
            logger.lraise(
                "unrecognized plot_cols type: {0}, should be list or dict".format(
                    type(plot_cols)
                )
            )

        missing = splot_cols - all_cols
        if len(missing) > 0:
            logger.lraise(
                "the following plot_cols are missing: {0}".format(",".join(missing))
            )

    logger.statement("plotting {0} histograms".format(len(plot_cols)))

    fig = plt.figure(figsize=figsize)
    if "fig_title" in kwargs:
        plt.figtext(0.5, 0.5, kwargs["fig_title"])
    else:
        plt.figtext(
            0.5,
            0.5,
            "pyemu.plot_utils.ensemble_helper()\n at {0}".format(str(datetime.now())),
            ha="center",
        )
    # plot_cols = list(plot_cols)
    # plot_cols.sort()
    labels = list(plot_cols.keys())
    labels.sort()
    logger.statement("saving pdf to {0}".format(filename))
    figs = []

    ax_count = 0

    # for label,plot_col in plot_cols.items():
    for label in labels:
        plot_col = plot_cols[label]
        logger.log("plotting reals for {0}".format(label))
        if ax_count % (nr * nc) == 0:
            plt.tight_layout()
            # pdf.savefig()
            # plt.close(fig)
            figs.append(fig)
            fig = plt.figure(figsize=figsize)
            axes = _get_page_axes()
            [ax.set_yticks([]) for ax in axes]
            ax_count = 0

        ax = axes[ax_count]

        if sync_bins:
            mx, mn = -1.0e30, 1.0e30
            for fc, en in ensembles.items():
                # for pc in plot_col:
                #     if pc in en.columns:
                #         emx,emn = en.loc[:,pc].max(),en.loc[:,pc].min()
                #         mx = max(mx,emx)
                #         mn = min(mn,emn)
                emn = np.nanmin(en.loc[:, plot_col].values)
                emx = np.nanmax(en.loc[:, plot_col].values)
                mx = max(mx, emx)
                mn = min(mn, emn)
            if mx == -1.0e30 and mn == 1.0e30:
                logger.warn("all NaNs for label: {0}".format(label))
                ax.set_title(
                    "{0}) {1}, count:{2} - all NaN".format(
                        abet[ax_count], label, len(plot_col)
                    ),
                    loc="left",
                )
                ax.set_yticks([])
                ax.set_xticks([])
                ax_count += 1
                continue
            plot_bins = np.linspace(mn, mx, num=bins)
            logger.statement("{0} min:{1:5G}, max:{2:5G}".format(label, mn, mx))
        else:
            plot_bins = bins
        for fc, en in ensembles.items():
            # for pc in plot_col:
            #    if pc in en.columns:
            #        try:
            #            en.loc[:,pc].hist(bins=plot_bins,facecolor=fc,
            #                                    edgecolor="none",alpha=0.5,
            #                                    density=True,ax=ax)
            #        except Exception as e:
            #            logger.warn("error plotting histogram for {0}:{1}".
            #                        format(pc,str(e)))
            vals = en.loc[:, plot_col].values.flatten()
            # print(plot_bins)
            # print(vals)

            ax.hist(
                vals,
                bins=plot_bins,
                edgecolor="none",
                alpha=0.5,
                density=True,
                facecolor=fc,
            )
            v = None
            if deter_vals is not None:
                for pc in plot_col:
                    if pc in deter_vals:
                        ylim = ax.get_ylim()
                        v = deter_vals[pc]
                        ax.plot([v, v], ylim, "k--", lw=1.5)
                        ax.set_ylim(ylim)

            if std_window is not None:
                try:
                    ylim = ax.get_ylim()
                    mn, st = (
                        en.loc[:, pc].mean(),
                        en.loc[:, pc].std() * (std_window / 2.0),
                    )

                    ax.plot([mn - st, mn - st], ylim, color=fc, lw=1.5, ls="--")
                    ax.plot([mn + st, mn + st], ylim, color=fc, lw=1.5, ls="--")
                    ax.set_ylim(ylim)
                    if deter_range and v is not None:
                        xmn = v - st
                        xmx = v + st
                        ax.set_xlim(xmn, xmx)
                except:
                    logger.warn("error plotting std window for {0}".format(pc))
        ax.grid()
        if len(ensembles) > 1:
            ax.set_title(
                "{0}) {1}, count: {2}".format(abet[ax_count], label, len(plot_col)),
                loc="left",
            )
        else:
            ax.set_title(
                "{0}) {1}, count:{2}\nmin:{3:3.1E}, max:{4:3.1E}".format(
                    abet[ax_count],
                    label,
                    len(plot_col),
                    np.nanmin(vals),
                    np.nanmax(vals),
                ),
                loc="left",
            )
        ax_count += 1

    for a in range(ax_count, nr * nc):
        axes[a].set_axis_off()
        axes[a].set_yticks([])
        axes[a].set_xticks([])

    plt.tight_layout()
    # pdf.savefig()
    # plt.close(fig)
    figs.append(fig)
    if filename is not None:
        plt.tight_layout()
        with PdfPages(filename) as pdf:
            for fig in figs:
                pdf.savefig(fig)
                plt.close(fig)
    logger.log("pyemu.plot_utils.ensemble_helper()")


def ensemble_change_summary(
    ensemble1,
    ensemble2,
    pst,
    bins=10,
    facecolor="0.5",
    logger=None,
    filename=None,
    **kwargs
):
    """helper function to plot first and second moment change histograms between two
    ensembles

    Args:
        ensemble1 (varies): filename or `pandas.DataFrame` or `pyemu.Ensemble`
        ensemble2 (varies): filename or `pandas.DataFrame` or `pyemu.Ensemble`
        pst (`pyemu.Pst`): control file
        facecolor (`str`): the histogram facecolor.
        filename (`str`): the name of the multi-pdf to create. If None, return figs without saving.  Default is None.

    Returns:
        [`matplotlib.Figure`]: a list of figures.  Returns None is
        `filename` is not None

    Example::

        pst = pyemu.Pst("my.pst")
        prior = pyemu.ParameterEnsemble.from_binary(pst=pst, filename="prior.jcb")
        post = pyemu.ParameterEnsemble.from_binary(pst=pst, filename="my.3.par.jcb")
        pyemu.plot_utils.ensemble_change_summary(prior,post)
        plt.show()


    """
    if logger is None:
        logger = Logger("Default_Loggger.log", echo=False)
    logger.log("plot ensemble change")

    if isinstance(ensemble1, str):
        ensemble1 = pd.read_csv(ensemble1, index_col=0)
    ensemble1.columns = ensemble1.columns.str.lower()

    if isinstance(ensemble2, str):
        ensemble2 = pd.read_csv(ensemble2, index_col=0)
    ensemble2.columns = ensemble2.columns.str.lower()

    # better to ensure this is caught by pestpp-ies ensemble csvs
    unnamed1 = [col for col in ensemble1.columns if "unnamed:" in col]
    if len(unnamed1) != 0:
        ensemble1 = ensemble1.iloc[
            :, :-1
        ]  # ensure unnamed col result of poor csv read only (ie last col)
    unnamed2 = [col for col in ensemble2.columns if "unnamed:" in col]
    if len(unnamed2) != 0:
        ensemble2 = ensemble2.iloc[
            :, :-1
        ]  # ensure unnamed col result of poor csv read only (ie last col)

    d = set(ensemble1.columns).symmetric_difference(set(ensemble2.columns))

    if len(d) != 0:
        logger.lraise(
            "ensemble1 does not have the same columns as ensemble2: {0}".format(
                ",".join(d)
            )
        )
    if "grouper" in kwargs:
        raise NotImplementedError()
    else:
        en_cols = set(ensemble1.columns)
        if len(en_cols.difference(set(pst.par_names))) == 0:
            par = pst.parameter_data.loc[en_cols, :]
            grouper = par.groupby(par.pargp).groups
            grouper["all"] = pst.adj_par_names
            li = par.loc[par.partrans == "log", "parnme"]
            ensemble1.loc[:, li] = ensemble1.loc[:, li].apply(np.log10)
            ensemble2.loc[:, li] = ensemble2.loc[:, li].apply(np.log10)
        elif len(en_cols.difference(set(pst.obs_names))) == 0:
            obs = pst.observation_data.loc[en_cols, :]
            grouper = obs.groupby(obs.obgnme).groups
            grouper["all"] = pst.nnz_obs_names
        else:
            logger.lraise("could not match ensemble cols with par or obs...")

    en1_mn, en1_std = ensemble1.mean(axis=0), ensemble1.std(axis=0)
    en2_mn, en2_std = ensemble2.mean(axis=0), ensemble2.std(axis=0)

    # mn_diff = 100.0 * ((en1_mn - en2_mn) / en1_mn)
    # std_diff = 100 * ((en1_std - en2_std) / en1_std)

    mn_diff = -1 * (en2_mn - en1_mn)
    std_diff = 100 * (((en1_std - en2_std) / en1_std))
    # set en1_std==0 to nan
    # std_diff[en1_std.index[en1_std==0]] = np.nan

    # diff = ensemble1 - ensemble2
    # mn_diff = diff.mean(axis=0)
    # std_diff = diff.std(axis=0)

    fig = plt.figure(figsize=figsize)
    if "fig_title" in kwargs:
        plt.figtext(0.5, 0.5, kwargs["fig_title"])
    else:
        plt.figtext(
            0.5,
            0.5,
            "pyemu.Pst.plot(kind='1to1')\nfrom pest control file '{0}'\n at {1}".format(
                pst.filename, str(datetime.now())
            ),
            ha="center",
        )
    # if plot_hexbin:
    #    pdfname = pst.filename.replace(".pst", ".1to1.hexbin.pdf")
    # else:
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
            # pdf.savefig()
            # plt.close(fig)
            figs.append(fig)
            fig = plt.figure(figsize=figsize)
            axes = _get_page_axes()
            ax_count = 0

        ax = axes[ax_count]
        mn_g.hist(ax=ax, facecolor=facecolor, alpha=0.5, edgecolor=None, bins=bins)
        # mx = max(mn_g.max(), mn_g.min(),np.abs(mn_g.max()),np.abs(mn_g.min())) * 1.2
        # ax.set_xlim(-mx,mx)

        # std_g.hist(ax=ax,facecolor='b',alpha=0.5,edgecolor=None)

        # ax.set_xlim(xlim)
        ax.set_yticklabels([])
        ax.set_xlabel("mean change", labelpad=0.1)
        ax.set_title(
            "{0}) mean change group:{1}, {2} entries\nmax:{3:10G}, min:{4:10G}".format(
                abet[ax_count], g, mn_g.shape[0], mn_g.max(), mn_g.min()
            ),
            loc="left",
        )
        ax.grid()
        ax_count += 1

        ax = axes[ax_count]
        std_g.hist(ax=ax, facecolor=facecolor, alpha=0.5, edgecolor=None, bins=bins)
        # std_g.hist(ax=ax,facecolor='b',alpha=0.5,edgecolor=None)

        # ax.set_xlim(xlim)
        ax.set_yticklabels([])
        ax.set_xlabel("sigma percent reduction", labelpad=0.1)
        ax.set_title(
            "{0}) sigma change group:{1}, {2} entries\nmax:{3:10G}, min:{4:10G}".format(
                abet[ax_count], g, mn_g.shape[0], std_g.max(), std_g.min()
            ),
            loc="left",
        )
        ax.grid()
        ax_count += 1

        logger.log("plotting change for {0}".format(g))

    for a in range(ax_count, nr * nc):
        axes[a].set_axis_off()
        axes[a].set_yticks([])
        axes[a].set_xticks([])

    plt.tight_layout()
    # pdf.savefig()
    # plt.close(fig)
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


def _process_ensemble_arg(ensemble, facecolor, logger):
    """private method to work out ensemble plot args"""
    ensembles = {}
    if isinstance(ensemble, pd.DataFrame) or isinstance(ensemble, pyemu.Ensemble):
        if not isinstance(facecolor, str):
            logger.lraise("facecolor must be str")
        ensembles[facecolor] = ensemble
    elif isinstance(ensemble, str):
        if not isinstance(facecolor, str):
            logger.lraise("facecolor must be str")

        logger.log("loading ensemble from csv file {0}".format(ensemble))
        en = pd.read_csv(ensemble, index_col=0)
        logger.statement("{0} shape: {1}".format(ensemble, en.shape))
        ensembles[facecolor] = en
        logger.log("loading ensemble from csv file {0}".format(ensemble))

    elif isinstance(ensemble, list):
        if isinstance(facecolor, list):
            if len(ensemble) != len(facecolor):
                logger.lraise("facecolor list len != ensemble list len")
        else:
            colors = ["m", "c", "b", "r", "g", "y"]

            facecolor = [colors[i] for i in range(len(ensemble))]
        ensembles = {}
        for fc, en_arg in zip(facecolor, ensemble):
            if isinstance(en_arg, str):
                logger.log("loading ensemble from csv file {0}".format(en_arg))
                en = pd.read_csv(en_arg, index_col=0)
                logger.log("loading ensemble from csv file {0}".format(en_arg))
                logger.statement("ensemble {0} gets facecolor {1}".format(en_arg, fc))

            elif isinstance(en_arg, pd.DataFrame) or isinstance(en_arg, pyemu.Ensemble):
                en = en_arg
            else:
                logger.lraise("unrecognized ensemble list arg:{0}".format(en_arg))
            ensembles[fc] = en

    elif isinstance(ensemble, dict):
        for fc, en_arg in ensemble.items():
            if isinstance(en_arg, pd.DataFrame) or isinstance(en_arg, pyemu.Ensemble):
                ensembles[fc] = en_arg
            elif isinstance(en_arg, str):
                logger.log("loading ensemble from csv file {0}".format(en_arg))
                en = pd.read_csv(en_arg, index_col=0)
                logger.log("loading ensemble from csv file {0}".format(en_arg))
                ensembles[fc] = en
            else:
                logger.lraise("unrecognized ensemble list arg:{0}".format(en_arg))
    try:
        for fc in ensembles:
            ensembles[fc].columns = ensembles[fc].columns.str.lower()
    except:
        logger.lraise("error processing ensemble")

    return ensembles


def ensemble_res_1to1(
    ensemble,
    pst,
    facecolor="0.5",
    logger=None,
    filename=None,
    skip_groups=[],
    base_ensemble=None,
    **kwargs
):
    """helper function to plot ensemble 1-to-1 plots showing the simulated range

    Args:
        ensemble (varies):  the ensemble argument can be a pandas.DataFrame or derived type or a str, which
            is treated as a fileanme.  Optionally, ensemble can be a list of these types or
            a dict, in which case, the keys are treated as facecolor str (e.g., 'b', 'y', etc).
        pst (`pyemu.Pst`): a control file instance
        facecolor (`str`): the histogram facecolor.  Only applies if `ensemble` is a single thing
        filename (`str`): the name of the pdf to create. If None, return figs
            without saving.  Default is None.
        base_ensemble (`varies`): an optional ensemble argument for the observations + noise ensemble.
            This will be plotted as a transparent red bar on the 1to1 plot.

    Note:

        the vertical bar on each plot the min-max range

    Example::


        pst = pyemu.Pst("my.pst")
        prior = pyemu.ObservationEnsemble.from_binary(pst=pst, filename="my.0.obs.jcb")
        post = pyemu.ObservationEnsemble.from_binary(pst=pst, filename="my.3.obs.jcb")
        pyemu.plot_utils.ensemble_res_1to1(ensemble={"0.5":prior, "b":post})
        plt.show()

    """
    def _get_plotlims(oen, ben, obsnames):
        if not isinstance(oen, dict):
            oen = {'g': oen.loc[:, obsnames]}
        if not isinstance(ben, dict):
            ben = {'g': ben.get(obsnames)}
        outofrange = False
        # work back from crazy values
        oemin = 1e32
        oemeanmin = 1e32
        oemax = -1e32
        oemeanmax = -1e32
        bemin = 1e32
        bemeanmin = 1e32
        bemax = -1e32
        bemeanmax = -1e32
        for _, oeni in oen.items():  # loop over ensembles
            oeni = oeni.loc[:, obsnames]  # slice group obs
            oemin = np.min([oemin, oeni.min().min()])
            oemax = np.max([oemax, oeni.max().max()])
            # get min and max of mean sim vals
            # (incase we want plot to ignore extremes)
            oemeanmin = np.min([oemeanmin, oeni.mean().min()])
            oemeanmax = np.max([oemeanmax, oeni.mean().max()])
        for _, beni in ben.items():  # same with base ensemble/obsval
            # work with either ensemble or obsval series
            beni = beni.get(obsnames)
            bemin = np.min([bemin, beni.min().min()])
            bemax = np.max([bemax, beni.max().max()])
            bemeanmin = np.min([bemeanmin, beni.mean().min()])
            bemeanmax = np.max([bemeanmax, beni.mean().max()])
        # get base ensemble range
        berange = bemax-bemin
        if berange == 0.:  # only one obs in group (probs)
            berange = bemeanmax * 1.1  # expand a little
        # add buffer to obs endpoints
        bemin = bemin - (berange*0.05)
        bemax = bemax + (berange*0.05)
        if oemax < bemin:  # sim well below obs
            oemin = oemeanmin  # set min to mean min
            # (sim captured but not extremes)
            outofrange = True
        if oemin > bemax:  # sim well above obs
            oemax = oemeanmax
            outofrange = True
        oerange = oemax - oemin
        if bemax > oemax + (0.1*oerange):  # obs max well above sim
            if not outofrange:  # but sim still in range
                # zoom to sim
                bemax = oemax + (0.1*oerange)
            else:  # use obs mean max
                bemax = bemeanmax
        if bemin < oemin - (0.1 * oerange):  # obs min well below sim
            if not outofrange:  # but sim still in range
                # zoom to sim
                bemin = oemin - (0.1 * oerange)
            else:
                bemin = bemeanmin
        pmin = np.min([oemin, bemin])
        pmax = np.max([oemax, bemax])
        return pmin, pmax


    if logger is None:
        logger = Logger("Default_Loggger.log", echo=False)
    logger.log("plot res_1to1")
    obs = pst.observation_data
    ensembles = _process_ensemble_arg(ensemble, facecolor, logger)

    if base_ensemble is not None:
        base_ensemble = _process_ensemble_arg(base_ensemble, "r", logger)

    if "grouper" in kwargs:
        raise NotImplementedError()
    else:
        grouper = obs.groupby(obs.obgnme).groups
        for skip_group in skip_groups:
            grouper.pop(skip_group)

    fig = plt.figure(figsize=figsize)
    if "fig_title" in kwargs:
        plt.figtext(0.5, 0.5, kwargs["fig_title"])
    else:
        plt.figtext(
            0.5,
            0.5,
            "pyemu.Pst.plot(kind='1to1')\nfrom pest control file '{0}'\n at {1}".format(
                pst.filename, str(datetime.now())
            ),
            ha="center",
        )

    figs = []
    ax_count = 0
    for g, names in grouper.items():
        logger.log("plotting 1to1 for {0}".format(g))
        # control file observation for group
        obs_g = obs.loc[names, :]
        # normally only look a non-zero weighted obs
        if "include_zero" not in kwargs or kwargs["include_zero"] is False:
            obs_g = obs_g.loc[obs_g.weight > 0, :]
        if obs_g.shape[0] == 0:
            logger.statement("no non-zero obs for group '{0}'".format(g))
            logger.log("plotting 1to1 for {0}".format(g))
            continue
        # if the first axis in page
        if ax_count % (nr * nc) == 0:
            if ax_count > 0:
                plt.tight_layout()
            figs.append(fig)
            fig = plt.figure(figsize=figsize)
            axes = _get_page_axes()
            ax_count = 0
        ax = axes[ax_count]

        if base_ensemble is None:
            # if obs not defined by obs+noise ensemble,
            # use min and max for obsval from control file
            pmin, pmax = _get_plotlims(ensembles, obs_g.obsval, obs_g.obsnme)
        else:
            # if obs defined by obs+noise use obs+noise min and max
            pmin, pmax = _get_plotlims(ensembles, base_ensemble, obs_g.obsnme)
            obs_gg = obs_g.sort_values(by="obsval")
            for c, en in base_ensemble.items():
                en_g = en.loc[:, obs_gg.obsnme]
                ex = en_g.max()
                en = en_g.min()
                # update y min and max for obs+noise ensembles
                if len(obs_gg.obsval) > 1:
                    ax.fill_between(obs_gg.obsval, en, ex, facecolor=c, alpha=0.2, zorder=2)
                else:
                    ax.plot([obs_gg.obsval, obs_gg.obsval], [en, ex], color=c, alpha=0.2, zorder=2)
        for c, en in ensembles.items():
            en_g = en.loc[:, obs_g.obsnme]
            # output mins and maxs
            ex = en_g.max()
            en = en_g.min()
            [
                ax.plot([ov, ov], [een, eex], color=c, zorder=1)
                for ov, een, eex in zip(obs_g.obsval.values, en.values, ex.values)
            ]
        ax.plot([pmin, pmax], [pmin, pmax], "k--", lw=1.0, zorder=3)
        xlim = (pmin, pmax)
        ax.set_xlim(pmin, pmax)
        ax.set_ylim(pmin, pmax)

        if max(np.abs(xlim)) > 1.0e5:
            ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%1.0e"))
            ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%1.0e"))
        ax.grid()

        ax.set_xlabel("observed", labelpad=0.1)
        ax.set_ylabel("simulated", labelpad=0.1)
        ax.set_title(
            "{0}) group:{1}, {2} observations".format(
                abet[ax_count], g, obs_g.shape[0]
            ),
            loc="left",
        )

        # Residual (RHS plot)
        ax_count += 1
        ax = axes[ax_count]
        # ax.scatter(obs_g.obsval, obs_g.res, marker='.', s=10, color='b')

        if base_ensemble is not None:
            obs_gg = obs_g.sort_values(by="obsval")
            for c, en in base_ensemble.items():
                en_g = en.loc[:, obs_gg.obsnme].subtract(obs_gg.obsval)
                ex = en_g.max()
                en = en_g.min()
                if len(obs_gg.obsval) > 1:
                    ax.fill_between(obs_gg.obsval, en, ex, facecolor=c, alpha=0.2, zorder=2)
                else:
                    # [ax.plot([ov, ov], [een, eex], color=c,alpha=0.3) for ov, een, eex in zip(obs_g.obsval.values, en.values, ex.values)]
                    ax.plot([obs_gg.obsval, obs_gg.obsval], [en, ex], color=c,
                            alpha=0.2, zorder=2)
        omn = []
        omx = []
        for c, en in ensembles.items():
            en_g = en.loc[:, obs_g.obsnme].subtract(obs_g.obsval, axis=1)
            ex = en_g.max()
            en = en_g.min()
            omn.append(en)
            omx.append(ex)
            [
                ax.plot([ov, ov], [een, eex], color=c, zorder=1)
                for ov, een, eex in zip(obs_g.obsval.values, en.values, ex.values)
            ]

        omn = pd.concat(omn).min()
        omx = pd.concat(omx).max()
        mx = max(np.abs(omn), np.abs(omx))  # ensure symmetric about y=0
        if obs_g.shape[0] == 1:
            mx *= 1.05
        else:
            mx *= 1.02
        if np.sign(omn) == np.sign(omx):
            # allow y axis asymm if all above or below
            mn = np.min([0, np.sign(omn) * mx])
            mx = np.max([0, np.sign(omn) * mx])
        else:
            mn = -mx
        ax.set_ylim(mn, mx)
        bmin = obs_g.obsval.values.min()
        bmax = obs_g.obsval.values.max()
        brange = (bmax - bmin)
        if brange == 0.:
            brange = obs_g.obsval.values.mean()
        bmin = bmin - 0.1*brange
        bmax = bmax + 0.1*brange
        xlim = (bmin, bmax)
        # show a zero residuals line
        ax.plot(xlim, [0, 0], "k--", lw=1.0, zorder=3)

        ax.set_xlim(xlim)
        ax.set_ylabel("residual", labelpad=0.1)
        ax.set_xlabel("observed", labelpad=0.1)
        ax.set_title(
            "{0}) group:{1}, {2} observations".format(
                abet[ax_count], g, obs_g.shape[0]
            ),
            loc="left",
        )
        ax.grid()
        if ax.get_xlim()[1] > 1.0e5:
            ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%1.0e"))

        ax_count += 1

        logger.log("plotting 1to1 for {0}".format(g))

    for a in range(ax_count, nr * nc):
        axes[a].set_axis_off()
        axes[a].set_yticks([])
        axes[a].set_xticks([])

    plt.tight_layout()
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


def plot_jac_test(
    csvin, csvout, targetobs=None, filetype=None, maxoutputpages=1, outputdirectory=None
):
    """helper function to plot results of the Jacobian test performed using the pest++
    program pestpp-swp.

    Args:
        csvin (`str`): name of csv file used as input to sweep, typically developed with
            static method pyemu.helpers.build_jac_test_csv()
        csvout (`str`): name of csv file with output generated by sweep, both input
            and output files can be specified in the pest++ control file
            with pyemu using: pest_object.pestpp_options["sweep_parameter_csv_file"] = jactest_in_file.csv
            pest_object.pestpp_options["sweep_output_csv_file"] = jactest_out_file.csv
        targetobs ([`str`]): list of observation file names to plot, each parameter used for jactest can
            have up to 32 observations plotted per page, throws a warning if more tha
            10 pages of output are requested per parameter. If none, all observations in
            the output csv file are used.
        filetype (`str`): file type to store output, if None, plt.show() is called.
        maxoutputpages (`int`): maximum number of pages of output per parameter.  Each page can
            hold up to 32 observation derivatives.  If value > 10, set it to
            10 and throw a warning.  If observations in targetobs > 32*maxoutputpages,
            then a random set is selected from the targetobs list (or all observations
            in the csv file if targetobs=None).
        outputdirectory (`str`):  directory to store results, if None, current working directory is used.
            If string is passed, it is joined to the current working directory and
            created if needed. If os.path is passed, it is used directly.

    Note:
        Used in conjunction with pyemu.helpers.build_jac_test_csv() and sweep to perform
        a Jacobian Test and then view the results. Can generate a lot of plots so easiest
        to put into a separate directory and view the files.

    """

    localhome = os.getcwd()
    # check if the output directory exists, if not make it
    if outputdirectory is not None and not os.path.exists(
        os.path.join(localhome, outputdirectory)
    ):
        os.mkdir(os.path.join(localhome, outputdirectory))
    if outputdirectory is None:
        figures_dir = localhome
    else:
        figures_dir = os.path.join(localhome, outputdirectory)

    # read the input and output files into pandas dataframes
    jactest_in_df = pd.read_csv(csvin, engine="python", index_col=0)
    jactest_in_df.index.name = "input_run_id"
    jactest_out_df = pd.read_csv(csvout, engine="python", index_col=1)

    # subtract the base run from every row, leaves the one parameter that
    # was perturbed in any row as only non-zero value. Set zeros to nan
    # so round-off doesn't get us and sum across rows to get a column of
    # the perturbation for each row, finally extract to a series. First
    # the input csv and then the output.
    base_par = jactest_in_df.loc["base"]
    delta_par_df = jactest_in_df.subtract(base_par, axis="columns")
    delta_par_df.replace(0, np.nan, inplace=True)
    delta_par_df.drop("base", axis="index", inplace=True)
    delta_par_df["change"] = delta_par_df.sum(axis="columns")
    delta_par = pd.Series(delta_par_df["change"])

    base_obs = jactest_out_df.loc["base"]
    delta_obs = jactest_out_df.subtract(base_obs)
    delta_obs.drop("base", axis="index", inplace=True)
    # if targetobs is None, then reset it to all the observations.
    if targetobs is None:
        targetobs = jactest_out_df.columns.tolist()[8:]
    delta_obs = delta_obs[targetobs]

    # get the Jacobian by dividing the change in observation by the change in parameter
    # for the perturbed parameters
    jacobian = delta_obs.divide(delta_par, axis="index")

    # use the index created by build_jac_test_csv to get a column of parameter names
    # and increments, then we can plot derivative vs. increment for each parameter
    extr_df = pd.Series(jacobian.index.values).str.extract(r"(.+)(_\d+$)", expand=True)
    extr_df[1] = pd.to_numeric(extr_df[1].str.replace("_", "")) + 1
    extr_df.rename(columns={0: "parameter", 1: "increment"}, inplace=True)
    extr_df.index = jacobian.index

    # make a dataframe for plotting the Jacobian by combining the parameter name
    # and increments frame with the Jacobian frame
    plotframe = pd.concat([extr_df, jacobian], axis=1, join="inner")

    # get a list of observations to keep based on maxoutputpages.
    if maxoutputpages > 10:
        print("WARNING, more than 10 pages of output requested per parameter")
        print("maxoutputpage reset to 10.")
        maxoutputpages = 10
    num_obs_plotted = np.min(np.array([maxoutputpages * 32, len(targetobs)]))
    if num_obs_plotted < len(targetobs):
        # get random sample
        index_plotted = np.random.choice(len(targetobs), num_obs_plotted, replace=False)
        obs_plotted = [targetobs[x] for x in index_plotted]
        real_pages = maxoutputpages
    else:
        obs_plotted = targetobs
        real_pages = int(targetobs / 32) + 1

    # make a subplot of derivative vs. increment one plot for each of the
    # observations in targetobs, and outputs grouped by parameter.
    for param, group in plotframe.groupby("parameter"):
        for page in range(0, real_pages):
            fig, axes = plt.subplots(8, 4, sharex=True, figsize=(10, 15))
            for row in range(0, 8):
                for col in range(0, 4):
                    count = 32 * page + 4 * row + col
                    if count < num_obs_plotted:
                        axes[row, col].scatter(
                            group["increment"], group[obs_plotted[count]]
                        )
                        axes[row, col].plot(
                            group["increment"], group[obs_plotted[count]], "r"
                        )
                        axes[row, col].set_title(obs_plotted[count])
                        axes[row, col].set_xticks([1, 2, 3, 4, 5])
                        axes[row, col].tick_params(direction="in")
                        if row == 3:
                            axes[row, col].set_xlabel("Increment")
            plt.tight_layout()

            if filetype is None:
                plt.show()
            else:
                plt.savefig(
                    os.path.join(
                        figures_dir, "{0}_jactest_{1}.{2}".format(param, page, filetype)
                    )
                )
            plt.close()
