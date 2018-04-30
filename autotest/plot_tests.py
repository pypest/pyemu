import os
#import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyemu

def plot_summary_test():

    try:
        import matplotlib.pyplot as plt
    except:
        return

    par_df = pd.read_csv(os.path.join("utils","freyberg_pp.par.usum.csv"),
                         index_col=0)
    idx = list(par_df.index.map(lambda x: x.startswith("HK")))
    par_df = par_df.loc[idx,:]
    ax = pyemu.plot_utils.plot_summary_distributions(par_df,label_post=True)
    plt.savefig(os.path.join("temp","hk_par.png"))
    plt.close()

    df = os.path.join("utils","freyberg_pp.pred.usum.csv")
    figs,axes = pyemu.plot_utils.plot_summary_distributions(df,subplots=True)
    #plt.show()
    for i,fig in enumerate(figs):
        plt.figure(fig.number)
        plt.savefig(os.path.join("temp","test_pred_{0}.png".format(i)))
        plt.close(fig)
    df = os.path.join("utils","freyberg_pp.par.usum.csv")
    figs, axes = pyemu.plot_utils.plot_summary_distributions(df,subplots=True)
    for i,fig in enumerate(figs):
        plt.figure(fig.number)
        plt.savefig(os.path.join("temp","test_par_{0}.png".format(i)))
        plt.close(fig)


def pst_plot_test():
    try:
        import matplotlib.pyplot as plt
    except:
        return
    pst = pyemu.Pst(os.path.join("pst", "pest.pst"))
    pst.parameter_data.loc[:, "partrans"] = "none"
    pst.plot(kind="prior")
    #plt.show()
    #return
    pst = pyemu.Pst(os.path.join("pst", "pest.pst"))

    pst.plot(kind="phi_progress")

    pst = pyemu.Pst(os.path.join("pst","freyberg_gr.pst"))
    par = pst.parameter_data
    par.loc[pst.par_names[:3],"pargp"] = "test"
    par.loc[pst.par_names[1:],"partrans"] = "fixed"
    #pst.plot()
    pst.parameter_data.loc[:,"partrans"] = "none"
    pst.plot(kind="prior", unique_only=False)
    pst.plot(kind="prior",unique_only=True)
    pst.plot(kind="prior", unique_only=True, fig_title="priors")
    pst.plot(kind="prior", unique_only=True, fig_title="priors",filename=os.path.join("temp","test.pdf"))

    #
    pst.plot(kind="1to1")
    pst.plot(kind="1to1",include_zero=True)
    pst.plot(kind="1to1", include_zero=True,fig_title="1to1")
    fig = pst.plot(kind="1to1", include_zero=True, fig_title="1to1")
    #
    #
    pst.plot(kind="obs_v_sim")
    pst.plot(kind="obs_v_sim",include_zero=True)
    pst.plot(kind="obs_v_sim", include_zero=True,fig_title="obs_v_sim")
    #
    ax = pst.plot(kind="phi_pie")
    ax = plt.subplot(111,aspect="equal")
    pst.plot(kind="phi_pie",ax=ax)
    # plt.show()



def ensemble_plot_test():
    try:
        import matplotlib.pyplot as plt
    except:
        return

    pst = pyemu.Pst(os.path.join("pst","pest.pst"))
    cov = pyemu.Cov.from_parameter_data(pst)
    num_reals = 100
    pe = pyemu.ParameterEnsemble.from_gaussian_draw(pst,cov,num_reals=num_reals,
                                                    use_homegrown=True)

    csv_file = os.path.join("temp", "pe.csv")
    pe.plot(filename=csv_file + ".pdf",plot_cols=pst.par_names[:10])

    pe.to_csv(csv_file)

    pyemu.plot_utils.ensemble_helper(pe, filename=csv_file + ".pdf",
                                     plot_cols=pst.par_names[:10])
    pyemu.plot_utils.ensemble_helper(csv_file, filename=csv_file + ".pdf",
                                     plot_cols=pst.par_names[:10])


    pst.parameter_data.loc[:,"partrans"] = "none"
    cov = pyemu.Cov.from_parameter_data(pst)
    pe = pyemu.ParameterEnsemble.from_gaussian_draw(pst, cov, num_reals=num_reals,
                                                    use_homegrown=True)


    pyemu.plot_utils.ensemble_helper([pe,csv_file],filename=csv_file+".pdf",
                                     plot_cols=pst.par_names[:10])

    pyemu.plot_utils.ensemble_helper([pe, csv_file], filename=csv_file + ".pdf",
                                     plot_cols=pst.par_names[:10],sync_bins=False)

    pyemu.plot_utils.ensemble_helper({"b":pe,"y":csv_file}, filename=csv_file + ".pdf",
                                     plot_cols=pst.par_names[:10])

    pyemu.plot_utils.ensemble_helper({"b":pe,"y":csv_file}, filename=csv_file + ".pdf",
                                     plot_cols=pst.par_names[:10], sync_bins=False)

    pyemu.plot_utils.ensemble_helper({"b": pe, "y": csv_file}, filename=csv_file + ".pdf",
                                     plot_cols=pst.par_names[:10], sync_bins=False,
                                     func_dict={pst.par_names[0]:np.log10})

    figs = pyemu.plot_utils.ensemble_helper({"b": pe, "y": csv_file}, filename=None,
                                     plot_cols=pst.par_names[:10], sync_bins=False,
                                     func_dict={pst.par_names[0]:np.log10})
    plt.close("all")
    deter_vals = pst.parameter_data.parval1.apply(np.log10).to_dict()
    pyemu.plot_utils.ensemble_helper({"b": pe, "y": csv_file}, filename=csv_file + ".pdf",
                                     plot_cols=pst.par_names[:10], sync_bins=False,
                                     func_dict={pst.par_names[0]: np.log10},
                                     deter_vals=deter_vals)


    pyemu.plot_utils.ensemble_helper({"b": pe, "y": csv_file}, filename=csv_file + ".pdf",
                                     plot_cols=pst.par_names[:10], sync_bins=False,
                                     func_dict={pst.par_names[0]: np.log10},
                                     deter_vals=deter_vals,deter_range=True)


def ensemble_1to1_test():
    try:
        import matplotlib.pyplot as plt
    except:
        return

    pst = pyemu.Pst(os.path.join("pst","pest.pst"))
    num_reals = 100

    oe1 = pyemu.ObservationEnsemble.from_id_gaussian_draw(pst,num_reals=num_reals)
    pst.observation_data.loc[pst.nnz_obs_names,"weight"] *= 10.0
    oe2 = pyemu.ObservationEnsemble.from_id_gaussian_draw(pst, num_reals=num_reals)
    print(oe1.loc[:,pst.nnz_obs_names].std())
    print(oe2.loc[:,pst.nnz_obs_names].std())

    pyemu.plot_utils.ensemble_res_1to1(oe1,pst,filename=os.path.join("temp","e1to1.pdf"))

    pyemu.plot_utils.ensemble_res_1to1({"0.5":oe1,"b":oe2},pst,filename=os.path.join("temp","e1to1.pdf"))


# def cov_test():
#     try:
#         import matplotlib.pyplot as plt
#     except:
#         return
#
#     import os
#     import numpy as np
#     import pyemu
#     pst_file = os.path.join("pst", "pest.pst")
#     pst = pyemu.Pst(pst_file)
#
#     tpl_file = os.path.join("utils", "pp_locs.tpl")
#     str_file = os.path.join("utils", "structure.dat")
#
#     cov = pyemu.helpers.geostatistical_prior_builder(pst_file, {str_file: tpl_file})
#     d1 = np.diag(cov.x)
#
#     df = pyemu.gw_utils.pp_tpl_to_dataframe(tpl_file)
#     #df.loc[:, "zone"] = np.arange(df.shape[0])
#     gs = pyemu.geostats.read_struct_file(str_file)
#     cov = pyemu.helpers.geostatistical_prior_builder(pst_file, {gs: df},
#                                                      sigma_range=4)
#
#     pyemu.plot_utils.par_cov_helper(cov,pst)
#     plt.show()


if __name__ == "__main__":
    #plot_summary_test()
    pst_plot_test()
    #ensemble_plot_test()
    #ensemble_1to1_test()
    #cov_test()

