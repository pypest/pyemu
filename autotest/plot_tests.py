import os
import shutil
import numpy as np
import pandas as pd
import pyemu


def plot_summary_test(tmp_path):

    try:
        import matplotlib.pyplot as plt
    except:
        return
    par_df_fname = os.path.join("utils","freyberg_pp.par.usum.csv")
    shutil.copy(par_df_fname, tmp_path)
    par_df_fname = os.path.join(tmp_path, "freyberg_pp.par.usum.csv")

    par_df = pd.read_csv(par_df_fname, index_col=0)
    idx = list(par_df.index.map(lambda x: x.startswith("HK")))
    par_df = par_df.loc[idx,:]
    ax = pyemu.plot_utils.plot_summary_distributions(par_df,label_post=True)
    plt.savefig(os.path.join(tmp_path,"hk_par.png"))
    plt.close()

    df = os.path.join("utils","freyberg_pp.pred.usum.csv")
    figs,axes = pyemu.plot_utils.plot_summary_distributions(df,subplots=True)
    #plt.show()
    for i,fig in enumerate(figs):
        plt.figure(fig.number)
        plt.savefig(os.path.join(tmp_path,"test_pred_{0}.png".format(i)))
        plt.close(fig)
    df = os.path.join("utils","freyberg_pp.par.usum.csv")
    figs, axes = pyemu.plot_utils.plot_summary_distributions(df,subplots=True)
    for i,fig in enumerate(figs):
        plt.figure(fig.number)
        plt.savefig(os.path.join(tmp_path,"test_par_{0}.png".format(i)))
        plt.close(fig)


def pst_plot_test(tmp_path):
    try:
        import matplotlib.pyplot as plt
    except:
        return
    pstfile = os.path.join("pst", "pest.pst")
    shutil.copy(pstfile,
                os.path.join(tmp_path, "pest.pst"))
    shutil.copy(pstfile.replace('.pst', '.rei'),
                os.path.join(tmp_path, "pest.rei"))
    shutil.copy(pstfile.replace('.pst', '.iobj'),
                os.path.join(tmp_path, "pest.iobj"))
    shutil.copy(os.path.join("pst", "freyberg_gr.pst"),
                os.path.join(tmp_path, "freyberg_gr.pst"))
    shutil.copy(os.path.join("pst", "freyberg_gr.rei"),
                os.path.join(tmp_path, "freyberg_gr.rei"))
    os.chdir(tmp_path)
    pst = pyemu.Pst("pest.pst")
    pst.plot()
    pst.parameter_data.loc[:, "partrans"] = "none"
    pst.plot()

    ax = pst.plot(kind="phi_pie")
    pst.observation_data.loc[pst.nnz_obs_names[::2],"obgnme"] = "test"
    print(pst.phi_components)
    pst.plot(kind="phi_pie")
    pst.plot(kind="prior")
    #plt.show()
    #return

    pst = pyemu.Pst("pest.pst")
    pst.plot(kind="phi_progress")

    pst = pyemu.Pst("freyberg_gr.pst")
    par = pst.parameter_data
    par.loc[pst.par_names[:3],"pargp"] = "test"
    par.loc[pst.par_names[1:],"partrans"] = "fixed"
    #pst.plot()
    pst.parameter_data.loc[:,"partrans"] = "none"
    pst.plot(kind="prior", unique_only=False)
    pst.plot(kind="prior",unique_only=True)
    pst.plot(kind="prior", unique_only=True, fig_title="priors")
    pst.plot(kind="prior", unique_only=True, fig_title="priors",
             filename=os.path.join(tmp_path,"test.pdf"))

    #
    pst.plot(kind="1to1")
    pst.plot(kind="1to1",include_zero=True)
    pst.plot(kind="1to1", include_zero=True,fig_title="1to1")
    fig = pst.plot(kind="1to1", include_zero=True, fig_title="1to1")
    pst.plot(kind="1to1", include_zero=True, fig_title="1to1",histogram=True)
    #
    #
    #
    ax = pst.plot(kind="phi_pie")
    ax = plt.subplot(111,aspect="equal")
    pst.plot(kind="phi_pie",ax=ax)
    # plt.show()


def ensemble_plot_test(tmp_path):
    try:
        import matplotlib.pyplot as plt
    except:
        return
    pstfile = os.path.join("pst", "pest.pst")
    shutil.copy(pstfile,
                os.path.join(tmp_path, "pest.pst"))
    os.chdir(tmp_path)
    pst = pyemu.Pst("pest.pst")
    cov = pyemu.Cov.from_parameter_data(pst)
    num_reals = 100
    pe = pyemu.ParameterEnsemble.from_gaussian_draw(pst,cov,num_reals=num_reals,fill=True)
    csv_file = "pe.csv"

    pe.plot(filename=csv_file + ".pdf",plot_cols=pst.par_names[:10])

    pd = pst.parameter_data.groupby("pargp").groups

    pyemu.plot_utils.ensemble_helper(
        pe,plot_cols=pd,
        filename=csv_file+".pdf",alpha=0.1
    )

    pe.to_csv(csv_file)
    pyemu.plot_utils.ensemble_helper(pe, filename=csv_file + ".pdf",
                                     plot_cols=pst.par_names[:10])
    pyemu.plot_utils.ensemble_helper(csv_file, filename=csv_file + ".pdf",
                                     plot_cols=pst.par_names[:10])


    pst.parameter_data.loc[:,"partrans"] = "none"
    cov = pyemu.Cov.from_parameter_data(pst)
    pe = pyemu.ParameterEnsemble.from_gaussian_draw(pst, cov, num_reals=num_reals)


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
                                     deter_vals=deter_vals)


    pyemu.plot_utils.ensemble_helper({"b": pe, "y": csv_file}, filename=csv_file + ".pdf",
                                     plot_cols=pst.par_names[:10], sync_bins=False,
                                     deter_vals=deter_vals,deter_range=True)


def ensemble_1to1_test(tmp_path):
    try:
        import matplotlib.pyplot as plt
    except:
        return
    shutil.copy(os.path.join("pst","pest.pst"),
                os.path.join(tmp_path, "pest.pst"))
    shutil.copy(os.path.join("pst","pest.rei"),
                os.path.join(tmp_path, "pest.rei"))
    os.chdir(tmp_path)
    #try:
    pst = pyemu.Pst("pest.pst")
    num_reals = 100

    oe1 = pyemu.ObservationEnsemble.from_gaussian_draw(pst,num_reals=num_reals)
    pst.observation_data.loc[pst.nnz_obs_names,"weight"] *= 10.0
    oe2 = pyemu.ObservationEnsemble.from_gaussian_draw(pst, num_reals=num_reals)

    pst.observation_data.loc[pst.nnz_obs_names, "weight"] /= 100.0
    oe_base = pyemu.ObservationEnsemble.from_gaussian_draw(pst, num_reals=num_reals)

    pst.observation_data.loc[pst.nnz_obs_names, "weight"] *= 1000.0
    oe_base2 = pyemu.ObservationEnsemble.from_gaussian_draw(pst, num_reals=num_reals)

    #print(oe1.loc[:,pst.nnz_obs_names].std())
    #print(oe2.loc[:,pst.nnz_obs_names].std())

    # pyemu.plot_utils.ensemble_res_1to1(oe1,pst,filename=os.path.join("temp","e1to1.pdf"))
    #
    # pyemu.plot_utils.ensemble_res_1to1({"0.5":oe1,"b":oe2},pst,filename=os.path.join("temp","e1to1.pdf"))

    pyemu.plot_utils.ensemble_res_1to1(
        {"0.5": oe1, "b": oe2},
        pst,
        filename="e1to1_noise.pdf",
        base_ensemble=oe_base
    )
    pyemu.plot_utils.ensemble_res_1to1(
        {"0.5": oe1, "b": oe2},
        pst,
        filename="e1to1_noise2.pdf",
        base_ensemble=oe_base2
    )

    pyemu.plot_utils.ensemble_res_1to1(
        {"0.5": oe1 + 10, "b": oe2 + 10},
        pst,
        filename="e1to1_noise3.pdf",
        base_ensemble=oe_base2
    )

    pyemu.plot_utils.ensemble_res_1to1(
        {"0.5": oe1 + 10, "b": oe2},
        pst,
        filename="e1to1_noise4.pdf",
        base_ensemble=oe_base2
    )

    pyemu.plot_utils.ensemble_res_1to1(
        {"0.5": oe1 * -10, "b": oe2*-10},
        pst,
        filename="e1to1_noise4.pdf",
        base_ensemble=oe_base2
    )
    pst.observation_data.loc[:, 'o_obgnme'] = pst.observation_data.obgnme
    pst.observation_data.loc[pst.nnz_obs_names[0], 'obgnme'] = 'solo1'
    pst.observation_data.loc[pst.nnz_obs_names[-1], 'obgnme'] = 'solo2'
    pyemu.plot_utils.ensemble_res_1to1(
        {"0.5": oe1 * -10, "b": oe2*-10},
        pst,
        filename="e1to1_noise5.pdf",
        base_ensemble=oe_base2
    )

    pyemu.plot_utils.ensemble_res_1to1(
        {"0.5": oe1+1, "b": oe2},
        pst,
        filename="e1to1_noise6.pdf"
    )

    oenan = oe2._df
    oenan[oenan == oenan.max()] = np.nan
    pyemu.plot_utils.ensemble_res_1to1(
        {"0.5": oe1, "b": oenan},
        pst,
        filename="e1to1_nans.pdf",
        base_ensemble=oe_base
    )

    pst.observation_data.loc[:, 'obgnme'] = pst.observation_data.o_obgnme
    pyemu.plot_utils.res_phi_pie(pst=pst,ensemble=oe1)
    pyemu.plot_utils.res_1to1(pst=pst, ensemble=oe1)


def ensemble_summary_test(tmp_path):
    try:
        import matplotlib.pyplot as plt
    except:
        return

    shutil.copy(os.path.join("pst", "pest.pst"),
                os.path.join(tmp_path, "pest.pst"))
    shutil.copy(os.path.join("pst", "pest.rei"),
                os.path.join(tmp_path, "pest.rei"))
    os.chdir(tmp_path)
    pst = pyemu.Pst("pest.pst")
    num_reals = 100

    oe1 = pyemu.ObservationEnsemble.from_gaussian_draw(pst,num_reals=num_reals)
    pst.observation_data.loc[pst.nnz_obs_names,"weight"] *= 10.0
    oe2 = pyemu.ObservationEnsemble.from_gaussian_draw(pst, num_reals=num_reals)
    #print(oe1.loc[:,pst.nnz_obs_names].std())
    #print(oe2.loc[:,pst.nnz_obs_names].std())

    pyemu.plot_utils.ensemble_change_summary(
        oe1,oe2,pst,
        filename="edeltasum.pdf"
    )
    pst.parameter_data.loc[:,"partrans"] = "none"
    cov1 = pyemu.Cov.from_parameter_data(pst,sigma_range=6)
    pe1 = pyemu.ParameterEnsemble.from_gaussian_draw(pst,cov1,num_reals=1000)

    cov2 = cov1 * 0.001
    pe2 = pyemu.ParameterEnsemble.from_gaussian_draw(pst,cov2,num_reals=1000)

    pyemu.plot_utils.ensemble_change_summary(
        pe1, pe2, pst,
        filename="edeltasum.pdf"
    )

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


def ensemble_change_test(tmp_path):
    import matplotlib.pyplot as plt
    os.chdir(tmp_path)
    pst = pyemu.Pst.from_par_obs_names(par_names=["p1","p2"])
    cov = pyemu.Cov(np.array([[1.0,0.0],[0.0,1.0]]),names=["p1","p2"])
    pe1 = pyemu.ParameterEnsemble.from_gaussian_draw(pst=pst,cov=cov,num_reals=5000)
    cov *= 0.1
    pe2 = pyemu.ParameterEnsemble.from_gaussian_draw(pst=pst,cov=cov,num_reals=5000)
    pyemu.plot_utils.ensemble_change_summary(pe1,pe2,pst=pst)
    print(pe1.mean(),pe1.std())
    print(pe2.mean(),pe2.std())
    pyemu.plot_utils.ensemble_change_summary(pe1,pe2,pst)
    #plt.show()


if __name__ == "__main__":
    # plot_summary_test('.')
    pst_plot_test('.')
    #ensemble_summary_test('.')
    #ensemble_plot_test()
    # ensemble_1to1_test('.')
    #ensemble_plot_test('.')
    #ensemble_change_test('.')

