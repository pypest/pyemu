import os
#import matplotlib.pyplot as plt
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

    pst = pyemu.Pst(os.path.join("pst","freyberg_gr.pst"))
    pst.plot()
    pst.plot(kind="prior", unique_only=False)
    pst.plot(kind="prior",unique_only=True)

    pst.plot(kind="1to1")
    pst.plot(kind="1to1",include_zero=True)

    pst.plot(kind="obs_v_sim")
    pst.plot(kind="obs_v_sim",include_zero=True)
    ax = pst.plot(kind="phi_pie")

    # ax = plt.subplot(111,aspect="equal")
    # pst.plot(kind="phi_pie",ax=ax)
    # plt.show()


def ensemble_plot_test():
    try:
        import matplotlib.pyplot as plt
    except:
        return

    #one en file

    #one en loaded

    #two en list

    #two en list one loaded

    #two en dict

    #two en dict one loaded



if __name__ == "__main__":
    #plot_summary_test()
    #pst_plot_test()
    ensemble_plot_test()

