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

    pst = pyemu.Pst(os.path.join("pst","pest.pst"))
    pst.plot()




if __name__ == "__main__":
    #plot_summary_test()
    pst_plot_test()

