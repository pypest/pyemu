import os
import sys

#sys.path.append(os.path.join("..","pyemu"))

from pyemu.prototypes.moouu import EvolAlg, EliteDiffEvol


if not os.path.exists("temp1"):
    os.mkdir("temp1")

def tenpar_test():
    import os
    import numpy as np
    import flopy
    import pyemu

    bd = os.getcwd()
    try:
        os.chdir(os.path.join("moouu","10par_xsec"))
        csv_files = [f for f in os.listdir('.') if f.endswith(".csv")]
        [os.remove(csv_file) for csv_file in csv_files]
        pst = pyemu.Pst("10par_xsec.pst")

        obj_names = pst.nnz_obs_names
        # pst.observation_data.loc[pst.obs_names[0],"obgnme"] = "greaterthan"
        # pst.observation_data.loc[pst.obs_names[0], "weight"] = 1.0
        # pst.observation_data.loc[pst.obs_names[0], "obsval"] *= 0.85
        # pst.observation_data.loc[pst.obs_names[-1], "obgnme"] = "greaterthan"
        # pst.observation_data.loc[pst.obs_names[-1], "weight"] = 1.0
        # pst.observation_data.loc[pst.obs_names[-1], "obsval"] *= 0.85

        # pst.observation_data.loc["h01_10", "obgnme"] = "greaterthan"
        # pst.observation_data.loc["h01_10", "weight"] = 1.0
        #pst.observation_data.loc["h01_10", "obsval"] *= 0.85


        par = pst.parameter_data
        #par.loc[:,"partrans"] = "none"

        obj_dict = {}
        obj_dict[obj_names[0]] = "min"
        obj_dict[obj_names[1]] = "max"


        # testing for reduce method
        # oe = pyemu.ObservationEnsemble.from_id_gaussian_draw(pst=pst, num_reals=5000)
        # logger = pyemu.Logger("temp.log")
        # obj_func = evol_proto.ParetoObjFunc(pst,obj_dict,logger)
        # df = obj_func.reduce_stack_with_risk_shift(oe,50,0.05)
        #
        # import matplotlib.pyplot as plt
        # ax = plt.subplot(111)
        # oe.iloc[:, -1].hist(ax=ax)
        # ylim = ax.get_ylim()
        # val = df.iloc[0,-1]
        # ax.plot([val, val], ylim)
        # ax.set_ylim(ylim)
        # plt.show()
        # print(df.shape)
        # return

        pe = pyemu.ParameterEnsemble.from_mixed_draws(pst=pst, how_dict={p: "uniform" for p in pst.adj_par_names[:2]},
                                                      num_reals=5,
                                                      partial=False)
        ea = EvolAlg(pst, num_slaves=8, port=4005, verbose=True)

        dv = pyemu.ParameterEnsemble.from_mixed_draws(pst=pst, how_dict={p: "uniform" for p in pst.adj_par_names[2:]},
                                                      num_reals=5,
                                                      partial=True)
        ea.initialize(obj_dict, par_ensemble=pe, dv_ensemble=dv, risk=0.5)

        # test the infeas calcs
        oe = ea.obs_ensemble
        is_feasible = ea.obj_func.is_feasible(oe)
        oe.loc[is_feasible.index,"feas"] = is_feasible
        obs = pst.observation_data
        for lt_obs in pst.less_than_obs_constraints:
            val = obs.loc[lt_obs,"obsval"]
            infeas = oe.loc[:,lt_obs] >= val
            assert np.all(~is_feasible.loc[infeas])

        for gt_obs in pst.greater_than_obs_constraints:
            val = obs.loc[gt_obs,"obsval"]
            infeas = oe.loc[:,gt_obs] <= val
            assert np.all(~is_feasible.loc[infeas])

        # test that the end members are getting max distance
        crowd_distance = ea.obj_func.crowd_distance(oe)
        for name,direction in ea.obj_func.obs_dict.items():
            assert crowd_distance.loc[oe.loc[:,name].idxmax()] >= ea.obj_func.max_distance,crowd_distance.loc[oe.loc[:,name].idxmax()]
            assert crowd_distance.loc[oe.loc[:, name].idxmin()] >= ea.obj_func.max_distance,crowd_distance.loc[oe.loc[:, name].idxmin()]
    except Exception as e:
        os.chdir(os.path.join("..",".."))
        raise Exception(str(e))

    os.chdir(os.path.join("..",".."))


def tenpar_dev():
    import os
    import numpy as np
    import matplotlib.pyplot as plt

    import flopy
    import pyemu

    os.chdir(os.path.join("moouu","10par_xsec"))
    csv_files = [f for f in os.listdir('.') if f.endswith(".csv")]
    [os.remove(csv_file) for csv_file in csv_files]
    pst = pyemu.Pst("10par_xsec.pst")
    #obj_names = pst.nnz_obs_names
    obj_names = ["h01_04", "h01_06", "h01_08"]

    # pst.observation_data.loc[pst.obs_names[0],"obgnme"] = "greaterthan"
    # pst.observation_data.loc[pst.obs_names[0], "weight"] = 1.0
    # pst.observation_data.loc[pst.obs_names[0], "obsval"] *= 0.85
    # pst.observation_data.loc[pst.obs_names[-1], "obgnme"] = "greaterthan"
    # pst.observation_data.loc[pst.obs_names[-1], "weight"] = 1.0
    # pst.observation_data.loc[pst.obs_names[-1], "obsval"] *= 0.85

    # pst.observation_data.loc["h01_10", "obgnme"] = "lessthan"
    # pst.observation_data.loc["h01_10", "weight"] = 1.0
    # pst.observation_data.loc["h01_10", "obsval"] *= 0.85


    par = pst.parameter_data
    par.loc[:,"partrans"] = "none"


    obj_dict = {}
    obj_dict[obj_names[0]] = "min"
    obj_dict[obj_names[1]] = "max"
    obj_dict[obj_names[2]] = "max"

    # testing for reduce method
    # oe = pyemu.ObservationEnsemble.from_id_gaussian_draw(pst=pst, num_reals=5000)
    # logger = pyemu.Logger("temp.log")
    # obj_func = evol_proto.ParetoObjFunc(pst,obj_dict,logger)
    # df = obj_func.reduce_stack_with_risk_shift(oe,50,0.05)
    #
    # import matplotlib.pyplot as plt
    # ax = plt.subplot(111)
    # oe.iloc[:, -1].hist(ax=ax)
    # ylim = ax.get_ylim()
    # val = df.iloc[0,-1]
    # ax.plot([val, val], ylim)
    # ax.set_ylim(ylim)
    # plt.show()
    # print(df.shape)
    # return
    dv_names = pst.adj_par_names[2:]
    par_names = pst.adj_par_names[:2]
    par.loc[dv_names, "parlbnd"] = 1.0
    par.loc[dv_names, "parubnd"] = 5.0

    pe = pyemu.ParameterEnsemble.from_mixed_draws(pst=pst, how_dict={p: "uniform" for p in par_names},
                                                 num_reals=1,
                                                 partial=False)


    dv = pyemu.ParameterEnsemble.from_mixed_draws(pst=pst, how_dict={p: "uniform" for p in dv_names},
                                                  num_reals=100,
                                                  partial=True)

    dv.index = ["p_{0}".format(i) for i in range(dv.shape[0])]
    ea = EliteDiffEvol(pst, num_slaves=30, port=4005, verbose=True)

    ea.initialize(obj_dict,par_ensemble=pe,dv_ensemble=dv,risk=0.5)

    #ax = plt.subplot(111)
    obj_org = ea.obs_ensemble.loc[:, obj_names].copy()
    dom_org = ea.obj_func.is_nondominated(obj_org)

    def plot_3obj(obj=None,label=False):

        fig = plt.figure(figsize=(9,3))
        ax = plt.subplot(131,aspect="equal")
        ax2 = plt.subplot(132, aspect="equal")
        ax3 = plt.subplot(133, aspect="equal")
        # obj_org = ea.obs_ensemble.loc[:, obj_names].copy()
        # dom_org = ea.obj_func.is_nondominated(obj_org)
        ax.scatter(obj_org.loc[:, obj_names[0]], obj_org.loc[:, obj_names[1]], color="0.5", marker='.', alpha=0.25,
                   s=8)
        ax.scatter(obj_org.loc[dom_org, obj_names[0]], obj_org.loc[dom_org, obj_names[1]], color="0.5", marker='+',
                   alpha=0.25, s=25)
        if label and obj is None:
            for i in obj_org.index:
               if i not in dom_org.index:
                   continue
               ax.text(obj_org.loc[i, obj_names[0]], obj_org.loc[i, obj_names[1]],str(i),ha="center",fontsize=5)

        ax2.scatter(obj_org.loc[:, obj_names[1]], obj_org.loc[:, obj_names[2]], color="0.5", marker='.', alpha=0.25,
                   s=8)
        ax2.scatter(obj_org.loc[dom_org, obj_names[1]], obj_org.loc[dom_org, obj_names[2]], color="0.5", marker='+',
                   alpha=0.25, s=25)
        if label and obj is None:
            for i in obj_org.index:
                if i not in dom_org.index:
                    continue
                ax2.text(obj_org.loc[i, obj_names[1]], obj_org.loc[i, obj_names[2]], str(i), ha="center", fontsize=5)

        ax3.scatter(obj_org.loc[:, obj_names[0]], obj_org.loc[:, obj_names[2]], color="0.5", marker='.', alpha=0.25,
                    s=8)
        ax3.scatter(obj_org.loc[dom_org, obj_names[0]], obj_org.loc[dom_org, obj_names[2]], color="0.5", marker='+',
                    alpha=0.25, s=25)
        if label and obj is None:
            for i in obj_org.index:
                if i not in dom_org.index:
                    continue
                ax3.text(obj_org.loc[i, obj_names[0]], obj_org.loc[i, obj_names[2]], str(i), ha="center", fontsize=5)

        if obj is not None:
            dom = ea.obj_func.is_nondominated(obj)
            ax.scatter(obj.loc[:, obj_names[0]], obj.loc[:, obj_names[1]], color="b", marker='.', alpha=0.25,
                       s=8)
            ax.scatter(obj.loc[dom, obj_names[0]], obj.loc[dom, obj_names[1]], color="b", marker='+',
                       alpha=0.25, s=25)

            if label:
                for i in obj.index:
                    if i not in dom.index:
                        continue
                    ax.text(obj.loc[i, obj_names[0]], obj.loc[i, obj_names[1]], str(i), ha="center", fontsize=5)

            ax2.scatter(obj.loc[:, obj_names[1]], obj.loc[:, obj_names[2]], color="b", marker='.', alpha=0.25,
                       s=8)
            ax2.scatter(obj.loc[dom, obj_names[1]], obj.loc[dom, obj_names[2]], color="b", marker='+',
                       alpha=0.25, s=25)

            if label:
                for i in obj.index:
                    if i not in dom.index:
                        continue
                    ax2.text(obj.loc[i, obj_names[1]], obj.loc[i, obj_names[2]], str(i), ha="center", fontsize=5)

            ax3.scatter(obj.loc[:, obj_names[0]], obj.loc[:, obj_names[2]], color="b", marker='.', alpha=0.25,
                        s=8)
            ax3.scatter(obj.loc[dom, obj_names[0]], obj.loc[dom, obj_names[2]], color="b", marker='+',
                        alpha=0.25, s=25)

            if label:
                for i in obj.index:
                    if i not in dom.index:
                        continue
                    ax3.text(obj.loc[i, obj_names[0]], obj.loc[i, obj_names[2]], str(i), ha="center", fontsize=5)



        for aax in [ax,ax2,ax3]:
            aax.set_xlim(0, 4.5)
            aax.set_ylim(0, 4.5)

        ax.set_title("iter {0}, total runs: {1}".format(0,ea.total_runs))
        ax.set_xlabel("{0}, dir:{1}".format(obj_names[0], obj_dict[obj_names[0]]))
        ax.set_ylabel("{0}, dir:{1}".format(obj_names[1], obj_dict[obj_names[1]]))
        ax2.set_xlabel("{0}, dir:{1}".format(obj_names[1], obj_dict[obj_names[1]]))
        ax2.set_ylabel("{0}, dir:{1}".format(obj_names[2], obj_dict[obj_names[2]]))
        ax3.set_xlabel("{0}, dir:{1}".format(obj_names[0], obj_dict[obj_names[0]]))
        ax3.set_ylabel("{0}, dir:{1}".format(obj_names[2], obj_dict[obj_names[2]]))

        plt.savefig("iter_{0:03d}.png".format(ea.iter_num),dpi=500)

        plt.close("all")

    plot_3obj()

    for i in range(10):

        ea.update()

        obj = ea.obs_ensemble.loc[:, obj_names]
        plot_3obj(obj)

    os.system("ffmpeg -r 2 -i iter_%03d.png -loop 0 -final_delay 100 -y shhh.mp4")
    return
    ax = plt.subplot(111)
    colors = ['r','y','g','b','m']
    risks = [0.05,0.25,0.51,0.75,0.95]
    for risk,color in zip(risks,colors):
        ea.initialize(obj_dict,par_ensemble=pe,dv_ensemble=dv,risk=risk,dv_names=pst.adj_par_names[2:])
        oe = ea.obs_ensemble
        # call the nondominated sorting
        is_nondom = ea.obj_func.is_nondominated(oe)
        obj = oe.loc[:,obj_names]
        obj.loc[is_nondom,"is_nondom"] = is_nondom
        #print(obj)

        stack = ea.last_stack
        plt.scatter(stack.loc[:, obj_names[0]], stack.loc[:, obj_names[1]], color="0.5", marker='.',s=10, alpha=0.25)

        plt.scatter(obj.loc[:,obj_names[0]],obj.loc[:,obj_names[1]],color=color,marker='.',alpha=0.25,s=8)
        ind = obj.loc[is_nondom,:]
        #plt.scatter(ind.iloc[:, 0], ind.iloc[:, 1], color="m", marker='.',s=8,alpha=0.5)
        isfeas = ea.obj_func.is_feasible(oe)

        isf = obj.loc[isfeas,:]
        #plt.scatter(isf.iloc[:, 0], isf.iloc[:, 1], color="g", marker='.', s=30, alpha=0.5)
        both = [True if s and d else False for s,d in zip(is_nondom,isfeas)]
        both = obj.loc[both,:]
        plt.scatter(both.loc[:, obj_names[0]], both.loc[:, obj_names[1]], color=color, marker='+', s=90,alpha=0.5)

    ax.set_xlabel("{0}, dir:{1}".format(obj_names[0],obj_dict[obj_names[0]]))
    ax.set_ylabel("{0}, dir:{1}".format(obj_names[1], obj_dict[obj_names[1]]))
    plt.savefig("risk_compare.pdf")
    #plt.show()

    os.chdir(os.path.join("..",".."))




if __name__ == "__main__":
    tenpar_test()
    #tenpar_dev()
