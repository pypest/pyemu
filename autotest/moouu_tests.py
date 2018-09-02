import os
import sys

sys.path.append(os.path.join("..","pyemu"))

from pyemu.prototypes.moouu import EvolAlg


if not os.path.exists("temp1"):
    os.mkdir("temp1")

def tenpar_test():
    import os
    import numpy as np
    import flopy
    import pyemu

    os.chdir(os.path.join("smoother","10par_xsec"))
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

    pst.observation_data.loc["h01_10", "obgnme"] = "greaterthan"
    pst.observation_data.loc["h01_10", "weight"] = 1.0
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
                                                  num_reals=10,
                                                  partial=False)
    ea = EvolAlg(pst, num_slaves=20, port=4005, verbose=True)

    dv = pyemu.ParameterEnsemble.from_mixed_draws(pst=pst, how_dict={p: "uniform" for p in pst.adj_par_names[2:]},
                                                  num_reals=10,
                                                  partial=True)

    import matplotlib.pyplot as plt
    ax = plt.subplot(111)
    colors = ['r','y','g','b','m']
    risks = [0.05,0.25,0.51,0.75,0.95]
    for risk,color in zip(risks,colors):
        ea.initialize(obj_dict,par_ensemble=pe,dv_ensemble=dv,risk=risk,dv_names=pst.adj_par_names[2:])
        oe = ea.obs_ensemble
        # call the nondominated sorting
        is_nondom = ea.obj_func.is_nondominated_continuous(oe)
        obj = oe.loc[:,obj_names]
        obj.loc[is_nondom,"is_nondom"] = is_nondom
        #print(obj)

        stack = ea.last_stack
        plt.scatter(stack.iloc[:, 0], stack.iloc[:, 1], color="0.5", marker='.',s=20, alpha=0.25)

        plt.scatter(obj.iloc[:,0],obj.iloc[:,1],color=color,marker='.',alpha=0.25,s=40)
        ind = obj.loc[is_nondom,:]
        #plt.scatter(ind.iloc[:, 0], ind.iloc[:, 1], color="m", marker='.',s=20,alpha=0.5)
        isfeas = ea.obj_func.is_feasible(oe)

        isf = obj.loc[isfeas,:]
        #plt.scatter(isf.iloc[:, 0], isf.iloc[:, 1], color="g", marker='.', s=30, alpha=0.5)
        both = [True if s and d else False for s,d in zip(is_nondom,isfeas)]
        both = obj.loc[both,:]
        plt.scatter(both.iloc[:, 0], both.iloc[:, 1], color=color, marker='+', s=90,alpha=0.5)
    plt.savefig("risk_compare.pdf")
    #plt.show()


    # test the infeas calcs
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

    os.chdir(os.path.join("..",".."))


if __name__ == "__main__":
    tenpar_test()
