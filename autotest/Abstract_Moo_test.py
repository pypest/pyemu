import os
#from pyemu.prototypes.Abstract_Moo import *
from pyemu.prototypes.moouu import EliteDiffEvol


def tenpar_test():
    import os
    import numpy as np
    import flopy
    import pyemu

    bd = os.getcwd()
    try:
        #os.chdir(os.path.join("moouu","10par_xsec"))
        os.chdir(os.path.join("moouu", 'StochasticProblemSuite'))
        csv_files = [f for f in os.listdir('.') if f.endswith(".csv")]
        [os.remove(csv_file) for csv_file in csv_files]
        #pst = pyemu.Pst("10par_xsec.pst")
        pst = pyemu.Pst('SRN.pst')
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
        obj_dict[obj_names[1]] = "min"



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
        pe = pyemu.ParameterEnsemble.from_mixed_draws(pst=pst, how_dict={p: "gaussian" for p in pst.adj_par_names[2:]},
                                                      num_reals=5,
                                                      partial=False)
        ea = EliteDiffEvol(pst, num_slaves=8, port=4005, verbose=True)

        dv = pyemu.ParameterEnsemble.from_mixed_draws(pst=pst, how_dict={p: "uniform" for p in pst.adj_par_names[:2]},
                                                      num_reals=5,
                                                      partial=True)

        ea.initialize(obj_dict,num_dv_reals=5,num_par_reals=1,risk=0.5)
        ea.initialize(obj_dict,par_ensemble=pe, dv_ensemble=dv, risk=0.5)

        #ea.update()


        # test the infeas calcs
    #     oe = ea.obs_ensemble
    #     ea.obj_func.is_nondominated_continuous(oe)
    #     ea.obj_func.is_nondominated_kung(oe)
    #     is_feasible = ea.obj_func.is_feasible(oe)
    #     oe.loc[is_feasible.index,"feas"] = is_feasible
    #     obs = pst.observation_data
    #     for lt_obs in pst.less_than_obs_constraints:
    #         val = obs.loc[lt_obs,"obsval"]
    #         infeas = oe.loc[:,lt_obs] >= val
    #         assert np.all(~is_feasible.loc[infeas])
    #
    #     for gt_obs in pst.greater_than_obs_constraints:
    #         val = obs.loc[gt_obs,"obsval"]
    #         infeas = oe.loc[:,gt_obs] <= val
    #         assert np.all(~is_feasible.loc[infeas])
    #
    #     # test that the end members are getting max distance
    #     crowd_distance = ea.obj_func.crowd_distance(oe)
    #     for name,direction in ea.obj_func.obs_dict.items():
    #         assert crowd_distance.loc[oe.loc[:,name].idxmax()] >= ea.obj_func.max_distance,crowd_distance.loc[oe.loc[:,name].idxmax()]
    #         assert crowd_distance.loc[oe.loc[:, name].idxmin()] >= ea.obj_func.max_distance,crowd_distance.loc[oe.loc[:, name].idxmin()]
    except Exception as e:
        os.chdir(os.path.join("..",".."))
        raise Exception(str(e))

if __name__ == "__main__":
    tenpar_test()


