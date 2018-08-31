import os
import sys

sys.path.append(os.path.join("..","pyemu"))

import evol_proto


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
    pst.observation_data.loc[pst.obs_names[0],"obgnme"] = "lessthan"
    pst.observation_data.loc[pst.obs_names[0], "weight"] = 1.0
    pst.observation_data.loc[pst.obs_names[-1], "obgnme"] = "greaterthan"
    pst.observation_data.loc[pst.obs_names[-1], "weight"] = 1.0

    par = pst.parameter_data
    #par.loc[:,"partrans"] = "none"

    pe = pyemu.ParameterEnsemble.from_mixed_draws(pst=pst,how_dict={p:"uniform" for p in pst.adj_par_names[:2]},
                                                  num_reals=100,
                                                  partial=False)

    ea = evol_proto.EvolAlg(pst,num_slaves=20,port=4005,verbose=True)
    obj_dict = {}
    obj_dict[obj_names[0]] = "min"
    obj_dict[obj_names[1]] = "max"
    ea.initialize(obj_dict,par_ensemble=pe,num_dv_reals=100,risk=0.51,dv_names=pst.adj_par_names[2:])

    #oe = pyemu.ObservationEnsemble.from_id_gaussian_draw(pst=pst,num_reals=500)
    oe = ea.obs_ensemble
    # call the nondominated sorting
    is_nondom = ea.obj_func.is_nondominated_continuous(oe)
    obj = oe.loc[:,obj_names]
    obj.loc[is_nondom,"is_nondom"] = is_nondom
    #print(obj)
    import matplotlib.pyplot as plt
    plt.scatter(obj.iloc[:,0],obj.iloc[:,1],color="0.5",marker='.',alpha=0.5)
    ind = obj.loc[is_nondom,:]
    plt.scatter(ind.iloc[:, 0], ind.iloc[:, 1], color="m", marker='.',s=20)
    plt.show()


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
