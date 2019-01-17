from pyemu.prototypes.NSGA_II import NSGA_II
import pyemu
import os


def test():
    os.chdir(os.path.join('moouu', 'StochasticProblemSuite', 'template'))
    pst = pyemu.Pst('StochasticParaboloid.pst')
    print(pst.par_names)
    dv_names = pst.par_names[:2]
    dvs = pyemu.ParameterEnsemble.from_mixed_draws(pst, how_dict={p:'uniform' for p in dv_names}, partial=True)
    print(pst.parameter_data)
    print(pst.add_transform_columns())
    print(pst.parameter_data)
    #print(dvs)
    #evolAlg = NSGA_II(pst, verbose=True, slave_dir='template')
    #obj_func_dict = {obj_func: 'min' for obj_func in pst.obs_names}
    #evolAlg.initialize(obj_func_dict=obj_func_dict, num_par_reals=5, num_dv_reals=5, dv_names=dv_names)


if __name__ == "__main__":
    test()