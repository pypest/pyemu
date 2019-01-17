from pyemu.prototypes.NSGA_II import NSGA_II
import pyemu
import os


def test():
    os.chdir(os.path.join('moouu', 'StochasticProblemSuite', 'template'))
    pst = pyemu.Pst('StochasticParaboloid.pst')
    print(pst.par_names)
    dv_names = pst.par_names[:2]
    evolAlg = NSGA_II(pst, verbose=True, slave_dir='template')
    obj_func_dict = {obj_func: 'min' for obj_func in pst.obs_names}
    evolAlg.initialize(obj_func_dict=obj_func_dict, num_par_reals=5, num_dv_reals=5, dv_names=dv_names)
    evolAlg.update()


if __name__ == "__main__":
    test()