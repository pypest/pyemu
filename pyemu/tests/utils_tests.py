
def fac2real_test():
    import os
    import pyemu
    pp_file = os.path.join("utils","points1.dat")
    factors_file = os.path.join("utils","factors1.dat")
    pyemu.utils.gw_utils.fac2real(pp_file,factors_file,
                                  out_file=os.path.join("utils","test.ref"))


def vario_test():
    import numpy as np
    import pyemu
    contribution = 0.1
    a = 2.0
    for const in [pyemu.utils.geostats.ExpVario,pyemu.utils.geostats.GauVario,
                  pyemu.utils.geostats.SphVario]:

        v = const(contribution,a)
        h = v.h_function(np.array([0.0]))
        assert h == contribution
        h = v.h_function(np.array([a*1000]))
        assert h == 0.0

        v2 = const(contribution,a,anisotropy=2.0,bearing=90.0)
        print(v2.h_function(np.array([a])))


def aniso_test():

    import pyemu
    contribution = 0.1
    a = 2.0
    for const in [pyemu.utils.geostats.ExpVario,pyemu.utils.geostats.GauVario,
                  pyemu.utils.geostats.SphVario]:

        v = const(contribution,a)
        v2 = const(contribution,a,anisotropy=2.0,bearing=90.0)
        v3 = const(contribution,a,anisotropy=2.0,bearing=0.0)
        pt0 = (0,0)
        pt1 = (1,0)
        assert v.covariance(pt0,pt1) == v2.covariance(pt0,pt1)

        pt0 = (0,0)
        pt1 = (0,1)
        assert v.covariance(pt0,pt1) == v3.covariance(pt0,pt1)

def geostruct_test():
    import pyemu
    v1 = pyemu.utils.geostats.ExpVario(0.1,2.0)
    v2 = pyemu.utils.geostats.GauVario(0.1,2.0)
    v3 = pyemu.utils.geostats.SphVario(0.1,2.0)

    g = pyemu.utils.geostats.GeoStruct(0.2,[v1,v2,v3])
    pt0 = (0,0)
    pt1 = (0,0)
    print(g.covariance(pt0,pt1))
    assert g.covariance(pt0,pt1) == 0.5

    pt0 = (0,0)
    pt1 = (1.0e+10,0)
    assert g.covariance(pt0,pt1) == 0.2

def struct_file_test():
    import os
    import pyemu
    struct = pyemu.utils.geostats.read_struct_file(
            os.path.join("utils","struct.dat"))[0]
    print(struct)
    pt0 = (0,0)
    pt1 = (0,0)
    assert struct.covariance(pt0,pt1) == struct.nugget + \
                                         struct.variograms[0].contribution


def covariance_matrix_test():
    import os
    import pandas as pd
    import pyemu

    pts = pd.read_csv(os.path.join("utils","points1.dat"),delim_whitespace=True,
                      header=None,names=["name","x","y"],usecols=[0,1,2])
    struct = pyemu.utils.geostats.read_struct_file(
            os.path.join("utils","struct.dat"))[0]
    struct.variograms[0].covariance_matrix(pts.x,pts.y,names=pts.name)

    print(struct.covariance_matrix(pts.x,pts.y,names=pts.name).x)

if __name__ == "__main__":
    #fac2real_test()
    vario_test()
    geostruct_test()
    aniso_test()
    struct_file_test()
    covariance_matrix_test()