
def fac2real_test():
    import os
    import pyemu
    pp_file = os.path.join("utils","points1.dat")
    factors_file = os.path.join("utils","factors1.dat")
    pyemu.utils.gw_utils.fac2real(pp_file,factors_file,
                                  out_file=os.path.join("utils","test.ref"))


def vario_test():
    import pyemu
    contribution = 0.1
    a = 2.0
    for const in [pyemu.utils.geostats.ExpVario,pyemu.utils.geostats.GauVario,
                  pyemu.utils.geostats.SphVario]:

        v = const(contribution,a)
        assert v.h_function(0.0) == contribution
        assert v.h_function(a*1000) == 0.0



def geostruct_test():
    import pyemu
    v1 = pyemu.utils.geostats.ExpVario(0.1,2.0)
    v2 = pyemu.utils.geostats.GauVario(0.1,2.0)
    v3 = pyemu.utils.geostats.SphVario(0.1,2.0)

    g = pyemu.utils.geostats.GeoStruct(0.2,[v1,v2,v3])
    pt0 = (0,0)
    pt1 = (0,0)
    print(g.covariance(pt0,pt1))


if __name__ == "__main__":
    #fac2real_test()
    #vario_test()
    geostruct_test()