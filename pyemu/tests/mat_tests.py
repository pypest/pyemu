
def mat_test():
    import os
    import numpy as np
    from pyemu.mat import Jco,Cov
    test_dir = os.path.join("mat")
    if not os.path.exists(test_dir):
        os.mkdir(test_dir)
    arr = np.arange(0,12)
    arr.resize(4,3)
    first = Jco(x=arr,col_names=["p1","p2","p3"],row_names=["o1","o2","o3","o4"])
    first.to_binary(os.path.join(test_dir,"test.bin"))
    first.from_binary(os.path.join(test_dir,"test.bin"))

    first = Jco(x=np.ones((4,3)),col_names=["p1","p2","p3"],row_names=["o1","o2","o3","o4"])
    second = Cov(x=np.ones((3,3))+1.0,names=["p1","p2","p3"],isdiagonal=False)
    third = Cov(x=np.ones((4,1))+2.0,names=["o1","o2","o3","o4"],isdiagonal=True)
    third.to_uncfile(os.path.join(test_dir,"test.unc"),
                     covmat_file=os.path.join(test_dir,"cov.mat"))
    third.from_uncfile(os.path.join(test_dir,"test.unc"))

    si = second.identity
    result = second - second.identity
    # add and sub
    newfirst = first.get(row_names=["o1"],col_names="p1")
    result = newfirst - first
    result = first - newfirst
    result = newfirst + first
    result = first + newfirst
    newfirst = first.get(row_names=["o1","o2"],col_names="p1")
    result = newfirst - first
    result = first - newfirst
    result = newfirst + first
    result = first + newfirst
    newfirst = first.get(row_names=["o1","o2"],col_names=["p1","p3"])
    result = newfirst - first
    result = first - newfirst
    result = newfirst + first
    result = first + newfirst

    # mul test
    result = first.T * third * first
    result = first * second

    newfirst = first.get(col_names="p1")
    result = newfirst * second
    result = second * newfirst.T

    newfirst = first.get(col_names=["p1","p2"])
    result = newfirst * second
    result = second * newfirst.T

    newfirst = first.get(row_names=["o1"])
    result = newfirst * second
    result = second * newfirst.T

    newfirst = first.get(row_names=["o1","o2"])
    result = newfirst * second
    result = second * newfirst.T
    result = newfirst.T * third * newfirst

    newthird = third.get(row_names=["o1"])
    result = first.T * newthird * first

    # drop testing
    second.drop("p2",axis=0)
    assert second.shape == (2, 2)

    third.drop("o1",axis=1)
    assert third.shape == (3, 3)

    first.drop("p1",axis=1)
    assert first.shape == (4,2)

    first.drop("o4",axis=0)
    assert first.shape == (3,2)

def load_jco_test():
    import os
    import pyemu
    jco = pyemu.Jco.from_binary(os.path.join("mat","base_pest.jco"))
    sc = pyemu.Schur(jco=os.path.join("mat","base_pest.jco"),
                     parcov=os.path.join("mat","parameters.unc"))
    print(sc.get_parameter_summary())

def extend_test():
    import numpy as np
    import pyemu
    first = pyemu.Cov(x=np.ones((3,3))+1.0,names=["p1","p2","p3"],isdiagonal=False)
    second = pyemu.Cov(x=np.ones((4,1))+2.0,names=["o1","o2","o3","o4"],isdiagonal=True)

    third = first.extend(second)
    print(third)
    assert third.x[:,0].sum() == 6
    assert third.x[0,:].sum() == 6
    assert third.x[:,2].sum() == 6
    assert third.x[2,:].sum() == 6
    assert third.x[:,3].sum() == 3
    assert third.x[3,:].sum() == 3
    assert third.x[:,6].sum() == 3
    assert third.x[6,:].sum() == 3


if __name__ == "__main__":
    #mat_test()
    #load_jco_test()
    extend_test()