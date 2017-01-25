import os
if not os.path.exists("temp"):
    os.mkdir("temp")


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


def drop_test():
    import numpy as np
    import pyemu
    arr = np.arange(0,12)
    arr.resize(4,3)
    first = pyemu.Jco(x=arr,col_names=["p2","p1","p3"],row_names=["o4","o1","o3","o2"])
    print(first)
    first.drop(["o2","o1"],axis=0)
    print(first)
    first.drop(["p1","p3"],axis=1)
    print(first)
    assert first.row_names == ["o4","o3"]
    assert first.col_names == ["p2"]
    t_array = np.atleast_2d(np.array([0,6])).transpose()
    print(first.x,t_array)
    assert np.array_equal(first.x,t_array)

def get_test():
    import numpy as np
    import pyemu
    arr = np.arange(0,12)
    arr.resize(4,3)
    first = pyemu.Jco(x=arr,col_names=["p2","p1","p3"],row_names=["o4","o1","o3","o2"])
    #print(first)
    #second = first.get(row_names=["o1","o3"])
    second = first.get(row_names=first.row_names)
    assert np.array_equal(first.x,second.x)

    cov1 = pyemu.Cov(x=np.atleast_2d(np.arange(10)).transpose(),names=["c{0}".format(i) for i in range(10)],isdiagonal=True)

    print(cov1)

    cov2 = cov1.get(row_names=cov1.row_names)
    print(cov2)




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


def pseudo_inv_test():
    import os
    import pyemu
    jco = pyemu.Jco.from_binary(os.path.join("mat","pest.jcb"))
    print(jco.shape)
    jpi = jco.pseudo_inv(maxsing=1)
    jpi = jco.pseudo_inv(maxsing=19)

def cov_identity_test():
    import os
    import numpy as np
    import pyemu
    n = 100
    names = ["name_{0}".format(i) for i in range(n)]
    arr = np.random.random(n*n)
    arr.resize((n,n))
    cov = pyemu.Cov(x=arr*arr.transpose(),names=names)
    cov *= 2.0
    cov_i = pyemu.Cov.identity_like(cov)
    cov_i *= 2.0
    assert cov_i.x[0,0] == 2.0

def hadamard_product_test():
    import os
    import numpy as np
    import pyemu
    jco = pyemu.Jco.from_binary(os.path.join("mat", "pest.jcb"))

    z = jco.zero2d
    #print(jco.shape)
    #print(z.shape)

    hp = jco.hadamard_product(z)
    assert hp.x.sum() == 0.0
    hp = z.hadamard_product(hp)
    assert hp.x.sum() == 0.0

    c = pyemu.Cov(x=np.ones((jco.shape[0],1)),names=jco.row_names,isdiagonal=True)
    r = pyemu.Matrix(x=np.random.rand(c.shape[0],c.shape[0]),
                     row_names=c.row_names,col_names=c.col_names)
    hp = c.hadamard_product(r)
    assert hp.x.sum() == np.diagonal(r.x).sum()
    hp = r.hadamard_product(c)
    assert hp.x.sum() == np.diagonal(r.x).sum()


def get_diag_test():
    import numpy as np
    import pyemu
    n = 100
    col_names = ["cname_{0}".format(i) for i in range(n)]
    row_names = ["rname_{0}".format(i) for i in range(n)]
    arr = np.random.random(n*n)
    arr.resize((n,n))
    mat = pyemu.Matrix(x=arr,row_names=row_names,
                       col_names=col_names)
    diag_mat = mat.get_diagonal_vector(col_name="test")
    s1 = np.diag(arr).sum()
    s2 = diag_mat.x.sum()
    assert s1 == s2

if __name__ == "__main__":
    #mat_test()
    #load_jco_test()
    #extend_test()
    #pseudo_inv_test()
    #drop_test()
    #get_test()
    #cov_identity_test()
    #hadamard_product_test()
    get_diag_test()