import os
if not os.path.exists("temp"):
    os.mkdir("temp")


def mat_test():
    import os
    import numpy as np
    from pyemu.mat import Jco,Cov,concat
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

    result = first.T.x * third
    result = 2.0 * third

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

    result.to_sparse()

    # drop testing
    second.drop("p2",axis=0)
    assert second.shape == (2, 2)

    third.drop("o1",axis=1)
    assert third.shape == (3, 3)

    first.drop("p1",axis=1)
    assert first.shape == (4,2)

    first.drop("o4",axis=0)
    assert first.shape == (3,2)

    try:
        concat([first,third])
    except:
        pass
    else:
        raise Exception()

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
    try:
        forth = pyemu.mat.concat([first,third])
    except:
        pass
    else:
        raise Exception()

    forth = pyemu.Matrix(x=first.x,row_names=first.row_names,col_names=[str(i) for i in range(first.shape[1])])
    x = pyemu.mat.concat([first,forth])
    print(x)

    fifth = pyemu.Matrix(x=first.x, row_names=[str(i) for i in range(first.shape[0])], col_names=first.col_names)
    x = pyemu.mat.concat([first,fifth])
    print(x)


def pseudo_inv_test():
    import os
    import pyemu
    jco = pyemu.Jco.from_binary(os.path.join("mat","pest.jcb"))
    print(jco.shape)
    jpi = jco.pseudo_inv(maxsing=1)
    jpi = jco.pseudo_inv(maxsing=19)

    u1,s1,v1 = jco.pseudo_inv_components(2)
    print(s1.shape)
    assert s1.shape[0] == 2
    u2, s2, v2 = jco.pseudo_inv_components(2,truncate=False)
    assert s2.shape == jco.shape

    d = u1 - u2[:,:2]
    assert d.x.max() == 0.0




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
    assert np.abs(hp.x.sum() - np.diagonal(r.x).sum()) < 1.0e-6
    hp = r.hadamard_product(c)
    assert np.abs(hp.x.sum() - np.diagonal(r.x).sum()) < 1.0e-6


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

def to_pearson_test():
    import os
    import numpy as np
    import pandas as pd
    from pyemu import Schur
    #w_dir = os.path.join("..","..","verification","10par_xsec","master_opt0")
    w_dir = "la"
    forecasts = ["h01_08","h02_08"]
    sc = Schur(jco=os.path.join(w_dir,"pest.jcb"))
    sc.posterior_parameter.to_pearson()

def sigma_range_test():
    import pyemu
    cov8 = pyemu.la.Cov.from_parbounds(os.path.join("mat", "base_pest.pst"), sigma_range=8.0)
    covdefault = pyemu.la.Cov.from_parbounds(os.path.join("mat", "base_pest.pst"))
    print(cov8.df().iloc[0, 0])
    print(covdefault.df().iloc[0, 0])
    assert covdefault.df().iloc[0, 0] > cov8.df().iloc[0, 0]


def cov_replace_test():
    import os
    import numpy as np
    import pyemu

    pst = pyemu.Pst(os.path.join("pst","pest.pst"))
    cov1 = pyemu.Cov.from_parameter_data(pst)
    cov2 = pyemu.Cov(x=cov1.x[:3],names=cov1.names[:3],isdiagonal=True) * 3
    cov1.replace(cov2)
    print(cov1.x[0],cov2.x[0])
    assert cov1.x[0] == cov2.x[0]

    cov2 = pyemu.Cov(x=np.ones(cov1.shape) * 2,names=cov1.names[::-1])
    cov2 = cov2.get(row_names=cov2.names[:3],col_names=cov2.names[:3])
    cov1.replace(cov2)
    assert cov1.x[-1,-1] == 2.0


def cov_scale_offset_test():
    import os
    import numpy as np
    import pyemu

    pst = pyemu.Pst(os.path.join("pst", "pest.pst"))


    par = pst.parameter_data
    par.loc[:,"partrans"] = "none"
    cov1 = pyemu.Cov.from_parameter_data(pst)
    par.loc[:,"offset"] = 100
    cov2 = pyemu.Cov.from_parameter_data(pst)

    d = np.abs((cov1.x - cov2.x)).sum()
    assert d == 0.0,d

    pyemu.Cov.from_parameter_data(pst,scale_offset=False)
    assert np.abs(cov1.x - cov2.x).sum() == 0.0


def from_names_test():
    import os
    import pyemu

    rnames = ["row_{0}".format(i) for i in range(20)]
    cnames = ["col_{0}".format(i) for i in range(40)]

    m = pyemu.Matrix.from_names(rnames,cnames)
    assert m.shape[0] == len(rnames)
    assert m.shape[1] == len(cnames)

    pst_name = os.path.join("pst","pest.pst")
    pst = pyemu.Pst(pst_name)
    j = pyemu.Jco.from_pst(pst)

    jj = pyemu.Jco.from_pst(pst_name)

    assert j.shape == jj.shape


    c = pyemu.Cov.from_names(rnames,cnames)
    assert type(c) == pyemu.Cov

def from_uncfile_test():
    import os
    import numpy as np
    import pyemu

    cov_full = pyemu.Cov.from_uncfile(os.path.join("mat","param.unc"))
    cov_kx = pyemu.Cov.from_ascii(os.path.join("mat","cov_kx.mat"))
    cov_full_kx = cov_full.get(row_names=cov_kx.row_names,col_names=cov_kx.col_names)
    assert np.abs((cov_kx - cov_full_kx).x).max() == 0.0


def copy_test():

    import os
    import numpy as np
    import pyemu

    cov_full = pyemu.Cov.from_uncfile(os.path.join("mat", "param.unc"))
    cov_copy = cov_full.copy()
    assert np.abs((cov_copy- cov_full).x).max() == 0.0
    cov_full = cov_full + 2.0
    #print(cov_full.row_names)
    assert np.abs((cov_copy - cov_full).x).max() == 2.0,np.abs((cov_copy - cov_full).x).max()



def indices_test():
    import os
    from datetime import datetime
    import numpy as np
    import pyemu

    nrow = 1000
    ncol = 1000

    rnames = ["row_{0}".format(i) for i in range(nrow)]
    cnames = ["col_{0}".format(i) for i in range(ncol)]

    m = pyemu.Matrix.from_names(rnames, cnames)
    assert m.shape[0] == len(rnames)
    assert m.shape[1] == len(cnames)


    try:
        m.indices(cnames, 0)
    except:
        pass
    else:
        raise Exception()

    cycles = 10
    s = datetime.now()
    for _ in range(cycles):
        idx1 = m.old_indices(cnames,1)
    t1 = (datetime.now() - s).total_seconds()

    s = datetime.now()
    for _ in range(cycles):
        idx2 = m.indices(cnames,1)
    t2 = (datetime.now() - s).total_seconds()
    print(t1,t2)
    assert np.allclose(idx1,idx2)


def coo_tests():
    import os
    from datetime import datetime
    import numpy as np
    import pyemu

    nrow = 100
    ncol = 1000

    rnames = ["row_{0}".format(i) for i in range(nrow)]
    cnames = ["col_{0}".format(i) for i in range(ncol)]

    x = np.random.random((nrow,ncol))

    m = pyemu.Matrix(x=x,row_names=rnames, col_names=cnames)
    assert m.shape[0] == len(rnames)
    assert m.shape[1] == len(cnames)

    mname = os.path.join("temp","temp.jcb")

    pyemu.mat.save_coo(m.to_sparse(), m.row_names, m.col_names, mname)
    mm = pyemu.Matrix.from_binary(mname)
    assert np.array_equal(m.x, mm.x)
    os.remove(mname)

    m.to_coo(mname)
    mm = pyemu.Matrix.from_binary(mname)
    assert np.array_equal(m.x,mm.x)
    os.remove(mname)

    m.to_coo(mname,chunk=1)
    mm = pyemu.Matrix.from_binary(mname)
    assert np.array_equal(m.x, mm.x)
    os.remove(mname)

    m.to_coo(mname,chunk=100000)
    mm = pyemu.Matrix.from_binary(mname)
    assert np.array_equal(m.x,mm.x)
    os.remove(mname)

    m.to_coo(mname,chunk=1000)
    mm = pyemu.Matrix.from_binary(mname)
    assert np.array_equal(m.x, mm.x)
    os.remove(mname)



    m.to_binary(mname)
    mm = pyemu.Matrix.from_binary(mname)
    assert np.array_equal(m.x, mm.x)
    os.remove(mname)

    m.to_binary(mname)
    mm = pyemu.Matrix.from_binary(mname)
    assert np.array_equal(m.x, mm.x)
    os.remove(mname)

    m.to_binary(mname)
    mm = pyemu.Matrix.from_binary(mname)
    assert np.array_equal(m.x, mm.x)
    os.remove(mname)

    m.to_binary(mname)
    mm = pyemu.Matrix.from_binary(mname)
    assert np.array_equal(m.x, mm.x)
    os.remove(mname)


def sparse_constructor_test():
    import os
    from datetime import datetime
    import numpy as np
    import pyemu

    nrow = 100
    ncol = 100

    rnames = ["row_{0}".format(i) for i in range(nrow)]
    cnames = ["col_{0}".format(i) for i in range(ncol)]

    x = np.random.random((nrow, ncol))

    m = pyemu.Matrix(x=x, row_names=rnames, col_names=cnames)

    sm = pyemu.SparseMatrix.from_matrix(m)

    mname = os.path.join("temp","test.jcb")
    m.to_binary(mname)
    sm = pyemu.SparseMatrix.from_binary(mname)

    sm.to_coo(mname)
    m1 = sm.to_matrix()
    m = pyemu.Matrix.from_binary(mname)
    assert np.array_equal(m1.x,m.x)

def sparse_extend_test():
    import os
    from datetime import datetime
    import numpy as np
    import pyemu

    nrow = 5
    ncol = 5

    rnames = ["row_{0}".format(i) for i in range(nrow)]
    cnames = ["col_{0}".format(i) for i in range(ncol)]

    x = np.random.random((nrow, ncol))

    m = pyemu.Matrix(x=x, row_names=rnames, col_names=cnames)

    sm = pyemu.SparseMatrix.from_matrix(m)

    try:
        sm.block_extend_ip(m)
    except:
        pass
    else:
        raise Exception()

    m = pyemu.Matrix(x,row_names=['t{0}'.format(i) for i in range(nrow)],col_names=m.col_names)
    try:
        sm.block_extend_ip(m)
    except:
        pass
    else:
        raise Exception()


    m = pyemu.Matrix(x,row_names=['r{0}'.format(i) for i in range(nrow)],
                     col_names=['r{0}'.format(i) for i in range(ncol)])
    sm.block_extend_ip(m)
    m1 = sm.to_matrix()
    d = m.x - m1.x[m.shape[0]:,m.shape[1]:]
    assert d.sum() == 0

    m = pyemu.Cov(x=np.atleast_2d(np.ones(nrow)),names=['p{0}'.format(i) for i in range(nrow)],isdiagonal=True)
    sm.block_extend_ip(m)
    d = m.as_2d - sm.to_matrix().x[-nrow:,-nrow:]
    assert d.sum() == 0

    m1 = pyemu.Matrix(x=x, row_names=rnames, col_names=cnames)

    sm1 = pyemu.SparseMatrix.from_matrix(m1)
    sm2 = pyemu.SparseMatrix.from_matrix(m)
    sm1.block_extend_ip(sm2)

    m2 = sm1.to_matrix()
    d = m.as_2d - m2.x[m.shape[0]:,m.shape[1]:]
    assert d.sum() == 0


def sparse_get_test():
    import os
    from datetime import datetime
    import numpy as np
    import pyemu

    nrow = 5
    ncol = 5

    rnames = ["row_{0}".format(i) for i in range(nrow)]
    cnames = ["col_{0}".format(i) for i in range(ncol)]

    x = np.random.random((nrow, ncol))

    m = pyemu.Matrix(x=x, row_names=rnames, col_names=cnames)

    sm = pyemu.SparseMatrix.from_matrix(m)
    m1 = sm.get_matrix(rnames[0],cnames)
    d = m1.x - m.x[0,:]
    assert d.sum() == 0

    sm = pyemu.SparseMatrix.from_matrix(m)
    m1 = sm.get_matrix(rnames[:2], cnames)
    d = m1.x - m.x[:2, :]
    assert d.sum() == 0

    sm = pyemu.SparseMatrix.from_matrix(m)
    m1 = sm.get_matrix(rnames, cnames[0])
    d = m1.x - m.x[:, 0]
    assert d.sum() == 0

    sm = pyemu.SparseMatrix.from_matrix(m)
    m1 = sm.get_matrix(rnames, cnames[:2])
    d = m1.x - m.x[:, :2]
    assert d.sum() == 0

    sm = pyemu.SparseMatrix.from_matrix(m)
    m1 = sm.get_matrix(rnames, cnames)
    d = m1.x - m.x
    assert d.sum() == 0


def sparse_get_sparse_test():
    import os
    from datetime import datetime
    import numpy as np
    import pyemu

    nrow = 5
    ncol = 5

    rnames = ["row_{0}".format(i) for i in range(nrow)]
    cnames = ["col_{0}".format(i) for i in range(ncol)]

    x = np.random.random((nrow, ncol))

    m = pyemu.Matrix(x=x, row_names=rnames, col_names=cnames)

    sm = pyemu.SparseMatrix.from_matrix(m)
    m1 = sm.get_matrix(rnames[0],cnames)
    d = m1.x - m.x[0,:]
    assert d.sum() == 0

    sm = pyemu.SparseMatrix.from_matrix(m)
    m1 = sm.get_sparse_matrix(rnames[:2], cnames).to_matrix()
    d = m1.x - m.x[:2, :]
    assert d.sum() == 0

    sm = pyemu.SparseMatrix.from_matrix(m)
    m1 = sm.get_sparse_matrix(rnames, cnames[0]).to_matrix()
    d = m1.x - m.x[:, 0]
    assert d.sum() == 0

    sm = pyemu.SparseMatrix.from_matrix(m)
    m1 = sm.get_sparse_matrix(rnames, cnames[:2]).to_matrix()
    d = m1.x - m.x[:, :2]
    assert d.sum() == 0

    sm = pyemu.SparseMatrix.from_matrix(m)
    m1 = sm.get_sparse_matrix(rnames, cnames).to_matrix()
    d = m1.x - m.x
    assert d.sum() == 0


def df_tests():
    import os
    import numpy as np
    import pandas as pd
    import pyemu

    nrow = 5
    ncol = 5

    rnames = ["row_{0}".format(i) for i in range(nrow)]
    cnames = ["col_{0}".format(i) for i in range(ncol)]

    x = np.random.random((nrow, ncol))

    m = pyemu.Matrix(x=x, row_names=rnames, col_names=cnames)

    df = pd.DataFrame(data=x,columns=cnames,index=rnames)

    #sub
    d = m - df
    assert d.x.max() == 0.0

    d = df - m.x #returns a df
    #print(d.max())

    # add
    d = (m + df) - (df * 2)
    assert d.x.max() == 0.0
    d = (df * 2) - (m + df).x #returns a df

    # mul
    d = (m * df.T) - np.dot(m.x,df.T.values)
    assert d.x.max() == 0.0

    # hadamard
    d = (m.hadamard_product(df)) - (m.x * df)
    assert d.x.max() == 0.0









if __name__ == "__main__":
    #df_tests()
    # cov_scale_offset_test()
    #coo_tests()
    # indices_test()
    #mat_test()
    # load_jco_test()
    # extend_test()
    # pseudo_inv_test()
    drop_test()
    # get_test()
    # cov_identity_test()
    # hadamard_product_test()
    # get_diag_test()
    # to_pearson_test()
    # sigma_range_test()
    # cov_replace_test()
    # from_names_test()
    #from_uncfile_test()
    # copy_test()
    # sparse_constructor_test()
    # sparse_extend_test()
    # sparse_get_test()
    # sparse_get_sparse_test()