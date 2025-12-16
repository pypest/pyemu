import os
import shutil

import pytest
from pathlib import Path

@pytest.fixture
def setup_empty_mat_temp(tmp_path):
    shutil.copy(os.path.join("pst", "pest.pst"), tmp_path)
    test_dir = os.path.join(tmp_path, "mat")
    if not os.path.exists(test_dir):
        os.mkdir(test_dir)
    bd = Path.cwd()
    os.chdir(tmp_path)
    try:
        yield "mat"
    except Exception as e:
        os.chdir(bd)
        raise e
    os.chdir(bd)


@pytest.fixture
def copy_mat_temp(tmp_path):
    shutil.copy(os.path.join("pst", "pest.pst"), tmp_path)
    test_dir = os.path.join(tmp_path, "mat")
    shutil.copytree("mat", test_dir)
    bd = Path.cwd()
    os.chdir(tmp_path)
    try:
        yield "mat"
    except Exception as e:
        os.chdir(bd)
        raise e
    os.chdir(bd)


@pytest.fixture
def copy_la_temp(tmp_path):
    test_dir = os.path.join(tmp_path, "la")
    shutil.copytree("la", test_dir)
    bd = Path.cwd()
    os.chdir(tmp_path)
    try:
        yield "la"
    except Exception as e:
        os.chdir(bd)
        raise e
    os.chdir(bd)


def mat_test(setup_empty_mat_temp):
    import os
    import numpy as np
    from pyemu.mat import Jco,Cov,concat
    test_dir = setup_empty_mat_temp
    arr = np.arange(0,12)
    arr.resize(4,3)
    first = Jco(x=arr, col_names=["p1","p2","p3"],
                row_names=["o1","o2","o3","o4"])
    first.to_binary(os.path.join(test_dir, "test.bin"))
    first.from_binary(os.path.join(test_dir, "test.bin"))

    first = Jco(x=np.ones((4,3)),col_names=["p1","p2","p3"],
                row_names=["o1","o2","o3","o4"])
    second = Cov(x=np.ones((3,3))+1.0,names=["p1","p2","p3"],
                 isdiagonal=False)
    third = Cov(x=np.ones((4,1))+2.0,names=["o1","o2","o3","o4"],
                isdiagonal=True)
    third.to_uncfile(os.path.join(test_dir, "test.unc"),
                     covmat_file=os.path.join(test_dir, "cov.mat"))
    third.from_uncfile(os.path.join(test_dir, "test.unc"))

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
    first = pyemu.Jco(x=arr,col_names=["p2","p1","p3"],
                      row_names=["o4","o1","o3","o2"])
    #print(first)
    #second = first.get(row_names=["o1","o3"])
    second = first.get(row_names=first.row_names)
    assert np.array_equal(first.x,second.x)

    cov1 = pyemu.Cov(x=np.atleast_2d(np.arange(10)).transpose(),
                     names=["c{0}".format(i) for i in range(10)],
                     isdiagonal=True)

    print(cov1)

    cov2 = cov1.get(row_names=cov1.row_names)
    print(cov2)


def load_jco_test(copy_mat_temp):
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


def pseudo_inv_test(copy_mat_temp):
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


def hadamard_product_test(copy_mat_temp):
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


def to_pearson_test(copy_la_temp):
    import os
    from pyemu import Schur
    #w_dir = os.path.join("..","..","verification","10par_xsec","master_opt0")
    w_dir = "la"
    forecasts = ["h01_08","h02_08"]
    sc = Schur(jco=os.path.join(w_dir,"pest.jcb"))
    sc.posterior_parameter.to_pearson()


def sigma_range_test(copy_mat_temp):
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


def from_uncfile_test(copy_mat_temp):
    import os
    import numpy as np
    import pyemu
    cov_full = pyemu.Cov.from_uncfile(os.path.join("mat", "param_path.unc"))

    #cov_full = pyemu.Cov.from_uncfile(os.path.join("mat","param.unc"))
    cov_kx = pyemu.Cov.from_ascii(os.path.join("mat","cov_kx.mat"))
    cov_full_kx = cov_full.get(row_names=cov_kx.row_names,col_names=cov_kx.col_names)
    assert np.abs((cov_kx - cov_full_kx).x).max() == 0.0


def icode_minus_one_test(copy_mat_temp):
    import os
    import numpy as np
    import pyemu
    pst = pyemu.Pst("pest.pst")
    c1 = pyemu.Cov.from_parameter_data(pst)
    n1 = os.path.join("mat","test.cov")
    c1.to_ascii(n1,icode=-1)
    c2 = pyemu.Cov.from_ascii(n1)
    d = c1 - c2
    print(d.x.max())
    assert d.x.max() == 0,d.x.max()

    c3 = pyemu.Cov.from_ascii(os.path.join("mat","cov_kx.mat"))
    try:
        c3.to_ascii(n1,icode=-1)
    except:
        pass
    else:
        raise Exception("should have failed")


def copy_test(copy_mat_temp):
    import os
    import numpy as np
    import pyemu
    cov_full = pyemu.Cov.from_uncfile(os.path.join("mat", "param_path.unc"))
    cov_copy = cov_full.copy()
    assert np.abs((cov_copy- cov_full).x).max() == 0.0
    cov_full = cov_full + 2.0
    #print(cov_full.row_names)
    assert np.abs((cov_copy - cov_full).x).max() == 2.0,np.abs((cov_copy - cov_full).x).max()


def indices_test():
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


def coo_test(setup_empty_mat_temp):
    import os
    import numpy as np
    import pyemu
    nrow = 100
    ncol = 1000
    wd = setup_empty_mat_temp
    rnames = ["row_{0}".format(i) for i in range(nrow)]
    cnames = ["col_{0}".format(i) for i in range(ncol)]

    x = np.random.random((nrow,ncol))

    m = pyemu.Matrix(x=x,row_names=rnames, col_names=cnames)
    assert m.shape[0] == len(rnames)
    assert m.shape[1] == len(cnames)

    mname = os.path.join(wd,"temp.jcb")

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


def df_test():
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


def dense_mat_format_test(setup_empty_mat_temp):
    import numpy as np
    import pyemu
    from datetime import datetime
    wd = setup_empty_mat_temp
    nrow = 100
    ncol = 500000

    long_str = ""
    for _ in range(35):
        long_str += "long"
    rnames = [long_str+"row_{0}".format(i) for i in range(nrow)]
    cnames = [long_str+"col_{0}".format(i) for i in range(ncol)]

    arr = np.random.random((nrow,ncol))
    matfile = os.path.join(wd, "dense.bin")
    m = pyemu.Matrix(x=arr, row_names=rnames, col_names=cnames)
    f = m.to_dense(matfile, close=True)
    row_names,row_offsets,col_names,success = pyemu.Matrix.get_dense_binary_info(matfile)
    assert success
    assert col_names == cnames
    assert row_names == rnames
    assert len(row_offsets) == len(row_names)

    only_rows = row_names[::5]
    data_rows_only, row_names_only,col_names_only = pyemu.Matrix.read_dense(matfile,only_rows=only_rows)

    assert len(row_names_only) == len(only_rows)
    assert data_rows_only.shape == (len(only_rows),ncol)
    assert len(col_names_only) == len(col_names)
    dif = np.abs(arr[::5,:] - data_rows_only)
    print(dif.max())
    assert dif.max() == 0.0

    m1 = pyemu.Matrix.from_binary(matfile)
    print(m1.shape)
    assert m1.shape == (nrow, ncol)
    d = np.abs(m.x - m1.x).sum()
    print(d)
    assert d < 1.0e-10

    matruncfile = os.path.join(wd, "dense_trunc.bin")
    f_in = open(matfile, "rb")
    f_out = open(matruncfile, "wb")
    f_out.write(f_in.read(ncol * (len(long_str)) + int(nrow / 2) * ncol * 8))
    f_in.close()
    f_out.close()
    try:
        m1 = pyemu.Matrix.from_binary(matruncfile,forgive=False)
    except:
        pass
    else:
        raise Exception("should have failed")

    m1 = pyemu.Matrix.from_binary(matruncfile, forgive=True)


    m = pyemu.Matrix(x=arr,row_names=rnames,col_names=cnames)
    f = m.to_dense(matfile,close=False)
    new_rnames = [r+"new" for r in rnames]
    m.row_names = new_rnames
    m.to_dense(f,close=True)

    m1 = pyemu.Matrix.from_binary(matfile)
    print(m1.shape)
    assert m1.shape == (nrow*2,ncol)
    arr2 = np.zeros((nrow*2,ncol))
    arr2[:nrow,:] = arr
    arr2[nrow:,:] = arr
    d = np.abs(arr2 - m1.x).sum()
    print(d)
    assert d < 1.0e-10,d

    s1 = datetime.now()
    for _ in range(1):
        m.to_dense(matfile)
        pyemu.Matrix.read_binary(matfile)
    e1 = datetime.now()
    s2 = datetime.now()
    jcbfile = os.path.join(wd, "dense.jcb")
    for _ in range(1):
        m.to_coo(jcbfile)
        pyemu.Matrix.read_binary(jcbfile)
    e2 = datetime.now()
    print((e1-s1).total_seconds())
    print((e2-s2).total_seconds())


def from_uncfile_firstlast_test(setup_empty_mat_temp):
    import os
    import pyemu
    wd = setup_empty_mat_temp
    pst = pyemu.Pst("pest.pst")
    c1 = pyemu.Cov.from_parameter_data(pst)
    unc_file = os.path.join(wd,"fistlast.unc")
    cov_file = "firstlast.cov"
    c1.to_uncfile(unc_file,covmat_file=cov_file)
    c2 = pyemu.Cov.from_uncfile(unc_file,pst=pst)
    lines = open(unc_file,'r').readlines()
    with open(unc_file,'w') as f:
        for line in lines[:-1]:
            f.write(line)
        f.write("first_parameter {0}\n".format(pst.par_names[0]))
        f.write("last_parameters {0}\n".format(pst.par_names[-1]))
        f.write(lines[-1])
    try:
        c2 = pyemu.Cov.from_uncfile(unc_file)
    except:
        pass
    else:
        raise Exception("should have failed")
    c2 = pyemu.Cov.from_uncfile(unc_file, pyemu.Pst("pest.pst"))
    with open(unc_file,'w') as f:
        for line in lines[:-1]:
            f.write(line)
        f.write("first_parameter {0}\n".format("junk1"))
        f.write("last_parameters {0}\n".format(pst.par_names[-1]))
        f.write(lines[-1])
    try:
        c2 = pyemu.Cov.from_uncfile(unc_file)
    except:
        pass
    else:
        raise Exception("should have failed")

    c1.to_uncfile(unc_file,covmat_file=cov_file)
    lines = open(unc_file, 'r').readlines()
    with open(unc_file, 'w') as f:
        for line in lines[:-1]:
            f.write(line)
        f.write("first_parameter {0}\n".format(pst.par_names[0]))
        f.write("last_parameters {0}\n".format("junk2"))
        f.write(lines[-1])
    try:
        c2 = pyemu.Cov.from_uncfile(unc_file)
    except:
        pass
    else:
        raise Exception("should have failed")

    c1.to_uncfile(unc_file, covmat_file=cov_file)
    lines = open(unc_file, 'r').readlines()
    with open(unc_file, 'w') as f:
        for line in lines[:-1]:
            f.write(line)
        f.write("first_parameter {0}\n".format(pst.par_names[0]))
        f.write("last_parameters {0}\n".format(pst.par_names[-2]))
        f.write(lines[-1])
    try:
        c2 = pyemu.Cov.from_uncfile(unc_file)
    except:
        pass
    else:
        raise Exception("should have failed")

    c1.to_uncfile(unc_file, covmat_file=cov_file)
    lines = open(unc_file, 'r').readlines()
    with open(unc_file, 'w') as f:
        for line in lines[:-1]:
            f.write(line)
        f.write("first_parameter {0}\n".format(pst.par_names[0]))
        f.write("last_parameters {0}\n".format(pst.par_names[-2]))
        f.write(lines[-1])
    try:
        c2 = pyemu.Cov.from_uncfile(unc_file)
    except:
        pass
    else:
        raise Exception("should have failed")


def trunc_names_test(tmp_path):
    import pyemu
    import pandas as pd

    a=pd.DataFrame({'a': [1,2,3], 'b': [4,5,6]})
    names = ['They','We','Reality']
    a['names'] =  names
    a.set_index('names',inplace=True, drop=True)
    am = pyemu.Matrix.from_dataframe(a)
    am.to_binary(tmp_path / 'a.binary.jcb')
    am.to_coo(tmp_path /'a.coo.jcb')
    # both read fine
    ar = pyemu.Matrix.from_binary(tmp_path / 'a.coo.jcb')
    ar = pyemu.Matrix.from_binary(tmp_path / 'a.binary.jcb')

    # use longer names
    names = ["They've done studies, you know. 60 percent of the time, it works every time.",
             "We’ll never survive! You’re only saying that because no one ever has.",
             "Reality is frequently inaccurate."]
    a['names'] =  [n.replace(" ","-",) for n in names]
    a.set_index('names',inplace=True, drop=True)
    am = pyemu.Matrix.from_dataframe(a)
    am.to_binary(tmp_path /'a.binary.jcb')
    am.to_coo(tmp_path /'a.coo.jcb')

    ar = pyemu.Matrix.from_binary(tmp_path /'a.coo.jcb')
    # long names save with .to_binary() can't be read with .from_binary()
    ar = pyemu.Matrix.from_binary(tmp_path /'a.binary.jcb')

if __name__ == "__main__":
    #df_tests()
    # cov_scale_offset_test()
    #coo_tests()
    # indices_test()
    #  mat_test()
    # load_jco_test()
    # extend_test()
    # pseudo_inv_test()
    #drop_test()
    # get_test()
    # cov_identity_test()
    # hadamard_product_test()
    # get_diag_test()
    # to_pearson_test()
    # sigma_range_test()
    # cov_replace_test()
    # from_names_test()
    # from_uncfile_test()
    # copy_test()
    # sparse_constructor_test()
    # sparse_extend_test()
    # sparse_get_test()
    # sparse_get_sparse_test()
    dense_mat_format_test(".")
    #icode_minus_one_test()
    #from_uncfile_firstlast_test()
    #trunc_names_test()
