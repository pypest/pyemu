import os
import shutil

import pyemu
import pandas as pd
import numpy as np

def add_base_test():
    pst = pyemu.Pst(os.path.join("pst", "pest.pst"))
    num_reals = 10
    pe = pyemu.ParameterEnsemble.from_gaussian_draw(pst,num_reals=num_reals)
    oe = pyemu.ObservationEnsemble.from_gaussian_draw(pst,num_reals=num_reals)
    pet = pe.copy()
    pet.transform()
    pe.add_base()
    assert pe.shape[0] == num_reals
    pet.add_base()
    assert pet.shape[0] == num_reals
    assert "base" in pe.index
    assert "base" in pet.index
    p = pe.loc["base",:]
    d = (pst.parameter_data.parval1 - pe.loc["base",:]).apply(np.abs)
    pst.add_transform_columns()
    d = (pst.parameter_data.parval1_trans - pet.loc["base", :]).apply(np.abs)
    assert d.max() == 0.0
    try:
        pe.add_base()
    except:
        pass
    else:
        raise Exception("should have failed")


    oe.add_base()
    assert oe.shape[0] == num_reals
    d = (pst.observation_data.loc[oe.columns,"obsval"] - oe.loc["base",:]).apply(np.abs)
    assert d.max() == 0
    try:
        oe.add_base()
    except:
        pass
    else:
        raise Exception("should have failed")

def nz_test():
    pst = pyemu.Pst(os.path.join("pst", "pest.pst"))
    num_reals = 10
    oe = pyemu.ObservationEnsemble.from_gaussian_draw(pst, fill=True, num_reals=num_reals)
    assert oe.shape[1] == pst.nobs
    oe_nz = oe.nonzero
    assert oe_nz.shape[1] == pst.nnz_obs
    assert list(oe_nz.columns.values) == pst.nnz_obs_names

def par_gauss_draw_consistency_test(tmp_path):

    pst = pyemu.Pst(os.path.join("pst","pest.pst"))
    pst.parameter_data.loc[pst.par_names[3::2],"partrans"] = "fixed"
    pst.parameter_data.loc[pst.par_names[0],"pargp"] = "test"

    num_reals = 5000

    pe1 = pyemu.ParameterEnsemble.from_gaussian_draw(pst,num_reals=num_reals)
    sigma_range = 4
    cov = pyemu.Cov.from_parameter_data(pst,sigma_range=sigma_range).to_2d()
    pe2 = pyemu.ParameterEnsemble.from_gaussian_draw(pst,cov=cov,num_reals=num_reals)
    pe3 = pyemu.ParameterEnsemble.from_gaussian_draw(pst,cov=cov,num_reals=num_reals,by_groups=False)

    pst.add_transform_columns()
    theo_mean = pst.parameter_data.parval1_trans
    adj_par = pst.parameter_data.loc[pst.adj_par_names,:].copy()
    ub,lb = adj_par.parubnd_trans,adj_par.parlbnd_trans
    theo_std = ((ub - lb) / sigma_range)

    for pe in [pe1,pe2,pe3]:
        assert pe.shape[0] == num_reals
        assert pe.shape[1] == pst.npar
        pe.transform()
        pe_mean = pe.loc[:,theo_mean.index].mean()
        d = (pe_mean - theo_mean).apply(np.abs)
        assert d.max() < 0.05,d.max()
        d = (pe.loc[:,pst.adj_par_names].std() - theo_std)
        assert d.max() < 0.05,d.max()

        # ensemble should be transformed so now lets test the I/O
        pe_org = pe.copy()

        pe.to_binary(os.path.join(tmp_path, "test.jcb"))
        pe = pyemu.ParameterEnsemble.from_binary(
            pst=pst, filename=os.path.join(tmp_path, "test.jcb")
        )
        pe.transform()
        pe._df.index = pe.index.map(np.int64)
        d = (pe - pe_org).apply(np.abs)
        assert d.max().max() < 1.0e-10, d.max().sort_values(ascending=False)

        pe.to_csv(os.path.join(tmp_path, "test.csv"))
        pe = pyemu.ParameterEnsemble.from_csv(
            pst=pst, filename=os.path.join(tmp_path, "test.csv")
        )
        pe.transform()
        d = (pe - pe_org).apply(np.abs)
        assert d.max().max() < 1.0e-10,d.max().sort_values(ascending=False)

def obs_gauss_draw_consistency_test(tmp_path):

    pst = pyemu.Pst(os.path.join("pst","pest.pst"))

    num_reals = 1000

    oe1 = pyemu.ObservationEnsemble.from_gaussian_draw(pst,num_reals=num_reals)
    cov = pyemu.Cov.from_observation_data(pst).to_2d()
    oe2 = pyemu.ObservationEnsemble.from_gaussian_draw(pst,cov=cov,num_reals=num_reals)
    oe3 = pyemu.ObservationEnsemble.from_gaussian_draw(pst,cov=cov,num_reals=num_reals,by_groups=False)

    theo_mean = pst.observation_data.obsval.copy()
    #theo_mean.loc[pst.nnz_obs_names] = 0.0
    theo_std = 1.0 / pst.observation_data.loc[pst.nnz_obs_names,"weight"]

    for oe in [oe1,oe2,oe3]:
        assert oe.shape[0] == num_reals
        assert oe.shape[1] == pst.nnz_obs
        d = (oe.mean() - theo_mean).apply(np.abs)
        assert d.max() < 0.01,d.sort_values()
        d = (oe.loc[:,pst.nnz_obs_names].std() - theo_std)
        assert d.max() < 0.01,d.sort_values()

        # ensemble should be transformed so now lets test the I/O
        oe_org = oe.copy()

        oe.to_binary(os.path.join(tmp_path, "test.jcb"))
        oe = pyemu.ObservationEnsemble.from_binary(
            pst=pst,  filename=os.path.join(tmp_path, "test.jcb"))
        oe._df.index = oe.index.map(np.int64)
        d = (oe - oe_org).apply(np.abs)
        assert d.max().max() < 1.0e-10, d.max().sort_values(ascending=False)

        oe.to_csv(os.path.join(tmp_path, "test.csv"))
        oe = pyemu.ObservationEnsemble.from_csv(
            pst=pst, filename=os.path.join(tmp_path, "test.csv"))
        d = (oe - oe_org).apply(np.abs)
        assert d.max().max() < 1.0e-10,d.max().sort_values(ascending=False)

def phi_vector_test():
    pst = pyemu.Pst(os.path.join("pst", "pest.pst"))
    num_reals = 10
    oe1 = pyemu.ObservationEnsemble.from_gaussian_draw(pst, num_reals=num_reals)
    pv = oe1.phi_vector

    for real in oe1.index:
        pst.res.loc[oe1.columns,"modelled"] = oe1.loc[real,:].values
        d = np.abs(pst.phi - pv.loc[real])
        assert d < 1.0e-10

def get_phi_vector_noise_obs_test():
    pst = pyemu.Pst('ends_master/freyberg6_run_ies.pst')
    oe = pyemu.ObservationEnsemble.from_csv(pst, 'ends_master/freyberg6_run_ies.0.obs.csv')
    phi = oe.phi_vector
    phi_noise = oe.get_phi_vector(noise_obs_filename='ends_master/freyberg6_run_ies.obs+noise.csv')
    assert np.isclose(phi.loc['base'],phi_noise['base'])
    assert (phi==phi_noise).sum() < len(phi)
    
def deviations_test():
    pst = pyemu.Pst(os.path.join("pst", "pest.pst"))
    num_reals = 10
    pe = pyemu.ParameterEnsemble.from_gaussian_draw(pst, num_reals=num_reals)
    oe = pyemu.ObservationEnsemble.from_gaussian_draw(pst, num_reals=num_reals)
    pe_devs = pe.get_deviations()
    oe_devs = oe.get_deviations()
    pe.add_base()
    pe_base_devs = pe.get_deviations(center_on="base")
    s = pe_base_devs.loc["base",:].apply(np.abs).sum()
    assert s == 0.0
    pe.transform()
    pe_base_devs = pe.get_deviations(center_on="base")
    s = pe_base_devs.loc["base", :].apply(np.abs).sum()
    assert s == 0.0

    oe.add_base()
    oe_base_devs = oe.get_deviations(center_on="base")
    s = oe_base_devs.loc["base", :].apply(np.abs).sum()
    assert s == 0.0


def as_pyemu_matrix_test():
    pst = pyemu.Pst(os.path.join("pst", "pest.pst"))
    num_reals = 10
    pe = pyemu.ParameterEnsemble.from_gaussian_draw(pst, num_reals=num_reals)
    pe.add_base()
    oe = pyemu.ObservationEnsemble.from_gaussian_draw(pst, num_reals=num_reals)
    oe.add_base()

    pe_mat = pe.as_pyemu_matrix()
    assert type(pe_mat) == pyemu.Matrix
    assert pe_mat.shape == pe.shape
    pe._df.index = pe._df.index.map(str)
    d = (pe_mat.to_dataframe() - pe._df).apply(np.abs).values.sum()
    assert d == 0.0

    oe_mat = oe.as_pyemu_matrix(typ=pyemu.Cov)
    assert type(oe_mat) == pyemu.Cov
    assert oe_mat.shape == oe.shape
    oe._df.index = oe._df.index.map(str)
    d = (oe_mat.to_dataframe() - oe._df).apply(np.abs).values.sum()
    assert d == 0.0


def dropna_test():
    pst = pyemu.Pst(os.path.join("pst", "pest.pst"))
    num_reals = 10
    pe = pyemu.ParameterEnsemble.from_gaussian_draw(pst, num_reals=num_reals)
    pe.iloc[::3,:] = np.nan
    ped = pe.dropna()
    assert type(ped) == pyemu.ParameterEnsemble
    assert ped.shape == pe._df.dropna().shape

def enforce_test():
    pst = pyemu.Pst(os.path.join("pst", "pest.pst"))

    # make sure sanity check is working
    num_reals = 10
    broke_pst = pst.get()
    broke_pst.parameter_data.loc[:, "parval1"] = broke_pst.parameter_data.parubnd
    pe = pyemu.ParameterEnsemble.from_gaussian_draw(broke_pst, num_reals=num_reals)
    try:
        pe.enforce(how="scale")
    except:
        pass
    else:
        raise Exception("should have failed")
    broke_pst.parameter_data.loc[:, "parval1"] = broke_pst.parameter_data.parlbnd
    pe = pyemu.ParameterEnsemble.from_gaussian_draw(broke_pst, num_reals=num_reals)
    try:
        pe.enforce(how="scale")
    except:
        pass
    else:
        raise Exception("should have failed")

    # check that all pars at parval1 values don't change
    num_reals = 1
    pe = pyemu.ParameterEnsemble.from_gaussian_draw(pst, num_reals=num_reals)
    pe._df.loc[:, :] = pst.parameter_data.parval1.values
    pe._df.loc[0, pst.par_names[0]] = pst.parameter_data.parlbnd.loc[pst.par_names[0]] * 0.5
    pe.enforce(how="scale")
    assert (pe.loc[0,pst.par_names[1:]] - pst.parameter_data.loc[pst.par_names[1:], "parval1"]).apply(np.abs).max() == 0

    #check that all pars are in bounds
    pe.reseed()
    pe = pyemu.ParameterEnsemble.from_gaussian_draw(pst, num_reals=num_reals)
    print(pe.head())
    pe.enforce(how="scale")
    for ridx in pe._df.index:
        real = pe._df.loc[ridx,pst.adj_par_names]
        ub_diff = pe.ubnd - real
        assert ub_diff.min() >= 0.0,ub_diff
        lb_diff = real - pe.lbnd
        assert lb_diff.min() >= 0.0,lb_diff


    pe = pyemu.ParameterEnsemble.from_gaussian_draw(pst, num_reals=num_reals)
    pe._df.loc[0, :] += pst.parameter_data.parubnd
    pe.enforce(how="scale")

    pe = pyemu.ParameterEnsemble.from_gaussian_draw(pst, num_reals=num_reals)
    pe._df.loc[0,:] += pst.parameter_data.parubnd
    pe.enforce()
    assert (pe._df.loc[0,:] - pst.parameter_data.parubnd).apply(np.abs).sum() == 0.0

    # mixed numpy types test
    pe = pyemu.ParameterEnsemble.from_gaussian_draw(pst, num_reals=num_reals)
    pe._df["mult1"] = pe._df["mult1"].astype("float")
    pe._df.loc[0,:] += pst.parameter_data.parubnd
    pe.enforce()
    assert (pe._df.loc[0,:] - pst.parameter_data.parubnd).apply(np.abs).sum() == 0.0

    # columns out of order test
    pe = pyemu.ParameterEnsemble.from_gaussian_draw(pst, num_reals=num_reals)
    pe._df.loc[0,:] += pst.parameter_data.parubnd
    cols_arr = pe._df.columns.values
    cols_out_of_order = np.append(cols_arr[1:], [cols_arr[0]])
    pe._df = pe._df[cols_out_of_order]
    pe.enforce()
    assert (pe._df.loc[0,:] - pst.parameter_data.parubnd).apply(np.abs).sum() == 0.0

    pe._df.loc[0, :] += pst.parameter_data.parubnd
    pe._df.loc[1:,:] = pst.parameter_data.parval1.values
    pe.enforce(how="drop")
    assert pe.shape[0] == num_reals - 1

    pe = pyemu.ParameterEnsemble.from_gaussian_draw(pst, num_reals=num_reals)
    pe.transform()
    pe.enforce()


def pnulpar_test(tmp_path):
    import os
    import pyemu

    ev = pyemu.ErrVar(jco=os.path.join("mc","freyberg_ord.jco"))
    ev.get_null_proj(maxsing=1).to_ascii(
        os.path.join(tmp_path, "ev_new_proj.mat")
    )
    pst = ev.pst
    par_dir = os.path.join("mc","prior_par_draws")
    par_files = [os.path.join(par_dir,f) for f in os.listdir(par_dir) if f.endswith('.par')]

    pe = pyemu.ParameterEnsemble.from_parfiles(pst=pst,parfile_names=par_files)
    real_num = [int(os.path.split(f)[-1].split('.')[0].split('_')[1]) for f in par_files]
    pe._df.index = real_num

    pe_proj = pe.project(ev.get_null_proj(maxsing=1), enforce_bounds='reset')

    par_dir = os.path.join("mc", "proj_par_draws")
    par_files = [os.path.join(par_dir, f) for f in os.listdir(par_dir) if f.endswith('.par')]
    real_num = [int(os.path.split(f)[-1].split('.')[0].split('_')[1]) for f in par_files]

    pe_pnul = pyemu.ParameterEnsemble.from_parfiles(pst=pst,parfile_names=par_files)

    pe_pnul._df.index = real_num
    pe_proj._df.sort_index(axis=1, inplace=True)
    pe_proj._df.sort_index(axis=0, inplace=True)
    pe_pnul._df.sort_index(axis=1, inplace=True)
    pe_pnul._df.sort_index(axis=0, inplace=True)

    diff = 100.0 * ((pe_proj._df - pe_pnul._df) / pe_proj._df)

    assert max(diff.max()) < 1.0e-4,diff


def from_parfiles_test(tmp_path):
    from pathlib import Path
    pst = os.path.join("mc","freyberg_ord.pst")
    pst = pyemu.Pst(pst)
    pars = pst.parameter_data
    par_dir = Path("mc","prior_par_draws")
    par_files = list(par_dir.glob('*.par'))
    pe = pyemu.ParameterEnsemble.from_parfiles(pst=pst, parfile_names=par_files)
    for pfile in par_files:
        real = pd.read_csv(pfile, sep=r'\s+', skiprows=1, index_col=0, header=None)
        with open(Path(tmp_path, pfile.name), 'w') as fp:
            fp.write('single point\n')
            sel = real.index.str.startswith('hkr')
            real.loc[sel].to_csv(fp, header=False, sep=' ')
    newparfiles = list(Path(tmp_path).glob('*.par'))
    pe = pyemu.ParameterEnsemble.from_parfiles(pst=pst, parfile_names=newparfiles)
    assert pe.shape == (len(newparfiles), len(pars))
    pst.parameter_data = pars.loc[pars.parnme.str.startswith('hkr')]
    pe = pyemu.ParameterEnsemble.from_parfiles(pst=pst, parfile_names=par_files)
    assert pe.shape == (len(par_files), len(pst.parameter_data))


def triangular_draw_test():
    import os
    import matplotlib.pyplot as plt
    import pyemu

    pst = pyemu.Pst(os.path.join("pst","pest.pst"))
    pst.parameter_data.loc[:,"partrans"] = "none"
    pe = pyemu.ParameterEnsemble.from_triangular_draw(pst,1000)

def uniform_draw_test():
    import os
    import numpy as np
    pst = pyemu.Pst(os.path.join("pst", "pest.pst"))
    pe = pyemu.ParameterEnsemble.from_uniform_draw(pst, 5000)


def fill_test():
    import os
    import numpy as np
    pst = pyemu.Pst(os.path.join("pst", "pest.pst"))
    num_reals = 100

    oe = pyemu.ObservationEnsemble.from_gaussian_draw(pst)
    assert oe.shape == (num_reals,pst.nnz_obs),oe.shape
    oe = pyemu.ObservationEnsemble.from_gaussian_draw(pst,fill=True)
    assert oe.shape == (num_reals,pst.nobs)
    std = oe.std().loc[pst.zero_weight_obs_names].apply(np.abs)
    assert std.max() < 1e-10
    print(std)

    obs = pst.observation_data
    obs.loc[:,"weight"] = 0.0
    oe = pyemu.ObservationEnsemble.from_gaussian_draw(pst, fill=True)
    assert oe.shape == (num_reals, pst.nobs)
    std = oe.std().apply(np.abs)
    print(std)
    assert std.max() < 1e-10
    return

    par = pst.parameter_data

    # first test that all ensembles are filled with parval1
    par.loc[:,"partrans"] = "fixed"

    pe = pyemu.ParameterEnsemble.from_uniform_draw(pst, num_reals=num_reals)
    assert pe.shape == (num_reals,pst.npar)
    std = pe._df.std().apply(np.abs)
    assert std.max() == 0.0

    pe = pyemu.ParameterEnsemble.from_triangular_draw(pst, num_reals=num_reals)
    assert pe.shape == (num_reals, pst.npar)
    std = pe._df.std().apply(np.abs)
    assert std.max() == 0.0

    pe = pyemu.ParameterEnsemble.from_gaussian_draw(pst, num_reals=num_reals)
    assert pe.shape == (num_reals,pst.npar)
    std = pe._df.std().apply(np.abs)
    assert std.max() == 0.0

    cov = pyemu.Cov.from_parameter_data(pst)
    pe = pyemu.ParameterEnsemble.from_gaussian_draw(pst,cov=cov, num_reals=num_reals)
    assert pe.shape == (num_reals,pst.npar)
    std = pe._df.std().apply(np.abs)
    assert std.max() == 0.0

    cov = cov.to_2d()
    pe = pyemu.ParameterEnsemble.from_gaussian_draw(pst, cov=cov, num_reals=num_reals)
    assert pe.shape == (num_reals, pst.npar)
    std = pe._df.std().apply(np.abs)
    assert std.max() == 0.0

    # test that fill=False is working
    pe = pyemu.ParameterEnsemble.from_uniform_draw(pst, num_reals=num_reals,fill=False)
    assert pe.shape == (num_reals, pst.npar_adj),pe.shape

    pe = pyemu.ParameterEnsemble.from_triangular_draw(pst, num_reals=num_reals,fill=False)
    assert pe.shape == (num_reals, pst.npar_adj),pe.shape

    pe = pyemu.ParameterEnsemble.from_gaussian_draw(pst, num_reals=num_reals,fill=False)
    assert pe.shape == (num_reals, pst.npar_adj),pe.shape

    cov = pyemu.Cov.from_parameter_data(pst)
    pe = pyemu.ParameterEnsemble.from_gaussian_draw(pst, cov=cov, num_reals=num_reals,fill=False)
    assert pe.shape == (num_reals, pst.npar_adj),pe.shape

    cov = cov.to_2d()
    pe = pyemu.ParameterEnsemble.from_gaussian_draw(pst, cov=cov, num_reals=num_reals,fill=False)
    assert pe.shape == (num_reals, pst.npar_adj),pe.shape

    # unfix one par
    par.loc[pst.par_names[0],"partrans"] = "log"
    pe = pyemu.ParameterEnsemble.from_uniform_draw(pst, num_reals=num_reals, fill=False)
    print(pe.shape)
    assert pe.shape == (num_reals, pst.npar_adj), pe.shape

    pe = pyemu.ParameterEnsemble.from_triangular_draw(pst, num_reals=num_reals, fill=False)
    assert pe.shape == (num_reals, pst.npar_adj), pe.shape

    pe = pyemu.ParameterEnsemble.from_gaussian_draw(pst, num_reals=num_reals, fill=False)
    assert pe.shape == (num_reals, pst.npar_adj), pe.shape

    cov = pyemu.Cov.from_parameter_data(pst)
    pe = pyemu.ParameterEnsemble.from_gaussian_draw(pst, cov=cov, num_reals=num_reals, fill=False)
    assert pe.shape == (num_reals, pst.npar_adj), pe.shape

    cov = cov.to_2d()
    pe = pyemu.ParameterEnsemble.from_gaussian_draw(pst, cov=cov, num_reals=num_reals, fill=False)
    assert pe.shape == (num_reals, pst.npar_adj), pe.shape

    par.loc[pst.par_names[1], "partrans"] = "tied"
    par.loc[pst.par_names[1], "partied"] = pst.par_names[0]

    pe = pyemu.ParameterEnsemble.from_uniform_draw(pst, num_reals=num_reals, fill=False)
    assert pe.shape == (num_reals, pst.npar_adj), pe.shape

    pe = pyemu.ParameterEnsemble.from_triangular_draw(pst, num_reals=num_reals, fill=False)
    assert pe.shape == (num_reals, pst.npar_adj), pe.shape

    pe = pyemu.ParameterEnsemble.from_gaussian_draw(pst, num_reals=num_reals, fill=False)
    assert pe.shape == (num_reals, pst.npar_adj), pe.shape

    cov = pyemu.Cov.from_parameter_data(pst)
    pe = pyemu.ParameterEnsemble.from_gaussian_draw(pst, cov=cov, num_reals=num_reals, fill=False)
    assert pe.shape == (num_reals, pst.npar_adj), pe.shape

    cov = cov.to_2d()
    pe = pyemu.ParameterEnsemble.from_gaussian_draw(pst, cov=cov, num_reals=num_reals, fill=False)
    assert pe.shape == (num_reals, pst.npar_adj), pe.shape


def emp_cov_test():
    import os
    import numpy as np
    import pyemu
    pst = pyemu.Pst(os.path.join("en", "pest.pst"))
    cov = pyemu.Cov.from_binary(os.path.join("en", "cov.jcb"))
    print(pst.npar, cov.shape)
    num_reals = 5000

    pe_eig = pyemu.ParameterEnsemble.from_gaussian_draw(pst, cov=cov, num_reals=num_reals, factor="cholesky")
    emp_cov = pe_eig.covariance_matrix()
    assert isinstance(emp_cov,pyemu.Cov)
    assert emp_cov.row_names == pst.adj_par_names
    cov_df = cov.to_dataframe()
    emp_df = emp_cov.to_dataframe()
    for p in pst.adj_par_names:
        print(p,cov_df.loc[p,p],emp_df.loc[p,p])
    diff = np.diag(cov.x) - np.diag(emp_cov.x)
    print(diff.max())
    assert diff.max() < 0.5,diff.max()

def test_factor_draw(tmp_path):
    import numpy as np
    import pyemu

    shutil.copytree("en", tmp_path/"en")
    pst = pyemu.Pst(str(tmp_path/"en"/"pest.pst"))
    cov = pyemu.Cov.from_binary(str(tmp_path/"en"/"cov.jcb"))
    print(pst.npar,cov.shape)
    num_reals = 1000
    pe_cho = pyemu.ParameterEnsemble.from_gaussian_draw(pst, cov=cov, num_reals=num_reals, factor="cholesky")
    pe_cho.transform()
    mn_cho = pe_cho.mean()
    sd_cho = pe_cho.std()
    del pe_cho

    pe_eig = pyemu.ParameterEnsemble.from_gaussian_draw(pst,cov=cov,num_reals=num_reals,factor="eigen")
    pe_eig.transform()
    emp_cov = pe_eig.covariance_matrix()
    mn_eig = pe_eig.mean()
    sd_eig = pe_eig.std()
    del pe_eig

    pe_svd = pyemu.ParameterEnsemble.from_gaussian_draw(pst, cov=cov, num_reals=num_reals, factor="svd")
    pe_svd.transform()
    # mn_svd = pe_svd.mean()
    # sd_svd = pe_svd.std()
    del pe_svd

    pst.add_transform_columns()
    # par = pst.parameter_data
    # df = cov.to_dataframe()
    # for p in pst.adj_par_names:
    #     print(p,par.loc[p,"parval1_trans"],mn_eig[p],mn_svd[p],np.sqrt(df.loc[p,p]),sd_eig[p],sd_svd[p])
    d = (mn_eig - mn_cho).apply(np.abs)
    print(d.max())
    assert d.max() < 0.5,d.sort_values()
    d = (sd_eig - sd_cho).apply(np.abs)
    print(d.max())
    assert d.max() < 0.5,d.sort_values()

    num_reals = 1000
    pe_cho2 = pyemu.ParameterEnsemble.from_gaussian_draw(pst, cov=cov, num_reals=num_reals)

    pe_cho3 = pyemu.ParameterEnsemble.from_gaussian_draw(pst, cov=emp_cov, num_reals=num_reals)


def emp_cov_draw_test():
    import os
    import numpy as np
    import pyemu

    pst = pyemu.Pst(os.path.join("en","pest.pst"))
    cov = pyemu.Cov.from_binary(os.path.join("en","cov.jcb"))
    num_reals = 1000
    pe_cho = pyemu.ParameterEnsemble.from_gaussian_draw(pst, cov=cov, num_reals=num_reals, factor="cholesky")

    emp_cov = pe_cho.covariance_matrix()
    num_reals = 1000
    pe_eig = pyemu.ParameterEnsemble.from_gaussian_draw(pst, cov=emp_cov, num_reals=num_reals, factor="eigen")
    pe_cho = pyemu.ParameterEnsemble.from_gaussian_draw(pst, cov=emp_cov, num_reals=num_reals, factor="cholesky")
    pe_eig.transform()
    pe_cho.transform()
    mn_eig = pe_eig.mean()
    mn_cho = pe_cho.mean()

    sd_eig = pe_eig.std()
    sd_cho = pe_cho.std()

    pst.add_transform_columns()
    par = pst.parameter_data
    df = cov.to_dataframe()
    for p in pst.adj_par_names:
        print(p,par.loc[p,"parval1_trans"],mn_eig[p],mn_cho[p],np.sqrt(df.loc[p,p]),sd_eig[p],sd_cho[p])
    d = (mn_eig - mn_cho).apply(np.abs)
    assert d.max() < 0.5,d.sort_values()
    d = (sd_eig - sd_cho).apply(np.abs)
    assert d.max() < 0.5,d.sort_values()

def mixed_par_draw_test():
    import os
    import matplotlib.pyplot as plt
    import pandas as pd
    import pyemu

    pst = pyemu.Pst(os.path.join("pst","pest.pst"))
    pname = pst.par_names[0]
    pst.parameter_data.loc[pname,"partrans"] = "none"
    npar = pst.npar
    num_reals = 100
    pe1 = pyemu.ParameterEnsemble.from_mixed_draws(pst, {}, num_reals=num_reals)

    pe2 = pyemu.ParameterEnsemble.from_mixed_draws(pst, {},default="uniform",num_reals=num_reals)
    pe3 = pyemu.ParameterEnsemble.from_mixed_draws(pst, {}, default="triangular", num_reals=num_reals)

    # ax = plt.subplot(111)
    # pe1.loc[:,pname].hist(ax=ax,alpha=0.5,bins=25)
    # pe2.loc[:, pname].hist(ax=ax,alpha=0.5,bins=25)
    # pe3.loc[:, pname].hist(ax=ax,alpha=0.5,bins=25)
    # plt.show()

    how = {}

    for p in pst.adj_par_names[:10]:
        how[p] = "gaussian"
    for p in pst.adj_par_names[12:30]:
        how[p] = "uniform"
    for p in pst.adj_par_names[40:100]:
        how[p] = "triangular"
    #for pnames in how.keys():
    #    pst.parameter_data.loc[::3,"partrans"] = "fixed"

    pe = pyemu.ParameterEnsemble.from_mixed_draws(pst,how)
    pst.parameter_data.loc[pname, "partrans"] = "none"
    how = {p:"uniform" for p in pst.par_names}
    how["junk"] = "uniform"
    try:
        pe = pyemu.ParameterEnsemble.from_mixed_draws(pst,how)
    except:
        pass
    else:
        raise Exception("should have failed")

    try:
        pe = pyemu.ParameterEnsemble.from_mixed_draws(pst,{p:"junk" for p in pst.par_names})
    except:
        pass
    else:
        raise Exception("should have failed")

    try:
        pe = pyemu.ParameterEnsemble.from_mixed_draws(pst,{},default="junk")
    except:
        pass
    else:
        raise Exception("should have failed")

    cov = pyemu.Cov.from_parameter_data(pst)
    cov.drop(pst.par_names[:2],0)
    how = {p:"gaussian" for p in pst.par_names}
    try:
        pe = pyemu.ParameterEnsemble.from_mixed_draws(pst, how,cov=cov)
    except:
        pass
    else:
        raise Exception("should have failed")

    how = {p: "uniform" for p in pst.par_names}
    pe = pyemu.ParameterEnsemble.from_mixed_draws(pst, how, cov=cov)

    #cov.drop(pst.par_names[:2], 0)
    assert pst.npar == npar


def binary_test(tmp_path):
    from datetime import datetime
    import numpy as np
    import pandas as pd
    import pyemu
    npar = 1000
    nobs = 100
    nreal = 100
    par_names = ["p{0}".format(i) for i in range(npar)]
    obs_names = ["o{0}".format(i) for i in range(nobs)]
    pst = pyemu.Pst.from_par_obs_names(par_names,obs_names)
    # array to write (mimicing ensemble)
    arr = np.random.random((nreal,npar))
    df = pd.DataFrame(data=arr,columns=par_names,index=[str(i) for i in range(nreal)])
    pe = pyemu.ParameterEnsemble(pst=pst, df=df)

    # explicitly write dense format
    s1 = datetime.now()
    pe.to_dense(os.path.join(tmp_path, "par.bin"))
    # load written
    pe1 = pyemu.ParameterEnsemble.from_binary(
        pst=pst,filename=os.path.join(tmp_path, "par.bin")
    )
    e1 = datetime.now()
    print(f"dense write and read took ({df.shape}) {(e1 - s1).total_seconds()}")
    d = (pe - pe1).apply(np.abs)
    print(f"max diff between read and write: {d.max().max()}")
    assert d.max().max() < 1.0e-10

    # should be regular write
    s2 = datetime.now()
    pe.to_binary(os.path.join(tmp_path, "par.bin"))
    pe1 = pyemu.ParameterEnsemble.from_binary(
        pst=pst,filename=os.path.join(tmp_path, "par.bin")
    )
    e2 = datetime.now()
    print(f"Regular write took: {(e2 - s2).total_seconds()}")
    d = (pe - pe1).apply(np.abs)
    print(f"max diff between read and write: {d.max().max()}")
    assert d.max().max() < 1.0e-10

    # big ensemble should default to dense .bin write
    pe3 = pd.DataFrame(np.random.rand(10, int(2e6)))
    pe3.columns = "parameter_number_" + pe3.columns.astype(str)
    pe3 = pe3.rename(index={9:'base'})
    pst = pyemu.Pst.from_par_obs_names(pe3.columns, obs_names)
    pe3 = pyemu.ParameterEnsemble(pst=pst, df=pe3)
    fname = pe3.to_binary(os.path.join(tmp_path, "paren.jcb"))
    assert fname.endswith('bin')
    pe4 = pyemu.ParameterEnsemble.from_binary(pst=pst, filename=fname)
    assert pe4.shape == pe3.shape
    d = (pe4._df - pe4._df).abs()
    print(d.max().max())
    assert d.max().max() < 1.0e-10
    pass


def mixed_par_draw_2_test():
    import os
    import matplotlib.pyplot as plt
    import pandas as pd
    import pyemu

    pst = pyemu.Pst(os.path.join("pst","ryan.pst"))
    #pname = pst.par_names[0]
    #pst.parameter_data.loc[pname,"partrans"] = "none"
    npar = pst.npar
    num_reals = 100
    pe1 = pyemu.ParameterEnsemble.from_mixed_draws(pst, {}, num_reals=num_reals)

    pe2 = pyemu.ParameterEnsemble.from_mixed_draws(pst, {},default="uniform",num_reals=num_reals)
    pe3 = pyemu.ParameterEnsemble.from_mixed_draws(pst, {}, default="triangular", num_reals=num_reals)

    # ax = plt.subplot(111)
    # pe1.loc[:,pname].hist(ax=ax,alpha=0.5,bins=25)
    # pe2.loc[:, pname].hist(ax=ax,alpha=0.5,bins=25)
    # pe3.loc[:, pname].hist(ax=ax,alpha=0.5,bins=25)
    # plt.show()

    how = {}

    for p in pst.adj_par_names[:10]:
        how[p] = "gaussian"
    for p in pst.adj_par_names[12:30]:
        how[p] = "uniform"
    for p in pst.adj_par_names[40:100]:
        how[p] = "triangular"
    #for pnames in how.keys():
    #    pst.parameter_data.loc[::3,"partrans"] = "fixed"

    pe = pyemu.ParameterEnsemble.from_mixed_draws(pst,how)
    how = {p:"uniform" for p in pst.par_names}
    how["junk"] = "uniform"
    try:
        pe = pyemu.ParameterEnsemble.from_mixed_draws(pst,how)
    except:
        pass
    else:
        raise Exception("should have failed")

    try:
        pe = pyemu.ParameterEnsemble.from_mixed_draws(pst,{p:"junk" for p in pst.par_names})
    except:
        pass
    else:
        raise Exception("should have failed")

    try:
        pe = pyemu.ParameterEnsemble.from_mixed_draws(pst,{},default="junk")
    except:
        pass
    else:
        raise Exception("should have failed")

    cov = pyemu.Cov.from_parameter_data(pst)
    cov.drop(pst.adj_par_names[:2],0)
    how = {p:"gaussian" for p in pst.par_names}
    try:
        pe = pyemu.ParameterEnsemble.from_mixed_draws(pst, how,cov=cov)
    except:
        pass
    else:
        raise Exception("should have failed")

    how = {p: "uniform" for p in pst.adj_par_names}
    pe = pyemu.ParameterEnsemble.from_mixed_draws(pst, how, cov=cov,fill=True,partial=False,enforce_bounds=True)

    #cov.drop(pst.par_names[:2], 0)
    assert pst.npar == npar


def _check_ens(origen, newen, noise=False, helptxt=""):
    origen.transform()
    newen.transform()
    stdiff = (newen.std() / origen.std()) - 1
    if noise:
        # if adding noise only check for signif std reductions
        voilations = stdiff.loc[stdiff < -0.1]
        assert len(voilations) == 0, f"'{helptxt}' std > 10% reduction:\n{voilations}"
    else:
        # else check for signif std increase or decrease
        voilations = stdiff.loc[(stdiff > 0.15) | (stdiff < -0.1)]
        assert len(voilations) == 0, f"'{helptxt}' std differences > 0.1:\n{voilations}"
    # new std should not be shrunk by more than 15%?

    # max mean change?
    # mean change 5% of range?
    mndiff = np.abs(newen.mean() - origen.mean()) / (origen.max() - origen.min())
    voilations = mndiff.loc[mndiff>0.05]
    assert len(voilations) == 0, f"{helptxt} mean differences > 5% of initial range:\n{voilations}"
    origen.back_transform()

def plot_ens_cdf(pst, ens, namestrs):
    colors = ["r", "y", "b", "g", "m", "c"][:len(ens)]
    paren = isinstance(ens[0], pyemu.en.ParameterEnsemble)
    if paren:
        for en in ens:
            if not en.istransformed:
                en.transform()

        pst.add_transform_columns()
        ubnd = pst.parameter_data.parubnd_trans.to_dict()
        lbnd = pst.parameter_data.parlbnd_trans.to_dict()
        fname = "check_parens.pdf"
        nameids = pst.adj_par_names
    else:
        ubnd = None
        lbnd = None
        fname = "check_obsens.pdf"
        nameids = pst.nnz_obs_names

    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    with PdfPages(fname) as pdf:
        for nm in nameids:
            fig,ax = plt.subplots(1,1,figsize=(10,10))
            for en,c,name in zip(ens, colors, namestrs):
                nmen = en.loc[:,nm]
                # ax.hist(par,fc=c,alpha=0.5,bins=10,density=True)
                ax.plot(np.sort(nmen), np.arange(nmen.shape[0])/nmen.shape[0], color=c, label=name)
            ylim = ax.get_ylim()
            if paren:
                ax.plot([ubnd[nm],ubnd[nm]],ylim,"k--",lw=3)
                ax.plot([lbnd[nm],lbnd[nm]],ylim,"k--",lw=3)

            ax.grid()
            ax.set_title(nm)
            ax.legend()
            plt.tight_layout()
            pdf.savefig()
            plt.close(fig)
            # print(nm)

def test_draw_new_pe(tmp_path, plot=False):
    import os
    import numpy as np
    import pyemu
    shutil.copy(os.path.join("en", "pest.pst"), tmp_path)
    shutil.copy(os.path.join("en", "cov.jcb"), tmp_path)
    pst = pyemu.Pst(os.path.join(tmp_path, "pest.pst"))
    cov = pyemu.Cov.from_binary(os.path.join(tmp_path, "cov.jcb"))
    # basesd = pd.Series(cov.x.diagonal(), index=cov.row_names)
    print(pst.npar, cov.shape)
    num_reals = 1000

    pe = pyemu.ParameterEnsemble.from_gaussian_draw(pst, cov=cov, num_reals=num_reals, factor="cholesky")

    sub_pe = pe.iloc[:int(num_reals), :]

    new_pe_nonoise = sub_pe.draw_new_ensemble(num_reals=num_reals, include_noise=False)
    _check_ens(pe, new_pe_nonoise, False, 'no noise')
    if not plot:
        del new_pe_nonoise
    new_pe_stdnoise = sub_pe.draw_new_ensemble(num_reals=num_reals, include_noise=True)
    _check_ens(pe, new_pe_stdnoise, True, 'std noise')
    if not plot:
        del new_pe_stdnoise
    new_pe_usernoise = sub_pe.draw_new_ensemble(num_reals=num_reals, include_noise=1. / np.sqrt(sub_pe.shape[0]))
    _check_ens(pe, new_pe_usernoise, True, 'user noise')
    if not plot:
        del new_pe_usernoise
    new_pe_ennoise = sub_pe.draw_new_ensemble(num_reals=num_reals, include_noise=True, noise_reals=pe)
    _check_ens(pe, new_pe_ennoise, False, 'ens noise')
    if not plot:
        del new_pe_ennoise

    if plot:
        pes = [pe,
               sub_pe,
               new_pe_nonoise,
               new_pe_stdnoise,
               new_pe_usernoise,
               new_pe_ennoise]
        names = ["original",
                 "sub_ensemble",
                 "new_no_noise",
                 "new_std_noise",
                 "new_user_noise",
                 "new_ens_noise"]
        plot_ens_cdf(pst, pes, names)


def test_draw_new_oe(tmp_path, plot=False):
    num_reals = 1000

    shutil.copy(os.path.join("en", "pest.pst"), tmp_path)
    shutil.copy(os.path.join("en", "cov.jcb"), tmp_path)
    pst = pyemu.Pst(os.path.join(tmp_path, "pest.pst"))
    cov = pyemu.Cov.from_binary(os.path.join(tmp_path, "cov.jcb"))
    obs = pst.observation_data
    obs.loc[pst.obs_names[:1000],"weight"] = 1.0
    cov = cov.x[:pst.nnz_obs,:pst.nnz_obs]
    cov = pyemu.Cov(x=cov,names=pst.nnz_obs_names)

    oe = pyemu.ObservationEnsemble.from_gaussian_draw(pst, cov=cov, num_reals=num_reals, factor="cholesky")
    sub_oe = oe.iloc[:int(num_reals),:]

    new_oe_nonoise = sub_oe.draw_new_ensemble(num_reals=num_reals)
    _check_ens(oe, new_oe_nonoise, False, 'no noise')
    if not plot:
        del new_oe_nonoise
    new_oe_stdnoise = sub_oe.draw_new_ensemble(num_reals=num_reals,include_noise=True)
    _check_ens(oe, new_oe_stdnoise, True,'std noise')
    if not plot:
        del new_oe_stdnoise
    new_oe_usernoise = sub_oe.draw_new_ensemble(num_reals=num_reals,include_noise=1./np.sqrt(sub_oe.shape[0]))
    _check_ens(oe, new_oe_usernoise, True,'user noise')
    if not plot:
        del new_oe_usernoise

    if plot:
        ens = [oe,
               sub_oe,
               new_oe_nonoise,
               new_oe_stdnoise,
               new_oe_usernoise]
        names = ["original",
                 "sub_ensemble",
                 "new_no_noise",
                 "new_std_noise",
                 "new_user_noise"]
        plot_ens_cdf(pst, ens, names)


if __name__ == "__main__":
    import cProfile
    import pstats
    # test_draw_new()
    #par_gauss_draw_consistency_test()
    #obs_gauss_draw_consistency_test()
    # phi_vector_test()
    #add_base_test()
    #nz_test()
    #deviations_test()
    # as_pyemu_matrix_test()
    # dropna_test()
    #enforce_test()
    #pnulpar_test()
    # triangular_draw_test()
    # uniform_draw_test()
    #fill_test()
    #emp_cov_test()
    #emp_cov_draw_test()
    #mixed_par_draw_2_test()
    #binary_test()
    #get_phi_vector_noise_obs_test()
    #test_factor_draw()
    #enforce_test()
    profiler = cProfile.Profile()
    profiler.enable()
    test_draw_new_oe('temp', True)
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('cumtime')
    stats.print_stats(10)


