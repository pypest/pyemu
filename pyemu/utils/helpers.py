""" module with high-level functions to help
perform complex tasks
"""

from __future__ import print_function, division
import os
import multiprocessing as mp
import warnings
from datetime import datetime
import struct
import shutil
import copy
import numpy as np
import scipy.sparse
import pandas as pd
pd.options.display.max_colwidth = 100

try:
    import flopy
except:
    pass

import pyemu
from pyemu.utils.os_utils import run, start_slaves

def remove_readonly(func, path, excinfo):
    """remove readonly dirs, apparently only a windows issue
    add to all rmtree calls: shutil.rmtree(**,onerror=remove_readonly), wk"""
    os.chmod(path, 128) #stat.S_IWRITE==128==normal
    func(path)

def run(cmd_str,cwd='.',verbose=False):
    """ an OS agnostic function to execute command

    Parameters
    ----------
    cmd_str : str
        the str to execute with os.system()

    cwd : str
        the directory to execute the command in

    verbose : bool
        flag to echo to stdout complete cmd str

    Note
    ----
    uses platform to detect OS and adds .exe or ./ as appropriate

    for Windows, if os.system returns non-zero, raises exception

    Example
    -------
    ``>>>import pyemu``

    ``>>>pyemu.helpers.run("pestpp pest.pst")``

    """
    warnings.warn("run() has moved to pyemu.os_utils")
    pyemu.os_utils.run(cmd_str=cmd_str,cwd=cwd,verbose=verbose)


def pilotpoint_prior_builder(pst, struct_dict,sigma_range=4):
    warnings.warn("'pilotpoint_prior_builder' has been renamed to "+\
                  "'geostatistical_prior_builder'")
    return geostatistical_prior_builder(pst=pst,struct_dict=struct_dict,
                                        sigma_range=sigma_range)

def sparse_geostatistical_prior_builder(pst, struct_dict,sigma_range=4,verbose=False):
    """ a helper function to construct a full prior covariance matrix using
    a mixture of geostastical structures and parameter bounds information.
    The covariance of parameters associated with geostatistical structures is defined
    as a mixture of GeoStruct and bounds.  That is, the GeoStruct is used to construct a
    pyemu.Cov, then the entire pyemu.Cov is scaled by the uncertainty implied by the bounds and
    sigma_range. Sounds complicated...

    Parameters
    ----------
    pst : pyemu.Pst
        a control file (or the name of control file)
    struct_dict : dict
        a python dict of GeoStruct (or structure file), and list of pp tpl files pairs
        If the values in the dict are pd.DataFrames, then they must have an
        'x','y', and 'parnme' column.  If the filename ends in '.csv',
        then a pd.DataFrame is loaded, otherwise a pilot points file is loaded.
    sigma_range : float
        a float representing the number of standard deviations implied by parameter bounds.
        Default is 4.0, which implies 95% confidence parameter bounds.
    verbose : bool
        flag for stdout.

    Returns
    -------
    Cov : pyemu.SparseMatrix
        a sparse covariance matrix that includes all adjustable parameters in the control
        file.

    Example
    -------
    ``>>>import pyemu``

    ``>>>pst = pyemu.Pst("pest.pst")``

    ``>>>sd = {"struct.dat":["hkpp.dat.tpl","vka.dat.tpl"]}``

    ``>>>cov = pyemu.helpers.sparse_geostatistical_prior_builder(pst,struct_dict=sd)``

    ``>>>cov.to_coo("prior.jcb")``

    """

    if isinstance(pst,str):
        pst = pyemu.Pst(pst)
    assert isinstance(pst,pyemu.Pst),"pst arg must be a Pst instance, not {0}".\
        format(type(pst))
    if verbose: print("building diagonal cov")
    full_cov = pyemu.Cov.from_parameter_data(pst,sigma_range=sigma_range)

    full_cov_dict = {n:float(v) for n,v in zip(full_cov.col_names,full_cov.x)}

    full_cov = None
    par = pst.parameter_data
    for gs,items in struct_dict.items():
        if verbose: print("processing ",gs)
        if isinstance(gs,str):
            gss = pyemu.geostats.read_struct_file(gs)
            if isinstance(gss,list):
                warnings.warn("using first geostat structure in file {0}".\
                              format(gs))
                gs = gss[0]
            else:
                gs = gss
        if not isinstance(items,list):
            items = [items]
        for item in items:
            if isinstance(item,str):
                assert os.path.exists(item),"file {0} not found".\
                    format(item)
                if item.lower().endswith(".tpl"):
                    df = pyemu.pp_utils.pp_tpl_to_dataframe(item)
                elif item.lower.endswith(".csv"):
                    df = pd.read_csv(item)
            else:
                df = item
            for req in ['x','y','parnme']:
                if req not in df.columns:
                    raise Exception("{0} is not in the columns".format(req))
            missing = df.loc[df.parnme.apply(
                    lambda x : x not in par.parnme),"parnme"]
            if len(missing) > 0:
                warnings.warn("the following parameters are not " + \
                              "in the control file: {0}".\
                              format(','.join(missing)))
                df = df.loc[df.parnme.apply(lambda x: x not in missing)]
            if "zone" not in df.columns:
                df.loc[:,"zone"] = 1
            zones = df.zone.unique()
            for zone in zones:
                df_zone = df.loc[df.zone==zone,:].copy()
                df_zone.sort_values(by="parnme",inplace=True)
                if verbose: print("build cov matrix")
                cov = gs.sparse_covariance_matrix(df_zone.x,df_zone.y,df_zone.parnme)
                if verbose: print("done")

                if verbose: print("getting diag var cov",df_zone.shape[0])
                #tpl_var = np.diag(full_cov.get(list(df_zone.parnme)).x).max()
                tpl_var = max([full_cov_dict[pn] for pn in df_zone.parnme])

                if verbose: print("scaling full cov by diag var cov")
                cov.x.data *= tpl_var

                if full_cov is None:
                    full_cov = cov
                else:
                    if verbose: print("extending SparseMatix")
                    full_cov.block_extend_ip(cov)


    if verbose: print("adding remaining parameters to diagonal")
    fset = set(full_cov.row_names)
    pset = set(pst.par_names)
    diff = list(pset.difference(fset))
    diff.sort()
    vals = np.array([full_cov_dict[d] for d in diff])
    i = np.arange(vals.shape[0])
    coo = scipy.sparse.coo_matrix((vals,(i,i)),shape=(vals.shape[0],vals.shape[0]))
    cov = pyemu.SparseMatrix(x=coo,row_names=diff,col_names=diff)
    full_cov.block_extend_ip(cov)

    return full_cov

def geostatistical_prior_builder(pst, struct_dict,sigma_range=4,
                                 par_knowledge_dict=None,verbose=False):
    """ a helper function to construct a full prior covariance matrix using
    a mixture of geostastical structures and parameter bounds information.
    The covariance of parameters associated with geostatistical structures is defined
    as a mixture of GeoStruct and bounds.  That is, the GeoStruct is used to construct a
    pyemu.Cov, then the entire pyemu.Cov is scaled by the uncertainty implied by the bounds and
    sigma_range. Sounds complicated...

    Parameters
    ----------
    pst : pyemu.Pst
        a control file (or the name of control file)
    struct_dict : dict
        a python dict of GeoStruct (or structure file), and list of pp tpl files pairs
        If the values in the dict are pd.DataFrames, then they must have an
        'x','y', and 'parnme' column.  If the filename ends in '.csv',
        then a pd.DataFrame is loaded, otherwise a pilot points file is loaded.
    sigma_range : float
        a float representing the number of standard deviations implied by parameter bounds.
        Default is 4.0, which implies 95% confidence parameter bounds.
    par_knowledge_dict : dict
        used to condition on existing knowledge about parameters.  This functionality is
        currently in dev - don't use it.
    verbose : bool
        stdout flag
    Returns
    -------
    Cov : pyemu.Cov
        a covariance matrix that includes all adjustable parameters in the control
        file.

    Example
    -------
    ``>>>import pyemu``

    ``>>>pst = pyemu.Pst("pest.pst")``

    ``>>>sd = {"struct.dat":["hkpp.dat.tpl","vka.dat.tpl"]}``

    ``>>>cov = pyemu.helpers.geostatistical_prior_builder(pst,struct_dict=sd)``

    ``>>>cov.to_ascii("prior.cov")``

    """

    if isinstance(pst,str):
        pst = pyemu.Pst(pst)
    assert isinstance(pst,pyemu.Pst),"pst arg must be a Pst instance, not {0}".\
        format(type(pst))
    if verbose: print("building diagonal cov")
    full_cov = pyemu.Cov.from_parameter_data(pst,sigma_range=sigma_range)

    full_cov_dict = {n:float(v) for n,v in zip(full_cov.col_names,full_cov.x)}
    #full_cov = None
    par = pst.parameter_data
    for gs,items in struct_dict.items():
        if verbose: print("processing ",gs)
        if isinstance(gs,str):
            gss = pyemu.geostats.read_struct_file(gs)
            if isinstance(gss,list):
                warnings.warn("using first geostat structure in file {0}".\
                              format(gs))
                gs = gss[0]
            else:
                gs = gss
        if not isinstance(items,list):
            items = [items]
        for item in items:
            if isinstance(item,str):
                assert os.path.exists(item),"file {0} not found".\
                    format(item)
                if item.lower().endswith(".tpl"):
                    df = pyemu.pp_utils.pp_tpl_to_dataframe(item)
                elif item.lower.endswith(".csv"):
                    df = pd.read_csv(item)
            else:
                df = item
            for req in ['x','y','parnme']:
                if req not in df.columns:
                    raise Exception("{0} is not in the columns".format(req))
            missing = df.loc[df.parnme.apply(
                    lambda x : x not in par.parnme),"parnme"]
            if len(missing) > 0:
                warnings.warn("the following parameters are not " + \
                              "in the control file: {0}".\
                              format(','.join(missing)))
                df = df.loc[df.parnme.apply(lambda x: x not in missing)]
            if "zone" not in df.columns:
                df.loc[:,"zone"] = 1
            zones = df.zone.unique()
            for zone in zones:
                df_zone = df.loc[df.zone==zone,:].copy()
                df_zone.sort_values(by="parnme",inplace=True)
                if verbose: print("build cov matrix")
                cov = gs.covariance_matrix(df_zone.x,df_zone.y,df_zone.parnme)
                if verbose: print("done")
                # find the variance in the diagonal cov
                if verbose: print("getting diag var cov",df_zone.shape[0])
                #tpl_var = np.diag(full_cov.get(list(df_zone.parnme)).x).max()
                tpl_var = max([full_cov_dict[pn] for pn in df_zone.parnme])
                #if np.std(tpl_var) > 1.0e-6:
                #    warnings.warn("pars have different ranges" +\
                #                  " , using max range as variance for all pars")
                #tpl_var = tpl_var.max()
                if verbose: print("scaling full cov by diag var cov")
                cov *= tpl_var
                if verbose: print("test for inversion")
                try:
                    ci = cov.inv
                except:
                    df_zone.to_csv("prior_builder_crash.csv")
                    raise Exception("error inverting cov {0}".
                                    format(cov.row_names[:3]))

                    if verbose: print('replace in full cov')
                full_cov.replace(cov)
                # d = np.diag(full_cov.x)
                # idx = np.argwhere(d==0.0)
                # for i in idx:
                #     print(full_cov.names[i])

    if par_knowledge_dict is not None:
        full_cov = condition_on_par_knowledge(full_cov,
                    par_knowledge_dict=par_knowledge_dict)
    return full_cov



def condition_on_par_knowledge(cov,par_knowledge_dict):
    """  experimental function to include conditional prior information
    for one or more parameters in a full covariance matrix
    """

    missing = []
    for parnme in par_knowledge_dict.keys():
        if parnme not in cov.row_names:
            missing.append(parnme)
    if len(missing):
        raise Exception("par knowledge dict parameters not found: {0}".\
                        format(','.join(missing)))
    # build the selection matrix and sigma epsilon
    #sel = cov.zero2d
    #sel = pyemu.Matrix(x=np.zeros((cov.shape[0],1)),row_names=cov.row_names,col_names=['sel'])
    sel = cov.zero2d
    sigma_ep = cov.zero2d
    for parnme,var in par_knowledge_dict.items():
        idx = cov.row_names.index(parnme)
        #sel.x[idx,:] = 1.0
        sel.x[idx,idx] = 1.0
        sigma_ep.x[idx,idx] = var
    #print(sigma_ep.x)
    #q = sigma_ep.inv
    #cov_inv = cov.inv
    print(sel)
    term2 = sel * cov * sel.T
    #term2 += sigma_ep
    #term2 = cov
    print(term2)
    term2 = term2.inv
    term2 *= sel
    term2 *= cov

    new_cov = cov - term2

    return new_cov





def kl_setup(num_eig,sr,struct,array_dict,basis_file="basis.jco",
             tpl_file="kl.tpl"):
    """setup a karhuenen-Loeve based parameterization for a given
    geostatistical structure.

    Parameters
    ----------
    num_eig : int
        number of basis vectors to retain in the reduced basis
    sr : flopy.reference.SpatialReference

    struct : str or pyemu.geostats.Geostruct
        geostatistical structure (or file containing one)

    array_dict : dict
        a dict of arrays to setup as KL-based parameters.  The key becomes the
        parameter name prefix. The total number of parameters is
        len(array_dict) * num_eig

    basis_file : str
        the name of the PEST-format binary file where the reduced basis will be saved

    tpl_file : str
        the name of the template file to make.  The template
        file is a csv file with the parameter names, the
        original factor values,and the template entries.
        The original values can be used to set the parval1
        entries in the control file

    Returns
    -------
    back_array_dict : dict
        a dictionary of back transformed arrays.  This is useful to see
        how much "smoothing" is taking place compared to the original
        arrays.

    Note
    ----
    requires flopy

    Example
    -------
    ``>>>import flopy``

    ``>>>import pyemu``

    ``>>>m = flopy.modflow.Modflow.load("mymodel.nam")``

    ``>>>a_dict = {"hk":m.lpf.hk[0].array}``

    ``>>>ba_dict = pyemu.helpers.kl_setup(10,m.sr,"struct.dat",a_dict)``

    """

    try:
        import flopy
    except Exception as e:
        raise Exception("error import flopy: {0}".format(str(e)))
    assert isinstance(sr,flopy.utils.SpatialReference)
    for name,array in array_dict.items():
        assert isinstance(array,np.ndarray)
        assert array.shape[0] == sr.nrow
        assert array.shape[1] == sr.ncol
        assert len(name) + len(str(num_eig)) <= 12,"name too long:{0}".\
            format(name)

    if isinstance(struct,str):
        assert os.path.exists(struct)
        gs = pyemu.utils.read_struct_file(struct)
    else:
        gs = struct
    names = []
    for i in range(sr.nrow):
        names.extend(["i{0:04d}j{1:04d}".format(i,j) for j in range(sr.ncol)])

    cov = gs.covariance_matrix(sr.xcentergrid.flatten(),
                               sr.ycentergrid.flatten(),
                               names=names)

    trunc_basis = cov.u[:,:num_eig].T
    #for i in range(num_eig):
    #    trunc_basis.x[i,:] *= cov.s.x[i]
    trunc_basis.to_binary(basis_file)
    #trunc_basis = trunc_basis.T

    back_array_dict = {}
    f = open(tpl_file,'w')
    f.write("ptf ~\n")
    f.write("name,org_val,new_val\n")
    for name,array in array_dict.items():
        mname = name+"mean"
        f.write("{0},{1:20.8E},~   {2}    ~\n".format(mname,0.0,mname))
        #array -= array.mean()
        array_flat = pyemu.Matrix(x=np.atleast_2d(array.flatten()).transpose()
                                  ,col_names=["flat"],row_names=names,
                                  isdiagonal=False)
        factors = trunc_basis * array_flat
        enames = ["{0}{1:04d}".format(name,i) for i in range(num_eig)]
        for n,val in zip(enames,factors.x):
            f.write("{0},{1:20.8E},~    {0}    ~\n".format(n,val[0]))
        back_array_dict[name] = (factors.T * trunc_basis).x.reshape(array.shape)
        #print(array_back)
        #print(factors.shape)

    return back_array_dict


def kl_apply(par_file, basis_file,par_to_file_dict,arr_shape):
    """ Applies a KL parameterization transform from basis factors to model
    input arrays.  Companion function to kl_setup()

    Parameters
    ----------
    par_file : str
        the csv file to get factor values from.  Must contain
        the following columns: name, new_val, org_val
    basis_file : str
        the binary file that contains the reduced basis

    par_to_file_dict : dict
        a mapping from KL parameter prefixes to array file names.

    Note
    ----
    This is the companion function to kl_setup.

    This function should be called during the forward run

    Example
    -------
    ``>>>import pyemu``

    ``>>>pyemu.helpers.kl_apply("kl.dat","basis.dat",{"hk":"hk_layer_1.dat",(100,100))``


    """
    df = pd.read_csv(par_file)
    assert "name" in df.columns
    assert "org_val" in df.columns
    assert "new_val" in df.columns

    df.loc[:,"prefix"] = df.name.apply(lambda x: x[:-4])
    for prefix in df.prefix.unique():
        assert prefix in par_to_file_dict.keys(),"missing prefix:{0}".\
            format(prefix)
    basis = pyemu.Matrix.from_binary(basis_file)
    assert basis.shape[1] == arr_shape[0] * arr_shape[1]
    arr_min = 1.0e-10 # a temp hack

    means = df.loc[df.name.apply(lambda x: x.endswith("mean")),:]
    print(means)
    df = df.loc[df.name.apply(lambda x: not x.endswith("mean")),:]
    for prefix,filename in par_to_file_dict.items():
        factors = pyemu.Matrix.from_dataframe(df.loc[df.prefix==prefix,["new_val"]])
        factors.autoalign = False

        #assert df_pre.shape[0] == arr_shape[0] * arr_shape[1]
        arr = (factors.T * basis).x.reshape(arr_shape)
        arr += means.loc[means.prefix==prefix,"new_val"].values
        arr[arr<arr_min] = arr_min
        np.savetxt(filename,arr,fmt="%20.8E")


def zero_order_tikhonov(pst, parbounds=True,par_groups=None,
                        reset=True):
    """setup preferred-value regularization

    Parameters
    ----------
    pst : pyemu.Pst
        the control file instance
    parbounds : bool
        flag to weight the prior information equations according
        to parameter bound width - approx the KL transform. Default
        is True
    par_groups : list
        parameter groups to build PI equations for.  If None, all
        adjustable parameters are used. Default is None

    reset : bool
        flag to reset the prior_information attribute of the pst
        instance.  Default is True

    Example
    -------
    ``>>>import pyemu``

    ``>>>pst = pyemu.Pst("pest.pst")``

    ``>>>pyemu.helpers.zero_order_tikhonov(pst)``

    """

    if par_groups is None:
        par_groups = pst.par_groups

    pilbl, obgnme, weight, equation = [], [], [], []
    for idx, row in pst.parameter_data.iterrows():
        pt = row["partrans"].lower()
        try:
            pt = pt.decode()
        except:
            pass
        if pt not in ["tied", "fixed"] and\
            row["pargp"] in par_groups:
            pilbl.append(row["parnme"])
            weight.append(1.0)
            ogp_name = "regul"+row["pargp"]
            obgnme.append(ogp_name[:12])
            parnme = row["parnme"]
            parval1 = row["parval1"]
            if pt == "log":
                parnme = "log(" + parnme + ")"
                parval1 = np.log10(parval1)
            eq = "1.0 * " + parnme + " ={0:15.6E}".format(parval1)
            equation.append(eq)

    if reset:
        pst.prior_information = pd.DataFrame({"pilbl": pilbl,
                                               "equation": equation,
                                               "obgnme": obgnme,
                                               "weight": weight})
    else:
        pi = pd.DataFrame({"pilbl": pilbl,
                          "equation": equation,
                          "obgnme": obgnme,
                          "weight": weight})
        pst.prior_information = pst.prior_information.append(pi)
    if parbounds:
        regweight_from_parbound(pst)
    if pst.control_data.pestmode == "estimation":
        pst.control_data.pestmode = "regularization"


def regweight_from_parbound(pst):
    """sets regularization weights from parameter bounds
    which approximates the KL expansion.  Called by
    zero_order_tikhonov().

    Parameters
    ----------
    pst : pyemu.Pst
        a control file instance

    """

    pst.parameter_data.index = pst.parameter_data.parnme
    pst.prior_information.index = pst.prior_information.pilbl
    for idx, parnme in enumerate(pst.prior_information.pilbl):
        if parnme in pst.parameter_data.index:
            row = pst.parameter_data.loc[parnme, :]
            lbnd,ubnd = row["parlbnd"], row["parubnd"]
            if row["partrans"].lower() == "log":
                weight = 1.0 / (np.log10(ubnd) - np.log10(lbnd))
            else:
                weight = 1.0 / (ubnd - lbnd)
            pst.prior_information.loc[parnme, "weight"] = weight
        else:
            print("prior information name does not correspond" +\
                  " to a parameter: " + str(parnme))


def first_order_pearson_tikhonov(pst,cov,reset=True,abs_drop_tol=1.0e-3):
    """setup preferred-difference regularization from a covariance matrix.
    The weights on the prior information equations are the Pearson
    correlation coefficients implied by covariance matrix.

    Parameters
    ----------
    pst : pyemu.Pst
        pst instance
    cov : pyemu.Cov
        covariance matrix instance
    reset : bool
        drop all other pi equations.  If False, append to
        existing pi equations
    abs_drop_tol : float
        tolerance to control how many pi equations are written.
        If the Pearson C is less than abs_drop_tol, the prior information
        equation will not be included in the control file

    Example
    -------
    ``>>>import pyemu``

    ``>>>pst = pyemu.Pst("pest.pst")``

    ``>>>cov = pyemu.Cov.from_ascii("prior.cov")``

    ``>>>pyemu.helpers.first_order_pearson_tikhonov(pst,cov,abs_drop_tol=0.25)``

    """
    assert isinstance(cov,pyemu.Cov)
    cc_mat = cov.to_pearson()
    #print(pst.parameter_data.dtypes)
    try:
        ptrans = pst.parameter_data.partrans.apply(lambda x:x.decode()).to_dict()
    except:
        ptrans = pst.parameter_data.partrans.to_dict()
    pi_num = pst.prior_information.shape[0] + 1
    pilbl, obgnme, weight, equation = [], [], [], []
    for i,iname in enumerate(cc_mat.row_names):
        if iname not in pst.adj_par_names:
            continue
        for j,jname in enumerate(cc_mat.row_names[i+1:]):
            if jname not in pst.adj_par_names:
                continue
            #print(i,iname,i+j+1,jname)
            cc = cc_mat.x[i,j+i+1]
            if cc < abs_drop_tol:
                continue
            pilbl.append("pcc_{0}".format(pi_num))
            iiname = str(iname)
            if str(ptrans[iname]) == "log":
                iiname = "log("+iname+")"
            jjname = str(jname)
            if str(ptrans[jname]) == "log":
                jjname = "log("+jname+")"
            equation.append("1.0 * {0} - 1.0 * {1} = 0.0".\
                            format(iiname,jjname))
            weight.append(cc)
            obgnme.append("regul_cc")
            pi_num += 1
    df = pd.DataFrame({"pilbl": pilbl,"equation": equation,
                       "obgnme": obgnme,"weight": weight})
    df.index = df.pilbl
    if reset:
        pst.prior_information = df
    else:
        pst.prior_information = pst.prior_information.append(df)

    if pst.control_data.pestmode == "estimation":
        pst.control_data.pestmode = "regularization"

def simple_tpl_from_pars(parnames, tplfilename='model.input.tpl'):
    """
    Make a template file just assuming a list of parameter names the values of which should be
    listed in order in a model input file
    Args:
        parnames: list of names from which to make a template file
        tplfilename: filename for TPL file (default: model.input.tpl)

    Returns:
        writes a file <tplfilename> with each parameter name on a line

    """
    with open(tplfilename, 'w') as ofp:
        ofp.write('ptf ~\n')
        [ofp.write('~{0:^12}~\n'.format(cname)) for cname in parnames]


def simple_ins_from_obs(obsnames, insfilename='model.output.ins'):
    """
    writes an instruction file that assumes wanting to read the values names in obsnames in order
    one per line from a model output file
    Args:
        obsnames: list of obsnames to read in
        insfilename: filename for INS file (default: model.output.ins)

    Returns:
        writes a file <insfilename> with each observation read off a line

    """
    with open(insfilename, 'w') as ofp:
        ofp.write('pif ~\n')
        [ofp.write('!{0}!\n'.format(cob)) for cob in obsnames]

def pst_from_parnames_obsnames(parnames, obsnames,
                               tplfilename='model.input.tpl', insfilename='model.output.ins'):
    """
    Creates a Pst object from a list of parameter names and a list of observation names.
    Default values are provided for the TPL and INS
    Args:
        parnames: list of names from which to make a template file
        obsnames: list of obsnames to read in
        tplfilename: filename for TPL file (default: model.input.tpl)
        insfilename: filename for INS file (default: model.output.ins)

    Returns:
        Pst object

    """
    simple_tpl_from_pars(parnames, tplfilename)
    simple_ins_from_obs(obsnames, insfilename)

    modelinputfilename = tplfilename.replace('.tpl','')
    modeloutputfilename = insfilename.replace('.ins','')

    return pyemu.Pst.from_io_files(tplfilename, modelinputfilename, insfilename, modeloutputfilename)



def start_slaves(slave_dir,exe_rel_path,pst_rel_path,num_slaves=None,slave_root="..",
                 port=4004,rel_path=None,local=True,cleanup=True,master_dir=None,
                 verbose=False,silent_master=False):
    """ start a group of pest(++) slaves on the local machine

    Parameters
    ----------
    slave_dir :  str
        the path to a complete set of input files
    exe_rel_path : str
        the relative path to the pest(++) executable from within the slave_dir
    pst_rel_path : str
        the relative path to the pst file from within the slave_dir
    num_slaves : int
        number of slaves to start. defaults to number of cores
    slave_root : str
        the root to make the new slave directories in
    rel_path: str
        the relative path to where pest(++) should be run from within the
        slave_dir, defaults to the uppermost level of the slave dir
    local: bool
        flag for using "localhost" instead of hostname on slave command line
    cleanup: bool
        flag to remove slave directories once processes exit
    master_dir: str
        name of directory for master instance.  If master_dir
        exists, then it will be removed.  If master_dir is None,
        no master instance will be started
    verbose : bool
        flag to echo useful information to stdout

    Note
    ----
    if all slaves (and optionally master) exit gracefully, then the slave
    dirs will be removed unless cleanup is false

    Example
    -------
    ``>>>import pyemu``

    start 10 slaves using the directory "template" as the base case and
    also start a master instance in a directory "master".

    ``>>>pyemu.helpers.start_slaves("template","pestpp","pest.pst",10,master_dir="master")``

    """

    warnings.warn("start_slaves has moved to pyemu.os_utils")
    pyemu.os_utils.start_slaves(slave_dir=slave_dir,exe_rel_path=exe_rel_path,pst_rel_path=pst_rel_path
                      ,num_slaves=num_slaves,slave_root=slave_root,port=port,rel_path=rel_path,
                      local=local,cleanup=cleanup,master_dir=master_dir,verbose=verbose,
                      silent_master=silent_master)


def read_pestpp_runstorage(filename,irun=0):
    """read pars and obs from a specific run in a pest++ serialized run storage file into
    pandas.DataFrame(s)

    Parameters
    ----------
    filename : str
        the name of the run storage file
    irun : int
        the run id to process.  Default is 0

    Returns
    -------
    par_df : pandas.DataFrame
        parameter information
    obs_df : pandas.DataFrame
        observation information

    """

    header_dtype = np.dtype([("n_runs",np.int64),("run_size",np.int64),("p_name_size",np.int64),
                      ("o_name_size",np.int64)])
    with open(filename,'rb') as f:
        header = np.fromfile(f,dtype=header_dtype,count=1)
        p_name_size,o_name_size = header["p_name_size"][0],header["o_name_size"][0]
        par_names = struct.unpack('{0}s'.format(p_name_size),
                                f.read(p_name_size))[0].strip().lower().decode().split('\0')[:-1]
        obs_names = struct.unpack('{0}s'.format(o_name_size),
                                f.read(o_name_size))[0].strip().lower().decode().split('\0')[:-1]
        n_runs,run_size = header["n_runs"],header["run_size"][0]
        assert irun <= n_runs
        run_start = f.tell()
        f.seek(run_start + (irun * run_size))
        r_status = np.fromfile(f,dtype=np.int8,count=1)
        info_txt = struct.unpack("41s",f.read(41))[0].strip().lower().decode()
        par_vals = np.fromfile(f,dtype=np.float64,count=len(par_names)+1)[1:]
        obs_vals = np.fromfile(f,dtype=np.float64,count=len(obs_names)+1)[:-1]
    par_df = pd.DataFrame({"parnme":par_names,"parval1":par_vals})

    par_df.index = par_df.pop("parnme")
    obs_df = pd.DataFrame({"obsnme":obs_names,"obsval":obs_vals})
    obs_df.index = obs_df.pop("obsnme")
    return par_df,obs_df


def parse_dir_for_io_files(d):
    """ a helper function to find template/input file pairs and
    instruction file/output file pairs.  the return values from this
    function can be passed straight to pyemu.Pst.from_io_files() classmethod
    constructor. Assumes the template file names are <input_file>.tpl and
    instruction file names are <output_file>.ins.

    Parameters
    ----------
    d : str
        directory to search for interface files

    Returns
    -------
    tpl_files : list
        list of template files in d
    in_files : list
        list of input files in d
    ins_files : list
        list of instruction files in d
    out_files : list
        list of output files in d

    """

    files = os.listdir(d)
    tpl_files = [f for f in files if f.endswith(".tpl")]
    in_files = [f.replace(".tpl","") for f in tpl_files]
    ins_files = [f for f in files if f.endswith(".ins")]
    out_files = [f.replace(".ins","") for f in ins_files]
    return tpl_files,in_files,ins_files,out_files


def pst_from_io_files(tpl_files,in_files,ins_files,out_files,pst_filename=None):
    """generate a Pst instance from the model io files.  If 'inschek'
    is available (either in the current directory or registered
    with the system variables) and the model output files are available
    , then the observation values in the control file will be set to the
    values of the model-simulated equivalents to observations.  This can be
    useful for testing

    Parameters
    ----------
    tpl_files : list
        list of pest template files
    in_files : list
        list of corresponding model input files
    ins_files : list
        list of pest instruction files
    out_files: list
        list of corresponding model output files
    pst_filename : str
        name of file to write the control file to.  If None,
        control file is not written.  Default is None

    Returns
    -------
    pst : pyemu.Pst


    Example
    -------
    ``>>>import pyemu``

    this will construct a new Pst instance from template and instruction files
    found in the current directory, assuming that the naming convention follows
    that listed in parse_dir_for_io_files()

    ``>>>pst = pyemu.helpers.pst_from_io_files(*pyemu.helpers.parse_dir_for_io_files('.'))``

    """
    par_names = set()
    if not isinstance(tpl_files,list):
        tpl_files = [tpl_files]
    if not isinstance(in_files,list):
        in_files = [in_files]
    assert len(in_files) == len(tpl_files),"len(in_files) != len(tpl_files)"

    for tpl_file in tpl_files:
        assert os.path.exists(tpl_file),"template file not found: "+str(tpl_file)
        #new_names = [name for name in pyemu.pst_utils.parse_tpl_file(tpl_file) if name not in par_names]
        #par_names.extend(new_names)
        new_names = pyemu.pst_utils.parse_tpl_file(tpl_file)
        par_names.update(new_names)

    if not isinstance(ins_files,list):
        ins_files = [ins_files]
    if not isinstance(out_files,list):
        out_files = [out_files]
    assert len(ins_files) == len(out_files),"len(out_files) != len(out_files)"


    obs_names = []
    for ins_file in ins_files:
        assert os.path.exists(ins_file),"instruction file not found: "+str(ins_file)
        obs_names.extend(pyemu.pst_utils.parse_ins_file(ins_file))

    new_pst = pyemu.pst_utils.generic_pst(list(par_names),list(obs_names))

    new_pst.template_files = tpl_files
    new_pst.input_files = in_files
    new_pst.instruction_files = ins_files
    new_pst.output_files = out_files

    #try to run inschek to find the observtion values
    pyemu.pst_utils.try_run_inschek(new_pst)

    if pst_filename:
        new_pst.write(pst_filename,update_regul=True)
    return new_pst


wildass_guess_par_bounds_dict = {"hk":[0.01,100.0],"vka":[0.1,10.0],
                                   "sy":[0.25,1.75],"ss":[0.1,10.0],
                                   "cond":[0.01,100.0],"flux":[0.25,1.75],
                                   "rech":[0.9,1.1],"stage":[0.9,1.1],
                                   }

class PstFromFlopyModel(object):
    """ a monster helper class to setup multiplier parameters for an
        existing MODFLOW model.  does all kinds of coolness like building a
        meaningful prior, assigning somewhat meaningful parameter groups and
        bounds, writes a forward_run.py script with all the calls need to
        implement multiplier parameters, run MODFLOW and post-process.

    Parameters
    ----------
    model : flopy.mbase
        a loaded flopy model instance. If model is an str, it is treated as a
        MODFLOW nam file (requires org_model_ws)
    new_model_ws : str
        a directory where the new version of MODFLOW input files and PEST(++)
        files will be written
    org_model_ws : str
        directory to existing MODFLOW model files.  Required if model argument
        is an str.  Default is None
    pp_props : list
        pilot point multiplier parameters for grid-based properties.
        A nested list of grid-scale model properties to parameterize using
        name, iterable pairs.  For 3D properties, the iterable is zero-based
        layer indices.  For example, ["lpf.hk",[0,1,2,]] would setup pilot point multiplier
        parameters for layer property file horizontal hydraulic conductivity for model
        layers 1,2, and 3.  For time-varying properties (e.g. recharge), the
        iterable is for zero-based stress period indices.  For example, ["rch.rech",[0,4,10,15]]
        would setup pilot point multiplier parameters for recharge for stress
        period 1,5,11,and 16.
    const_props : list
        constant (uniform) multiplier parameters for grid-based properties.
        A nested list of grid-scale model properties to parameterize using
        name, iterable pairs.  For 3D properties, the iterable is zero-based
        layer indices.  For example, ["lpf.hk",[0,1,2,]] would setup constant (uniform) multiplier
        parameters for layer property file horizontal hydraulic conductivity for model
        layers 1,2, and 3.  For time-varying properties (e.g. recharge), the
        iterable is for zero-based stress period indices.  For example, ["rch.rech",[0,4,10,15]]
        would setup constant (uniform) multiplier parameters for recharge for stress
        period 1,5,11,and 16.
    temporal_bc_props : list
        boundary condition stress-period level multiplier parameters.
        A nested list of boundary condition elements to parameterize using
        name, iterable pairs.  The iterable is zero-based stress-period indices.
        For example, to setup multipliers for WEL flux and for RIV conductance,
        bc_props = [["wel.flux",[0,1,2]],["riv.cond",None]] would setup
        multiplier parameters for well flux for stress periods 1,2 and 3 and
        would setup one single river conductance multipler parameter that is applied
        to all stress periods
    spatial_bc_props : list
        boundary condition spatial multiplier parameters.
        A nested list of boundary condition elements to parameterize using
        names (e.g. [["riv.cond",0],["wel.flux",1] to setup up cell-based parameters for
        each boundary condition element listed.  These multipler parameters are applied across
        all stress periods.  For this to work, there must be the same number of entries
        for all stress periods.  If more than one BC of the same type is in a single
        cell, only one parameter is used to multiply all BCs in the same cell.
    grid_props : list
        grid-based (every active model cell) multiplier parameters.
        A nested list of grid-scale model properties to parameterize using
        name, iterable pairs.  For 3D properties, the iterable is zero-based
        layer indices (e.g., ["lpf.hk",[0,1,2,]] would setup a multiplier
        parameter for layer property file horizontal hydraulic conductivity for model
        layers 1,2, and 3 in every active model cell).  For time-varying properties (e.g. recharge), the
        iterable is for zero-based stress period indices.  For example, ["rch.rech",[0,4,10,15]]
        would setup grid-based multiplier parameters in every active model cell
        for recharge for stress period 1,5,11,and 16.
    grid_geostruct : pyemu.geostats.GeoStruct
        the geostatistical structure to build the prior parameter covariance matrix
        elements for grid-based parameters.  If None, a generic GeoStruct is created
        using an "a" parameter that is 10 times the max cell size.  Default is None
    pp_space : int
        number of grid cells between pilot points.  If None, use the default
        in pyemu.pp_utils.setup_pilot_points_grid.  Default is None
    zone_props : list
        zone-based multiplier parameters.
        A nested list of grid-scale model properties to parameterize using
        name, iterable pairs.  For 3D properties, the iterable is zero-based
        layer indices (e.g., ["lpf.hk",[0,1,2,]] would setup a multiplier
        parameter for layer property file horizontal hydraulic conductivity for model
        layers 1,2, and 3 for unique zone values in the ibound array.
        For time-varying properties (e.g. recharge), the iterable is for
        zero-based stress period indices.  For example, ["rch.rech",[0,4,10,15]]
        would setup zone-based multiplier parameters for recharge for stress
        period 1,5,11,and 16.
    pp_geostruct : pyemu.geostats.GeoStruct
        the geostatistical structure to use for building the prior parameter
        covariance matrix for pilot point parameters.  If None, a generic
        GeoStruct is created using pp_space and grid-spacing information.
        Default is None
    par_bounds_dict : dict
        a dictionary of model property/boundary condition name, upper-lower bound pairs.
        For example, par_bounds_dict = {"hk":[0.01,100.0],"flux":[0.5,2.0]} would
        set the bounds for horizontal hydraulic conductivity to
        0.001 and 100.0 and set the bounds for flux parameters to 0.5 and
        2.0.  For parameters not found in par_bounds_dict,
        pyemu.helpers.wildass_guess_par_bounds_dict is
        used to set somewhat meaningful bounds.  Default is None
    temporal_bc_geostruct : pyemu.geostats.GeoStruct
        the geostastical struture to build the prior parameter covariance matrix
        for time-varying boundary condition multiplier parameters.  This GeoStruct
        express the time correlation so that the 'a' parameter is the length of
        time that boundary condition multiplier parameters are correlated across.
        If None, then a generic GeoStruct is created that uses an 'a' parameter
        of 3 stress periods.  Default is None
    spatial_bc_geostruct : pyemu.geostats.GeoStruct
        the geostastical struture to build the prior parameter covariance matrix
        for spatially-varying boundary condition multiplier parameters.
        If None, a generic GeoStruct is created using an "a" parameter that
        is 10 times the max cell size.  Default is None.
    remove_existing : bool
        a flag to remove an existing new_model_ws directory.  If False and
        new_model_ws exists, an exception is raised.  If True and new_model_ws
        exists, the directory is destroyed - user beware! Default is False.
    k_zone_dict : dict
        a dictionary of zero-based layer index, zone array pairs.  Used to
        override using ibound zones for zone-based parameterization.  If None,
        use ibound values greater than zero as zones.
    use_pp_zones : bool
         a flag to use ibound zones (or k_zone_dict, see above) as pilot
         point zones.  If False, ibound values greater than zero are treated as
         a single zone for pilot points.  Default is False
     obssim_smp_pairs: list
        a list of observed-simulated PEST-type SMP file pairs to get observations
        from and include in the control file.  Default is []
    external_tpl_in_pairs : list
        a list of existing template file, model input file pairs to parse parameters
        from and include in the control file.  Default is []
    external_ins_out_pairs : list
        a list of existing instruction file, model output file pairs to parse
        observations from and include in the control file.  Default is []
    extra_pre_cmds : list
        a list of preprocessing commands to add to the forward_run.py script
        commands are executed with os.system() within forward_run.py. Default
        is [].
    extra_post_cmds : list
        a list of post-processing commands to add to the forward_run.py script.
        Commands are executed with os.system() within forward_run.py.
        Default is [].
    tmp_files : list
        a list of temporary files that should be removed at the start of the forward
        run script.  Default is [].
    model_exe_name : str
        binary name to run modflow.  If None, a default from flopy is used,
        which is dangerous because of the non-standard binary names
        (e.g. MODFLOW-NWT_x64, MODFLOWNWT, mfnwt, etc). Default is None.
    build_prior : bool
        flag to build prior covariance matrix. Default is Triue
    sfr_obs : bool
        flag to include observations of flow and aquifer exchange from
        the sfr ASCII output file



    Returns
    -------
    PstFromFlopyModel : PstFromFlopyModel

    Attributes
    ----------
    pst : pyemu.Pst


    Note
    ----
    works a lot better of TEMPCHEK, INSCHEK and PESTCHEK are available in the
    system path variable

    """

    def __init__(self,model,new_model_ws,org_model_ws=None,pp_props=[],const_props=[],
                 temporal_bc_props=[],grid_props=[],grid_geostruct=None,pp_space=None,
                 zone_props=[],pp_geostruct=None,par_bounds_dict=None,sfr_pars=False,
                 temporal_bc_geostruct=None,remove_existing=False,k_zone_dict=None,
                 mflist_waterbudget=True,mfhyd=True,hds_kperk=[],use_pp_zones=False,
                 obssim_smp_pairs=None,external_tpl_in_pairs=None,
                 external_ins_out_pairs=None,extra_pre_cmds=None,
                 extra_model_cmds=None,extra_post_cmds=None,
                 tmp_files=None,model_exe_name=None,build_prior=True,
                 sfr_obs=False, all_wells=False,bc_props=[],
                 spatial_bc_props=[],spatial_bc_geostruct=None):

        self.logger = pyemu.logger.Logger("PstFromFlopyModel.log")
        self.log = self.logger.log

        self.logger.echo = True
        self.zn_suffix = "_zn"
        self.gr_suffix = "_gr"
        self.pp_suffix = "_pp"
        self.cn_suffix = "_cn"
        self.arr_org = "arr_org"
        self.arr_mlt = "arr_mlt"
        self.bc_org = "bc_org"
        self.bc_mlt = "bc_mlt"
        self.forward_run_file = "forward_run.py"

        self.remove_existing = remove_existing
        self.external_tpl_in_pairs = external_tpl_in_pairs
        self.external_ins_out_pairs = external_ins_out_pairs

        self.setup_model(model, org_model_ws, new_model_ws)
        self.add_external()

        self.arr_mult_dfs = []
        self.par_bounds_dict = par_bounds_dict
        self.pp_props = pp_props
        self.pp_space = pp_space
        self.pp_geostruct = pp_geostruct
        self.use_pp_zones = use_pp_zones

        self.const_props = const_props

        self.grid_props = grid_props
        self.grid_geostruct = grid_geostruct

        self.zone_props = zone_props

        if len(bc_props) > 0:
            if len(temporal_bc_props) > 0:
                self.logger.lraise("bc_props and temporal_bc_props. "+\
                                   "bc_props is deprecated and replaced by temporal_bc_props")
            self.logger.warn("bc_props is deprecated and replaced by temporal_bc_props")
            temporal_bc_props = bc_props

        self.temporal_bc_props = temporal_bc_props
        self.temporal_bc_geostruct = temporal_bc_geostruct
        self.spatial_bc_props = spatial_bc_props
        self.spatial_bc_geostruct = spatial_bc_geostruct


        self.obssim_smp_pairs = obssim_smp_pairs
        self.hds_kperk = hds_kperk
        self.sfr_obs = sfr_obs
        self.frun_pre_lines = []
        self.frun_model_lines = []
        self.frun_post_lines = []
        self.tmp_files = []
        if tmp_files is not None:
            if not isinstance(tmp_files,list):
                tmp_files = [tmp_files]
            self.tmp_files.extend(tmp_files)

        if k_zone_dict is None:
            self.k_zone_dict = {k:self.m.bas6.ibound[k].array for k in np.arange(self.m.nlay)}
        else:
            for k,arr in k_zone_dict.items():
                if k not in np.arange(self.m.nlay):
                    self.logger.lraise("k_zone_dict layer index not in nlay:{0}".
                                       format(k))
                if arr.shape != (self.m.nrow,self.m.ncol):
                    self.logger.lraise("k_zone_dict arr for k {0} has wrong shape:{1}".
                                       format(k,arr.shape))
            self.k_zone_dict = k_zone_dict

        # add any extra commands to the forward run lines

        for alist,ilist in zip([self.frun_pre_lines,self.frun_model_lines,self.frun_post_lines],
                               [extra_pre_cmds,extra_model_cmds,extra_post_cmds]):
            if ilist is None:
                continue

            if not isinstance(ilist,list):
                ilist = [ilist]
            for cmd in ilist:
                self.logger.statement("forward_run line:{0}".format(cmd))
                alist.append("pyemu.helpers.run('{0}')\n".format(cmd))

        # add the model call

        if model_exe_name is None:
            model_exe_name = self.m.exe_name
            self.logger.warn("using flopy binary to execute the model:{0}".format(model))
        line = "pyemu.helpers.run('{0} {1} 1>{1}.stdout 2>{1}.stderr')".format(model_exe_name,self.m.namefile)
        self.logger.statement("forward_run line:{0}".format(line))
        self.frun_model_lines.append(line)

        self.tpl_files,self.in_files = [],[]
        self.ins_files,self.out_files = [],[]
        self.all_wells = all_wells
        self.setup_mult_dirs()

        self.mlt_files = []
        self.org_files = []
        self.m_files = []
        self.mlt_counter = {}
        self.par_dfs = {}
        self.mlt_dfs = []
        if self.all_wells:
            #self.setup_all_wells()
            self.logger.lraise("all_wells has been deprecated and replaced with"+\
                               " spatial_bc_pars.")
        self.setup_bc_pars()

        self.setup_array_pars()

        if sfr_pars:
            self.setup_sfr_pars()
        self.setup_observations()
        self.build_pst()
        if build_prior:
            self.parcov = self.build_prior()
        else:
            self.parcov = None
        self.log("saving intermediate _setup_<> dfs into {0}".
                 format(self.m.model_ws))
        for tag,df in self.par_dfs.items():
            df.to_csv(os.path.join(self.m.model_ws,"_setup_par_{0}_{1}.csv".
                                   format(tag.replace(" ",'_'),self.pst_name)))
        for tag,df in self.obs_dfs.items():
            df.to_csv(os.path.join(self.m.model_ws,"_setup_obs_{0}_{1}.csv".
                                   format(tag.replace(" ",'_'),self.pst_name)))
        self.log("saving intermediate _setup_<> dfs into {0}".
                 format(self.m.model_ws))

        self.logger.statement("all done")


    def setup_sfr_obs(self):
        """setup sfr ASCII observations"""
        if not self.sfr_obs:
            return

        if self.m.sfr is None:
            self.logger.lraise("no sfr package found...")
        org_sfr_out_file = os.path.join(self.org_model_ws,"{0}.sfr.out".format(self.m.name))
        if not os.path.exists(org_sfr_out_file):
            self.logger.lraise("setup_sfr_obs() error: could not locate existing sfr out file: {0}".
                               format(org_sfr_out_file))
        new_sfr_out_file = os.path.join(self.m.model_ws,os.path.split(org_sfr_out_file)[-1])
        shutil.copy2(org_sfr_out_file,new_sfr_out_file)
        seg_group_dict = None
        if isinstance(self.sfr_obs,dict):
            seg_group_dict = self.sfr_obs

        df = pyemu.gw_utils.setup_sfr_obs(new_sfr_out_file,seg_group_dict=seg_group_dict,
                                          model=self.m,include_path=True)
        if df is not None:
            self.obs_dfs["sfr"] = df
        self.frun_post_lines.append("pyemu.gw_utils.apply_sfr_obs()")


    def setup_sfr_pars(self):
        """setup multiplier parameters for sfr segment data"""
        assert self.m.sfr is not None,"can't find sfr package..."
        df = pyemu.gw_utils.setup_sfr_seg_parameters(self.m.namefile,self.m.model_ws)
        self.par_dfs["sfr"] = df

        self.frun_pre_lines.append("pyemu.gw_utils.apply_sfr_seg_parameters()")
        self.tpl_files.append("sfr_seg_pars.dat.tpl")
        self.in_files.append("sfr_seg_pars.dat")


    def setup_mult_dirs(self):
        """ setup the directories to use for multiplier parameterization.  Directories
        are make within the PstFromFlopyModel.m.model_ws directory

        """
        # setup dirs to hold the original and multiplier model input quantities
        set_dirs = []
#        if len(self.pp_props) > 0 or len(self.zone_props) > 0 or \
#                        len(self.grid_props) > 0:
        if self.pp_props is not None or \
                        self.zone_props is not None or \
                        self.grid_props is not None or\
                        self.const_props is not None:
            set_dirs.append(self.arr_org)
            set_dirs.append(self.arr_mlt)
 #       if len(self.bc_props) > 0:
        if len(self.temporal_bc_props) > 0 or len(self.spatial_bc_props) > 0:
            set_dirs.append(self.bc_org)
        if len(self.spatial_bc_props):
            set_dirs.append(self.bc_mlt)

        for d in set_dirs:
            d = os.path.join(self.m.model_ws,d)
            self.log("setting up '{0}' dir".format(d))
            if os.path.exists(d):
                if self.remove_existing:
                    shutil.rmtree(d,onerror=remove_readonly)
                else:
                    raise Exception("dir '{0}' already exists".
                                    format(d))
            os.mkdir(d)
            self.log("setting up '{0}' dir".format(d))

    def setup_model(self,model,org_model_ws,new_model_ws):
        """ setup the flopy.mbase instance for use with multipler parameters.
        Changes model_ws, sets external_path and writes new MODFLOW input
        files

        Parameters
        ----------
        model : flopy.mbase
            flopy model instance
        org_model_ws : str
            the orginal model working space
        new_model_ws : str
            the new model working space

        """
        split_new_mws = [i for i in os.path.split(new_model_ws) if len(i) > 0]
        if len(split_new_mws) != 1:
            self.logger.lraise("new_model_ws can only be 1 folder-level deep:{0}".
                               format(str(split_new_mws)))

        if isinstance(model,str):
            self.log("loading flopy model")
            try:
                import flopy
            except:
                raise Exception("from_flopy_model() requires flopy")
            # prepare the flopy model
            self.org_model_ws = org_model_ws
            self.new_model_ws = new_model_ws
            self.m = flopy.modflow.Modflow.load(model,model_ws=org_model_ws,
                                                check=False,verbose=True,forgive=False)
            self.log("loading flopy model")
        else:
            self.m = model
            self.org_model_ws = str(self.m.model_ws)
            self.new_model_ws = new_model_ws

        self.log("updating model attributes")
        self.m.array_free_format = True
        self.m.free_format_input = True
        self.m.external_path = '.'
        self.log("updating model attributes")
        if os.path.exists(new_model_ws):
            if not self.remove_existing:
                self.logger.lraise("'new_model_ws' already exists")
            else:
                self.logger.warn("removing existing 'new_model_ws")
                shutil.rmtree(new_model_ws,onerror=remove_readonly)
        self.m.change_model_ws(new_model_ws,reset_external=True)
        self.m.exe_name = self.m.exe_name.replace(".exe",'')
        self.m.exe = self.m.version
        self.log("writing new modflow input files")
        self.m.write_input()
        self.log("writing new modflow input files")

    def get_count(self,name):
        """ get the latest counter for a certain parameter type.

        Parameters
        ----------
        name : str
            the parameter type

        Returns
        -------
        count : int
            the latest count for a parameter type

        Note
        ----
        calling this function increments the counter for the passed
        parameter type

        """
        if name not in self.mlt_counter:
            self.mlt_counter[name] = 1
            c = 0
        else:
            c = self.mlt_counter[name]
            self.mlt_counter[name] += 1
            #print(name,c)
        return c

    def prep_mlt_arrays(self):
        """  prepare multipler arrays.  Copies existing model input arrays and
        writes generic (ones) multiplier arrays

        """
        par_props = [self.pp_props,self.grid_props,
                         self.zone_props,self.const_props]
        par_suffixs = [self.pp_suffix,self.gr_suffix,
                       self.zn_suffix,self.cn_suffix]

        # Need to remove props and suffixes for which no info was provided (e.g. still None)
        del_idx = []
        for i,cp in enumerate(par_props):
            if cp is None:
                del_idx.append(i)
        for i in del_idx[::-1]:
            del(par_props[i])
            del(par_suffixs[i])

        mlt_dfs = []
        for par_prop,suffix in zip(par_props,par_suffixs):
            if len(par_prop) == 2:
                if not isinstance(par_prop[0],list):
                    par_prop = [par_prop]
            if len(par_prop) == 0:
                continue
            for pakattr,k_org in par_prop:
                attr_name = pakattr.split('.')[1]
                pak,attr = self.parse_pakattr(pakattr)
                ks = np.arange(self.m.nlay)
                if isinstance(attr,flopy.utils.Transient2d):
                    ks = np.arange(self.m.nper)
                try:
                    k_parse = self.parse_k(k_org,ks)
                except Exception as e:
                    self.logger.lraise("error parsing k {0}:{1}".
                                       format(k_org,str(e)))
                org,mlt,mod,layer = [],[],[],[]
                c = self.get_count(attr_name)
                mlt_prefix = "{0}{1}".format(attr_name,c)
                mlt_name = os.path.join(self.arr_mlt,"{0}.dat{1}"
                                        .format(mlt_prefix,suffix))
                for k in k_parse:
                    # horrible kludge to avoid passing int64 to flopy
                    # this gift may give again...
                    if type(k) is np.int64:
                        k = int(k)
                    if isinstance(attr,flopy.utils.Util2d):
                        fname = self.write_u2d(attr)

                        layer.append(k)
                    elif isinstance(attr,flopy.utils.Util3d):
                        fname = self.write_u2d(attr[k])
                        layer.append(k)
                    elif isinstance(attr,flopy.utils.Transient2d):
                        fname = self.write_u2d(attr.transient_2ds[k])
                        layer.append(0) #big assumption here
                    mod.append(os.path.join(self.m.external_path,fname))
                    mlt.append(mlt_name)
                    org.append(os.path.join(self.arr_org,fname))
                df = pd.DataFrame({"org_file":org,"mlt_file":mlt,"model_file":mod,"layer":layer})
                df.loc[:,"suffix"] = suffix
                df.loc[:,"prefix"] = mlt_prefix
                mlt_dfs.append(df)
        if len(mlt_dfs) > 0:
            mlt_df = pd.concat(mlt_dfs,ignore_index=True)
            return mlt_df

    def write_u2d(self, u2d):
        """ write a flopy.utils.Util2D instance to an ASCII text file using the
        Util2D filename

        Parameters
        ----------
        u2d : flopy.utils.Util2D

        Returns
        -------
        filename : str
            the name of the file written (without path)

        """
        filename = os.path.split(u2d.filename)[-1]
        np.savetxt(os.path.join(self.m.model_ws,self.arr_org,filename),
                   u2d.array,fmt="%15.6E")
        return filename

    def write_const_tpl(self,name,tpl_file,zn_array):
        """ write a template file a for a constant (uniform) multiplier parameter

        Parameters
        ----------
        name : str
            the base parameter name
        tpl_file : str
            the template file to write
        zn_array : numpy.ndarray
            an array used to skip inactive cells

        Returns
        -------
        df : pandas.DataFrame
            a dataframe with parameter information

        """
        parnme = []
        with open(os.path.join(self.m.model_ws,tpl_file),'w') as f:
            f.write("ptf ~\n")
            for i in range(self.m.nrow):
                for j in range(self.m.ncol):
                    if zn_array[i,j] < 1:
                        pname = " 1.0  "
                    else:
                        pname = "{0}{1}".format(name,self.cn_suffix)
                        if len(pname) > 12:
                            self.logger.lraise("zone pname too long:{0}".\
                                               format(pname))
                        parnme.append(pname)
                        pname = " ~   {0}    ~".format(pname)
                    f.write(pname)
                f.write("\n")
        df = pd.DataFrame({"parnme":parnme},index=parnme)
        #df.loc[:,"pargp"] = "{0}{1}".format(self.cn_suffixname)
        df.loc[:,"pargp"] = self.cn_suffix.replace('_','')
        df.loc[:,"tpl"] = tpl_file
        return df

    def write_grid_tpl(self,name,tpl_file,zn_array):
        """ write a template file a for grid-based multiplier parameters

        Parameters
        ----------
        name : str
            the base parameter name
        tpl_file : str
            the template file to write
        zn_array : numpy.ndarray
            an array used to skip inactive cells

        Returns
        -------
        df : pandas.DataFrame
            a dataframe with parameter information

        """
        parnme,x,y = [],[],[]
        with open(os.path.join(self.m.model_ws,tpl_file),'w') as f:
            f.write("ptf ~\n")
            for i in range(self.m.nrow):
                for j in range(self.m.ncol):
                    if zn_array[i,j] < 1:
                        pname = ' 1.0 '
                    else:
                        pname = "{0}{1:03d}{2:03d}".format(name,i,j)
                        if len(pname) > 12:
                            self.logger.lraise("grid pname too long:{0}".\
                                               format(pname))
                        parnme.append(pname)
                        pname = ' ~     {0}   ~ '.format(pname)
                        x.append(self.m.sr.xcentergrid[i,j])
                        y.append(self.m.sr.ycentergrid[i,j])
                    f.write(pname)
                f.write("\n")
        df = pd.DataFrame({"parnme":parnme,"x":x,"y":y},index=parnme)
        df.loc[:,"pargp"] = "{0}{1}".format(self.gr_suffix.replace('_',''),name)
        df.loc[:,"tpl"] = tpl_file
        return df

    @staticmethod
    def write_zone_tpl(model, name, tpl_file, zn_array, zn_suffix, logger=None):
        """ write a template file a for zone-based multiplier parameters

        Parameters
        ----------
        model : flopy model object
            model from which to obtain workspace information, nrow, and ncol
        name : str
            the base parameter name
        tpl_file : str
            the template file to write
        zn_array : numpy.ndarray
            an array used to skip inactive cells

        logger : a logger object
            optional - a logger object to document errors, etc.
        Returns
        -------
        df : pandas.DataFrame
            a dataframe with parameter information

        """
        parnme = []
        with open(os.path.join(model.model_ws, tpl_file), 'w') as f:
            f.write("ptf ~\n")
            for i in range(model.nrow):
                for j in range(model.ncol):
                    if zn_array[i,j] < 1:
                        pname = " 1.0  "
                    else:
                        pname = "{0}_zn{1}".format(name, zn_array[i, j])
                        if len(pname) > 12:
                            if logger is not None:
                                logger.lraise("zone pname too long:{0}".\
                                               format(pname))
                        parnme.append(pname)
                        pname = " ~   {0}    ~".format(pname)
                    f.write(pname)
                f.write("\n")
        df = pd.DataFrame({"parnme":parnme}, index=parnme)
        df.loc[:, "pargp"] = "{0}{1}".format(zn_suffix.replace("_",''), name)
        return df

    def grid_prep(self):
        """ prepare grid-based parameterizations

        """
        if len(self.grid_props) == 0:
            return

        if self.grid_geostruct is None:
            self.logger.warn("grid_geostruct is None,"\
                  " using ExpVario with contribution=1 and a=(max(delc,delr)*10")
            dist = 10 * float(max(self.m.dis.delr.array.max(),
                                           self.m.dis.delc.array.max()))
            v = pyemu.geostats.ExpVario(contribution=1.0,a=dist)
            self.grid_geostruct = pyemu.geostats.GeoStruct(variograms=v)

    def pp_prep(self, mlt_df):
        """ prepare pilot point based parameterizations

        Parameters
        ----------
        mlt_df : pandas.DataFrame
            a dataframe with multiplier array information

        Note
        ----
        calls pyemu.pp_utils.setup_pilot_points_grid()


        """
        if len(self.pp_props) == 0:
            return
        if self.pp_space is None:
            self.logger.warn("pp_space is None, using 10...\n")
            self.pp_space=10
        if self.pp_geostruct is None:
            self.logger.warn("pp_geostruct is None,"\
                  " using ExpVario with contribution=1 and a=(pp_space*max(delr,delc))")
            pp_dist = self.pp_space * float(max(self.m.dis.delr.array.max(),
                                           self.m.dis.delc.array.max()))
            v = pyemu.geostats.ExpVario(contribution=1.0,a=pp_dist)
            self.pp_geostruct = pyemu.geostats.GeoStruct(variograms=v)

        pp_df = mlt_df.loc[mlt_df.suffix==self.pp_suffix,:]
        layers = pp_df.layer.unique()
        pp_dict = {l:list(pp_df.loc[pp_df.layer==l,"prefix"].unique()) for l in layers}
        # big assumption here - if prefix is listed more than once, use the lowest layer index
        for i,l in enumerate(layers):
            p = set(pp_dict[l])
            for ll in layers[i+1:]:
                pp = set(pp_dict[ll])
                d = pp - p
                pp_dict[ll] = list(d)


        pp_array_file = {p:m for p,m in zip(pp_df.prefix,pp_df.mlt_file)}
        self.logger.statement("pp_dict: {0}".format(str(pp_dict)))

        self.log("calling setup_pilot_point_grid()")
        if self.use_pp_zones:
            ib = self.k_zone_dict
        else:
            ib = {k:self.m.bas6.ibound[k].array for k in range(self.m.nlay)}

            for k,i in ib.items():
                if np.any(i<0):
                    u,c = np.unique(i[i>0], return_counts=True)
                    counts = dict(zip(u,c))
                    mx = -1.0e+10
                    imx = None
                    for u,c in counts.items():
                        if c > mx:
                            mx = c
                            imx = u
                    self.logger.warn("resetting negative ibound values for PP zone"+ \
                                     "array in layer {0} : {1}".format(k+1,u))
                    i[i<0] = u
        pp_df = pyemu.pp_utils.setup_pilotpoints_grid(self.m,
                                         ibound=ib,
                                         use_ibound_zones=self.use_pp_zones,
                                         prefix_dict=pp_dict,
                                         every_n_cell=self.pp_space,
                                         pp_dir=self.m.model_ws,
                                         tpl_dir=self.m.model_ws,
                                         shapename=os.path.join(
                                                 self.m.model_ws,"pp.shp"))
        self.logger.statement("{0} pilot point parameters created".
                              format(pp_df.shape[0]))
        self.logger.statement("pilot point 'pargp':{0}".
                              format(','.join(pp_df.pargp.unique())))
        self.log("calling setup_pilot_point_grid()")

        # calc factors for each layer
        pargp = pp_df.pargp.unique()
        pp_dfs_k = {}
        fac_files = {}
        pp_df.loc[:,"fac_file"] = np.NaN
        for pg in pargp:
            ks = pp_df.loc[pp_df.pargp==pg,"k"].unique()
            if len(ks) == 0:
                self.logger.lraise("something is wrong in fac calcs for par group {0}".format(pg))
            if len(ks) == 1:

                ib_k = ib[ks[0]]
            if len(ks) != 1:
                #self.logger.lraise("something is wrong in fac calcs for par group {0}".format(pg))
                self.logger.warn("multiple k values for {0},forming composite zone array...".format(pg))
                ib_k = np.zeros((self.m.nrow,self.m.ncol))
                for k in ks:
                    t = ib[k].copy()
                    t[t<1] = 0
                    ib_k[t>0] = t[t>0]
            k = int(ks[0])
            if k not in pp_dfs_k.keys():
                self.log("calculating factors for k={0}".format(k))

                fac_file = os.path.join(self.m.model_ws,"pp_k{0}.fac".format(k))
                var_file = fac_file.replace(".fac",".var.dat")
                self.logger.statement("saving krige variance file:{0}"
                                      .format(var_file))
                self.logger.statement("saving krige factors file:{0}"\
                                      .format(fac_file))
                pp_df_k = pp_df.loc[pp_df.pargp==pg]
                ok_pp = pyemu.geostats.OrdinaryKrige(self.pp_geostruct,pp_df_k)
                ok_pp.calc_factors_grid(self.m.sr,var_filename=var_file,
                                        zone_array=ib_k)
                ok_pp.to_grid_factors_file(fac_file)
                fac_files[k] = fac_file
                self.log("calculating factors for k={0}".format(k))
                pp_dfs_k[k] = pp_df_k

        for k,fac_file in fac_files.items():
            #pp_files = pp_df.pp_filename.unique()
            fac_file = os.path.split(fac_file)[-1]
            pp_prefixes = pp_dict[k]
            for pp_prefix in pp_prefixes:
                self.log("processing pp_prefix:{0}".format(pp_prefix))
                if pp_prefix not in pp_array_file.keys():
                    self.logger.lraise("{0} not in self.pp_array_file.keys()".
                                       format(pp_prefix,','.
                                              join(pp_array_file.keys())))


                out_file = os.path.join(self.arr_mlt,os.path.split(pp_array_file[pp_prefix])[-1])

                pp_files = pp_df.loc[pp_df.pp_filename.apply(lambda x: pp_prefix in x),"pp_filename"]
                if pp_files.unique().shape[0] != 1:
                    self.logger.lraise("wrong number of pp_files found:{0}".format(','.join(pp_files)))
                pp_file = os.path.split(pp_files.iloc[0])[-1]
                pp_df.loc[pp_df.pargp==pp_prefix,"fac_file"] = fac_file
                pp_df.loc[pp_df.pargp==pp_prefix,"pp_file"] = pp_file
                pp_df.loc[pp_df.pargp==pp_prefix,"out_file"] = out_file

        pp_df.loc[:,"pargp"] = pp_df.pargp.apply(lambda x: "pp_{0}".format(x))
        out_files = mlt_df.loc[mlt_df.mlt_file.
                    apply(lambda x: x.endswith(self.pp_suffix)),"mlt_file"]
        mlt_df.loc[:,"fac_file"] = np.NaN
        mlt_df.loc[:,"pp_file"] = np.NaN
        for out_file in out_files:
            pp_df_pf = pp_df.loc[pp_df.out_file==out_file,:]
            fac_files = pp_df_pf.fac_file
            if fac_files.unique().shape[0] != 1:
                self.logger.lraise("wrong number of fac files:{0}".format(str(fac_files.unique())))
            fac_file = fac_files.iloc[0]
            pp_files = pp_df_pf.pp_file
            if pp_files.unique().shape[0] != 1:
                self.logger.lraise("wrong number of pp files:{0}".format(str(pp_files.unique())))
            pp_file = pp_files.iloc[0]
            mlt_df.loc[mlt_df.mlt_file==out_file,"fac_file"] = fac_file
            mlt_df.loc[mlt_df.mlt_file==out_file,"pp_file"] = pp_file
        self.par_dfs[self.pp_suffix] = pp_df

        mlt_df.loc[mlt_df.suffix==self.pp_suffix,"tpl_file"] = np.NaN


    def setup_array_pars(self):
        """ main entry point for setting up array multipler parameters

        """
        mlt_df = self.prep_mlt_arrays()
        if mlt_df is None:
            return
        mlt_df.loc[:,"tpl_file"] = mlt_df.mlt_file.apply(lambda x: os.path.split(x)[-1]+".tpl")
        #mlt_df.loc[mlt_df.tpl_file.apply(lambda x:pd.notnull(x.pp_file)),"tpl_file"] = np.NaN
        mlt_files = mlt_df.mlt_file.unique()
        #for suffix,tpl_file,layer,name in zip(self.mlt_df.suffix,
        #                                 self.mlt_df.tpl,self.mlt_df.layer,
        #                                     self.mlt_df.prefix):
        par_dfs = {}
        for mlt_file in mlt_files:
            suffixes = mlt_df.loc[mlt_df.mlt_file==mlt_file,"suffix"]
            if suffixes.unique().shape[0] != 1:
                self.logger.lraise("wrong number of suffixes for {0}"\
                                   .format(mlt_file))
            suffix = suffixes.iloc[0]

            tpl_files = mlt_df.loc[mlt_df.mlt_file==mlt_file,"tpl_file"]
            if tpl_files.unique().shape[0] != 1:
                self.logger.lraise("wrong number of tpl_files for {0}"\
                                   .format(mlt_file))
            tpl_file = tpl_files.iloc[0]
            layers = mlt_df.loc[mlt_df.mlt_file==mlt_file,"layer"]
            #if layers.unique().shape[0] != 1:
            #    self.logger.lraise("wrong number of layers for {0}"\
            #                       .format(mlt_file))
            layer = layers.iloc[0]
            names = mlt_df.loc[mlt_df.mlt_file==mlt_file,"prefix"]
            if names.unique().shape[0] != 1:
                self.logger.lraise("wrong number of names for {0}"\
                                   .format(mlt_file))
            name = names.iloc[0]
            #ib = self.k_zone_dict[layer]
            df = None
            if suffix == self.cn_suffix:
                self.log("writing const tpl:{0}".format(tpl_file))
                df = self.write_const_tpl(name,tpl_file,self.m.bas6.ibound[layer].array)
                self.log("writing const tpl:{0}".format(tpl_file))

            elif suffix == self.gr_suffix:
                self.log("writing grid tpl:{0}".format(tpl_file))
                df = self.write_grid_tpl(name,tpl_file,self.m.bas6.ibound[layer].array)
                self.log("writing grid tpl:{0}".format(tpl_file))

            elif suffix == self.zn_suffix:
                self.log("writing zone tpl:{0}".format(tpl_file))
                df = self.write_zone_tpl(self.m, name, tpl_file, self.k_zone_dict[layer], self.zn_suffix, self.logger)
                self.log("writing zone tpl:{0}".format(tpl_file))
            if df is None:
                continue
            if suffix not in par_dfs:
                par_dfs[suffix] = [df]
            else:
                par_dfs[suffix].append(df)
        for suf,dfs in par_dfs.items():
            self.par_dfs[suf] = pd.concat(dfs)

        if self.pp_suffix in mlt_df.suffix.values:
            self.log("setting up pilot point process")
            self.pp_prep(mlt_df)
            self.log("setting up pilot point process")

        if self.gr_suffix in mlt_df.suffix.values:
            self.log("setting up grid process")
            self.grid_prep()
            self.log("setting up grid process")

        mlt_df.to_csv(os.path.join(self.m.model_ws,"arr_pars.csv"))
        ones = np.ones((self.m.nrow,self.m.ncol))
        for mlt_file in mlt_df.mlt_file.unique():
            self.log("save test mlt array {0}".format(mlt_file))
            np.savetxt(os.path.join(self.m.model_ws,mlt_file),
                       ones,fmt="%15.6E")
            self.log("save test mlt array {0}".format(mlt_file))
            tpl_files = mlt_df.loc[mlt_df.mlt_file == mlt_file, "tpl_file"]
            if tpl_files.unique().shape[0] != 1:
                self.logger.lraise("wrong number of tpl_files for {0}" \
                                   .format(mlt_file))
            tpl_file = tpl_files.iloc[0]
            if pd.notnull(tpl_file):
                self.tpl_files.append(tpl_file)
                self.in_files.append(mlt_file)

        # for tpl_file,mlt_file in zip(mlt_df.tpl_file,mlt_df.mlt_file):
        #     if pd.isnull(tpl_file):
        #         continue
        #     self.tpl_files.append(tpl_file)
        #     self.in_files.append(mlt_file)

        os.chdir(self.m.model_ws)
        try:
            apply_array_pars()
        except Exception as e:
            os.chdir("..")
            self.logger.lraise("error test running apply_array_pars():{0}".
                               format(str(e)))
        os.chdir("..")
        line = "pyemu.helpers.apply_array_pars()\n"
        self.logger.statement("forward_run line:{0}".format(line))
        self.frun_pre_lines.append(line)

    def setup_observations(self):
        """ main entry point for setting up observations

        """
        obs_methods = [self.setup_water_budget_obs,self.setup_hyd,
                       self.setup_smp,self.setup_hob,self.setup_hds,
                       self.setup_sfr_obs]
        obs_types = ["mflist water budget obs","hyd file",
                     "external obs-sim smp files","hob","hds","sfr"]
        self.obs_dfs = {}
        for obs_method, obs_type in zip(obs_methods,obs_types):
            self.log("processing obs type {0}".format(obs_type))
            obs_method()
            self.log("processing obs type {0}".format(obs_type))


    def build_prior(self, fmt="ascii",filename=None,droptol=None, chunk=None, sparse=False,
                    sigma_range=6):
        """ build a prior parameter covariance matrix.

        Parameters
        ----------
            fmt : str
                the format to save the cov matrix.  Options are "ascii","binary","uncfile", "coo".
                default is "ascii"
            filename : str
                the filename to save the prior cov matrix to.  If None, the name is formed using
                model nam_file name.  Default is None.
            droptol : float
                tolerance for dropping near-zero values when writing compressed binary.
                Default is None
            chunk : int
                chunk size to write in a single pass - for binary only
            sparse : bool
                flag to build a pyemu.SparseMatrix format cov matrix.  Default is False
            sigma_range : float
                number of standard deviations represented by the parameter bounds.  Default
                is 6.

        Returns
        -------
            cov : pyemu.Cov
            a full covariance matrix

        """

        fmt = fmt.lower()
        acc_fmts = ["ascii","binary","uncfile","none","coo"]
        if fmt not in acc_fmts:
            self.logger.lraise("unrecognized prior save 'fmt':{0}, options are: {1}".
                               format(fmt,','.join(acc_fmts)))

        self.log("building prior covariance matrix")
        struct_dict = {}
        if self.pp_suffix in self.par_dfs.keys():
            pp_df = self.par_dfs[self.pp_suffix]
            pp_dfs = []
            for pargp in pp_df.pargp.unique():
                gp_df = pp_df.loc[pp_df.pargp==pargp,:]
                p_df = gp_df.drop_duplicates(subset="parnme")
                pp_dfs.append(p_df)
            #pp_dfs = [pp_df.loc[pp_df.pargp==pargp,:].copy() for pargp in pp_df.pargp.unique()]
            struct_dict[self.pp_geostruct] = pp_dfs
        if self.gr_suffix in self.par_dfs.keys():
            gr_df = self.par_dfs[self.gr_suffix]
            gr_dfs = []
            for pargp in gr_df.pargp.unique():
                gp_df = gr_df.loc[gr_df.pargp==pargp,:]
                p_df = gp_df.drop_duplicates(subset="parnme")
                gr_dfs.append(p_df)
            #gr_dfs = [gr_df.loc[gr_df.pargp==pargp,:].copy() for pargp in gr_df.pargp.unique()]
            struct_dict[self.grid_geostruct] = gr_dfs
        if "temporal_bc" in self.par_dfs.keys():
            bc_df = self.par_dfs["temporal_bc"]
            bc_df.loc[:,"y"] = 0
            bc_df.loc[:,"x"] = bc_df.timedelta.apply(lambda x: x.days)
            bc_dfs = []
            for pargp in bc_df.pargp.unique():
                gp_df = bc_df.loc[bc_df.pargp==pargp,:]
                p_df = gp_df.drop_duplicates(subset="parnme")
                #print(p_df)
                bc_dfs.append(p_df)
            #bc_dfs = [bc_df.loc[bc_df.pargp==pargp,:].copy() for pargp in bc_df.pargp.unique()]
            struct_dict[self.temporal_bc_geostruct] = bc_dfs
        if "spatial_bc" in self.par_dfs.keys():
            bc_df = self.par_dfs["spatial_bc"]
            bc_dfs = []
            for pargp in bc_df.pargp.unique():
                gp_df = bc_df.loc[bc_df.pargp==pargp,:]
                #p_df = gp_df.drop_duplicates(subset="parnme")
                #print(p_df)
                bc_dfs.append(gp_df)
            struct_dict[self.spatial_bc_geostruct] = bc_dfs
        if len(struct_dict) > 0:
            if sparse:
                cov = pyemu.helpers.sparse_geostatistical_prior_builder(self.pst,
                                                                        struct_dict=struct_dict,
                                                                        sigma_range=sigma_range)

            else:
                cov = pyemu.helpers.geostatistical_prior_builder(self.pst,
                                                             struct_dict=struct_dict,
                                                             sigma_range=sigma_range)
        else:
            cov = pyemu.Cov.from_parameter_data(self.pst,sigma_range=sigma_range)

        if filename is None:
            filename = os.path.join(self.m.model_ws,self.pst_name+".prior.cov")
        if fmt != "none":
            self.logger.statement("saving prior covariance matrix to file {0}".format(filename))
        if fmt == 'ascii':
            cov.to_ascii(filename)
        elif fmt == 'binary':
            cov.to_binary(filename,droptol=droptol,chunk=chunk)
        elif fmt == 'uncfile':
            cov.to_uncfile(filename)
        elif fmt == 'coo':
            cov.to_coo(filename,droptol=droptol,chunk=chunk)
        self.log("building prior covariance matrix")
        return cov

    def build_pst(self,filename=None):
        """ build the pest control file using the parameterizations and
        observations.

        Parameters
        ----------
            filename : str
                the filename to save the pst to.  If None, the name if formed from
                the model namfile name.  Default is None.

        Note
        ----
        calls pyemu.Pst.from_io_files

        calls PESTCHEK

        """
        self.logger.statement("changing dir in to {0}".format(self.m.model_ws))
        os.chdir(self.m.model_ws)
        tpl_files = copy.deepcopy(self.tpl_files)
        in_files = copy.deepcopy(self.in_files)
        try:
            files = os.listdir('.')
            new_tpl_files = [f for f in files if f.endswith(".tpl") and f not in tpl_files]
            new_in_files = [f.replace(".tpl",'') for f in new_tpl_files]
            tpl_files.extend(new_tpl_files)
            in_files.extend(new_in_files)
            ins_files = [f for f in files if f.endswith(".ins")]
            out_files = [f.replace(".ins",'') for f in ins_files]
            for tpl_file,in_file in zip(tpl_files,in_files):
                if tpl_file not in self.tpl_files:
                    self.tpl_files.append(tpl_file)
                    self.in_files.append(in_file)

            for ins_file,out_file in zip(ins_files,out_files):
                if ins_file not in self.ins_files:
                    self.ins_files.append(ins_file)
                    self.out_files.append(out_file)
            self.log("instantiating control file from i/o files")
            pst = pyemu.Pst.from_io_files(tpl_files=self.tpl_files,
                                          in_files=self.in_files,
                                          ins_files=self.ins_files,
                                          out_files=self.out_files)
            self.log("instantiating control file from i/o files")
        except Exception as e:
            os.chdir("..")
            self.logger.lraise("error build Pst:{0}".format(str(e)))
        os.chdir('..')
        # more customization here
        par = pst.parameter_data
        for name,df in self.par_dfs.items():
            if "parnme" not in df.columns:
                continue
            df.index = df.parnme
            for col in par.columns:
                if col in df.columns:
                    par.loc[df.parnme,col] = df.loc[:,col]
        par.loc[:,"parubnd"] = 10.0
        par.loc[:,"parlbnd"] = 0.1

        for name,df in self.par_dfs.items():
            if "parnme" not in df:
                continue
            df.index = df.parnme
            for col in ["parubnd","parlbnd","pargp"]:
                if col in df.columns:
                    par.loc[df.index,col] = df.loc[:,col]

        for tag,[lw,up] in wildass_guess_par_bounds_dict.items():
            par.loc[par.parnme.apply(lambda x: x.startswith(tag)),"parubnd"] = up
            par.loc[par.parnme.apply(lambda x: x.startswith(tag)),"parlbnd"] = lw


        if self.par_bounds_dict is not None:
            for tag,[lw,up] in self.par_bounds_dict.items():
                par.loc[par.parnme.apply(lambda x: x.startswith(tag)),"parubnd"] = up
                par.loc[par.parnme.apply(lambda x: x.startswith(tag)),"parlbnd"] = lw



        obs = pst.observation_data
        for name,df in self.obs_dfs.items():
            if "obsnme" not in df.columns:
                continue
            df.index = df.obsnme
            for col in df.columns:
                if col in obs.columns:
                    obs.loc[df.obsnme,col] = df.loc[:,col]

        self.pst_name = self.m.name+".pst"
        pst.model_command = ["python forward_run.py"]
        pst.control_data.noptmax = 0

        self.log("writing forward_run.py")
        self.write_forward_run()
        self.log("writing forward_run.py")

        if filename is None:
            filename = os.path.join(self.m.model_ws,self.pst_name)
        self.logger.statement("writing pst {0}".format(filename))

        pst.write(filename)
        self.pst = pst

        self.log("running pestchek on {0}".format(self.pst_name))
        os.chdir(self.m.model_ws)
        try:
            run("pestchek {0} >pestchek.stdout".format(self.pst_name))
        except Exception as e:
            self.logger.warn("error running pestchek:{0}".format(str(e)))
        for line in open("pestchek.stdout"):
            self.logger.statement("pestcheck:{0}".format(line.strip()))
        os.chdir("..")
        self.log("running pestchek on {0}".format(self.pst_name))

    def add_external(self):
        """ add external (existing) template files and instrution files to the
        Pst instance

        """
        if self.external_tpl_in_pairs is not None:
            if not isinstance(self.external_tpl_in_pairs,list):
                external_tpl_in_pairs = [self.external_tpl_in_pairs]
            for tpl_file,in_file in self.external_tpl_in_pairs:
                if not os.path.exists(tpl_file):
                    self.logger.lraise("couldn't find external tpl file:{0}".\
                                       format(tpl_file))
                self.logger.statement("external tpl:{0}".format(tpl_file))
                shutil.copy2(tpl_file,os.path.join(self.m.model_ws,
                                                   os.path.split(tpl_file)[-1]))
                if os.path.exists(in_file):
                    shutil.copy2(in_file,os.path.join(self.m.model_ws,
                                                   os.path.split(in_file)[-1]))

        if self.external_ins_out_pairs is not None:
            if not isinstance(self.external_ins_out_pairs,list):
                external_ins_out_pairs = [self.external_ins_out_pairs]
            for ins_file,out_file in self.external_ins_out_pairs:
                if not os.path.exists(ins_file):
                    self.logger.lraise("couldn't find external ins file:{0}".\
                                       format(ins_file))
                self.logger.statement("external ins:{0}".format(ins_file))
                shutil.copy2(ins_file,os.path.join(self.m.model_ws,
                                                   os.path.split(ins_file)[-1]))
                if os.path.exists(out_file):
                    shutil.copy2(out_file,os.path.join(self.m.model_ws,
                                                   os.path.split(out_file)[-1]))
                    self.logger.warn("obs listed in {0} will have values listed in {1}"
                                     .format(ins_file,out_file))
                else:
                    self.logger.warn("obs listed in {0} will have generic values")

    def write_forward_run(self):
        """ write the forward run script forward_run.py

        """
        with open(os.path.join(self.m.model_ws,self.forward_run_file),'w') as f:
            f.write("import os\nimport numpy as np\nimport pandas as pd\nimport flopy\n")
            f.write("import pyemu\n")
            for tmp_file in self.tmp_files:
                f.write("try:\n")
                f.write("   os.remove('{0}')\n".format(tmp_file))
                f.write("except Exception as e:\n")
                f.write("   print('error removing tmp file:{0}')\n".format(tmp_file))
            for line in self.frun_pre_lines:
                f.write(line+'\n')
            for line in self.frun_model_lines:
                f.write(line+'\n')
            for line in self.frun_post_lines:
                f.write(line+'\n')

    def parse_k(self,k,vals):
        """ parse the iterable from a property or boundary condition argument

        Parameters
        ----------
        k : int or iterable int
            the iterable
        vals : iterable of ints
            the acceptable values that k may contain

        Returns
        -------
        k_vals : iterable of int
            parsed k values

        """
        try:
            k = int(k)
        except:
            pass
        else:
            assert k in vals,"k {0} not in vals".format(k)
            return [k]
        if k is None:
            return vals
        else:
            try:
                k_vals = vals[k]
            except Exception as e:
                raise Exception("error slicing vals with {0}:{1}".
                                format(k,str(e)))
            return k_vals

    def parse_pakattr(self,pakattr):
        """ parse package-iterable pairs from a property or boundary condition
        argument

        Parameters
        ----------
        pakattr : iterable len 2


        Returns
        -------
        pak : flopy.PakBase
            the flopy package from the model instance
        attr : (varies)
            the flopy attribute from pak.  Could be Util2D, Util3D,
            Transient2D, or MfList
        attrname : (str)
            the name of the attribute for MfList type.  Only returned if
            attr is MfList.  For example, if attr is MfList and pak is
            flopy.modflow.ModflowWel, then attrname can only be "flux"

        """

        raw = pakattr.lower().split('.')
        if len(raw) != 2:
            self.logger.lraise("pakattr is wrong:{0}".format(pakattr))
        pakname = raw[0]
        attrname = raw[1]
        pak = self.m.get_package(pakname)
        if pak is None:
            if pakname == "extra":
                self.logger.statement("'extra' pak detected:{0}".format(pakattr))
                ud = flopy.utils.Util3d(self.m,(self.m.nlay,self.m.nrow,self.m.ncol),np.float32,1.0,attrname)
                return "extra",ud

            self.logger.lraise("pak {0} not found".format(pakname))
        if hasattr(pak,attrname):
            attr = getattr(pak,attrname)
            return pak,attr
        elif hasattr(pak,"stress_period_data"):
            dtype = pak.stress_period_data.dtype
            if attrname not in dtype.names:
                self.logger.lraise("attr {0} not found in dtype.names for {1}.stress_period_data".\
                                  format(attrname,pakname))
            attr = pak.stress_period_data
            return pak,attr,attrname
        else:
            self.logger.lraise("unrecognized attr:{0}".format(attrname))


    def setup_bc_pars(self):
        tdf = self.setup_temporal_bc_pars()
        sdf = self.setup_spatial_bc_pars()
        if tdf is None and sdf is None:
            return
        os.chdir(self.m.model_ws)
        try:
            apply_bc_pars()
        except Exception as e:
            os.chdir("..")
            self.logger.lraise("error test running apply_bc_pars():{0}".format(str(e)))
        os.chdir('..')
        line = "pyemu.helpers.apply_bc_pars()\n"
        self.logger.statement("forward_run line:{0}".format(line))
        self.frun_pre_lines.append(line)

    def setup_temporal_bc_pars(self):
        """ main entry point for setting up boundary condition multiplier
        parameters

        """
        if len(self.temporal_bc_props) == 0:
            return
        self.log("processing temporal_bc_props")
        # if not isinstance(self.bc_prop_dict,dict):
        #     self.logger.lraise("bc_prop_dict must be 'dict', not {0}".
        #                        format(str(type(self.bc_prop_dict))))
        bc_filenames = []
        bc_cols = []
        bc_pak = []
        bc_k = []
        bc_dtype_names = []
        bc_parnme = []
        if len(self.temporal_bc_props) == 2:
            if not isinstance(self.temporal_bc_props[0],list):
                self.temporal_bc_props = [self.temporal_bc_props]
        for pakattr,k_org in self.temporal_bc_props:
            pak,attr,col = self.parse_pakattr(pakattr)
            k_parse = self.parse_k(k_org,np.arange(self.m.nper))
            c = self.get_count(pakattr)
            for k in k_parse:
                bc_filenames.append(self.bc_helper(k,pak,attr,col))
                bc_cols.append(col)
                pak_name = pak.name[0].lower()
                bc_pak.append(pak_name)
                bc_k.append(k)
                bc_dtype_names.append(','.join(attr.dtype.names))

                bc_parnme.append("{0}{1}_{2:03d}".format(pak_name,col,c))

        df = pd.DataFrame({"filename":bc_filenames,"col":bc_cols,
                           "kper":bc_k,"pak":bc_pak,
                           "dtype_names":bc_dtype_names,
                          "parnme":bc_parnme})
        tds = pd.to_timedelta(np.cumsum(self.m.dis.perlen.array),unit='d')
        dts = pd.to_datetime(self.m._start_datetime) + tds
        df.loc[:,"datetime"] = df.kper.apply(lambda x: dts[x])
        df.loc[:,"timedelta"] = df.kper.apply(lambda x: tds[x])
        df.loc[:,"val"] = 1.0
        #df.loc[:,"kper"] = df.kper.apply(np.int)
        #df.loc[:,"parnme"] = df.apply(lambda x: "{0}{1}_{2:03d}".format(x.pak,x.col,x.kper),axis=1)
        df.loc[:,"tpl_str"] = df.parnme.apply(lambda x: "~   {0}   ~".format(x))
        df.loc[:,"bc_org"] = self.bc_org
        df.loc[:,"model_ext_path"] = self.m.external_path
        df.loc[:,"pargp"] = df.parnme.apply(lambda x: x.split('_')[0])
        names = ["filename","dtype_names","bc_org","model_ext_path","col","kper","pak","val"]
        df.loc[:,names].\
            to_csv(os.path.join(self.m.model_ws,"temporal_bc_pars.dat"),sep=' ')
        df.loc[:,"val"] = df.tpl_str
        tpl_name = os.path.join(self.m.model_ws,'temporal_bc_pars.dat.tpl')
        f_tpl =  open(tpl_name,'w')
        f_tpl.write("ptf ~\n")
        f_tpl.flush()
        df.loc[:,names].to_csv(f_tpl,sep=' ',quotechar=' ')
        f_tpl.close()
        self.par_dfs["temporal_bc"] = df

        if self.temporal_bc_geostruct is None:
            v = pyemu.geostats.ExpVario(contribution=1.0,a=180.0) # 180 correlation length
            self.temporal_bc_geostruct = pyemu.geostats.GeoStruct(variograms=v)
        self.log("processing temporal_bc_props")
        return True

    def setup_spatial_bc_pars(self):
        """helper to setup crazy numbers of well flux multipliers"""
        if len(self.spatial_bc_props) == 0:
            return
        self.log("processing spatial_bc_props")

        bc_filenames = []
        bc_cols = []
        bc_pak = []
        bc_k = []
        bc_dtype_names = []
        bc_parnme = []
        if len(self.spatial_bc_props) == 2:
            if not isinstance(self.spatial_bc_props[0], list):
                self.spatial_bc_props = [self.spatial_bc_props]
        for pakattr, k_org in self.spatial_bc_props:
            pak, attr, col = self.parse_pakattr(pakattr)
            k_parse = self.parse_k(k_org, np.arange(self.m.nlay))
            if len(k_parse) > 1:
                self.logger.lraise("spatial_bc_pars error: each set of spatial bc pars can only be applied "+\
                                   "to a single layer (e.g. [wel.flux,0].\n"+\
                                   "You passed [{0},{1}], implying broadcasting to layers {2}".
                                   format(pakattr,k_org,k_parse))
            for k in range(self.m.nper):
                bc_filenames.append(self.bc_helper(k, pak, attr, col))
                bc_cols.append(col)
                pak_name = pak.name[0].lower()
                bc_pak.append(pak_name)
                bc_k.append(k_parse[0])
                #bc_dtype_names.append(list(attr.dtype.names))
                bc_dtype_names.append(','.join(attr.dtype.names))

                #bc_parnme.append("{0}{1}_{2:03d}".format(pak_name, col, c))
        info_df = pd.DataFrame({"filename": bc_filenames, "col": bc_cols,
                           "k": bc_k, "pak": bc_pak,
                           "dtype_names": bc_dtype_names})
        info_df.loc[:,"bc_mlt"] = self.bc_mlt
        info_df.loc[:,"bc_org"] = self.bc_org
        info_df.loc[:,"model_ext_path"] = self.m.external_path

        # check that all files for a given package have the same number of entries
        info_df.loc[:,"itmp"] = np.NaN
        pak_dfs = {}
        for pak in info_df.pak.unique():
            df_pak = info_df.loc[info_df.pak==pak,:]
            itmp = []
            for filename in df_pak.filename:
                names = df_pak.dtype_names.iloc[0].split(',')
                fdf = pd.read_csv(os.path.join(self.m.model_ws,filename),
                                  delim_whitespace=True,header=None,names=names)
                for c in ['k','i','j']:
                    fdf.loc[:,c] -= 1
                itmp.append(fdf.shape[0])
                pak_dfs[pak] = fdf
            info_df.loc[info_df.pak==pak,"itmp"] = itmp
            if np.unique(np.array(itmp)).shape[0] != 1:
                info_df.to_csv("spatial_bc_trouble.csv")
                self.logger.lraise("spatial_bc_pars() error: must have same number of "+\
                                   "entries for every stress period for {0}".format(pak))

        # make the pak dfs have unique model indices
        for pak,df in pak_dfs.items():
            df.loc[:,"idx"] = df.apply(lambda x: "{0:02.0f}{1:04.0f}{2:04.0f}".format(x.k,x.i,x.j),axis=1)
            if df.idx.unique().shape[0] != df.shape[0]:
                self.logger.warn("duplicate entries in bc pak {0}...collapsing".format(pak))
                df.drop_duplicates(subset="idx",inplace=True)
            df.index = df.idx
            pak_dfs[pak] = df

        # write template files - find which cols are parameterized...
        par_dfs = []
        for pak,df in pak_dfs.items():
            pak_df = info_df.loc[info_df.pak==pak,:]
            # reset all non-index cols to 1.0
            for col in df.columns:
                if col not in ['k','i','j','inode']:
                    df.loc[:,col] = 1.0
            in_file = os.path.join(self.bc_mlt,pak+".csv")
            tpl_file = os.path.join(pak + ".csv.tpl")
            # save an all "ones" mult df for testing
            df.to_csv(os.path.join(self.m.model_ws,in_file))
            parnme,pargp = [],[]
            x = df.apply(lambda x: self.m.sr.xcentergrid[int(x.i),int(x.j)],axis=1).values
            y = df.apply(lambda x: self.m.sr.ycentergrid[int(x.i),int(x.j)],axis=1).values
            for col in pak_df.col.unique():
                col_df = pak_df.loc[pak_df.col==col]
                k_vals = col_df.k.unique()
                npar = col_df.k.apply(lambda x: x in k_vals).shape[0]
                if npar == 0:
                    continue
                names = df.index.map(lambda x: "{0}{1}{2}".format(pak[0],col[0],x))

                df.loc[:,col] = names.map(lambda x: "~   {0}   ~".format(x))
                df.loc[df.k.apply(lambda x: x not in k_vals),col] = 1.0
                par_df = pd.DataFrame({"parnme": names,"x":x,"y":y,"k":df.k.values}, index=names)
                par_df = par_df.loc[par_df.k.apply(lambda x: x in k_vals)]
                if par_df.shape[0] == 0:
                    self.logger.lraise("no parameters found for spatial bc k,pak,attr {0}, {1}, {2}".
                                       format(k_vals,pak,col))

                par_df.loc[:,"pargp"] = df.k.apply(lambda x : "{0}{1}_k{2:02.0f}".format(pak,col,int(x))).values
                par_df.loc[:,"tpl_file"] = tpl_file
                par_df.loc[:,"in_file"] = in_file
                par_dfs.append(par_df)
            with open(os.path.join(self.m.model_ws,tpl_file),'w') as f:
                f.write("ptf ~\n")
                f.flush()
                df.to_csv(f)
            self.tpl_files.append(tpl_file)
            self.in_files.append(in_file)

        par_df = pd.concat(par_dfs)
        self.par_dfs["spatial_bc"] = par_df
        info_df.to_csv(os.path.join(self.m.model_ws,"spatial_bc_pars.dat"),sep=' ')
        if self.spatial_bc_geostruct is None:
            dist = 10 * float(max(self.m.dis.delr.array.max(),
                                  self.m.dis.delc.array.max()))
            v = pyemu.geostats.ExpVario(contribution=1.0, a=dist)
            self.spatial_bc_geostruct = pyemu.geostats.GeoStruct(variograms=v)
        return True


    # def setup_all_wells(self):
    #     """helper to setup crazy numbers of well flux multipliers"""
    #     self.log("setup all wells parameterization")
    #     if self.m.wel is None:
    #         self.logger.lraise("setup_all_wells() requires a wel pak")
    #     pak, attr, col = self.parse_pakattr("wel.flux")
    #     k_parse = self.parse_k(np.arange(self.m.nper), np.arange(self.m.nper))
    #     bc_filenames = []
    #     for k in k_parse:
    #         bc_filenames.append(self.bc_helper(k, pak, attr, col))
    #     parnme = []
    #     for bc_filename in bc_filenames:
    #         bc_filename = os.path.split(bc_filename)[-1]
    #         kper = int(bc_filename.split('.')[0].split('_')[1])
    #         df = pd.read_csv(os.path.join(self.m.model_ws,self.bc_org,bc_filename),
    #                          delim_whitespace=True,header=None,names=['l','r','c','flux'])
    #         mlt_file = os.path.join(self.m.model_ws, self.bc_mlt, bc_filename)
    #         df.loc[:,"flux"] = 1.0
    #         df.to_csv(mlt_file,sep=' ',index=False,header=False)
    #         df.loc[:,"parnme"] = ["wf{0:04d}_{1:03d}".format(i,kper) for i in range(df.shape[0])]
    #         parnme.extend(list(df.parnme))
    #         df.loc[:,"tpl_str"] = df.parnme.apply(lambda x: "~  {0}   ~".format(x))
    #         tpl_file = os.path.join(self.m.model_ws,bc_filename+".tpl")
    #
    #
    #         self.logger.statement("writing tpl file "+tpl_file)
    #         with open(tpl_file,'w') as f:
    #             f.write("ptf ~\n")
    #             f.write(df.loc[:,['l','r','c','tpl_str']].to_string(index=False,header=False)+'\n')
    #         self.tpl_files.append(os.path.split(tpl_file)[-1])
    #         self.in_files.append(os.path.join(self.bc_mlt, bc_filename))
    #
    #     df = pd.DataFrame({"parnme":parnme}, index=parnme)
    #     df.loc[:,"parubnd"] = 1.2
    #     df.loc[:,"parlbnd"] = 0.8
    #     df.loc[:,"pargp"] = df.parnme.apply(lambda x: "wflux_{0}".format(x.split('_')[-1]))
    #     self.par_dfs["wel_flux"] = df
    #     self.frun_pre_lines.append("pyemu.helpers.apply_all_wells()")
    #     self.log("setup all wells parameterization")


    def bc_helper(self,k,pak,attr,col):
        """ helper to setup boundary condition multiplier parameters for a given
        k, pak, attr set.

        Parameters
        ----------
        k : int or iterable of int
            the zero-based stress period indices
        pak : flopy.PakBase=
            the MODFLOW package
        attr : MfList
            the MfList instance
        col : str
            the column name in the MfList recarray to parameterize

        """
        filename = attr.get_filename(k)
        filename_model = os.path.join(self.m.external_path,filename)
        shutil.copy2(os.path.join(self.m.model_ws,filename_model),
                     os.path.join(self.m.model_ws,self.bc_org,filename))
        return filename_model


    def setup_hds(self):
        """ setup modflow head save file observations for given kper (zero-based
        stress period index) and k (zero-based layer index) pairs using the
        kperk argument.

        Note
        ----
            this can setup a shit-ton of observations

            this is useful for dataworth analyses or for monitoring
            water levels as forecasts



        """
        if self.hds_kperk is None or len(self.hds_kperk) == 0:
            return
        from .gw_utils import setup_hds_obs
        # if len(self.hds_kperk) == 2:
        #     try:
        #         if len(self.hds_kperk[0] == 2):
        #             pass
        #     except:
        #         self.hds_kperk = [self.hds_kperk]
        oc = self.m.get_package("OC")
        if oc is None:
            raise Exception("can't find OC package in model to setup hds grid obs")
        if not oc.savehead:
            raise Exception("OC not saving hds, can't setup grid obs")
        hds_unit = oc.iuhead
        hds_file = self.m.get_output(unit=hds_unit)
        assert os.path.exists(os.path.join(self.org_model_ws,hds_file)),\
        "couldn't find existing hds file {0} in org_model_ws".format(hds_file)
        shutil.copy2(os.path.join(self.org_model_ws,hds_file),
                     os.path.join(self.m.model_ws,hds_file))
        inact = None
        if self.m.lpf is not None:
            inact = self.m.lpf.hdry
        elif self.m.upw is not None:
            inact = self.m.upw.hdry
        if inact is None:
            skip = lambda x: np.NaN if x == self.m.bas6.hnoflo else x
        else:
            skip = lambda x: np.NaN if x == self.m.bas6.hnoflo or x == inact else x
        print(self.hds_kperk)
        frun_line, df = setup_hds_obs(os.path.join(self.m.model_ws,hds_file),
                      kperk_pairs=self.hds_kperk,skip=skip)
        self.obs_dfs["hds"] = df
        self.frun_post_lines.append("pyemu.gw_utils.apply_hds_obs('{0}')".format(hds_file))
        self.tmp_files.append(hds_file)

    def setup_smp(self):
        """ setup observations from PEST-style SMP file pairs

        """
        if self.obssim_smp_pairs is None:
            return
        if len(self.obssim_smp_pairs) == 2:
            if isinstance(self.obssim_smp_pairs[0],str):
                self.obssim_smp_pairs = [self.obssim_smp_pairs]
        for obs_smp,sim_smp in self.obssim_smp_pairs:
            self.log("processing {0} and {1} smp files".format(obs_smp,sim_smp))
            if not os.path.exists(obs_smp):
                self.logger.lraise("couldn't find obs smp: {0}".format(obs_smp))
            if not os.path.exists(sim_smp):
                self.logger.lraise("couldn't find sim smp: {0}".format(sim_smp))
            new_obs_smp = os.path.join(self.m.model_ws,
                                              os.path.split(obs_smp)[-1])
            shutil.copy2(obs_smp,new_obs_smp)
            new_sim_smp = os.path.join(self.m.model_ws,
                                              os.path.split(sim_smp)[-1])
            shutil.copy2(sim_smp,new_sim_smp)
            pyemu.pst_utils.smp_to_ins(new_sim_smp)

    def setup_hob(self):
        """ setup observations from the MODFLOW HOB package


        """

        if self.m.hob is None:
            return
        hob_out_unit = self.m.hob.iuhobsv
        #hob_out_fname = os.path.join(self.m.model_ws,self.m.get_output_attribute(unit=hob_out_unit))
        hob_out_fname = os.path.join(self.org_model_ws,self.m.get_output_attribute(unit=hob_out_unit))

        if not os.path.exists(hob_out_fname):
            self.logger.warn("could not find hob out file: {0}...skipping".format(hob_out_fname))
            return
        hob_df = pyemu.gw_utils.modflow_hob_to_instruction_file(hob_out_fname)
        self.obs_dfs["hob"] = hob_df
        self.tmp_files.append(os.path.split(hob_out_fname))

    def setup_hyd(self):
        """ setup observations from the MODFLOW HYDMOD package


        """
        if self.m.hyd is None:
            return
        org_hyd_out = os.path.join(self.org_model_ws,self.m.name+".hyd.bin")
        if not os.path.exists(org_hyd_out):
            self.logger.warn("can't find existing hyd out file:{0}...skipping".
                               format(org_hyd_out))
            return
        new_hyd_out = os.path.join(self.m.model_ws,os.path.split(org_hyd_out)[-1])
        shutil.copy2(org_hyd_out,new_hyd_out)
        df = pyemu.gw_utils.modflow_hydmod_to_instruction_file(new_hyd_out)
        df.loc[:,"obgnme"] = df.obsnme.apply(lambda x: '_'.join(x.split('_')[:-1]))
        line = "pyemu.gw_utils.modflow_read_hydmod_file('{0}')".\
            format(os.path.split(new_hyd_out)[-1])
        self.logger.statement("forward_run line: {0}".format(line))
        self.frun_post_lines.append(line)
        self.obs_dfs["hyd"] = df
        self.tmp_files.append(os.path.split(new_hyd_out)[-1])

    def setup_water_budget_obs(self):
        """ setup observations from the MODFLOW list file for
        volume and flux water buget information

        """
        org_listfile = os.path.join(self.org_model_ws,self.m.lst.file_name[0])
        if os.path.exists(org_listfile):
            shutil.copy2(org_listfile,os.path.join(self.m.model_ws,
                                                   self.m.lst.file_name[0]))
        else:
            self.logger.warn("can't find existing list file:{0}...skipping".
                               format(org_listfile))
            return
        list_file = os.path.join(self.m.model_ws,self.m.lst.file_name[0])
        flx_file = os.path.join(self.m.model_ws,"flux.dat")
        vol_file = os.path.join(self.m.model_ws,"vol.dat")
        df = pyemu.gw_utils.setup_mflist_budget_obs(list_file,
                                                            flx_filename=flx_file,
                                                            vol_filename=vol_file,
                                                            start_datetime=self.m.start_datetime)
        if df is not None:
            self.obs_dfs["wb"] = df
        #line = "try:\n    os.remove('{0}')\nexcept:\n    pass".format(os.path.split(list_file)[-1])
        #self.logger.statement("forward_run line:{0}".format(line))
        #self.frun_pre_lines.append(line)
        self.tmp_files.append(os.path.split(list_file)[-1])
        line = "pyemu.gw_utils.apply_mflist_budget_obs('{0}',flx_filename='{1}',vol_filename='{2}',start_datetime='{3}')".\
                format(os.path.split(list_file)[-1],
                       os.path.split(flx_file)[-1],
                       os.path.split(vol_file)[-1],
                       self.m.start_datetime)
        self.logger.statement("forward_run line:{0}".format(line))
        self.frun_post_lines.append(line)


def apply_array_pars():
    """ a function to apply array-based multipler parameters.  Used to implement
    the parameterization constructed by PstFromFlopyModel during a forward run

    Note
    ----
    requires "arr_pars.csv" - this file is written by PstFromFlopy

    the function should be added to the forward_run.py script

    """
    df = pd.read_csv("arr_pars.csv")
    # for fname in df.model_file:
    #     try:
    #         os.remove(fname)
    #     except:
    #         print("error removing mult array:{0}".format(fname))

    if 'pp_file' in df.columns:
        for pp_file,fac_file,mlt_file in zip(df.pp_file,df.fac_file,df.mlt_file):
            if pd.isnull(pp_file):
                continue
            pyemu.geostats.fac2real(pp_file=pp_file,factors_file=fac_file,
                                    out_file=mlt_file,lower_lim=1.0e-10)

    for model_file in df.model_file.unique():
        # find all mults that need to be applied to this array
        df_mf = df.loc[df.model_file==model_file,:]
        results = []
        org_file = df_mf.org_file.unique()
        if org_file.shape[0] != 1:
            raise Exception("wrong number of org_files for {0}".
                            format(model_file))
        org_arr = np.loadtxt(org_file[0])

        for mlt in df_mf.mlt_file:
            org_arr *= np.loadtxt(mlt)
        if "upper_bound" in df.columns:
            ub_vals = df_mf.upper_bound.value_counts().dropna().to_dict()
            if len(ub_vals) == 0:
                pass
            elif len(ub_vals) > 1:
                raise Exception("different upper bound values for {0}".format(org_file))
            else:
                ub = list(ub_vals.keys())[0]
                org_arr[org_arr>ub] = ub
        if "lower_bound" in df.columns:
            lb_vals = df_mf.lower_bound.value_counts().dropna().to_dict()
            if len(lb_vals) == 0:
                pass
            elif len(lb_vals) > 1:
                raise Exception("different lower bound values for {0}".format(org_file))
            else:
                lb = list(lb_vals.keys())[0]
                org_arr[org_arr < lb] = lb

        np.savetxt(model_file,org_arr,fmt="%15.6E",delimiter='')

def apply_bc_pars():
    """ a function to apply boundary condition multiplier parameters.  Used to implement
    the parameterization constructed by PstFromFlopyModel during a forward run

    Note
    ----
    requires "bc_pars.csv"

    should be added to the forward_run.py script

    """
    temp_file = "temporal_bc_pars.dat"
    spat_file = "spatial_bc_pars.dat"

    temp_df,spat_df = None,None
    if os.path.exists(temp_file):
        temp_df = pd.read_csv(temp_file, delim_whitespace=True)
        temp_df.loc[:,"split_filename"] = temp_df.filename.apply(lambda x: os.path.split(x)[-1])
        org_dir = temp_df.bc_org.iloc[0]
        model_ext_path = temp_df.model_ext_path.iloc[0]
    if os.path.exists(spat_file):
        spat_df = pd.read_csv(spat_file, delim_whitespace=True)
        spat_df.loc[:,"split_filename"] = spat_df.filename.apply(lambda x: os.path.split(x)[-1])
        mlt_dir = spat_df.bc_mlt.iloc[0]
        org_dir = spat_df.bc_org.iloc[0]
        model_ext_path = spat_df.model_ext_path.iloc[0]
    if temp_df is None and spat_df is None:
        raise Exception("apply_bc_pars() - no key dfs found, nothing to do...")
    # load the spatial mult dfs
    sp_mlts = {}
    if spat_df is not None:

        for f in os.listdir(mlt_dir):
            df = pd.read_csv(os.path.join(mlt_dir,f),index_col=0)
            df.index = df.apply(lambda x: "{0:02.0f}{1:04.0f}{2:04.0f}".format(x.k,x.i,x.j),axis=1)
            pak = f.split(".")[0].lower()
            if pak in sp_mlts.keys():
                raise Exception("duplciate multplier csv for pak {0}".format(pak))
            sp_mlts[pak] = df

    org_files = os.listdir(org_dir)
    #for fname in df.filename.unique():
    for fname in org_files:

        #print(temp_df.filename)
        #print(spat_df.filename)
        names = None
        if temp_df is not None and fname in temp_df.split_filename.values:
            temp_df_fname = temp_df.loc[temp_df.split_filename==fname,:]
            if temp_df_fname.shape[0] > 0:
                names = temp_df_fname.dtype_names.iloc[0].split(',')
            #print(names)
        if spat_df is not None and fname in spat_df.split_filename.values:
            spat_df_fname = spat_df.loc[spat_df.split_filename == fname, :]
            if spat_df_fname.shape[0] > 0:
                names = spat_df_fname.dtype_names.iloc[0].split(',')
        if names is not None:

            df_list = pd.read_csv(os.path.join(org_dir,fname),
                                  delim_whitespace=True,header=None,names=names)
            df_list.loc[:,"idx"] = df_list.apply(lambda x: "{0:02.0f}{1:04.0f}{2:04.0f}".format(x.k-1,x.i-1,x.j-1),axis=1)
            df_list.index = df_list.idx
            pak_name = fname.split('_')[0].lower()
            if pak_name in sp_mlts:
                mlt_df = sp_mlts[pak_name]
                mlt_df_ri = mlt_df.reindex(df_list.index)
                for col in df_list.columns:
                    if col in ["k","i","j","inode"]:
                        continue
                    if col in mlt_df.columns:
                       # print(mlt_df.loc[mlt_df.index.duplicated(),:])
                       # print(df_list.loc[df_list.index.duplicated(),:])
                        df_list.loc[:,col] *= mlt_df_ri.loc[:,col].values

            if temp_df is not None and fname in temp_df.split_filename.values:
                temp_df_fname = temp_df.loc[temp_df.split_filename == fname, :]
                for col,val in zip(temp_df_fname.col,temp_df_fname.val):
                     df_list.loc[:,col] *= val
            #fmts = {}
            fmts = ''
            for name in names:
                if name in ["i","j","k","inode"]:
                    #fmts[name] = pyemu.pst_utils.IFMT
                    #fmts[name] = lambda x: " {0:>9.0f}".format(x)
                    fmts += " %9d"
                else:
                    #fmts[name] = pyemu.pst_utils.FFMT
                    #fmts[name] = lambda x: " {0:>9G}".format(x)
                    fmts += " %9G"

        np.savetxt(os.path.join(model_ext_path,fname),df_list.loc[:,names].as_matrix(),fmt=fmts)
        #with open(os.path.join(model_ext_path,fname),'w') as f:
        #    f.write(df_list.loc[:,names].to_string(header=False,index=False,formatters=fmts)+'\n')
            #df_list.to_csv(os.path.join(model_ext_path,fname),index=False,header=False)

# def apply_all_wells():
#
#     mlt_dir = "bc_mlt"
#     org_dir = "bc_org"
#     names = ["l","r","c","flux"]
#     fmt = {"flux": pyemu.pst_utils.FFMT}
#     for c in names[:-1]:
#         fmt[c] = pyemu.pst_utils.IFMT
#     if not os.path.exists(mlt_dir) or not os.path.exists(org_dir):
#         return
#     mlt_files = os.listdir(mlt_dir)
#     for f in mlt_files:
#         if not "wel" in f.lower():
#             continue
#         org_file = os.path.join(org_dir,f)
#         mlt_file = os.path.join(mlt_dir,f)
#         df_org = pd.read_csv(org_file,header=None,delim_whitespace=True,names=names)
#         df_mlt = pd.read_csv(mlt_file,header=None,delim_whitespace=True,names=names)
#         assert df_org.shape == df_mlt.shape
#         df_org.iloc[:,-1] *= df_mlt.iloc[:,-1]
#         with open(f,'w') as fi:
#             fi.write(df_org.to_string(index=False,header=False,formatters=fmt)+'\n')


def plot_flopy_par_ensemble(pst,pe,num_reals=None,model=None,fig_axes_generator=None,
                            pcolormesh_transform=None):
    """function to plot ensemble of parameter values for a flopy/modflow model.  Assumes
    the FlopytoPst helper was used to setup the model and the forward run.

    Parameters
    ----------
        pst : Pst instance

        pe : ParameterEnsemble instance

        num_reals : int
            number of realizations to process.  If None, all realizations are processed.
            default is None
        model : flopy.mbase
            model instance used for masking inactive areas and also geo-locating plots.  If None,
            generic plots are made.  Default is None.
        fig_axes_generator : function
            a function that returns a pyplot.figure and list of pyplot.axes instances for plots.
            If None, a generic 8.5inX11in figure with 3 rows and 2 cols is used.
        pcolormesh_transform : cartopy.csr.CRS
            transform to map pcolormesh plot into correct location.  Requires model arg


    Note
    ----
        Needs "arr_pars.csv" to run

    Todo
    ----
        add better support for log vs nonlog- currently just logging everything
        add support for cartopy basemap and stamen tiles

    """
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        raise Exception("error import matplotlib: {0}".format(str(e)))
    from matplotlib.backends.backend_pdf import PdfPages
    assert os.path.exists("arr_pars.csv"),"couldn't find arr_pars.csv, can't continue"
    df = pd.read_csv("arr_pars.csv")
    if isinstance(pst,str):
        pst = pyemu.Pst(pst)
    if isinstance(pe,str):
        pe = pd.read_csv(pe,index_col=0)
    if num_reals is None:
        num_reals = pe.shape[0]
    arr_dict,arr_mx,arr_mn = {},{},{}

    islog = True

    ib,sr = None,None
    if model is not None:
        if isinstance(model,str):
            model = flopy.modflow.Modflow.load(model,load_only=[],verbose=False,check=False)
        if model.bas6 is not None:
            ib = {k:model.bas6.ibound[k].array for k in range(model.nlay)}
            sr = model.sr
        if model.btn is not None:
            raise NotImplementedError()

    for i in range(num_reals):
        print(datetime.now(),"processing realization number {0} of {1}".format(i+1,num_reals))
        pst.parameter_data.loc[pe.columns,"parval1"] = pe.iloc[i,:]
        pst.write_input_files()
        apply_array_pars()
        for model_file,k in zip(df.model_file,df.layer):
            arr = np.loadtxt(model_file)
            if islog:
                arr = np.log10(arr)

            if ib is not None:
                arr = np.ma.masked_where(ib[k]<1,arr)
            arr = np.ma.masked_invalid(arr)
            if model_file not in arr_dict.keys():
                arr_dict[model_file] = []
                arr_mx[model_file] = -1.0e+10
                arr_mn[model_file] = 1.0e+10
            arr_mx[model_file] = max(arr_mx[model_file],arr.max())
            arr_mn[model_file] = min(arr_mn[model_file],arr.min())

            arr_dict[model_file].append(arr)




    def _get_fig_and_axes():
        nrow, ncol = 3, 2
        fig = plt.figure(figsize=(8.5, 11))
        axes = [plt.subplot(nrow, ncol, ii + 1) for ii in range(nrow * ncol)]
        [ax.set_xticklabels([]) for ax in axes]
        [ax.set_yticklabels([]) for ax in axes]
        return fig,axes

    if fig_axes_generator is None:
        fig_axes_generator = _get_fig_and_axes

    for model_file,arrs in arr_dict.items():
        k = df.loc[df.model_file==model_file,"layer"].values[0]
        print("plotting arrays for {0}".format(model_file))
        mx,mn = arr_mx[model_file],arr_mn[model_file]
        with PdfPages(model_file+".pdf") as pdf:
            fig,axes = fig_axes_generator()
            ax_count = 0
            for i,arr in enumerate(arrs):

                if ax_count >= len(axes):

                    pdf.savefig()
                    plt.close(fig)
                    fig, axes = fig_axes_generator()
                    ax_count = 0
                    #ax = plt.subplot(111)
                if sr is not None:
                    p = axes[ax_count].pcolormesh(sr.xcentergrid,sr.ycentergrid,arr,vmax=mx,vmin=mn,
                                                  transform=pcolormesh_transform)
                else:
                    p = axes[ax_count].imshow(arr,vmax=mx,vmin=mn)
                plt.colorbar(p,ax=axes[ax_count])
                if islog:
                    arr = 10.*arr
                axes[ax_count].set_title("real index:{0}, max:{1:5.2E}, min:{2:5.2E}".\
                                         format(i,arr.max(),arr.min()),fontsize=6)

                ax_count += 1
            stack = np.array(arrs)
            arr_dict = {}
            for lab,arr in zip(["mean","std"],[np.nanmean(stack,axis=0),np.nanstd(stack,axis=0)]):
                if ib is not None:
                    arr = np.ma.masked_where(ib[k]<1,arr)
                arr = np.ma.masked_invalid(arr)
                arr_dict[lab] = arr
                if ax_count >= len(axes):

                    pdf.savefig()
                    plt.close(fig)
                    fig, axes = fig_axes_generator()
                    ax_count = 0
                if sr is not None:
                    p = axes[ax_count].pcolormesh(sr.xcentergrid,sr.ycentergrid,arr,
                                                  transform=pcolormesh_transform)
                else:
                    p = axes[ax_count].imshow(arr)
                plt.colorbar(p,ax=axes[ax_count])
                axes[ax_count].set_title("{0}, max:{1:5.2E}, min:{2:5.2E}".format(lab,arr.max(),arr.min()),fontsize=6)
                ax_count += 1
            pdf.savefig()
            plt.close("all")
            if sr is not None:
                for i,arr in enumerate(arrs):
                    arr_dict[str(i)] = arr
                flopy.export.shapefile_utils.write_grid_shapefile(model_file+".shp",sr,array_dict=arr_dict)

    return


def _istextfile(filename, blocksize=512):


    """
        Function found from:
        https://eli.thegreenplace.net/2011/10/19/perls-guess-if-file-is-text-or-binary-implemented-in-python

        Returns True if file is most likely a text file
        Returns False if file is most likely a binary file

        Uses heuristics to guess whether the given file is text or binary,
        by reading a single block of bytes from the file.
        If more than 30% of the chars in the block are non-text, or there
        are NUL ('\x00') bytes in the block, assume this is a binary file.
    """

    import sys
    PY3 = sys.version_info[0] == 3

    # A function that takes an integer in the 8-bit range and returns
    # a single-character byte object in py3 / a single-character string
    # in py2.
    #
    int2byte = (lambda x: bytes((x,))) if PY3 else chr

    _text_characters = (
        b''.join(int2byte(i) for i in range(32, 127)) +
        b'\n\r\t\f\b')
    block = open(filename,'rb').read(blocksize)
    if b'\x00' in block:
        # Files with null bytes are binary
        return False
    elif not block:
        # An empty file is considered a valid text file
        return True

    # Use translate's 'deletechars' argument to efficiently remove all
    # occurrences of _text_characters from the block
    nontext = block.translate(None, _text_characters)
    return float(len(nontext)) / len(block) <= 0.30



def plot_summary_distributions(df,ax=None,label_post=False,label_prior=False,
                               subplots=False,figsize=(11,8.5),pt_color='b'):
    """ helper function to plot gaussian distrbutions from prior and posterior
    means and standard deviations

    Parameters
    ----------
    df : pandas.DataFrame
        a dataframe and csv file.  Must have columns named:
        'prior_mean','prior_stdev','post_mean','post_stdev'.  If loaded
        from a csv file, column 0 is assumed to tbe the index
    ax: matplotlib.pyplot.axis
        If None, and not subplots, then one is created
        and all distributions are plotted on a single plot
    label_post: bool
        flag to add text labels to the peak of the posterior
    label_prior: bool
        flag to add text labels to the peak of the prior
    subplots: (boolean)
        flag to use subplots.  If True, then 6 axes per page
        are used and a single prior and posterior is plotted on each
    figsize: tuple
        matplotlib figure size

    Returns
    -------
    figs : list
        list of figures
    axes : list
        list of axes

    Note
    ----
    This is useful for demystifying FOSM results

    if subplots is False, a single axis is returned

    Example
    -------
    ``>>>import matplotlib.pyplot as plt``

    ``>>>import pyemu``

    ``>>>pyemu.helpers.plot_summary_distributions("pest.par.usum.csv")``

    ``>>>plt.show()``
    """
    warnings.warn("pyemu.helpers.plot_summary_distributions() has moved to plot_utils")
    from pyemu import plot_utils
    return plot_utils.plot_summary_distributions(df=df,ax=ax,label_post=label_post,
                                                 label_prior=label_prior,subplots=subplots,
                                                 figsize=figsize,pt_color=pt_color)



def gaussian_distribution(mean, stdev, num_pts=50):
    """ get an x and y numpy.ndarray that spans the +/- 4
    standard deviation range of a gaussian distribution with
    a given mean and standard deviation. useful for plotting

    Parameters
    ----------
    mean : float
        the mean of the distribution
    stdev : float
        the standard deviation of the distribution
    num_pts : int
        the number of points in the returned ndarrays.
        Default is 50

    Returns
    -------
    x : numpy.ndarray
        the x-values of the distribution
    y : numpy.ndarray
        the y-values of the distribution

    """
    warnings.warn("pyemu.helpers.gaussian_distribution() has moved to plot_utils")
    from pyemu import plot_utils
    return plot_utils.gaussian_distribution(mean=mean,stdev=stdev,num_pts=num_pts)


def build_jac_test_csv(pst,num_steps,par_names=None,forward=True):
    """ build a dataframe of jactest inputs for use with sweep

    Parameters
    ----------
    pst : pyemu.Pst

    num_steps : int
        number of pertubation steps for each parameter
    par_names : list
        names of pars to test.  If None, all adjustable pars are used
        Default is None
    forward : bool
        flag to start with forward pertubations.  Default is True

    Returns
    -------
    df : pandas.DataFrame
        the index of the dataframe is par name and the parval used.

    """
    if isinstance(pst,str):
        pst = pyemu.Pst(pst)
    #pst.add_transform_columns()
    pst.build_increments()
    incr = pst.parameter_data.increment.to_dict()
    irow = 0
    par = pst.parameter_data
    if par_names is None:
        par_names = pst.adj_par_names
    total_runs = num_steps * len(par_names) + 1
    idx = ["base"]
    for par_name in par_names:
        idx.extend(["{0}_{1}".format(par_name,i) for i in range(num_steps)])
    df = pd.DataFrame(index=idx, columns=pst.par_names)
    li = par.partrans == "log"
    lbnd = par.parlbnd.copy()
    ubnd = par.parubnd.copy()
    lbnd.loc[li] = lbnd.loc[li].apply(np.log10)
    ubnd.loc[li] = ubnd.loc[li].apply(np.log10)
    lbnd = lbnd.to_dict()
    ubnd = ubnd.to_dict()

    org_vals = par.parval1.copy()
    org_vals.loc[li] = org_vals.loc[li].apply(np.log10)
    if forward:
        sign = 1.0
    else:
        sign = -1.0

    # base case goes in as first row, no perturbations
    df.loc["base",pst.par_names] = par.parval1.copy()
    irow = 1
    full_names = ["base"]
    for jcol, par_name in enumerate(par_names):
        org_val = org_vals.loc[par_name]
        last_val = org_val
        for step in range(num_steps):
            vals = org_vals.copy()
            i = incr[par_name]


            val = last_val + (sign * incr[par_name])
            if val > ubnd[par_name]:
                sign = -1.0
                val = org_val + (sign * incr[par_name])
                if val < lbnd[par_name]:
                    raise Exception("parameter {0} went out of bounds".
                                    format(par_name))
            elif val < lbnd[par_name]:
                sign = 1.0
                val = org_val + (sign * incr[par_name])
                if val > ubnd[par_name]:
                    raise Exception("parameter {0} went out of bounds".
                                    format(par_name))

            vals.loc[par_name] = val
            vals.loc[li] = 10**vals.loc[li]
            df.loc[idx[irow],pst.par_names] = vals
            full_names.append("{0}_{1:<15.6E}".format(par_name,vals.loc[par_name]).strip())

            irow += 1
            last_val = val
    df.index = full_names
    return df









