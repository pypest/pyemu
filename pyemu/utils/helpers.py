from __future__ import print_function, division
import os
import multiprocessing as mp
import subprocess as sp
import platform
import time
import warnings
import struct
import socket
import shutil
import copy
import numpy as np
import pandas as pd
pd.options.display.max_colwidth = 100

try:
    import flopy
except:
    pass

import pyemu


def run(cmd_str,cwd='.'):
    exe_name = cmd_str.split()[0]
    if "window" in platform.platform().lower():
        if not exe_name.lower().endswith("exe"):
            raw = cmd_str.split()
            raw[0] = exe_name + ".exe"
            cmd_str = ' '.join(raw)
    else:
        if os.path.exists(exe_name) and not exe_name.startswith('./'):
            cmd_str = "./" + cmd_str
    print("run():{0}".format(cmd_str))
    bwd = os.getcwd()
    os.chdir(cwd)
    try:
        ret_val = os.system(cmd_str)
    except Exception as e:
        os.chdir(bwd)
        raise Exception("run() raise :{0}".format(str(e)))
    os.chdir(bwd)
    if "window" in platform.platform().lower():
        if ret_val != 0:
            raise Exception("run() returned non-zero")


def pilotpoint_prior_builder(pst, struct_dict,sigma_range=4):
    warnings.warn("'pilotpoint_prior_builder' has been renamed to "+\
                  "'geostatistical_prior_builder'")
    return geostatistical_prior_builder(pst=pst,struct_dict=struct_dict,
                                        sigma_range=sigma_range)

def geostatistical_prior_builder(pst, struct_dict,sigma_range=4):
    """ a helper function to construct a full prior covariance matrix using
    a mixture of geostastical structures an parameter bounds information.
    Parameters:
        pst : pyemu.Pst instance (or the name of a pst file)
        struct_dict : a python dict of geostat structure file : list of pp tpl files
            if the values in the dict are pd.DataFrames, then they must have an
            'x','y', and 'parnme' column.  If the filename ends in '.csv',
            then a pd.DataFrame is loaded.
        sigma_range : float representing the number of standard deviations implied by parameter bounds
    Returns:
        Cov : pyemu.Cov instance

    """
    if isinstance(pst,str):
        pst = pyemu.Pst(pst)
    assert isinstance(pst,pyemu.Pst),"pst arg must be a Pst instance, not {0}".\
        format(type(pst))
    full_cov = pyemu.Cov.from_parameter_data(pst,sigma_range=sigma_range)
    par = pst.parameter_data
    for gs,items in struct_dict.items():
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
                    df = pyemu.gw_utils.pp_tpl_to_dataframe(item)
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
                cov = gs.covariance_matrix(df_zone.x,df_zone.y,df_zone.parnme)
                # find the variance in the diagonal cov
                tpl_var = np.diag(full_cov.get(list(df_zone.parnme),
                                               list(df_zone.parnme)).x)
                if np.std(tpl_var) > 1.0e-6:
                    warnings.warn("pilot points pars have different ranges" +\
                                  " , using max range as variance for all pars")
                tpl_var = tpl_var.max()
                cov *= tpl_var
                try:
                    ci = cov.inv
                except:
                    df_zone.to_csv("prior_builder_crash.csv")
                    raise Exception("error inverting cov {0}".format(cov.row_names[:3]))
                full_cov.replace(cov)
    return full_cov


def kl_setup(num_eig,sr,struct_file,array_dict,basis_file="basis.dat",
             tpl_file="kl.tpl"):
    """setup a karhuenen-Loeve based parameterization for a given
    geostatistical structure.
    Parameters:
        num_eig (int) : number of basis vectors to retain in the reduced basis

        struct_file (str) : a pest-style geostatistical structure file

        array_dict (dict(str:ndarray)): a dict of arrays to setup as KL-based
                                        parameters.  The key becomes the
                                        parameter name prefix. The total
                                        number of parameters is
                                        len(array_dict) * num_eig

        basis_file (str) : the name of the binary file where the reduced
                           basis will be saved

        tpl_file (str) : the name of the template file to make.  The template
                         file is a csv file with the parameter names, the
                         original factor values,and the template entries.
                         The original values can be used to set the parval1
                         entries in the control file
    Returns:
        back_array_dict (dict(str:ndarray)) : a dictionary of back transformed
                                              arrays.  This is useful to see
                                              how much "smoothing" is taking
                                              place compared to the original
                                              arrays.
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
    assert os.path.exists(struct_file)

    gs = pyemu.utils.read_struct_file(struct_file)
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
     input arrays
     Parameters:
        par_file (str) : the csv file to get factor values from.  Must contain
                        the following columns: name, new_val, org_val
        basis_file (str): the binary file that contains the reduced basis

        par_to_file_dict (dict(str:str)): a mapping from KL parameter prefixes
                                          to array file names.
    Returns:
        None

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


def zero_order_tikhonov(pst, parbounds=True,par_groups=None):
        """setup preferred-value regularization
        Parameters:
        ----------
            pst (Pst instance) : the control file instance
            parbounds (bool) : weight the prior information equations according
                to parameter bound width - approx the KL transform
            par_groups (list(str)) : parameter groups to build PI equations
               for.  If None, all adjustable parameters are used
        Returns:
        -------
            None
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

        pst.prior_information = pd.DataFrame({"pilbl": pilbl,
                                               "equation": equation,
                                               "obgnme": obgnme,
                                               "weight": weight})
        if parbounds:
            regweight_from_parbound(pst)


def regweight_from_parbound(pst):
    """sets regularization weights from parameter bounds
        which approximates the KL expansion
    Parameters:
    ----------
        pst (Pst) : a control file instance
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
        Parameters:
        ----------
            pst (pyemu.Pst) : pst instance
            cov (pyemu.Cov) : covariance matrix instance
            reset (bool) : drop all other pi equations.  If False, append to
                existing pi equations
            abs_drop_tol (float) : tolerance to control how many pi
                equations are written.  If the pearson cc is less than
                abs_drop_tol, it will not be included in the pi equations
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


def start_slaves(slave_dir,exe_rel_path,pst_rel_path,num_slaves=None,slave_root="..",
                 port=4004,rel_path=None,local=True,cleanup=True,master_dir=None):
    """ start a group of pest(++) slaves on the local machine

    Parameters:
    ----------
        slave_dir : (str) the path to a complete set of input files

        exe_rel_path : (str) the relative path to the pest(++)
                        executable from within the slave_dir
        pst_rel_path : (str) the relative path to the pst file
                        from within the slave_dir

        num_slaves : (int) number of slaves to start. defaults to number of cores

        slave_root : (str) the root to make the new slave directories in

        rel_path: (str) the relative path to where pest(++) should be run
                  from within the slave_dir, defaults to the uppermost level of the slave dir
        local: (bool) flag for using "localhost" instead of hostname on slave command line
        cleanup: (bool) flag to remove slave directories once processes exit
        master_dir: (str) name of directory for master instance.  If master_dir
                    exists, then it will be removed.  If master_dir is None,
                    no master instance will be started
    """

    assert os.path.isdir(slave_dir)
    assert os.path.isdir(slave_root)
    if num_slaves is None:
        num_slaves = mp.cpu_count()
    else:
        num_slaves = int(num_slaves)
    #assert os.path.exists(os.path.join(slave_dir,rel_path,exe_rel_path))
    exe_verf = True

    if rel_path:
        if not os.path.exists(os.path.join(slave_dir,rel_path,exe_rel_path)):
            #print("warning: exe_rel_path not verified...hopefully exe is in the PATH var")
            exe_verf = False
    else:
        if not os.path.exists(os.path.join(slave_dir,exe_rel_path)):
            #print("warning: exe_rel_path not verified...hopefully exe is in the PATH var")
            exe_verf = False
    if rel_path is not None:
        assert os.path.exists(os.path.join(slave_dir,rel_path,pst_rel_path))
    else:
        assert os.path.exists(os.path.join(slave_dir,pst_rel_path))
    if local:
        hostname = "localhost"
    else:
        hostname = socket.gethostname()

    base_dir = os.getcwd()
    port = int(port)

    if master_dir is not None:
        if master_dir != '.' and os.path.exists(master_dir):
            try:
                shutil.rmtree(master_dir)#, onerror=del_rw)
            except Exception as e:
                raise Exception("unable to remove existing master dir:" + \
                                "{0}\n{1}".format(master_dir,str(e)))
        if master_dir != '.':
            try:
                shutil.copytree(slave_dir,master_dir)
            except Exception as e:
                raise Exception("unable to copy files from slave dir: " + \
                                "{0} to new slave dir: {1}\n{2}".\
                                format(slave_dir,master_dir,str(e)))

        args = [exe_rel_path, pst_rel_path, "/h", ":{0}".format(port)]
        if rel_path is not None:
            cwd = os.path.join(master_dir,rel_path)
        else:
            cwd = master_dir
        print("master:{0} in {1}".format(' '.join(args),cwd))
        try:
            os.chdir(cwd)
            master_p = sp.Popen(args)#,stdout=sp.PIPE,stderr=sp.PIPE)
            os.chdir(base_dir)
        except Exception as e:
            raise Exception("error starting master instance: {0}".\
                            format(str(e)))
        time.sleep(1.5) # a few cycles to let the master get ready


    tcp_arg = "{0}:{1}".format(hostname,port)
    procs = []
    slave_dirs = []
    for i in range(num_slaves):
        new_slave_dir = os.path.join(slave_root,"slave_{0}".format(i))
        if os.path.exists(new_slave_dir):
            try:
                shutil.rmtree(new_slave_dir)#, onerror=del_rw)
            except Exception as e:
                raise Exception("unable to remove existing slave dir:" + \
                                "{0}\n{1}".format(new_slave_dir,str(e)))
        try:
            shutil.copytree(slave_dir,new_slave_dir)
        except Exception as e:
            raise Exception("unable to copy files from slave dir: " + \
                            "{0} to new slave dir: {1}\n{2}".format(slave_dir,new_slave_dir,str(e)))
        try:
            if exe_verf:
                # if rel_path is not None:
                #     exe_path = os.path.join(rel_path,exe_rel_path)
                # else:
                exe_path = exe_rel_path
            else:
                exe_path = exe_rel_path
            args = [exe_path, pst_rel_path, "/h", tcp_arg]
            #print("starting slave in {0} with args: {1}".format(new_slave_dir,args))
            if rel_path is not None:
                cwd = os.path.join(new_slave_dir,rel_path)
            else:
                cwd = new_slave_dir

            os.chdir(cwd)
            print("slave:{0} in {1}".format(' '.join(args),cwd))
            with open(os.devnull,'w') as f:
                p = sp.Popen(args,stdout=f,stderr=f)
            procs.append(p)
            os.chdir(base_dir)
        except Exception as e:
            raise Exception("error starting slave: {0}".format(str(e)))
        slave_dirs.append(new_slave_dir)

    if master_dir is not None:
        # while True:
        #     line = master_p.stdout.readline()
        #     if line != '':
        #         print(str(line.strip())+'\r',end='')
        #     if master_p.poll() is not None:
        #         print(master_p.stdout.readlines())
        #         break
        master_p.wait()
        time.sleep(1.5) # a few cycles to let the slaves end gracefully
        # kill any remaining slaves
        for p in procs:
            p.kill()

    for p in procs:
        p.wait()
    if cleanup:
        for dir in slave_dirs:
            shutil.rmtree(dir)


def plot_summary_distributions(df,ax=None,label_post=False,label_prior=False,
                               subplots=False,figsize=(11,8.5)):
    """ helper function to plot gaussian distrbutions from prior and posterior
    means and standard deviations
    :param df: a dataframe and csv file.  Must have columns named:
    'prior_mean','prior_stdev','post_mean','post_stdev'.  If loaded
    from a csv file, column 0 is assumed to tbe the index
    :param ax: matplotlib axis.  If None, and not subplots, then one is created
    and all distributions are plotted on a single plot
    :param label_post: flag to add text labels to the peak of the posterior
    :param label_prior: flag to add text labels to the peak of the prior
    :param subplots: flag to use subplots.  If True, then 6 axes per page
    are used and a single prior and posterior is plotted on each
    :param figsize: matplotlib figure size
    :return: if subplots, list of figs, list of axes, else, a single axis
    """
    import matplotlib.pyplot as plt
    if isinstance(df,str):
        df = pd.read_csv(df,index_col=0)
    if ax is None and not subplots:
        fig = plt.figure(figsize=figsize)
        ax = plt.subplot(111)
        ax.grid()


    if "post_stdev" not in df.columns and "post_var" in df.columns:
        df.loc[:,"post_stdev"] = df.post_var.apply(np.sqrt)
    if "prior_stdev" not in df.columns and "prior_var" in df.columns:
        df.loc[:,"prior_stdev"] = df.prior_var.apply(np.sqrt)
    if "prior_expt" not in df.columns and "prior_mean" in df.columns:
        df.loc[:,"prior_expt"] = df.prior_mean
    if "post_expt" not in df.columns and "post_mean" in df.columns:
        df.loc[:,"post_expt"] = df.post_mean

    if subplots:
        fig = plt.figure(figsize=figsize)
        ax = plt.subplot(2,3,1)
        ax_per_page = 6
        ax_count = 0
        axes = []
        figs = []
    for name in df.index:
        x,y = gaussian_distribution(df.loc[name,"post_expt"],
                                    df.loc[name,"post_stdev"])
        ax.fill_between(x,0,y,facecolor='b',edgecolor="none",alpha=0.25)
        if label_post:
            mx_idx = np.argmax(y)
            xtxt,ytxt = x[mx_idx],y[mx_idx] * 1.001
            ax.text(xtxt,ytxt,name,ha="center",alpha=0.5)

        x,y = gaussian_distribution(df.loc[name,"prior_expt"],
                                    df.loc[name,"prior_stdev"])
        ax.plot(x,y,color='k',lw=2.0,dashes=(2,1))
        if label_prior:
            mx_idx = np.argmax(y)
            xtxt,ytxt = x[mx_idx],y[mx_idx] * 1.001
            ax.text(xtxt,ytxt,name,ha="center",alpha=0.5)
        #ylim = list(ax.get_ylim())
        #ylim[1] *= 1.2
        #ylim[0] = 0.0
        #ax.set_ylim(ylim)
        if subplots:
            ax.set_title(name)
            ax_count += 1
            ax.set_yticklabels([])
            axes.append(ax)
            if name == df.index[-1]:
                break
            if ax_count >= ax_per_page:
                figs.append(fig)
                fig = plt.figure(figsize=figsize)
                ax_count = 0
            ax = plt.subplot(2,3,ax_count+1)
    if subplots:
        figs.append(fig)
        return figs, axes
    ylim = list(ax.get_ylim())
    ylim[1] *= 1.2
    ylim[0] = 0.0
    ax.set_ylim(ylim)
    ax.set_yticklabels([])
    return ax


def gaussian_distribution(mean,stdev,num_pts=50):
    """for plotting
    """
    xstart = mean - (4.0 * stdev)
    xend = mean + (4.0 * stdev)
    x = np.linspace(xstart,xend,num_pts)
    y = (1.0/np.sqrt(2.0*np.pi*stdev*stdev)) * np.exp(-1.0 * ((x - mean)**2)/(2.0*stdev*stdev))
    return x,y


def read_pestpp_runstorage(filename,irun=0):
    """read pars and obs from a specific run in a pest++ serialized run storage file"""
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

    Parameters:
    ----------
        tpl_files : list[str]
            list of pest template files
        in_files : list[str]
            list of corresponding model input files
        ins_files : list[str]
            list of pest instruction files
        out_files: list[str]
            list of corresponding model output files
        pst_filename : str (optional)
            name of file to write the control file to
    Returns:
    -------
        Pst instance
    """
    par_names = []
    if not isinstance(tpl_files,list):
        tpl_files = [tpl_files]
    if not isinstance(in_files,list):
        in_files = [in_files]
    assert len(in_files) == len(tpl_files),"len(in_files) != len(tpl_files)"

    for tpl_file in tpl_files:
        assert os.path.exists(tpl_file),"template file not found: "+str(tpl_file)
        new_names = [name for name in pyemu.pst_utils.parse_tpl_file(tpl_file) if name not in par_names]
        par_names.extend(new_names)

    if not isinstance(ins_files,list):
        ins_files = [ins_files]
    if not isinstance(out_files,list):
        out_files = [out_files]
    assert len(ins_files) == len(out_files),"len(out_files) != len(out_files)"


    obs_names = []
    for ins_file in ins_files:
        assert os.path.exists(ins_file),"instruction file not found: "+str(ins_file)
        obs_names.extend(pyemu.pst_utils.parse_ins_file(ins_file))

    new_pst = pyemu.pst_utils.generic_pst(par_names,obs_names)

    new_pst.template_files = tpl_files
    new_pst.input_files = in_files
    new_pst.instruction_files = ins_files
    new_pst.output_files = out_files

    #try to run inschek to find the observtion values
    pyemu.pst_utils.try_run_inschek(new_pst)

    if pst_filename:
        new_pst.write(pst_filename,update_regul=True)
    return new_pst


wildass_guess_par_bounds_dict = {"hk":[0.01,100.0],"vka":[0.01,100.0],
                                   "sy":[0.25,1.75],"ss":[0.01,100.0],
                                   "cond":[0.01,100.0],"flux":[0.25,1.75],
                                   "rech":[0.75,1.25],"stage":[0.9,1.1],
                                   }

class PstFromFlopyModel(object):

    def __init__(self,nam_file,org_model_ws,new_model_ws,pp_props=None,const_props=None,
                 bc_props=None,grid_props=None,grid_geostruct=None,pp_space=None,
                 zone_props=None,pp_geostruct=None,par_bounds_dict=None,
                 bc_geostruct=None,remove_existing=False,k_zone_dict=None,
                 mflist_waterbudget=True,mfhyd=True,use_pp_zones=False,
                 obssim_smp_pairs=None,external_tpl_in_pairs=None,
                 external_ins_out_pairs=None,extra_pre_cmds=None,
                 extra_model_cmds=None,extra_post_cmds=None):
        """ a monster helper class to setup multiplier parameters for an
        existing MODFLOW model.  does all kinds of coolness like building a
        meaningful prior, assigning somewhat meaningful parameter groups and
        bounds, writes a forward_run.py script with all the calls need to
        implement multiplier parameters, run MODFLOW and post-process.


        """
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
        self.forward_run_file = "forward_run.py"

        self.remove_existing = remove_existing
        self.external_tpl_in_pairs = external_tpl_in_pairs
        self.external_ins_out_pairs = external_ins_out_pairs
        self.add_external()

        self.arr_mult_dfs = []
        self.par_bounds_dict = par_bounds_dict
        self.pp_props = pp_props
        self.pp_space = pp_space
        self.pp_geostruct = pp_geostruct
        self.use_pp_zones = use_pp_zones

        self.const_props = const_props
        self.bc_props = bc_props
        self.bc_geostruct = bc_geostruct

        self.grid_props = grid_props
        self.grid_geostruct = grid_geostruct

        self.zone_props = zone_props

        self.obssim_smp_pairs = obssim_smp_pairs
        self.frun_pre_lines = []
        self.frun_model_lines = []
        self.frun_post_lines = []

        self.setup_model(nam_file,org_model_ws,new_model_ws)

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
                alist.append("pyemu.run('{0}')\n".format(cmd))

        # add the model call
        line = "pyemu.helpers.run('{0} {1} 1>{1}.stdout 2>{1}.stderr')".format(self.m.exe_name,self.m.namefile)
        self.logger.statement("forward_run line:{0}".format(line))
        self.frun_model_lines.append(line)

        self.tpl_files,self.in_files = [],[]
        self.ins_files,self.out_files = [],[]

        self.setup_mult_dirs()

        self.mlt_files = []
        self.org_files = []
        self.m_files = []
        self.mlt_counter = {}
        self.par_dfs = {}
        self.mlt_dfs = []
        self.setup_bc_pars()
        self.setup_array_pars()
        self.setup_observations()
        self.build_pst()
        self.build_prior()
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

    def setup_mult_dirs(self):
        # setup dirs to hold the original and multiplier model input quantities
        set_dirs = []
        if len(self.pp_props) > 0 or len(self.zone_props) > 0 or \
                        len(self.grid_props) > 0:
            set_dirs.append(self.arr_org)
            set_dirs.append(self.arr_mlt)
        if len(self.bc_props) > 0:
            set_dirs.append(self.bc_org)
        for d in set_dirs:
            d = os.path.join(self.m.model_ws,d)
            self.log("setting up '{0}' dir".format(d))
            if os.path.exists(d):
                if self.remove_existing:
                    shutil.rmtree(d)
                else:
                    raise Exception("dir '{0}' already exists".
                                    format(d))
            os.mkdir(d)
            self.log("setting up '{0}' dir".format(d))

    def setup_model(self,nam_file,org_model_ws,new_model_ws):
        split_new_mws = [i for i in os.path.split(new_model_ws) if len(i) > 0]
        if len(split_new_mws) != 1:
            self.logger.lraise("new_model_ws can only be 1 folder-level deep:{0}".
                               format(str(os.path.split(split_new_mws))))

        self.log("loading flopy model")
        try:
            import flopy
        except:
            raise Exception("from_flopy_model() requires flopy")
        # prepare the flopy model
        self.org_model_ws = org_model_ws
        self.new_model_ws = new_model_ws
        self.m = flopy.modflow.Modflow.load(nam_file,model_ws=org_model_ws)
        self.m.array_free_format = True
        self.m.free_format_input = True
        self.m.external_path = '.'
        self.log("loading flopy model")
        if os.path.exists(new_model_ws):
            if not self.remove_existing:
                self.logger.lraise("'new_model_ws' already exists")
            else:
                self.logger.warn("removing existing 'new_model_ws")
                shutil.rmtree(new_model_ws)
        self.m.change_model_ws(new_model_ws,reset_external=True)

        self.log("writing new modflow input files")
        self.m.write_input()
        self.log("writing new modflow input files")

    def get_count(self,name):
        if name not in self.mlt_counter:
            self.mlt_counter[name] = 1
            c = 0
        else:
            c = self.mlt_counter[name]
            self.mlt_counter[name] += 1
            #print(name,c)
        return c

    def prep_mlt_arrays(self):
        par_props = [self.pp_props,self.grid_props,
                         self.zone_props,self.const_props]
        par_suffixs = [self.pp_suffix,self.gr_suffix,
                       self.zn_suffix,self.cn_suffix]
        mlt_dfs = []
        for par_props,suffix in zip(par_props,par_suffixs):
            if len(par_props) == 2:
                if not isinstance(par_props[0],list):
                    par_props = [par_props]

            for pakattr,k_org in par_props:
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
        mlt_df = pd.concat(mlt_dfs,ignore_index=True)
        return pd.concat(mlt_dfs)

    def write_u2d(self, u2d):
        filename = os.path.split(u2d.filename)[-1]
        np.savetxt(os.path.join(self.m.model_ws,self.arr_org,filename),
                   u2d.array,fmt="%15.6E")
        return filename

    def write_const_tpl(self,name,tpl_file,zn_array):
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
        df.loc[:,"pargp"] = self.cn_suffix
        df.loc[:,"tpl"] = tpl_file
        return df

    def write_grid_tpl(self,name,tpl_file,zn_array):
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
        df.loc[:,"pargp"] = "{0}{1}".format(self.gr_suffix,name)
        df.loc[:,"tpl"] = tpl_file
        return df

    def write_zone_tpl(self,name,tpl_file,zn_array):
        parnme = []
        with open(os.path.join(self.m.model_ws,tpl_file),'w') as f:
            f.write("ptf ~\n")
            for i in range(self.m.nrow):
                for j in range(self.m.ncol):
                    if zn_array[i,j] < 1:
                        pname = " 1.0  "
                    else:
                        pname = "{0}_zn{1}".format(name,zn_array[i,j])
                        if len(pname) > 12:
                            self.logger.lraise("zone pname too long:{0}".\
                                               format(pname))
                        parnme.append(pname)
                        pname = " ~   {0}    ~".format(pname)
                    f.write(pname)
                f.write("\n")
        df = pd.DataFrame({"parnme":parnme},index=parnme)
        df.loc[:,"pargp"] = "{0}{1}".format(self.zn_suffix,name)
        return df

    def grid_prep(self):
        if self.grid_geostruct is None:
            self.logger.warn("grid_geostruct is None,"\
                  " using ExpVario with contribution=1 and a=(max(delc,delr)*10")
            dist = 10 * float(max(self.m.dis.delr.array.max(),
                                           self.m.dis.delc.array.max()))
            v = pyemu.geostats.ExpVario(contribution=1.0,a=dist)
            self.grid_geostruct = pyemu.geostats.GeoStruct(variograms=v)

    def pp_prep(self,mlt_df):
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
        pp_dict = {l:list(pp_df.loc[pp_df.layer==l,"prefix"]) for l in layers}
        pp_array_file = {p:m for p,m in zip(pp_df.prefix,pp_df.mlt_file)}
        self.logger.statement("pp_dict: {0}".format(str(pp_dict)))

        self.log("calling setup_pilot_point_grid()")
        if self.use_pp_zones:
            ib = self.k_zone_dict
        else:
            ib = {k:self.m.bas6.ibound[k].array for k in range(self.m.nlay)}
        pp_df = pyemu.gw_utils.setup_pilotpoints_grid(self.m,
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
            if len(ks) != 1:
                self.logger.lraise("something is wrong in fac calcs")
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
                                        zone_array=self.k_zone_dict[k])
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

    def setup_array_pars(self):
        mlt_df = self.prep_mlt_arrays()
        mlt_df.loc[:,"tpl_file"] = mlt_df.mlt_file.apply(lambda x: os.path.split(x)[-1]+".tpl")
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
            if layers.unique().shape[0] != 1:
                self.logger.lraise("wrong number of layers for {0}"\
                                   .format(mlt_file))
            layer = layers.iloc[0]
            names = mlt_df.loc[mlt_df.mlt_file==mlt_file,"prefix"]
            if names.unique().shape[0] != 1:
                self.logger.lraise("wrong number of names for {0}"\
                                   .format(mlt_file))
            name = names.iloc[0]
            ib = self.k_zone_dict[layer]
            df = None
            if suffix == self.cn_suffix:
                self.log("writing const tpl:{0}".format(tpl_file))
                df = self.write_const_tpl(name,tpl_file,ib)
                self.log("writing const tpl:{0}".format(tpl_file))

            elif suffix == self.gr_suffix:
                self.log("writing grid tpl:{0}".format(tpl_file))
                df = self.write_grid_tpl(name,tpl_file,ib)
                self.log("writing grid tpl:{0}".format(tpl_file))

            elif suffix == self.zn_suffix:
                self.log("writing zone tpl:{0}".format(tpl_file))
                df = self.write_zone_tpl(name,tpl_file,ib)
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
        obs_methods = [self.setup_water_budget_obs,self.setup_hyd,
                       self.setup_smp,self.setup_hob]
        obs_types = ["mflist water budget obs","hyd file",
                     "external obs-sim smp files","hob"]
        self.obs_dfs = {}
        for obs_method, obs_type in zip(obs_methods,obs_types):
            self.log("processing obs type {0}".format(obs_type))
            obs_method()
            self.log("processing obs type {0}".format(obs_type))

    def build_prior(self):
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
        if "bc" in self.par_dfs.keys():
            bc_df = self.par_dfs["bc"]
            bc_df.loc[:,"y"] = 0
            bc_df.loc[:,"x"] = bc_df.timedelta.apply(lambda x: x.days)
            bc_dfs = []
            for pargp in bc_df.pargp.unique():
                gp_df = bc_df.loc[bc_df.pargp==pargp,:]
                p_df = gp_df.drop_duplicates(subset="parnme")
                #print(p_df)
                bc_dfs.append(p_df)
            #bc_dfs = [bc_df.loc[bc_df.pargp==pargp,:].copy() for pargp in bc_df.pargp.unique()]
            struct_dict[self.bc_geostruct] = bc_dfs
        if len(struct_dict) > 0:
            cov = pyemu.helpers.pilotpoint_prior_builder(self.pst,
                                                         struct_dict=struct_dict,
                                                         sigma_range=6)
        else:
            cov = pyemu.Cov.from_parameter_data(self.pst,sigma_range=6)
        cov.to_ascii(os.path.join(self.m.model_ws,self.pst_name+".prior.cov"))
        self.log("building prior covariance matrix")

    def build_pst(self):
        self.log("changing dir in to {0}".format(self.m.model_ws))
        os.chdir(self.m.model_ws)
        try:
            files = os.listdir('.')
            tpl_files = [f for f in files if f.endswith(".tpl")]
            in_files = [f.replace(".tpl",'') for f in tpl_files]
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
            pst = pyemu.Pst.from_io_files(tpl_files=self.tpl_files,
                                          in_files=self.in_files,
                                          ins_files=self.ins_files,
                                          out_files=self.out_files)

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

        self.pst_name = self.m.name+"_pest.pst"
        pst.model_command = ["python forward_run.py"]
        pst.control_data.noptmax = 0

        self.log("writing forward_run.py")
        self.write_forward_run()
        self.log("writing forward_run.py")

        pst_path = os.path.join(self.m.model_ws,self.pst_name)
        self.logger.statement("writing pst {0}".format(pst_path))

        pst.write(pst_path)
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
        if self.external_tpl_in_pairs is not None:
            if not isinstance(self.external_tpl_in_pairs,list):
                external_tpl_in_pairs = [self.external_tpl_in_pairs]
            for tpl_file,in_file in self.external_tpl_in_pairs:
                if not os.path.exists(tpl_file):
                    self.logger.lraise("couldn't find external tpl file:{0}".\
                                       format(tpl_file))
                self.logger.statement("external tpl:{0}".format(tpl_file))
                shutil.copy2(tpl_file,os.path.join(self.m.model_ws,
                                                   os.path.split(tpl_file)))
                if os.path.exists(in_file):
                    shutil.copy2(in_file,os.path.join(self.m.model_ws,
                                                   os.path.split(in_file)))

        if self.external_ins_out_pairs is not None:
            if not isinstance(self.external_ins_out_pairs,list):
                external_ins_out_pairs = [self.external_ins_out_pairs]
            for ins_file,out_file in self.external_ins_out_pairs:
                if not os.path.exists(ins_file):
                    self.logger.lraise("couldn't find external ins file:{0}".\
                                       format(ins_file))
                self.logger.statement("external ins:{0}".format(ins_file))
                shutil.copy2(ins_file,os.path.join(self.m.model_ws,
                                                   os.path.split(ins_file)))
                if os.path.exists(out_file):
                    shutil.copy2(out_file,os.path.join(self.m.model_ws,
                                                   os.path.split(out_file)))
                    self.logger.warn("obs listed in {0} will have values listed in {1}"
                                     .format(ins_file,out_file))
                else:
                    self.logger.warn("obs listed in {0} will have generic values")

    def write_forward_run(self):
        with open(os.path.join(self.m.model_ws,self.forward_run_file),'w') as f:
            f.write("import os\nimport numpy as np\nimport pandas as pd\nimport flopy\n")
            f.write("import pyemu\n")
            for line in self.frun_pre_lines:
                f.write(line+'\n')
            for line in self.frun_model_lines:
                f.write(line+'\n')
            for line in self.frun_post_lines:
                f.write(line+'\n')

    def parse_k(self,k,vals):
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
        raw = pakattr.lower().split('.')
        if len(raw) != 2:
            self.logger.lraise("pakattr is wrong:{0}".format(pakattr))
        pakname = raw[0]
        attrname = raw[1]
        pak = self.m.get_package(pakname)
        if pak is None:
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
            self.logger.lraise("unrecognized attr:{1}".format(attrname))

    def setup_bc_pars(self):
        if len(self.bc_props) == 0:
            return

        self.log("processing bc_props")
        # if not isinstance(self.bc_prop_dict,dict):
        #     self.logger.lraise("bc_prop_dict must be 'dict', not {0}".
        #                        format(str(type(self.bc_prop_dict))))
        bc_filenames = []
        bc_cols = []
        bc_pak = []
        bc_k = []
        bc_dtype_names = []
        bc_parnme = []
        if len(self.bc_props) == 2:
            if not isinstance(self.bc_props[0],list):
                self.bc_props = [self.bc_props]
        for pakattr,k_org in self.bc_props:
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
        self.log("processing bc_prop_dict")
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
            to_csv(os.path.join(self.m.model_ws,"bc_pars.dat"),sep=' ')
        df.loc[:,"val"] = df.tpl_str
        tpl_name = os.path.join(self.m.model_ws,'bc_pars.dat.tpl')
        f_tpl =  open(tpl_name,'w')
        f_tpl.write("ptf ~\n")
        f_tpl.flush()
        df.loc[:,names].to_csv(f_tpl,sep=' ')

        self.par_dfs["bc"] = df
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

        if self.bc_geostruct is None:
            v = pyemu.geostats.ExpVario(contribution=1.0,a=180.0) # 180 correlation length
            self.bc_geostruct = pyemu.geostats.GeoStruct(variograms=v)

    def bc_helper(self,k,pak,attr,col):
        filename = attr.get_filename(k)
        filename_model = os.path.join(self.m.external_path,filename)
        shutil.copy2(os.path.join(self.m.model_ws,filename_model),
                     os.path.join(self.m.model_ws,self.bc_org,filename))
        return filename_model

    def setup_smp(self):
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
        if self.m.hob is None:
            return
        hob_out_unit = self.m.hob.iuhobsv
        hob_out_fname = os.path.join(self.m.model_ws,self.m.get_output_attribute(unit=hob_out_unit))
        if not os.path.exists(hob_out_fname):
            self.logger.warn("could not find hob out file: {0}...skipping".format(hob_out_fname))
            return
        hob_df = pyemu.gw_utils.modflow_hob_to_instruction_file(hob_out_fname)
        self.obs_dfs["hob"] = hob_df

    def setup_hyd(self):
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

    def setup_water_budget_obs(self):
        org_listfile = os.path.join(self.org_model_ws,self.m.lst.file_name[0])
        if os.path.exists(org_listfile):
            shutil.copy2(org_listfile,os.path.join(self.new_model_ws,
                                                   self.m.name+".list"))
        else:
            self.logger.warn("can't find existing list file:{0}...skipping".
                               format(org_listfile))
            return
        list_file = os.path.join(self.m.model_ws,self.m.name+".list")
        flx_file = os.path.join(self.m.model_ws,"flux.dat")
        vol_file = os.path.join(self.m.model_ws,"vol.dat")
        df = pyemu.gw_utils.setup_mflist_budget_obs(list_file,
                                                            flx_filename=flx_file,
                                                            vol_filename=vol_file,
                                                            start_datetime=self.m.start_datetime)
        if df is not None:
            self.obs_dfs["wb"] = df
        line = "try:\n    os.remove('{0}')\nexcept:\n    pass".format(os.path.split(list_file)[-1])
        self.logger.statement("forward_run line:{0}".format(line))
        self.frun_pre_lines.append(line)
        line = "pyemu.gw_utils.apply_mflist_budget_obs('{0}',flx_filename='{1}',vol_filename='{2}',start_datetime='{3}')".\
                format(os.path.split(list_file)[-1],
                       os.path.split(flx_file)[-1],
                       os.path.split(vol_file)[-1],
                       self.m.start_datetime)
        self.logger.statement("forward_run line:{0}".format(line))
        self.frun_post_lines.append(line)


def apply_array_pars():
    df = pd.read_csv("arr_pars.csv")
    # for fname in df.model_file:
    #     try:
    #         os.remove(fname)
    #     except:
    #         print("error removing mult array:{0}".format(fname))
    for pp_file,fac_file,mlt_file in zip(df.pp_file,df.fac_file,df.mlt_file):
        if pd.isnull(pp_file):
            continue
        pyemu.gw_utils.fac2real(pp_file=pp_file,factors_file=fac_file,
                                out_file=mlt_file)

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
        np.savetxt(model_file,org_arr,fmt="%15.6E")

def apply_bc_pars():
    df = pd.read_csv("bc_pars.dat",delim_whitespace=True)
    for fname in df.filename.unique():
        df_fname = df.loc[df.filename==fname,:]
        names = df_fname.dtype_names.iloc[0].split(',')
        bc_org = df_fname.bc_org.iloc[0]
        model_ext_path = df_fname.model_ext_path.iloc[0]
        df_list = pd.read_csv(os.path.join(bc_org,fname),
                              delim_whitespace=True,header=None,names=names)
        for col,val in zip(df_fname.col,df_fname.val):
            df_list.loc[:,col] *= val
        fmts = {}
        for name in names:
            if name in ["i","j","k","inode"]:
                fmts[name] = pyemu.pst_utils.IFMT
            else:
                fmts[name] = pyemu.pst_utils.FFMT
        with open(os.path.join(model_ext_path,fname),'w') as f:
            f.write(df_list.to_string(header=False,index=False,formatters=fmts)+'\n')
            #df_list.to_csv(os.path.join(model_ext_path,fname),index=False,header=False)

