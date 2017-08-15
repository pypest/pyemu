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


def run(cmd_str):
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
    ret_val = os.system(cmd_str)
    if "window" in platform.platform().lower():
        if ret_val != 0:
            raise Exception("run() returned non-zero")


def pilotpoint_prior_builder(pst, struct_dict,sigma_range=4):
    """ a helper function to construct a full prior covariance matrix.
    Parameters:
        pst : pyemu.Pst instance
        struct_dict : a python dict of geostat structure file : list of pp tpl files
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
    for gs,tpl_files in struct_dict.items():
        if isinstance(gs,str):
            gss = pyemu.geostats.read_struct_file(gs)
            if isinstance(gss,list):
                warnings.warn("using first geostat structure in file {0}".\
                              format(gs))
                gs = gss[0]
            else:
                gs = gss
        if not isinstance(tpl_files,list):
            tpl_files = [tpl_files]
        for tpl_file in tpl_files:
            if isinstance(tpl_file,str):
                assert os.path.exists(tpl_file),"pp template file {0} not found".\
                    format(tpl_file)
                pp_df = pyemu.gw_utils.pp_tpl_to_dataframe(tpl_file)
            else:
                pp_df = tpl_file
            missing = pp_df.loc[pp_df.parnme.apply(
                    lambda x : x not in par.parnme),"parnme"]
            if len(missing) > 0:
                warnings.warn("the following parameters in tpl {0} are not " + \
                              "in the control file: {1}".\
                              format(tpl_file,','.join(missing)))
                pp_df = pp_df.loc[pp_df.parnme.apply(lambda x: x not in missing)]
            if "zone" not in pp_df.columns:
                pp_df.loc[:,"zone"] = 1
            zones = pp_df.zone.unique()
            for zone in zones:
                pp_zone = pp_df.loc[pp_df.zone==zone,:]
                cov = gs.covariance_matrix(pp_zone.x,pp_zone.y,pp_zone.parnme)
                # find the variance in the diagonal cov
                tpl_var = np.diag(full_cov.get(list(pp_zone.parnme),
                                               list(pp_zone.parnme)).x)
                if np.std(tpl_var) > 1.0e-6:
                    warnings.warn("pilot points pars have different ranges" +\
                                  " , using max range as variance for all pars")
                tpl_var = tpl_var.max()
                cov *= tpl_var
                ci = cov.inv
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


def plot_summary_distributions(df,ax=None,label_post=False,label_prior=False):
    import matplotlib.pyplot as plt
    if ax is None:
        fig = plt.figure(figsize=(10,10))
        ax = plt.subplot(111)
        ax.grid()
    if "post_stdev" not in df.columns and "post_var" in df.columns:
        df.loc[:,"post_stdev"] = df.post_var.apply(np.sqrt)
    if "prior_stdev" not in df.columns and "prior_var" in df.columns:
        df.loc[:,"prior_stdev"] = df.prior_var.apply(np.sqrt)

    for name,mean,stdev in zip(df.index,df.post_expt,df.post_stdev):
        x,y = gaussian_distribution(mean,stdev)
        ax.fill_between(x,0,y,facecolor='b',edgecolor="none",alpha=0.25)
        if label_post:
            mx_idx = np.argmax(y)
            xtxt,ytxt = x[mx_idx],y[mx_idx] * 1.001
            ax.text(xtxt,ytxt,name,ha="center",alpha=0.5)


    for mean,stdev in zip(df.prior_expt,df.prior_stdev):
        x,y = gaussian_distribution(mean,stdev)
        ax.plot(x,y,color='k',lw=2.0,dashes=(2,1))
        if label_prior:
            mx_idx = np.argmax(y)
            xtxt,ytxt = x[mx_idx],y[mx_idx] * 1.001
            ax.text(xtxt,ytxt,name,ha="center",alpha=0.5)

    ylim = list(ax.get_ylim())
    ylim[1] *= 1.2
    ax.set_ylim(ylim)
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

#TODO write runtime helpers to apply pilot points, array mults and bc_mults
#TODO write a file mapping multiplier bc and array parameters to mlt arrays and org arrays
#TODO download binaries from pestpp github based on platform

# def pst_from_flopy_model(nam_file,org_model_ws,new_model_ws,pp_pakattr_list=None,const_pakattr_list=None,bc_pakattr_list=None,
#                          grid_pakattr_list=None,grid_geostruct=None,pp_space=None,pp_bounds=None,
#                          pp_geostruct=None,bc_geostruct=None,remove_existing=False):
#
#     return PstFromFlopyModel(nam_file=nam_file,org_model_ws=org_model_ws,new_model_ws=new_model_ws,
#                              pp_pakattr_list=pp_pakattr_list,const_pakattr_list=const_pakattr_list,
#                              bc_pakattr_list=bc_pakattr_list,grid_pakattr_list=grid_pakattr_list,
#                              grid_geostruct=None,pp_space=None,pp_bounds=None,
#                              pp_geostruct=None,bc_geostruct=None,remove_existing=False)

wildass_guess_par_bounds_dict = {"hk":[0.01,100.0],"vka":[0.01,100.0],
                                   "sy":[0.25,1.75],"ss":[0.01,100.0],
                                   "cond":[0.01,100.0],"flux":[0.25,1.75],
                                   "rech":[0.75,1.25],"stage":[0.9,1.1],
                                   }

class PstFromFlopyModel(object):

    def __init__(self,nam_file,org_model_ws,new_model_ws,pp_prop_dict=None,const_prop_dict=None,
                 bc_prop_dict=None,grid_prop_dict=None,grid_geostruct=None,pp_space=None,
                 zone_prop_dict=None,pp_geostruct=None,par_bounds_dict=None,
                 bc_geostruct=None,remove_existing=False,
                 mflist_waterbudget=True,mfhyd=True,
                 obssim_smp_pairs=None,external_tpl_in_pairs=None,
                 external_ins_out_pairs=None,extra_pre_cmds=None,
                 extra_model_cmds=None,extra_post_cmds=None):
        """ a monster helper function to setup multiplier parameters for an
        existing MODFLOW model.  does all kinds of coolness like building a
        meaningful prior, assigning somewhat meaningful parameter groups and
        bounds, writes a forward_run.py script with all the calls need to
        implement multiplier parameters, run MODFLOW and post-process.

        :param nam_file: MODFLOW nam file
        :param org_model_ws: the path to the existing model files
        :param new_model_ws: the path to write new model files in. A new set
        of model files will be written using open/close and free format for all
        input files
        :param pp_prop_dict: a dictionary of index (zero-based) keys and
         list of (package,property) pairs to apply multiplier pilot points parameters to.
         example: pp_prop_dict = {':':[("lpf","hk"),("lpf","ss")],1:[("lpf","vka)]}
         would apply pilot points to hk and ss for all layers (':') and pilot points to vka
         in layer 2
        :param const_prop_dict: a dictionary of index (zero-based) keys and
         list of (package,property) pairs to apply constant (uniform in space)
         parameters to. example: const_prop_dict = {':':[("lpf","hk"),("rch","rech")]}
         would construct a multipler for hk in all layers and a multiplier for
         recharge for all stress periods
        :param bc_prop_dict: a dictionary of index (zero-based) keys and
         list of (package,property) pairs to apply boundary condition multipliers to
          example: bc_prop_dict = {':':[("riv","cond"),("riv","stage"),("wel","flux")]}
          would construct river package conductance and stage multipliers for each
          stress period and well package flux multipliers for each stress period
        :param grid_prop_dict:a dictionary of index (zero-based) keys and
         list of (package,property) pairs to apply grid-scale (every active model cell)
         parameters. example: grid_prop_dict = {':':[("lpf","hk"),("lpf","ss")],1:[("lpf","vka)]}
         would setup a metric shit ton of parameters!
         parameters to
        :param grid_geostruct: a pyemu.geostats.GeoStruct to use when constructing the
        prior for grid-scale parameters.  If None, a decent, generic GeoStruct will be used

        :param pp_space: number of grid cells between pilot points.  If None, 10 is used
        :param zone_prop_dict:a dictionary of index (zero-based) keys and
         list of (package,property) pairs to apply zoned parameters to.  This relies on
         the ibound array for zones.example: zone_prop_dict =
         {':':[("lpf","hk"),("lpf","ss")],1:[("lpf","vka)]} would apply zone
          multiplier parameters to hk and ss for all layers (':') and to vka
         in layer 2
        :param pp_geostruct:a pyemu.geostats.GeoStruct to use when constructing the
        prior for pilot point parameters.  If None, a decent, generic GeoStruct will be used
        :param par_bounds_dict: a dictionary of {tag:(lower,upper)} pairs to use when
        setting parameter bounds.  example: par_bounds_dict = {"hk":(0.01,100.0),"ss":(0.5,1.5)}
        would set all parameters starting with "hk" to have lower and upper bound of 0.01 and
        100.0 and all parameters starting with "ss" to have a lower and upper bound of 0.5 and 1.5
        :param bc_geostruct:a pyemu.geostats.GeoStruct to use when constructing the
        prior for temporal boundary condition parameters.  In this context, the
        'a' variogram parameter represents the temporal correlation length.
        If None, a decent, generic GeoStruct will be used
        :param remove_existing: a safety flag.  If new_model_ws exists and
        remove_existing is False, an Exception will be raised
        :param mflist_waterbudget: flag to use flux and volumetric entires in the
        modflow list file as observations.
        :param mfhyd: flag to use modflow HYD package binary outputs as observations
        :param obssim_smp_pairs: a list of pairs of observed and simulated SMP files
        to use as observations.
        :param external_tpl_in_pairs: existing template and model input file pairs
        to add to the pest control file.  template files must be "found" from the location
        where the setup script is running.  They will be copied in the new_model_ws.
        :param external_ins_out_pairs: existing instruction and model output file pairs
        to add to the pest control file.  instruction files must be "found" from the
        location where the setup script is running.  They will be copied into new_model_ws
        :param extra_pre_cmds: additional pre-processor commands to add to the
        forward_run.py script.  example "my_extra_preprocessor.py"
        :param extra_model_cmds: additional model run commands to add to the
        forward_run.py script.  example "mp6 modpath.in"
        :param extra_post_cmds: additional post-processing commands to add to the
        forward_run.py script. example "mod2obs <mod2obs.rsp"
        :return:
        """
        self.logger = pyemu.logger.Logger("PstFromFlopyModel.log")
        self.log = self.logger.log

        self.logger.echo = True

        self.arr_org = "arr_org"
        self.arr_mlt = "arr_mlt"
        self.bc_org = "bc_org"
        self.forward_run_file = "forward_run.py"

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
            if not remove_existing:
                self.logger.lraise("'new_model_ws' already exists")
            else:
                self.logger.warn("removing existing 'new_model_ws")
                shutil.rmtree(new_model_ws)
        self.m.change_model_ws(new_model_ws,reset_external=True)

        self.log("writing new modflow input files")
        self.m.write_input()
        self.log("writing new modflow input files")

        # setup dirs to hold the original and multiplier model input quantities
        set_dirs = []
        if len(pp_prop_dict) > 0 or len(zone_prop_dict) > 0 or len(grid_prop_dict) > 0:
            set_dirs.append(self.arr_org)
            set_dirs.append(self.arr_mlt)
        if len(bc_prop_dict) > 0:
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

        # add any extra commands to the forward run lines
        self.frun_pre_lines = []
        self.frun_model_lines = []
        self.frun_post_lines = []
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

        self.external_tpl_in_pairs = external_tpl_in_pairs
        self.external_ins_out_pairs = external_ins_out_pairs
        self.add_external()

        self.arr_mult_dfs = []
        self.par_bounds_dict = par_bounds_dict
        self.pp_prop_dict = pp_prop_dict
        self.pp_space = pp_space
        self.pp_geostruct = pp_geostruct

        self.const_prop_dict = const_prop_dict
        self.bc_prop_dict = bc_prop_dict
        self.bc_geostruct = bc_geostruct

        self.grid_prop_dict = grid_prop_dict
        self.grid_geostruct = grid_geostruct

        self.zone_prop_dict = zone_prop_dict

        self.obssim_smp_pairs = obssim_smp_pairs

        self.process_parameters()

        self.process_observations()

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


    def process_parameters(self):
        par_dicts = [self.pp_prop_dict,self.bc_prop_dict,
                     self.const_prop_dict,self.grid_prop_dict,
                     self.zone_prop_dict]
        par_methods = [self.setup_pp,self.setup_bc,self.setup_const,
                       self.setup_grid,self.setup_zone]
        par_types = ["pilot points","boundary conditions","constants",
                     "grid","zones"]

        #TODO: check for dups by tracking k:prop pairs
        self.k_prop_list = []
        self.par_dfs = {}
        for par_dict,par_method,par_type in zip(par_dicts,par_methods,par_types):
            self.log("processing {0} parameters".format(par_type))
            par_method()
            self.log("processing {0} parameters".format(par_type))


    def process_observations(self):
        obs_methods = [self.setup_water_budget_obs,self.setup_hyd,self.setup_smp]
        obs_types = ["mflist water budget obs","hyd file","external obs-sim smp files"]
        self.obs_dfs = {}
        for obs_method, obs_type in zip(obs_methods,obs_types):
            self.log("processing obs type {0}".format(obs_type))
            obs_method()
            self.log("processing obs type {0}".format(obs_type))


    def build_prior(self):
        self.log("building prior covariance matrix")
        struct_dict = {}
        if "pp" in self.par_dfs.keys():
            pp_df = self.par_dfs["pp"]
            pp_dfs = [pp_df.loc[pp_df.pargp==pargp,:].copy() for pargp in pp_df.pargp.unique()]
            struct_dict[self.pp_geostruct] = pp_dfs
        if "grid" in self.par_dfs.keys():
            gr_df = self.par_dfs["grid"]
            gr_dfs = [gr_df.loc[gr_df.pargp==pargp,:].copy() for pargp in gr_df.pargp.unique()]
            struct_dict[self.grid_geostruct] = gr_dfs
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

    def setup_bc(self):
        if len(self.bc_prop_dict) == 0:
            return

        self.log("processing bc_prop_dict")
        if not isinstance(self.bc_prop_dict,dict):
            self.logger.lraise("bc_prop_dict must be 'dict', not {0}".
                               format(str(type(self.bc_prop_dict))))
        bc_filenames = []
        bc_cols = []
        bc_pak = []
        bc_k = []
        bc_dtype_names = []
        for pakattr,k_org in self.bc_prop_dict.items():
            pak,attr,col = self.parse_pakattr(pakattr)
            k_parse = self.parse_k(k_org,np.arange(self.m.nper))
            for k in k_parse:
                bc_filenames.append(self.bc_helper(k,pak,attr,col))
                bc_cols.append(col)
                bc_pak.append(pak.name[0].lower())
                bc_k.append(k)
                bc_dtype_names.append(','.join(attr.dtype.names))
        self.log("processing bc_prop_dict")
        df = pd.DataFrame({"filename":bc_filenames,"col":bc_cols,
                           "kper":bc_k,"pak":bc_pak,"dtype_names":bc_dtype_names})
        df.loc[:,"val"] = 1.0
        #df.loc[:,"kper"] = df.kper.apply(np.int)
        df.loc[:,"parnme"] = df.apply(lambda x: "{0}{1}_{2:03d}".format(x.pak,x.col,x.kper),axis=1)
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
        #f_tpl.close()
        #f_tpl = open(tpl_name,'a')
        #names[-1] = "tpl_str"
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

    def bc_helper(self,k,pak,attr,col):
        filename = attr.get_filename(k)
        filename_model = os.path.join(self.m.external_path,filename)
        shutil.copy2(os.path.join(self.m.model_ws,filename_model),
                     os.path.join(self.m.model_ws,self.bc_org,filename))
        return filename_model


    def setup_const(self):
        if len(self.const_prop_dict) == 0:
            return

        self.log("processing const_prop_dict")
        if not isinstance(self.const_prop_dict,dict):
            self.logger.lraise("const_prop_dict must be 'dict', not {0}".
                               format(str(type(self.const_prop_dict))))
        self.const_paks = {}
        parnme = []
        for pakattr,k_org in self.const_prop_dict.items():
            pak,attr = self.parse_pakattr(pakattr)

            if isinstance(attr,flopy.utils.Transient2d):
                try:
                    k_parse = self.parse_k(k_org,np.arange(self.m.nper))
                except Exception as e:
                    self.logger.lraise("error parsing k {0}:{1}".format(k_org,str(e)))
            else:
                try:
                    k_parse = self.parse_k(k_org,np.arange(self.m.nlay))
                except Exception as e:
                    self.logger.lraise("error parsing k {0}:{1}".format(k_org,str(e)))
            for k in k_parse:
                parnme.append(self.const_helper(k,pak,attr))
        self.log("processing const_prop_dict")
        df = pd.DataFrame({"parnme":parnme},index=parnme)
        df.loc[:,"pargp"] = "const"
        self.par_dfs["constants"] = df
        for panme,pak in self.const_paks.items():
            with open(pak.fn_path+".tpl",'w') as f:
                f.write("ptf ~\n")
                pak.write_file(f=f)

    def const_helper(self,k,pak,attr):
        if pak.name[0] not in self.const_paks:
            self.const_paks[pak.name[0]] = copy.copy(pak)

        if isinstance(attr,flopy.utils.Util2d):
            return self.const_util2d_helper(attr,k)
        elif isinstance(attr,flopy.utils.Util3d):
            return self.const_util2d_helper(attr.util_2ds[k],k)
        elif isinstance(attr,flopy.utils.Transient2d):
            return self.const_util2d_helper(attr.transient_2ds[k],k)
        elif isinstance(attr,flopy.utils.MfList):
            self.logger.lraise('MfList support not implemented for const')
        else:
            self.logger.lraise("unrecognized pakattr:{0},{1}".format(str(pak),str(attr)))

    def const_util2d_helper(self,attr,k):
        pname = os.path.split(attr.filename)[-1].split('.')[0].lower()
        attr.cnstnt = "~   {0}   ~".format(pname)
        return pname




    def setup_pp(self):
        if len(self.pp_prop_dict) == 0:
            return

        self.log("processing pp_prop_dict")
        if not isinstance(self.pp_prop_dict,dict):
            self.logger.lraise("pp_prop_dict must be 'dict', not {0}".
                               format(str(type(self.pp_prop_dict))))
        self.pp_dict = {}
        self.pp_array_file = {}
        for pakattr,k_org in self.pp_prop_dict.items():
            pak,attr = self.parse_pakattr(pakattr)
            ks = np.arange(self.m.nlay)
            if isinstance(attr,flopy.utils.Transient2d):
                ks = np.arange(self.m.nper)
            try:
                k_parse = self.parse_k(k_org,ks)
            except Exception as e:
                self.logger.lraise("error parsing k {0}:{1}".format(k_org,str(e)))
            for k in k_parse:
                self.pp_helper(k,pak,attr)
        self.log("processing pp_prop_dict")
        self.log("calling setup_pilot_point_grid()")
        if self.pp_space is None:
            self.logger.warn("pp_space is None, using 10...\n")
            self.pp_space=10

        pp_df = pyemu.gw_utils.setup_pilotpoints_grid(self.m,
                                         prefix_dict=self.pp_dict,
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

        if self.pp_geostruct is None:
            self.logger.warn("pp_geostruct is None,"\
                  " using ExpVario with contribution=1 and a=(pp_space*max(delr,delc))")
            pp_dist = self.pp_space * float(max(self.m.dis.delr.array.max(),
                                           self.m.dis.delc.array.max()))
            v = pyemu.geostats.ExpVario(contribution=1.0,a=pp_dist)
            self.pp_geostruct = pyemu.geostats.GeoStruct(variograms=v)

        # calc factors for each layer
        pargp = pp_df.pargp.unique()
        pp_dfs_k = {}
        fac_files = {}
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
                ok_pp.calc_factors_grid(self.m.sr,var_filename=var_file)
                ok_pp.to_grid_factors_file(fac_file)
                fac_files[k] = fac_file
                self.log("calculating factors for k={0}".format(k))
                pp_dfs_k[k] = pp_df_k

        # add lines to the forward run script
        for k,fac_file in fac_files.items():
            #pp_files = pp_df.pp_filename.unique()
            fac_file = os.path.split(fac_file)[-1]
            pp_prefixes = self.pp_dict[k]
            for pp_prefix in pp_prefixes:
                self.log("proecssing pp_prefix:{0}".format(pp_prefix))
                if pp_prefix not in self.pp_array_file.keys():
                    self.logger.lraise("{0} not in self.pp_array_file.keys()".
                                       format(pp_prefix,','.
                                              join(self.pp_array_file.keys())))
                out_file = os.path.join(self.arr_mlt,os.path.split(self.pp_array_file[pp_prefix])[-1])
                pp_files = pp_df.loc[pp_df.pp_filename.apply(lambda x: pp_prefix in x),"pp_filename"]
                if pp_files.unique().shape[0] != 1:
                    self.logger.lraise("wrong number of pp_files found:{0}".format(','.join(pp_files)))
                pp_file = os.path.split(pp_files[0])[-1]

                #line = "try:\n    os.remove('{0}')\nexcept:\n    pass".format(os.path.join(self.m.external_path,os.path.split(out_file)[-1]))
                #self.logger.statement("forward_run line:{0}".format(line))
                #self.frun_pre_lines.append(line)


                line = "try:\n    os.remove('{0}')\nexcept:\n    pass".\
                    format(os.path.join(self.m.external_path,os.path.split(out_file)[-1]))
                self.logger.statement("forward_line: {0}".format(line))
                self.frun_pre_lines.append(line)
                line = "pyemu.gw_utils.fac2real('{0}',factors_file='{1}',out_file='{2}')".\
                    format(pp_file,fac_file,out_file)
                self.logger.statement("forward_run line:{0}".format(line))
                self.frun_pre_lines.append(line)

                line = "org_arr = np.loadtxt('{0}')".\
                    format(os.path.join(self.arr_org,os.path.split(out_file)[-1]))
                self.logger.statement("forward_run line:{0}".format(line))
                self.frun_pre_lines.append(line)

                line = "mlt_arr = np.loadtxt('{0}')".format(out_file)
                self.logger.statement("forward_run line:{0}".format(line))
                self.frun_pre_lines.append(line)

                line = "np.savetxt('{0}',org_arr * mlt_arr,fmt='%15.6E')".\
                    format(os.path.join(self.m.external_path,os.path.split(out_file)[-1]))
                self.logger.statement("forward_run line:{0}".format(line))
                self.frun_pre_lines.append(line)
                self.log("processing pp_prefix:{0}".format(pp_prefix))
        pp_df.loc[:,"pargp"] = pp_df.pargp.apply(lambda x: "pp_{0}".format(x))
        self.par_dfs["pp"] = pp_df
        #self.pp_dict = None
        #self.pp_array_file = None

    def pp_helper(self,k,pak,attr):

        if isinstance(attr,flopy.utils.Util2d):
            name,filename = self.pp_util2d_helper(attr,k)
        elif isinstance(attr,flopy.utils.Util3d):
            name,filename =  self.pp_util2d_helper(attr.util_2ds[k],k)
        elif isinstance(attr,flopy.utils.Transient2d):
            pak.attr = attr.from_4d(self.m,pak.name[0],{attr.name_base.replace("_",''):attr.array})
            #self.logger.warn("only setting up one set of pilot points for all "+\
            #      "stress periods for pakattr:{0}".format(attr.name_base))
            #kper = list(attr.transient_2ds.keys())[0]
            name,filename = self.pp_util2d_helper(attr.transient_2ds[k],k)
            k = 0
        elif isinstance(attr,flopy.utils.MfList):
            self.logger.lraise('MfList support not implemented for pilot points')
        else:
            self.logger.lraise("unrecognized pak,attr:{0},{1}".
                               format(str(pak),str(attr)))
        # need to find what the external filename that flopy writes
        if k not in self.pp_dict.keys():
            self.pp_dict[k] = []
        self.pp_dict[k].append(name)
        self.pp_array_file[name] = filename


    def pp_util2d_helper(self,u2d,k):
        name = u2d.name.split('_')[0]+"{0:02d}".format(k)
        filename = u2d.filename
        if filename is None:
            self.logger.lraise("filename is None for {0}".format(u2d.name))
        filename = os.path.join(self.m.model_ws,self.arr_org,filename)
        self.logger.statement("resetting 'how' to openclose for {0}".format(name))
        self.logger.statement("{0} being written to array file {1}".format(name,filename))
        u2d.how = "openclose"
        # write original array into arr_org
        self.logger.statement("saving array:{0}".format(filename))
        np.savetxt(filename,u2d.array)

        self.k_prop_list.append((k,filename))
        return name,filename



    def setup_grid(self):
        if len(self.grid_prop_dict) == 0:
            return

        self.log("processing grid_prop_dict")
        if not isinstance(self.grid_prop_dict,dict):
            self.logger.lraise("grid_prop_dict must be 'dict', not {0}".
                               format(str(type(self.grid_prop_dict))))
        dfs = []
        for pakattr,k_org in self.grid_prop_dict.items():

            pak,attr = self.parse_pakattr(pakattr)
            ks = np.arange(self.m.nlay)
            if isinstance(attr,flopy.utils.Transient2d):
                ks = np.arange(self.m.nper)
            try:
                k_parse = self.parse_k(k_org,ks)
            except Exception as e:
                self.logger.lraise("error parsing k {0}:{1}".format(k_org,str(e)))
            for k in k_parse:
                dfs.append(self.grid_helper(k,pak,attr))
        self.log("processing grid_prop_dict")

        if self.grid_geostruct is None:
            self.logger.warn("grid_geostruct is None,"\
                  " using ExpVario with contribution=1 and a=(max(delc,delr)*10")
            dist = 10 * float(max(self.m.dis.delr.array.max(),
                                           self.m.dis.delc.array.max()))
            v = pyemu.geostats.ExpVario(contribution=1.0,a=dist)
            self.grid_geostruct = pyemu.geostats.GeoStruct(variograms=v)
        self.par_dfs["grid"] = pd.concat(dfs)

    def grid_helper(self,k,pak,attr):
        if isinstance(attr,flopy.utils.Util2d):
            return self.grid_util2d_helper(attr,k)
        elif isinstance(attr,flopy.utils.Util3d):
            return self.grid_util2d_helper(attr.util_2ds[k],k)
        elif isinstance(attr,flopy.utils.Transient2d):
            #self.logger.warn("only setting up one set of grid pars for all "+\
            #      "stress periods for pakattr:{0}".format(attr.name_base))
            pak.attr = attr.from_4d(self.m,pak.name[0],{attr.name_base.replace("_",''):attr.array})
            #kper = list(attr.transient_2ds.keys())[0]
            return self.grid_util2d_helper(attr.transient_2ds[k],k)

        elif isinstance(attr,flopy.utils.MfList):
            self.logger.lraise('MfList support not implemented for grid pars')
        else:
            self.logger.lraise("unrecognized pak,attr:{0},{1}".
                               format(str(pak),str(attr)))

    def grid_util2d_helper(self,u2d,k):
        name = u2d.name.split('_')[0]+"{0:02d}".format(k)
        filename = u2d.filename
        if filename is None:
            self.logger.lraise("filename is None for {0}".format(u2d.name))
        filename = os.path.join(self.m.model_ws,self.arr_org,filename)
        self.logger.statement("resetting 'how' to openclose for {0}".format(name))
        self.logger.statement("{0} being written to array file {1}".format(name,filename))
        u2d.how = "openclose"
        # write original array into arr_org
        self.logger.statement("saving array:{0}".format(filename))
        np.savetxt(filename,u2d.array)
        # write the template file
        tpl_file = os.path.join(self.m.model_ws,os.path.split(filename)[-1]+".tpl")
        ib = self.m.bas6.ibound[k].array
        parnme,x,y = [],[],[]
        with open(tpl_file,'w') as f:
            f.write("ptf ~\n")
            for i in range(self.m.nrow):
                for j in range(self.m.ncol):
                    if ib[i,j] < 1:
                        pname = ' 1.0 '
                    else:
                        pname = "{0}_{2:03d}{3:03d}".format(name,k,i,j)
                        if len(pname) > 12:
                            self.logger.lraise("grid pname too long:{0}".\
                                               format(pname))
                        parnme.append(pname)
                        pname = ' ~     {0}   ~ '.format(pname)
                        x.append(self.m.sr.xcentergrid[i,j])
                        y.append(self.m.sr.ycentergrid[i,j])
                    f.write(pname)
                f.write("\n")
        line = "try:\n    os.remove('{0}')\nexcept:\n    pass".\
            format(os.path.join(self.m.external_path,os.path.split(filename)[-1]))
        self.log("forward_run line:{0}".format(line))
        self.frun_pre_lines.append(line)
        line = "arr_org = np.loadtxt('{0}')".\
            format(os.path.join(self.arr_org,os.path.split(filename)[-1]))
        self.log("forward_run line:{0}".format(line))
        self.frun_pre_lines.append(line)
        line = "arr_mlt = np.loadtxt('{0}')".\
            format(os.path.join(self.arr_mlt,os.path.split(filename)[-1]))
        self.log("forward_run line:{0}".format(line))
        self.frun_pre_lines.append(line)

        line = "np.savetxt('{0}',arr_mlt * arr_org,fmt='%15.6E')\n".\
            format(os.path.join(self.m.external_path,os.path.split(filename)[-1]))
        self.log("forward_run line:{0}".format(line))
        self.frun_pre_lines.append(line)

        # track these tpl-in pairs because they don't follow the standard spec
        self.tpl_files.append(os.path.split(tpl_file)[-1])
        self.in_files.append(os.path.join(self.arr_mlt,os.path.split(filename)[-1]))
        self.k_prop_list.append((k,filename))
        df = pd.DataFrame({"parnme":parnme,"x":x,"y":y},index=parnme)
        df.loc[:,"pargp"] = "gr_{0}".format(name)
        return df

    def setup_zone(self):
        if len(self.zone_prop_dict) == 0:
            return

        self.log("processing zone_prop_dict")
        if not isinstance(self.zone_prop_dict,dict):
            self.logger.lraise("zone_prop_dict must be 'dict', not {0}".
                               format(str(type(self.zone_prop_dict))))
        dfs = []
        for pakattr,k_org in self.grid_prop_dict.items():
            pak,attr = self.parse_pakattr(pakattr)
            ks = np.arange(self.m.nlay)
            if isinstance(attr,flopy.utils.Transient2d):
                ks = np.arange(self.m.nper)
            try:
                k_parse = self.parse_k(k_org,ks)
            except Exception as e:
                self.logger.lraise("error parsing k {0}:{1}".format(k_org,str(e)))
            for k in k_parse:
                dfs.append(self.grid_helper(k,pak,attr))
        self.log("processing zone_prop_dict")
        self.par_dfs["zone"] = pd.concat(dfs)

    def zone_helper(self,k,pak,attr):

        if isinstance(attr,flopy.utils.Util2d):
            return self.zone_util2d_helper(attr,k)
        elif isinstance(attr,flopy.utils.Util3d):
            return self.zone_util2d_helper(attr.util_2ds[k],k)
        elif isinstance(attr,flopy.utils.Transient2d):
            self.logger.warn("only setting up one set of zone pars for all "+\
                  "stress periods for pakattr:{0}".format(attr.name_base))
            kper = list(attr.transient_2ds.keys())[0]
            return self.zone_util2d_helper(attr.transient_2ds[kper],k)

        elif isinstance(attr,flopy.utils.MfList):
            self.logger.lraise('MfList support not implemented for zone pars')
        else:
            self.logger.lraise("unrecognized pak,attr:{0},{1}".
                               format(str(pak),str(attr)))

    def zone_util2d_helper(self,u2d,k):
        name = u2d.name.split('_')[0]+"{0:02d}".format(k)
        filename = u2d.filename
        if filename is None:
            self.logger.lraise("filename is None for {0}".format(u2d.name))
        filename = os.path.join(self.m.model_ws,self.arr_org,filename)
        self.logger.statement("resetting 'how' to openclose for {0}".format(name))
        self.logger.statement("{0} being written to array file {1}".format(name,filename))
        u2d.how = "openclose"
        # write original array into arr_org
        self.logger.statement("saving array:{0}".format(filename))
        np.savetxt(filename,u2d.array)
        # write the template file
        tpl_file = os.path.join(self.m.model_ws,os.path.split(filename)[-1]+".tpl")
        ib = self.m.bas6.ibound[k].array
        parnme = []
        with open(tpl_file,'w') as f:
            f.write("ptf ~\n")
            for i in range(self.m.nrow):
                for j in range(self.m.ncol):
                    if ib[i,j] < 1:
                        pname = " 1.0  "
                    else:
                        pname = "{0}_zn{1}".format(name,ib[i,j])
                        if len(pname) > 12:
                            self.logger.lraise("zone pname too long:{0}".\
                                               format(pname))
                        parnme.append(pname)
                        pname = " ~   {0}    ~".format(pname)
                    f.write(pname)
                f.write("\n")
        line = "try:\n    os.remove('{0}')\nexcept:\n    pass".\
            format(os.path.join(self.m.external_path,os.path.split(filename)[-1]))
        self.log("forward_run line:{0}".format(line))
        self.frun_pre_lines.append(line)
        line = "arr_org = np.loadtxt('{0}')".\
            format(os.path.join(self.arr_org,os.path.split(filename)[-1]))
        self.log("forward_run line:{0}".format(line))
        self.frun_pre_lines.append(line)
        line = "arr_mlt = np.loadtxt('{0}')".\
            format(os.path.join(self.arr_mlt,os.path.split(filename)[-1]))
        self.log("forward_run line:{0}".format(line))
        self.frun_pre_lines.append(line)

        line = "np.savetxt('{0}',arr_mlt * arr_org,fmt='%15.6E')\n".\
            format(os.path.join(self.m.external_path,os.path.split(filename)[-1]))
        self.log("forward_run line:{0}".format(line))
        self.frun_pre_lines.append(line)

        # track these tpl-in pairs because they don't follow the standard spec
        self.tpl_files.append(os.path.split(tpl_file)[-1])
        self.in_files.append(os.path.join(self.arr_mlt,os.path.split(filename)[-1]))
        self.k_prop_list.append((k,filename))
        df = pd.DataFrame({"parnme":parnme},index=parnme)
        df.loc[:,"pargp"] = "zn_{0}".format(name)
        return df

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
        pass

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

