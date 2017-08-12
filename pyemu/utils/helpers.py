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
import numpy as np
import pandas as pd
pd.options.display.max_colwidth = 100

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
        new_names = [name for name in parse_tpl_file(tpl_file) if name not in par_names]
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



def pst_from_flopy_model(m,pp_pakattr_list=None,const_pakattr_list=None,bc_pakattr_list=None,
                         pp_space=None,pp_bounds=None,pp_geostruct=None,bc_geostruct=None,
                         remove_existing=False):
    try:
        import flopy
    except:
        raise Exception("from_flopy_model() requires flopy")

    assert m.array_free_format == True
    assert m.free_format_input == True
    assert m.external_path is not None
    f_frun = open(os.path.join(m.model_ws,"forward_run.py"),'w')
    f_frun.write("import os\nimport numpy as np\nimport pandas as pd\nimport flopy\n")
    f_frun.write("import pyemu\n")

    def flopy_pp_helper(pakattr):
        tpl_names = []
        if isinstance(pakattr,flopy.utils.Util2d):
            tpl_names.append(flopy_pp_util2d_helper(pakattr))
        elif isinstance(pakattr,flopy.utils.Util3d):
            for u2d in pakattr.util_2ds:
                tpl_names.append(flopy_pp_util2d_helper(u2d))
        elif isinstance(pakattr,flopy.utils.Transient2d):
            print("WARNING: only setting up one set of pilot points for all "+\
                  "stress periods for pakattr:{0}".format(pakattr.name_base))

            #for u2d in pakattr:
            #    tpl_names.append(flopy_pp_util2d_helper(u2d))
        elif isinstance(pakattr,flopy.utils.MfList):
            raise Exception('MfList support not implemented for pilot points')
        else:
            raise Exception("unrecognized pakattr:{0}".format(str(pakattr)))
        return tpl_names

    def flopy_pp_util2d_helper(u2d):
        name = u2d.name
        filename = os.path.join(m.model_ws,"arr_org",name+".dat")
        print(name,filename,u2d.how)
        assert u2d.how == "openclose","u2d '{0}' must use openclose format".format(name)
        # write original array into arr_org

        np.savetxt(filename,u2d.array)

        #setup pilot points
        # need to find what the external filename that flopy writes
        return "test"



    # def parse_pakattr(pakattr):
    #     if isinstance(pakattr,list) or isinstance(pakattr,tuple):
    #         assert len(pakattr) == 2,"pakattr: '{0}' must be iterable of len 2".\
    #             format(str(pakattr))
    #         pakname = pakattr[0].lower()
    #         attrname = pakattr[1].lower()
    #         pak = m.get_package(pakname)
    #         if pak is None:
    #             raise Exception("pak {0} not found".format(pakname))
    #         if hasattr(pak,attrname):
    #             pakattr = getattr(pak,attrname)
    #         elif hasattr(pak,"stress_period_data"):
    #             dtype = pak.stress_period_data.dtype
    #             if attrname not in dtype.names:
    #                 raise Exception("attr {0} not found in dtype.names for {1}.stress_period_data".\
    #                                 format(attrname,pakname))
    #             pakattr = pak.stress_period_data
    #         else:
    #             raise Exception("unrecognized attr:{1}".format(attrname))
    #     return pakattr
    #

    tpl_names = []
    # handle pilot point parameters
    if pp_pakattr_list is not None:
        if pp_space is None:
            print("WARNING: pp_space is None, using 10...\n")
            pp_space=10
        if pp_geostruct is None:
            print("WARNING: pp_geostruct is None,"\
                  " using ExpVario with contribution=1 and a=(pp_space*3")
            pp_dist = pp_space * float(max(m.dis.delr.array.max(),
                                           m.dis.delc.array.max()))
            v = pyemu.geostats.ExpVario(contribution=1.0,a=pp_dist)
            pp_geostruct = pyemu.geostats.GeoStruct(variograms=v)

        # make arr_org and arr_mult dirs for storage
        for d in [os.path.join(m.model_ws,"arr_org"),
                  os.path.join(m.model_ws,"arr_mlt")]:
            if os.path.exists(d):
                if remove_existing:
                    shutil.rmtree(d)
                else:
                    raise Exception("pilot point dir '{0}' already exists".format(d))
            os.mkdir(d)
        if not isinstance(pp_pakattr_list,list):
            pp_pakattr_list = list(pp_pakattr_list)

        for pakattr in pp_pakattr_list:
            tpl_names.extend(flopy_pp_helper(pakattr))









