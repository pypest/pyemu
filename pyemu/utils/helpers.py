from __future__ import print_function, division
import os
import multiprocessing as mp
import subprocess as sp
import time
import struct
import socket
import shutil
import numpy as np
import pandas as pd
pd.options.display.max_colwidth = 100

import pyemu

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
        arr += means.loc[means.prefix==prefix,"new_val"]
        arr[arr<arr_min] = arr_min
        np.savetxt(filename,arr,fmt="%20.8E")







def zero_order_tikhonov(pst, parbounds=True):
        """setup preferred-value regularization
        Parameters:
        ----------
            pst (Pst instance) : the control file instance
            parbounds (bool) : weight the prior information equations according
                to parameter bound width - approx the KL transform
        Returns:
        -------
            None
        """
        pass
        pilbl, obgnme, weight, equation = [], [], [], []
        for idx, row in pst.parameter_data.iterrows():
            if row["partrans"].lower() not in ["tied", "fixed"]:
                pilbl.append(row["parnme"])
                weight.append(1.0)
                ogp_name = "regul"+row["pargp"]
                obgnme.append(ogp_name[:12])
                parnme = row["parnme"]
                parval1 = row["parval1"]
                if row["partrans"].lower() == "log":
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
        ptrans = pst.parameter_data.partrans.to_dict()
        pi_num = pst.prior_information.shape[0] + 1
        pilbl, obgnme, weight, equation = [], [], [], []
        for i,iname in enumerate(cc_mat.row_names):
            for j,jname in enumerate(cc_mat.row_names[i+1:]):
                #print(i,iname,i+j+1,jname)
                cc = cc_mat.x[i,j+i+1]
                if cc < abs_drop_tol:
                    continue
                pilbl.append("pcc_{0}".format(pi_num))
                iiname = str(iname)
                if ptrans[iname] == "log":
                    iiname = "log("+iname+")"
                jjname = str(jname)
                if ptrans[jname] == "log":
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




