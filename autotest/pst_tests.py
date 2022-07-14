import os
import platform
import shutil
import numpy as np
if not os.path.exists("temp"):
    os.mkdir("temp")


def from_io_with_inschek_test():
    import os
    from pyemu import Pst, pst_utils
    # creation functionality
    dir = os.path.join("..", "verification", "10par_xsec", "template_mac")
    pst = Pst(os.path.join(dir, "pest.pst"))

    tpl_files = [os.path.join(dir, f) for f in pst.template_files]
    out_files = [os.path.join(dir, f) for f in pst.output_files]
    ins_files = [os.path.join(dir, f) for f in pst.instruction_files]
    in_files = [os.path.join(dir, f) for f in pst.input_files]

    new_pst = Pst.from_io_files(tpl_files, in_files,
                                ins_files, out_files,
                                pst_filename=os.path.join("temp", "test.pst"))
    print(new_pst.observation_data)
    return


def tpl_ins_test():
    import os
    from pyemu import Pst, pst_utils, helpers
    # creation functionality
    dir = os.path.join("..", "verification", "henry", "misc")
    files = os.listdir(dir)
    tpl_files, ins_files = [], []
    for f in files:
        if f.lower().endswith(".tpl") and "coarse" not in f:
            tpl_files.append(os.path.join(dir, f))
        if f.lower().endswith(".ins"):
            ins_files.append(os.path.join(dir, f))

    out_files = [f.replace(".ins", ".junk") for f in ins_files]
    in_files = [f.replace(".tpl", ".junk") for f in tpl_files]

    helpers.pst_from_io_files(tpl_files, in_files,
                                ins_files, out_files,
                                pst_filename=os.path.join("temp", "test.pst"))
    return

def res_covreg_test():
    import os
    import numpy as np
    from pyemu import Pst, pst_utils
    # test Pst.res with cov mat regularization
    pst_dir = os.path.join("pst")

    p = Pst(os.path.join(pst_dir, "pest_regulcov.pst"))
    assert "regulp" in p.res.group.unique()
    assert 1 - p.phi / ((p.res.residual * p.res.weight)**2).sum() < 1.0e-6
    

def res_test():
    import os
    import numpy as np
    from pyemu import Pst, pst_utils
    # residual functionality testing
    pst_dir = os.path.join("pst")

    p = Pst(os.path.join(pst_dir, "pest.pst"))
    phi_comp = p.phi_components
    assert "regul_p" in phi_comp
    assert "regul_m" in phi_comp

    p.adjust_weights_discrepancy(original_ceiling=False)
    d = np.abs(p.phi - p.nnz_obs)
    assert d < 1.0E-5

    p = Pst(os.path.join(pst_dir, "pest.pst"))
    p.adjust_weights_discrepancy(original_ceiling=False,bygroups=True)

    d = np.abs(p.phi - p.nnz_obs)
    assert d < 1.0E-5
    p.adjust_weights(obsgrp_dict={"head": 50})
    assert np.abs(p.phi_components["head"] - 50) < 1.0e-6

    # get()
    new_p = p.get()
    new_p.prior_information = p.prior_information
    new_file = os.path.join("temp", "new.pst")
    new_p.write(new_file)

    p_load = Pst(new_file, resfile=p.resfile)
    for gname in p.phi_components:
        d = np.abs(p.phi_components[gname] - p_load.phi_components[gname])
        assert d < 1.0e-5


def pst_manip_test():
    import os
    from pyemu import Pst
    pst_dir = os.path.join("pst")
    org_path = os.path.join(pst_dir, "pest.pst")
    new_path = os.path.join("temp", "pest1.pst")
    pst = Pst(org_path)
    pst.control_data.pestmode = "regularisation"
    pst.write(new_path)
    pst = Pst(new_path)
    pst.svd_data.maxsing = 1
    pst.write(new_path)
    pst = Pst(new_path)
    pst.write(new_path,version=2)
    pst = Pst(new_path)


def load_test():
    import os
    from pyemu import Pst, pst_utils
    pst_dir = os.path.join("pst")
    temp_dir = "temp"
    if not os.path.exists(temp_dir):
        os.mkdir(temp_dir)
    # just testing all sorts of different pst files
    pst_files = os.listdir(pst_dir)
    exceptions = []
    load_fails = []
    for pst_file in pst_files:
        if "pest_tied_tester" not in pst_file:
            continue
        if pst_file.endswith(".pst") and not "comments" in pst_file and \
                not "missing" in pst_file:
            print(pst_file)
            try:
                p = Pst(os.path.join(pst_dir, pst_file))
            except Exception as e:
                exceptions.append(pst_file + " read fail: " + str(e))
                load_fails.append(pst_file)
                continue
            out_name = os.path.join(temp_dir, pst_file)
            print(out_name)
            # p.write(out_name)
            try:
                p.write(out_name)
            except Exception as e:
                exceptions.append(pst_file + " write fail: " + str(e))
                continue
            print(pst_file)
            try:
                p = Pst(out_name)
            except Exception as e:
                exceptions.append(pst_file + " reload fail: " + str(e))
                continue
            print(out_name)
            # p.write(out_name)
            try:
                p.write(out_name,version=2)
            except Exception as e:
                exceptions.append(pst_file + " v2 write fail: " + str(e))
                continue

            p = Pst(out_name)
            try:
                p = Pst(out_name)
            except Exception as e:
                exceptions.append(pst_file + " v2 reload fail: " + str(e))
                continue


    # with open("load_fails.txt",'w') as f:
    #    [f.write(pst_file+'\n') for pst_file in load_fails]
    if len(exceptions) > 0:
        raise Exception('\n'.join(exceptions))

def comments_test():
    import os
    import pyemu

    pst = pyemu.Pst(os.path.join("pst", "comments.pst"))
    pst.with_comments = True
    pst.write(os.path.join("temp", "comments.pst"),version=1)
    pst1 = pyemu.Pst(os.path.join("temp", "comments.pst"))
    assert pst1.parameter_data.extra.dropna().shape[0] == pst.parameter_data.extra.dropna().shape[0]
    pst1.with_comments = False
    pst1.write(os.path.join("temp", "comments.pst"),version=1)
    pst2 = pyemu.Pst(os.path.join("temp", "comments.pst"))
    assert pst2.parameter_data.dropna().shape[0] == 0




def tied_test():
    import os
    import pyemu
    pst_dir = os.path.join("pst")
    pst = pyemu.Pst(os.path.join(pst_dir, "br_opt_no_zero_weighted.pst"))
    print(pst.tied)
    pst.write(os.path.join("temp", "pest_tied_tester_1.pst"))

    par = pst.parameter_data
    par.loc[pst.par_names[::3], "partrans"] = "tied"
    par.loc[:,"partied"] = "none"
    #pst.write(os.path.join("temp", "pest_tied_tester_1.pst"))
    try:
        pst.write(os.path.join("temp", "pest_tied_tester_1.pst"))
    except:
        pass
    else:
        raise Exception()
    par.loc[pst.par_names[::3], "partied"] = pst.par_names[0]

    pst = pyemu.Pst(os.path.join("pst", "pest.pst"))
    print(pst.tied)
    par = pst.parameter_data
    par.loc[pst.par_names[2], "partrans"] = "tied"
    print(pst.tied)
    par.loc[pst.par_names[2], "partied"] = "junk1"
    try:
        pst.write(os.path.join("temp", "test.pst"))
    except:
        pass
    else:
        raise Exception()
    #pst = pyemu.Pst(os.path.join("temp", "test.pst"))

    pst = pyemu.Pst(os.path.join("pst", "pest.pst"))
    print(pst.tied)
    par = pst.parameter_data
    par.loc[pst.par_names[2], "partrans"] = "tied"
    par.loc[pst.par_names[2], "partied"] = "junk"
    try:

        pst.write(os.path.join("temp", "test.pst"))
    except:
        pass
    else:
        raise Exception()
    #pst = pyemu.Pst(os.path.join("temp", "test.pst"))




def derivative_increment_tests():
    import os
    import pyemu

    pst = pyemu.Pst(os.path.join("pst", "inctest.pst"))
    pst.calculate_pertubations()


def pestpp_args_test():
    import os
    import pyemu
    pst_dir = os.path.join("pst")
    pst = pyemu.Pst(os.path.join(pst_dir, "br_opt_no_zero_weighted.pst"))
    pst.pestpp_options["lambdas"] = "0.1,0.2,0.3"
    pst.write(os.path.join("temp", "temp.pst"))
    pst = pyemu.Pst(os.path.join("temp", "temp.pst"))
    print(pst.pestpp_options)


def reweight_test():
    import os
    import numpy as np
    from pyemu import Pst, pst_utils
    pst_dir = os.path.join("pst")
    p = Pst(os.path.join(pst_dir, "pest.pst"))
    obsgrp_dict = {"pred": 1.0, "head": 1.0, "conc": 1.0}
    p.adjust_weights(obsgrp_dict=obsgrp_dict)
    assert np.abs(p.phi - 3.0) < 1.0e-5, p.phi

    obs = p.observation_data
    obs.loc[obs.obgnme == "pred", "weight"] = 0.0
    assert np.abs(p.phi - 2.0) < 1.0e-5, p.phi

    obs_dict = {"pd_one": 1.0, "pd_ten": 1.0}
    p.adjust_weights(obs_dict=obs_dict)
    assert np.abs(p.phi - 4.0) < 1.0e-5, p.phi


def reweight_res_test():
    import os
    import numpy as np
    import pyemu
    pst = pyemu.Pst(os.path.join("pst", "pest.pst"))
    print(pst.res.loc[pst.nnz_obs_names, :])
    print(pst.phi, pst.nnz_obs)
    pst.adjust_weights_discrepancy(bygroups=True)
    print(pst.phi, pst.nnz_obs)
    assert np.abs(pst.phi - pst.nnz_obs) < 1.0e-6


def regul_rectify_test():
    import os
    import pyemu
    pst = pyemu.Pst(os.path.join("pst", "inctest.pst"))
    pyemu.helpers.zero_order_tikhonov(pst)
    assert pst.prior_information.shape[0] == pst.npar_adj
    pst._update_control_section()
    assert pst.control_data.nprior == pst.prior_information.shape[0]
    fix_names = pst.adj_par_names[::2]
    pst.parameter_data.loc[fix_names, "partrans"] = "fixed"
    pst.rectify_pi()
    assert pst.prior_information.shape[0] == pst.npar_adj
    pst._update_control_section()
    assert pst.control_data.nprior == pst.prior_information.shape[0]


def nnz_groups_test():
    import os
    import pyemu
    pst_dir = os.path.join("pst")
    pst = pyemu.Pst(os.path.join(pst_dir, "br_opt_no_zero_weighted.pst"))
    org_og = pst.obs_groups
    org_nnz_og = pst.nnz_obs_groups
    obs = pst.observation_data
    obs.loc[obs.obgnme == org_og[0], "weight"] = 0.0
    new_og = pst.obs_groups
    new_nnz_og = pst.nnz_obs_groups
    assert org_og[0] not in new_nnz_og


def adj_group_test():
    import os
    import pyemu
    pst_dir = os.path.join("pst")
    pst = pyemu.Pst(os.path.join(pst_dir, "pest.pst"))
    par = pst.parameter_data
    par.loc[par.pargp.apply(lambda x: x in pst.par_groups[1:]),"partrans"] = "fixed"
    assert pst.adj_par_groups == [pst.par_groups[0]]

def regdata_test():
    import os
    import pyemu

    pst = pyemu.Pst(os.path.join("pst", "pest.pst"))
    phimlim = 10.0
    pst.reg_data.phimlim = phimlim
    pst.control_data.pestmode = "regularization"
    pst.write(os.path.join("temp", "pest_regultest.pst"))
    pst_new = pyemu.Pst(os.path.join("temp", "pest_regultest.pst"))
    assert pst_new.reg_data.phimlim == phimlim









def add_pi_test():
    import os
    import pyemu
    pst = pyemu.Pst(os.path.join("pst", "pest.pst"))
    pst.prior_information = pst.null_prior
    par_names = pst.parameter_data.parnme[:10]
    pst.add_pi_equation(par_names, coef_dict={par_names[1]: -1.0})
    pst.write(os.path.join("temp", "test.pst"))

    pst = pyemu.Pst(os.path.join("temp", "test.pst"))
    print(pst.prior_information)


def setattr_test():
    import os
    import pyemu
    pst = pyemu.Pst(os.path.join("pst", "pest.pst"))
    pst.model_command = 'test'
    assert isinstance(pst.model_command, list)
    pst.model_command = ["test", "test1"]
    assert isinstance(pst.model_command, list)
    pst.write(os.path.join("temp", "test.pst"))
    pst = pyemu.Pst(os.path.join("temp", "test.pst"))
    assert isinstance(pst.model_command, list)


def add_pars_test():
    import os
    import pyemu
    pst = pyemu.Pst(os.path.join("pst", "pest.pst"))
    npar = pst.npar
    tpl_file = os.path.join("temp", "crap.in.tpl")
    with open(tpl_file, 'w') as f:
        f.write("ptf ~\n")
        f.write("  ~junk1   ~\n")
        f.write("  ~ {0}  ~\n".format(pst.parameter_data.parnme[0]))
    print(pst.npar)
    pst.add_parameters(tpl_file, "crap.in", pst_path="temp")
    assert npar + 1 == pst.npar
    assert "junk1" in pst.parameter_data.parnme
    assert os.path.join("temp", "crap.in") in pst.input_files
    assert os.path.join("temp", "crap.in.tpl") in pst.template_files

    pyemu.helpers.zero_order_tikhonov(pst)
    nprior,npar = pst.nprior,pst.npar
    pst.parameter_data.loc[pst.par_names[0], "partrans"] = "tied"
    pst.parameter_data.loc[pst.par_names[0], "partied"] = "junk1"
    try:
        pst.drop_parameters(tpl_file,"temp")
    except:
        pass
    else:
        raise Exception("should have failed")

    pst.parameter_data.loc[pst.par_names[0], "partrans"] = "log"
    pst.drop_parameters(tpl_file, "temp")
    print(pst.npar,npar)
    print(pst.nprior,nprior)
    print(pst.par_names)
    assert pst.npar == npar - 2
    assert pst.nprior == nprior - 2



def add_obs_test():
    import os
    import pyemu
    pst = pyemu.Pst(os.path.join("pst", "pest.pst"))
    nobs = pst.nobs
    ins_file = os.path.join("temp", "crap.out.ins")
    out_file = os.path.join("temp", "crap.out")
    oval = 1234.56
    with open(ins_file, 'w') as f:
        f.write("pif ~\n")
        # f.write("  ~junk1   ~\n")
        # f.write("  ~ {0}  ~\n".format(pst.parameter_data.parnme[0]))
        f.write("l1 w  !{0}!\n".format("crap1"))
    with open(out_file, "w") as f:
        f.write("junk1  {0:8.2f} \n".format(oval))
    pst.add_observations(ins_file, out_file)
    assert nobs + 1 == pst.nobs
    assert "crap1" in pst.observation_data.obsnme
    assert os.path.join("temp", "crap.out") in pst.output_files, str(pst.output_files)
    assert os.path.join("temp", "crap.out.ins") in pst.instruction_files
    print(pst.observation_data.loc["crap1", "obsval"], oval)
    nobs = pst.nobs
    pst.drop_observations(ins_file,pst_path="temp")
    assert pst.nobs == nobs - 1
    assert ins_file not in pst.model_output_data.pest_file.to_list()


def test_write_input_files():
    import os
    import shutil
    import numpy as np
    import pyemu
    from pyemu import Pst, pst_utils
    # creation functionality
    dir = os.path.join("..", "verification", "10par_xsec", "template_mac")
    if os.path.exists("temp_dir"):
        shutil.rmtree("temp_dir")
    shutil.copytree(dir, "temp_dir")
    os.chdir("temp_dir")
    pst = Pst(os.path.join("pest.pst"))
    pst.write_input_files()
    arr1 = np.loadtxt(pst.input_files[0])
    print(pst.parameter_data.parval1)
    pst.parameter_data.loc[:, "parval1"] *= 10.0
    pst.write_input_files()
    arr2 = np.loadtxt(pst.input_files[0])
    assert (arr1 * 10).sum() == arr2.sum()
    os.chdir("..")


def res_stats_test():
    import os
    import pyemu

    import os
    import numpy as np
    from pyemu import Pst, pst_utils
    # residual functionality testing
    pst_dir = os.path.join("pst")

    p = pyemu.pst_utils.generic_pst(["p1"], ["o1"])
    try:
        p.get_res_stats()
    except:
        pass
    else:
        raise Exception()

    p = Pst(os.path.join(pst_dir, "pest.pst"))
    phi_comp = p.phi_components
    # print(phi_comp)
    df = p.get_res_stats()
    assert np.abs(df.loc["rss", "all"] - p.phi) < 1.0e-6, "{0},{1}".format(df.loc["rss", "all"], p.phi)
    for pc in phi_comp.keys():
        assert phi_comp[pc] == p.phi_components[pc]


def write_tables_test():
    import os
    import pyemu

    pst = pyemu.Pst(os.path.join("pst", "freyberg_gr.pst"))
    group_names = {"w0": "wells t"}
    pst.write_par_summary_table(group_names=group_names)
    pst.write_obs_summary_table(group_names={"calhead": "calibration heads"})
    pst.write_par_summary_table(filename='testpar.xlsx', group_names=group_names)
    pst.write_par_summary_table(filename='testpar2.xlsx', group_names=group_names, report_in_linear_space=True)   
    pst.write_obs_summary_table(filename = 'testobs.xlsx', group_names={"calhead": "calibration heads"})
    

def test_e_clean():
    import os
    import pyemu

    pst_name = os.path.join("pst", "test_missing_e.pst")
    try:
        pst = pyemu.Pst(pst_name)
    except:
        pass
    else:
        raise Exception()

    clean_name = os.path.join("temp", "clean.pst")
    pyemu.pst_utils.clean_missing_exponent(pst_name, clean_name)
    pst = pyemu.Pst(clean_name)


def run_test():
    import os
    import pyemu

    pst = pyemu.Pst(os.path.join("pst", "pest.pst"))
    # pst.run("pestchek")
    pst.write(os.path.join("temp", "test.pst"))


    try:
        pst.run("pestchek")
    except:
        print("error calling pestchek")


def rectify_pgroup_test():
    import os
    import pyemu
    pst = pyemu.Pst(os.path.join("pst", "pest.pst"))
    npar = pst.npar
    tpl_file = os.path.join("temp", "crap.in.tpl")
    with open(tpl_file, 'w') as f:
        f.write("ptf ~\n")
        f.write("  ~junk1   ~\n")
        f.write("  ~ {0}  ~\n".format(pst.parameter_data.parnme[0]))
    # print(pst.parameter_groups)

    pst.add_parameters(tpl_file, "crap.in", pst_path="temp")

    # print(pst.parameter_groups)
    pst.rectify_pgroups()
    # print(pst.parameter_groups)

    pst.parameter_groups.loc["pargp", "inctyp"] = "absolute"
    print(pst.parameter_groups)
    pst.write(os.path.join('temp', "test.pst"))
    print(pst.parameter_groups)


def try_process_ins_test():
    import os
    import numpy as np
    import pandas as pd
    import pyemu

    ins_file = os.path.join("ins","primary.dat.ins")
    i = pyemu.pst_utils.InstructionFile(ins_file)
    df2 = i.read_output_file(ins_file.replace(".ins", ""))
    print(df2)

    ins_file = os.path.join("utils", "BH.mt3d.processed.ins")
    i = pyemu.pst_utils.InstructionFile(ins_file)
    df2 = i.read_output_file(ins_file.replace(".ins",""))
    df2.loc[df2.obsval>1.0e+10,"obsval"] = np.NaN


    # df1 = pyemu.pst_utils._try_run_inschek(ins_file,ins_file.replace(".ins",""))
    df1 = pd.read_csv(ins_file.replace(".ins", ".obf"), delim_whitespace=True, names=["obsnme", "obsval"], index_col=0)
    df1.loc[df1.obsval > 1.0e+10, "obsval"] = np.NaN
    print(df1.max())
    print(df2.max())
    # df1.index = df1.obsnme
    df1.loc[:, "obsnme"] = df1.index
    df1.index = df1.obsnme
    # df1 = df1.loc[df.obsnme,:]
    diff = df2.obsval - df1.obsval
    print(diff.max(), diff.min())
    print(diff.sum())
    assert diff.sum() < 1.0e+10




def sanity_check_test():
    import os
    import pyemu
    pst = pyemu.Pst(os.path.join("pst", "pest.pst"))
    pst.parameter_data.loc[:, "parnme"] = "crap"

    try:
        pst.write(os.path.join("temp", "test.pst"))
    except:
        pass
    else:
        raise Exception("should have failed")

    pst = pyemu.Pst(os.path.join("pst", "pest.pst"))
    pst.observation_data.loc[:, "obsnme"] = "crap"
    try:
        pst.write(os.path.join("temp", "test.pst"))
    except:
        pass
    else:
        raise Exception("should have failed")

    pst = pyemu.Pst(os.path.join("pst", "pest.pst"))
    pst.parameter_data.loc[:, "partrans"] = "tied"
    pst.parameter_data.loc[:,"partied"] = pst.parameter_data.loc[:,"parnme"]
    try:
        pst.write(os.path.join("temp", "test.pst"))
    except:
        pass
    else:
        raise Exception("should have failed")

    pst = pyemu.Pst(os.path.join("pst", "pest.pst"))
    pst.parameter_data.loc[:, "partrans"] = "tied"
    pst.parameter_data.loc[:, "partied"] = pst.par_names[0]
    pst.parameter_data.loc[pst.par_names[0], "partrans"] = "fixed"
    try:
        pst.write(os.path.join("temp", "test.pst"))
    except:
        pass
    else:
        raise Exception("should have failed")



def csv_to_ins_test():
    import os
    import numpy as np
    import pandas as pd
    import pyemu

    cnames = ["col{0}".format(i) for i in range(10)]
    rnames = ["row{0}".format(i) for i in range(10)]
    df = pd.DataFrame(index=rnames,columns=cnames)
    df.loc[:,:] = np.random.random(df.shape)
    df.to_csv(os.path.join("temp", "temp.csv"))
    names = pyemu.pst_utils.csv_to_ins_file(df, ins_filename=os.path.join("temp", "temp.csv.ins"),
                                            only_cols=cnames[0],prefix="test")
    obnames = pyemu.pst_utils.parse_ins_file(os.path.join("temp", "temp.csv.ins"))
    assert len(names) == df.shape[0] == len(obnames), names
    for name in names.obsnme:
        assert name.startswith("test"),name

    names = pyemu.pst_utils.csv_to_ins_file(df, ins_filename=os.path.join("temp", "temp.csv.ins"),
                                            only_cols=cnames[0:2])
    assert len(names) == df.shape[0]*2, names

    names = pyemu.pst_utils.csv_to_ins_file(df, ins_filename=os.path.join("temp", "temp.csv.ins"),
                                            only_rows=rnames[0])
    assert len(names) == df.shape[1], names

    names = pyemu.pst_utils.csv_to_ins_file(df, ins_filename=os.path.join("temp", "temp.csv.ins"),
                                            only_rows=rnames[0:2])
    assert len(names) == df.shape[1] * 2, names

    names = pyemu.pst_utils.csv_to_ins_file(df,ins_filename=os.path.join("temp","temp.csv.ins"))
    assert len(names) == df.shape[0] * df.shape[1]

    df.columns = ["col" for i in range(df.shape[1])]
    names = pyemu.pst_utils.csv_to_ins_file(df, ins_filename=os.path.join("temp", "temp.csv.ins"))
    assert len(names) == df.shape[0] * df.shape[1]

    names = pyemu.pst_utils.csv_to_ins_file(df, ins_filename=os.path.join("temp", "temp.csv.ins"),
                                            only_cols="col")
    assert len(names) == df.shape[0] * df.shape[1]

    df.index = ["row" for i in range(df.shape[0])]
    names = pyemu.pst_utils.csv_to_ins_file(df, ins_filename=os.path.join("temp", "temp.csv.ins"))
    assert len(names) == df.shape[0] * df.shape[1]

    names = pyemu.pst_utils.csv_to_ins_file(df, ins_filename=os.path.join("temp", "temp.csv.ins"),
                                            only_cols="col",only_rows="row")
    assert len(names) == df.shape[0] * df.shape[1]



def lt_gt_constraint_names_test():
    import os
    import pyemu
    import os
    import pyemu
    pst = pyemu.Pst(os.path.join("pst", "pest.pst"))
    obs = pst.observation_data
    obs.loc[:,"weight"] = 1.0
    pst.observation_data.loc[pst.obs_names[:4],"obgnme"] = "lessjunk"
    pst.observation_data.loc[pst.obs_names[4:8], "obgnme"] = "l_junk"
    pst.observation_data.loc[pst.obs_names[8:12], "obgnme"] = "greaterjunk"
    pst.observation_data.loc[pst.obs_names[12:16], "obgnme"] = "g_junk"
    assert pst.less_than_obs_constraints.shape[0] == 8
    assert pst.greater_than_obs_constraints.shape[0] == 8

    obs.loc[:, "weight"] = 0.0
    assert pst.less_than_obs_constraints.shape[0] == 0
    assert pst.greater_than_obs_constraints.shape[0] == 0

    pi = pst.prior_information
    pi.loc[pst.prior_names[:4],"obgnme"] = "lessjunk"
    pi.loc[pst.prior_names[4:8], "obgnme"] = "l_junk"
    pi.loc[pst.prior_names[8:12], "obgnme"] = "greaterjunk"
    pi.loc[pst.prior_names[12:16], "obgnme"] = "g_junk"
    assert pst.less_than_pi_constraints.shape[0] == 8
    assert pst.greater_than_pi_constraints.shape[0] == 8

    pi.loc[:, "weight"] = 0.0
    assert pst.less_than_pi_constraints.shape[0] == 0
    assert pst.greater_than_pi_constraints.shape[0] == 0


def new_format_test_2():
    import pyemu
    pst_dir = "newpst"
    pst_files = [f for f in os.listdir(pst_dir) if f.endswith(".pst")]
    b_d = os.getcwd()
    os.chdir(pst_dir)
    #try:
    for pst_file in pst_files:
        print(pst_file)
        if os.path.exists("temp_pst"):
            shutil.rmtree("temp_pst")
        os.makedirs("temp_pst")
        if "fail" in pst_file:
            try:
                pst = pyemu.Pst(os.path.join(pst_file))

            except:
                pass
            else:
                raise Exception("should have failed on {0}".format(pst_file))
        else:
            pst = pyemu.Pst(os.path.join(pst_file))
            pst.write(os.path.join("temp_pst","pst_test.pst"))
            new_pst = pyemu.Pst(os.path.join("temp_pst","pst_test.pst"))
            d = set(pst.par_names).symmetric_difference(new_pst.par_names)
            assert len(d) == 0,d
            d = set(pst.obs_names).symmetric_difference(new_pst.obs_names)
            assert len(d) == 0,d
            d = set(pst.template_files).symmetric_difference(new_pst.template_files)
            assert len(d) == 0, d
            assert pst.nnz_obs == new_pst.nnz_obs
            assert pst.npar_adj == new_pst.npar_adj

            new_pst.write(os.path.join("temp_pst","pst_test.pst"),version=2)

            assert os.path.exists(os.path.join("temp_pst","pst_test.par_data.csv"))
            assert os.path.exists(os.path.join("temp_pst", "pst_test.obs_data.csv"))
            assert os.path.exists(os.path.join("temp_pst", "pst_test.tplfile_data.csv"))
            assert os.path.exists(os.path.join("temp_pst", "pst_test.insfile_data.csv"))

            new_pst = pyemu.Pst(os.path.join("temp_pst","pst_test.pst"))
            d = set(pst.par_names).symmetric_difference(new_pst.par_names)
            assert len(d) == 0, d
            d = set(pst.obs_names).symmetric_difference(new_pst.obs_names)
            assert len(d) == 0, d
            d = set(pst.template_files).symmetric_difference(new_pst.template_files)
            assert len(d) == 0, d
            assert pst.nnz_obs == new_pst.nnz_obs
            print(pst.npar_adj,new_pst.npar_adj)
            assert pst.npar_adj == new_pst.npar_adj
    os.chdir(b_d)

def new_format_test():
    import numpy as np
    import pyemu
    pst_files = [f for f in os.listdir("pst") if f.endswith(".pst")]
    for pst_file in pst_files:
        try:
            pst = pyemu.Pst(os.path.join("pst", pst_file))
        except:
            print("error loading",pst_file)
            continue
        print(pst_file)
        npar,nobs,npr = pst.npar,pst.nobs,pst.nprior
        ppo = pst.pestpp_options
        pst.write("test.pst",version=2)



        pst_new = pyemu.Pst("test.pst")
        npar1, nobs1, npr1 = pst_new.npar, pst_new.nobs, pst_new.nprior
        ppo1 = pst_new.pestpp_options

        assert len(ppo) == len(ppo1),"{0},{1}".format(ppo,ppo1)
        assert npar == npar1
        assert nobs == nobs1
        assert npr == npr1,"{0}: {1},{2}".format(pst_file,npr,npr1)
        assert len(pst.template_files) == len(pst_new.template_files)
        assert len(pst.input_files) == len(pst_new.input_files)
        assert len(pst.instruction_files) == len(pst_new.instruction_files)
        assert len(pst.output_files) == len(pst_new.output_files)

        pst_new.write("test.pst",version=1)
        pst_new = pyemu.Pst("test.pst")
        npar1, nobs1, npr1 = pst_new.npar, pst_new.nobs, pst_new.nprior
        ppo1 = pst_new.pestpp_options
        assert len(ppo) == len(ppo1)
        assert npar == npar1
        assert nobs == nobs1
        assert npr == npr1, "{0}: {1},{2}".format(pst_file, npr, npr1)
        assert len(pst.template_files) == len(pst_new.template_files)
        assert len(pst.input_files) == len(pst_new.input_files)
        assert len(pst.instruction_files) == len(pst_new.instruction_files)
        assert len(pst.output_files) == len(pst_new.output_files)
        pst_new.write("test.pst",version=2)
        pst_new = pyemu.Pst("test.pst")
        npar1, nobs1, npr1 = pst_new.npar, pst_new.nobs, pst_new.nprior
        ppo1 = pst_new.pestpp_options
        assert len(ppo) == len(ppo1)
        assert npar == npar1
        assert nobs == nobs1
        assert npr == npr1, "{0}: {1},{2}".format(pst_file, npr, npr1)
        assert len(pst.template_files) == len(pst_new.template_files)
        assert len(pst.input_files) == len(pst_new.input_files)
        assert len(pst.instruction_files) == len(pst_new.instruction_files)
        assert len(pst.output_files) == len(pst_new.output_files)


    pst_new.parameter_groups.loc[:,:] = np.NaN
    pst_new.parameter_groups.dropna(inplace=True)
    pst_new.write("test.pst",version=2)
    pst_new = pyemu.Pst("test.pst")


    pst_new.parameter_data.loc[:,"counter"] = 1
    pst_new.observation_data.loc[:,"x"] = 999.0
    pst_new.observation_data.loc[:,'y'] = 888.0
    pst_new.write("test.pst",version=2)
    pst_new = pyemu.Pst("test.pst")
    assert "counter" in pst_new.parameter_data.columns
    assert "x" in pst_new.observation_data.columns
    assert "y" in pst_new.observation_data.columns

    # lines = open("test.pst").readlines()
    # for i,line in enumerate(lines):
    #     lines[i] = line.replace("header=True","header=False")
    # with open("test.pst",'w') as f:
    #     [f.write(line) for line in lines]
    # try:
    #     pst_new = pyemu.Pst("test.pst")
    # except:
    #     pass
    # else:
    #     raise Exception()


def change_limit_test():
    import numpy as np
    import pyemu
    pst = pyemu.Pst(os.path.join("pst","pest.pst"))
    #print(pst.parameter_data)
    cols = ["parval1", "rel_upper", "rel_lower", "fac_upper", "fac_lower","chg_upper","chg_lower"]
    pst.control_data.relparmax = 3
    pst.control_data.facparmax = 3
    par = pst.parameter_data

    par.loc[:,"parval1"] = 1.0
    df = pst.get_par_change_limits()
    assert df.rel_upper.mean() == 4.0
    assert df.rel_lower.mean() == -2.0
    assert df.fac_upper.mean() == 3.0
    assert np.abs(df.fac_lower.mean() -  0.33333) < 1.0e-3

    pst.control_data.facorig = 2.0
    par.loc[:,"partrans"] = "none"
    df = pst.get_par_change_limits()
    assert df.rel_upper.mean() == 8.0
    assert df.rel_lower.mean() == -4.0
    assert df.fac_upper.mean() == 6.0
    assert np.abs(df.fac_lower.mean() - 0.66666) < 1.0e-3

    pst.control_data.facorig = 0.001
    par.loc[:, "partrans"] = "none"
    par.loc[:, "parval1"] = -1.0
    df = pst.get_par_change_limits()
    #print(df.loc[:, cols])
    assert df.rel_upper.mean() == 2.0
    assert df.rel_lower.mean() == -4.0
    assert df.fac_lower.mean() == -3.0
    assert np.abs(df.fac_upper.mean() + 0.33333) < 1.0e-3

    print(df.loc[:,["eff_upper","eff_lower"]])
    print(df.loc[:,cols])


def process_output_files_test():

    import os
    import numpy as np
    from pyemu import Pst, pst_utils

    # ins_file = os.path.join("utils","hauraki_transient.mt3d.processed.ins")
    # out_file = ins_file.replace(".ins","")
    #
    # i = pst_utils.InstructionFile(ins_file)
    # print(i.read_output_file(out_file))
    # return

    ins_dir = "ins"
    ins_files = [os.path.join(ins_dir,f) for f in os.listdir(ins_dir) if f.endswith(".ins")]
    ins_files.sort()
    out_files = [f.replace(".ins","") for f in ins_files]
    print(ins_files)

    i4 = pst_utils.InstructionFile(ins_files[4])
    s4 = i4.read_output_file(out_files[4])
    print(s4)
    assert s4.loc["h01_03", "obsval"] == 3.481,s4.loc["h01_03", "obsval"]
    assert s4.loc["h02_10", "obsval"] == 11.1,s4.loc["h02_10", "obsval"]

    i4 = pst_utils.InstructionFile(ins_files[3])
    s4 = i4.read_output_file(out_files[3])
    print(s4)
    assert s4.loc["h01_02", "obsval"] == 1.024
    assert s4.loc["h01_10", "obsval"] == 4.498

    i5 = pst_utils.InstructionFile(ins_files[5])
    s5 = i5.read_output_file(out_files[5])
    print(s5)
    assert s5.loc["obs3_1","obsval"] == 1962323.838381853
    assert s5.loc["obs3_2","obsval"] == 1012443.579448909



    i3 = pst_utils.InstructionFile(ins_files[2])
    s3 = i3.read_output_file(out_files[2])
    #print(s3)
    assert s3.loc["test","obsval"] == 1.23456
    assert s3.loc["h01_02","obsval"] == 1.024

    i1 = pst_utils.InstructionFile(ins_files[0])
    s1 = i1.read_output_file(out_files[0])
    a1 = np.loadtxt(out_files[0]).flatten()
    assert np.abs(s1.obsval.values - a1).sum() == 0.0

    i2 = pst_utils.InstructionFile(ins_files[1])
    s2 = i2.read_output_file(out_files[1])
    assert s2.loc["h01_02","obsval"] == 1.024

def new_format_path_mechanics_test():
    import pyemu

    l_path = "d1/d2/d3/d4/test.pst"
    w_path = "d1\\d2\\d3\\d4\\test.pst"
    true_path = os.path.join("d1","d2","d3","d4")
    test_path = pyemu.Pst._parse_path_agnostic(w_path)
    assert test_path[0] == true_path
    assert test_path[1] == "test.pst"
    test_path = pyemu.Pst._parse_path_agnostic(l_path)
    assert test_path[0] == true_path
    assert test_path[1] == "test.pst"

    l_eline = "d1/d2/d3/d4/Test.dat sep W"
    filename, options = pyemu.Pst._parse_external_line(l_eline,pst_path=".")
    assert filename == "Test.dat"
    assert "sep" in options,options
    assert options["sep"] == "w",options
    l_eline = "d1/d2/d3/d4/Test.dat sep W"
    filename, options = pyemu.Pst._parse_external_line(l_eline, pst_path="template")
    assert filename == os.path.join("template","Test.dat")
    assert "sep" in options, options
    assert options["sep"] == "w", options

    w_eline = "d1\\d2\\d3\\d4\\Test.dat sep W"
    filename, options = pyemu.Pst._parse_external_line(w_eline, pst_path=".")
    assert filename == "Test.dat"
    assert "sep" in options, options
    assert options["sep"] == "w", options

    w_eline = "d1\\d2\\d3\\d4\\Test.dat sep W"
    filename, options = pyemu.Pst._parse_external_line(w_eline, pst_path="template")
    assert filename == os.path.join("template","Test.dat")
    assert "sep" in options, options
    assert options["sep"] == "w", options


def ctrl_data_test():
    import os
    import numpy as np
    import pyemu
    pst = pyemu.Pst(os.path.join("pst","sm.pst"))
    pst.write(os.path.join("pst","test.pst"))

    pst2 = pyemu.Pst(os.path.join("pst", "test.pst"))

    #print(pst.control_data._df.passed)
    #print(pst2.control_data._df.passed)
    for i in pst.control_data._df.index:
        if pst.control_data._df.loc[i,"passed"] != pst2.control_data._df.loc[i,"passed"]:
            print(i)
    assert np.all(pst.control_data._df.passed == pst2.control_data._df.passed)

    pst2.write(os.path.join("pst","test2.pst"),version=2)
    pst3 = pyemu.Pst(os.path.join("pst", "test2.pst"))

def read_in_tpl_test():
    import pyemu
    tpl_d = "tpl"
    df = pyemu.pst_utils.try_read_input_file_with_tpl(os.path.join(tpl_d,"test1.dat.tpl"))
    print(df)
    assert df.parval1["p1"] == df.parval1["p2"]
    assert df.parval1["p3"] == df.parval1["p4"]
    assert df.parval1["p5"] == df.parval1["p6"]
    assert df.parval1["p5"] == df.parval1["p7"]

def read_in_tpl_test2():
    import pyemu
    tpl_d = "tpl"
    df = pyemu.pst_utils.try_read_input_file_with_tpl(os.path.join(tpl_d,"test2.dat.tpl"))
    assert np.isclose(df.loc['par1'].parval1, 8.675309)

def write2_nan_test():
    import numpy as np
    import pyemu
    import os
    import pandas


    pst = pyemu.Pst(os.path.join("pst", "pest.pst"))
    pst.observation_data.loc[pst.obs_names[0], "weight"] = 1.0e+1000
    try:
        pst.write("test.pst", version=2)
    except:
        pass
    else:
        raise Exception("should have failed")
    try:
        pst.write("test.pst", version=1)
    except:
        pass
    else:
        raise Exception("should have failed")

    pst = pyemu.Pst(os.path.join("pst", "pest.pst"))
    pst.observation_data.loc[pst.obs_names[0], "weight"] = 1.0e-1000
    assert pst.observation_data.loc[pst.obs_names[0], "weight"] == 0.0

    pst = pyemu.Pst(os.path.join("pst", "pest.pst"))
    pst.control_data.nphinored = 1000
    pst.write("test.pst",version=2)

    pst = pyemu.Pst(os.path.join("test.pst"))
    print(pst.control_data.nphinored)

    pst.write("test.pst", version=2)

    pst = pyemu.Pst(os.path.join("test.pst"))
    assert pst.control_data.nphinored == 1000

    pst = pyemu.Pst(os.path.join("pst", "pest.pst"))
    pyemu.helpers.zero_order_tikhonov(pst)
    pst.prior_information.loc[pst.prior_names[0], "weight"] = np.NaN
    try:
        pst.write("test.pst", version=1)
    except:
        pass
    else:
        raise Exception("should have failed")
    try:
        pst.write("test.pst", version=2)
    except:
        pass
    else:
        raise Exception("should have failed")



    pst = pyemu.Pst(os.path.join("pst", "pest.pst"))
    pst.model_output_data.loc[pst.instruction_files[0], "pest_file"] = np.NaN
    try:
        pst.write("test.pst", version=1)
    except:
        pass
    else:
        raise Exception("should have failed")
    try:
        pst.write("test.pst", version=2)
    except:
        pass
    else:
        raise Exception("should have failed")

    pst = pyemu.Pst(os.path.join("pst", "pest.pst"))
    pst.model_input_data.loc[pst.template_files[0], "pest_file"] = np.NaN
    try:
        pst.write("test.pst", version=1)
    except:
        pass
    else:
        raise Exception("should have failed")
    try:
        pst.write("test.pst", version=2)
    except:
        pass
    else:
        raise Exception("should have failed")

    pst = pyemu.Pst(os.path.join("pst","pest.pst"))
    pst.parameter_data.loc[pst.par_names[0],"parval1"] = np.NaN
    try:
        pst.write("test.pst",version=2)
    except:
        pass
    else:
        raise Exception("should have failed")
    try:
        pst.write("test.pst",version=1)
    except:
        pass
    else:
        raise Exception("should have failed")

    pst = pyemu.Pst(os.path.join("pst", "pest.pst"))
    pst.parameter_groups.loc[pst.parameter_groups.pargpnme[0], "derinc"] = np.NaN
    try:
        pst.write("test.pst", version=2)
    except:
        pass
    else:
        raise Exception("should have failed")
    try:
        pst.write("test.pst", version=1)
    except:
        pass
    else:
        raise Exception("should have failed")

    pst = pyemu.Pst(os.path.join("pst", "pest.pst"))
    pst.observation_data.loc[pst.obs_names[0], "weight"] = np.NaN
    try:
        pst.write("test.pst", version=2)
    except:
        pass
    else:
        raise Exception("should have failed")
    try:
        pst.write("test.pst", version=1)
    except:
        pass
    else:
        raise Exception("should have failed")


def rename_pars_test():
    import pyemu
    org_d = os.path.join("..","examples","henry")
    new_d = "henry"
    if os.path.exists(new_d):
        shutil.rmtree(new_d)
    shutil.copytree(org_d,new_d)
    pst = pyemu.Pst(os.path.join(new_d,"pest.pst"))
    pyemu.helpers.zero_order_tikhonov(pst)
    name_dict = {"mult1":"first_multiplier","kr01c01":"hk_r:1_c:1"}
    #print(pst.par_names)
    pst.rename_parameters(name_dict,pst_path=new_d)

    found = []
    for eq in pst.prior_information.equation:
        for old,new in name_dict.items():
            if old in eq:
                raise Exception(old)
            elif new in eq:
                found.append(new)
    assert len(found) == len(name_dict)

    snames = set(pst.par_names)
    for old,new in name_dict.items():
        assert old not in snames
        assert new in snames
    found = []
    for tpl_file in pst.model_input_data.loc[:, "pest_file"].apply(
        lambda x: x.replace("\\", os.path.sep)):
        if not os.path.exists(os.path.join(new_d,tpl_file)):
            continue
        t = set(pyemu.pst_utils.parse_tpl_file(os.path.join(new_d,tpl_file)))
        for old,new in name_dict.items():
            assert old not in t
            if new in t:
                found.append(new)
    found = set(found)
    assert len(found) == len(name_dict)
    pst.write(os.path.join(new_d,"test.pst"))
    pst.write(os.path.join(new_d, "test.pst"),version=2)

    name_dict = {"junk":"sux"}
    try:
        pst.rename_parameters(name_dict, pst_path=new_d)
    except:
        pass
    else:
        raise Exception("should have failed")


def rename_obs_test():
    import pyemu
    org_d = os.path.join("..","examples","henry")
    new_d = "henry"
    if os.path.exists(new_d):
        shutil.rmtree(new_d)
    shutil.copytree(org_d,new_d)
    pst = pyemu.Pst(os.path.join(new_d,"pest.pst"))
    print(pst.obs_names)
    return
    name_dict = {"h_obs01_1":"head_site:01_kper:0","c_obs13_2":"concen_site:01_kper:1"}
    pst.rename_observations(name_dict,pst_path=new_d)
    snames = set(pst.obs_names)
    for old,new in name_dict.items():
        assert old not in snames
        assert new in snames
    found = []
    for ins_file in pst.model_output_data.loc[:, "pest_file"].apply(
        lambda x: x.replace("\\", os.path.sep)):
        if not os.path.exists(os.path.join(new_d,ins_file)):
            continue
        i = pyemu.pst_utils.InstructionFile(os.path.join(new_d,ins_file))

        t = i.obs_name_set
        for old,new in name_dict.items():
            assert old not in t
            if new in t:
                found.append(new)
    found = set(found)
    assert len(found) == len(name_dict)
    pst.write(os.path.join(new_d,"test.pst"))
    pst.write(os.path.join(new_d, "test.pst"),version=2)

    name_dict = {"junk":"sux"}
    try:
        pst.rename_observations(name_dict, pst_path=new_d)
    except:
        pass
    else:
        raise Exception("should have failed")



def pst_ctl_opt_args_test():
    import pyemu
    pst = pyemu.Pst.from_par_obs_names()
    print(pst.control_data.numcom)
    pst.write("test3.pst")

    lines = open("test3.pst",'r').readlines()
    assert lines[3].strip()[-1] == "1"


    pst = pyemu.Pst(os.path.join("pst","pestpp_old.pst"))
    for i in range(10):
        pst.model_command.append("test{0}".format(i))
    pst.write("test.pst")

    pst2 = pyemu.Pst("test.pst")

    assert pst.control_data.numcom == pst2.control_data.numcom

    pst2.write("test2.pst",version=2)

    pst3 = pyemu.Pst("test2.pst")
    #print(pst3.control_data.numcom)
    assert pst.control_data.numcom != pst3.control_data.numcom


def invest():
    import pyemu
    pst = pyemu.Pst(os.path.join("pst","comments_pesthp.pst"))
    pst.write(os.path.join("newpst","comments_pesthp.pst"))
    





if __name__ == "__main__":
    
    # write2_nan_test()
    #process_output_files_test()
    # change_limit_test()
    new_format_test()
    # lt_gt_constraint_names_test()
    #csv_to_ins_test()
    #ctrl_data_test()
    #change_limit_test()
    #new_format_test_2()
    # try_process_ins_test()
    # write_tables_test()
    # res_stats_test()
    # test_write_input_files()
    #add_obs_test()
    #add_pars_test()
    # setattr_test()

    # add_pi_test()
    # regdata_test()
    # nnz_groups_test()
    # adj_group_test()
    # regul_rectify_test()
    # derivative_increment_tests()
    # tied_test()

    #pst_manip_test()
    #tpl_ins_test()
    #comments_test()
    #test_e_clean()
    # load_test()
    #res_test()
    #
    # from_io_with_inschek_test()
    # pestpp_args_test()
    # reweight_test()
    # reweight_res_test()
    # run_test()
    # rectify_pgroup_test()
    #sanity_check_test()
    #change_limit_test()
    #write_tables_test()
    #pi_helper_test()
    #ctrl_data_test()
    #new_format_test_2()
    #try_process_ins_test()
    #tpl_ins_test()
    #process_output_files_test()
    #comments_test()
    #read_in_tpl_test()
    #read_in_tpl_test2()
    # tied_test()

    #comments_test()
    #csv_to_ins_test()

    #rename_pars_test()
    #rename_obs_test()
    #pst_ctl_opt_args_test()
    #invest()
    # pst_ctl_opt_args_test()

