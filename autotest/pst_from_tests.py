import os
from pathlib import Path
import platform

import numpy as np
import pandas as pd
import pyemu
from pyemu import os_utils
from pyemu.utils import PstFrom, pp_file_to_dataframe, write_pp_file
import shutil
import pytest

ext = ''
local_bins = False  # change if wanting to test with local binary exes
if local_bins:
    bin_path = os.path.join("..", "..", "bin")
    if "linux" in platform.platform().lower():
        pass
        bin_path = os.path.join(bin_path, "linux")
    elif "darwin" in platform.platform().lower() or 'macos' in platform.platform().lower():
        pass
        bin_path = os.path.join(bin_path, "mac")
    else:
        bin_path = os.path.join(bin_path, "win")
        ext = '.exe'
else:
    bin_path = ''
    if "windows" in platform.platform().lower():
        ext = '.exe'

mf_exe_path = os.path.join(bin_path, "mfnwt")
mt_exe_path = os.path.join(bin_path, "mt3dusgs")
usg_exe_path = os.path.join(bin_path, "mfusg")
mf6_exe_path = os.path.join(bin_path, "mf6")
pp_exe_path = os.path.join(bin_path, "pestpp-glm")
ies_exe_path = os.path.join(bin_path, "pestpp-ies")
swp_exe_path = os.path.join(bin_path, "pestpp-swp")

mf_exe_name = os.path.basename(mf_exe_path)
mf6_exe_name = os.path.basename(mf6_exe_path)


def _get_port():
    import socket
    sock = socket.socket()
    sock.bind(('', 0))
    return sock.getsockname()[1]


def _gen_dummy_obs_file(ws='.', sep=',', ext=None):
    import pandas as pd
    ffn = "somefakeobs"
    if ext is None:
        if sep == ',':
            fnme = f'{ffn}.csv'
        else:
            fnme = f'{ffn}.dat'
    else:
        fnme = f'{ffn}.{ext}'
    text = pyemu.__doc__.split(' ', 100)
    t = []
    c = 15
    for s in text[:15]:
        s = s.strip().replace('\n', '')
        if len(s) > 1 and s not in t:
            t.append(s)
        else:
            t.append(text[c])
            c += 1
    np.random.seed(314)
    df = pd.DataFrame(
        np.random.rand(15,2)*1000,
        columns=['no', 'yes'],
        index=t
    )
    df.index.name = 'idx'
    df.to_csv(os.path.join(ws, fnme), sep=sep)
    return fnme, df


def setup_tmp(od, tmp_path, sub=None):
    basename = Path(od).name
    if sub is not None:
        new_d = Path(tmp_path, basename, sub)
    else:
        new_d = Path(tmp_path, basename)
    if new_d.exists():
        shutil.rmtree(new_d)
    Path(tmp_path).mkdir(exist_ok=True)
    # creation functionality
    shutil.copytree(od, new_d)
    return new_d

# @pytest.fixture
# def freybergmf6_2_pstfrom(tmp_path):
#     import numpy as np
#     import pandas as pd
#     pd.set_option('display.max_rows', 500)
#     pd.set_option('display.max_columns', 500)
#     pd.set_option('display.width', 1000)
#     try:
#         import flopy
#     except:
#         return
#
#     org_model_ws = os.path.join('..', 'examples', 'freyberg_mf6')
#     tmp_model_ws = setup_tmp(org_model_ws, tmp_path)
#     bd = Path.cwd()
#     os.chdir(tmp_path)
#     try:
#         tmp_model_ws = tmp_model_ws.relative_to(tmp_path)
#         sim = flopy.mf6.MFSimulation.load(sim_ws=str(tmp_model_ws))
#         m = sim.get_model()
#         sim.set_all_data_external(check_data=False)
#         sim.write_simulation()
#
#         # SETUP pest stuff...
#         os_utils.run("{0} ".format(mf6_exe_path), cwd=tmp_model_ws)
#         template_ws = "new_temp"
#         if os.path.exists(template_ws):
#             shutil.rmtree(template_ws)
#         # sr0 = m.sr
#         # sr = pyemu.helpers.SpatialReference.from_namfile(
#         #     os.path.join(tmp_model_ws, "freyberg6.nam"),
#         #     delr=m.dis.delr.array, delc=m.dis.delc.array)
#         sr = m.modelgrid
#         # set up PstFrom object
#         pf = PstFrom(original_d=tmp_model_ws, new_d=template_ws,
#                      remove_existing=True,
#                      longnames=True, spatial_reference=sr,
#                      zero_based=False, start_datetime="1-1-2018",
#                      chunk_len=1)
#         yield pf
#     except Exception as e:
#         os.chdir(bd)
#         raise e
#     os.chdir(bd)


# @pytest.fixture
# def freybergnwt_2_pstfrom(tmp_path):
#     import numpy as np
#     import pandas as pd
#     pd.set_option('display.max_rows', 500)
#     pd.set_option('display.max_columns', 500)
#     pd.set_option('display.width', 1000)
#     try:
#         import flopy
#     except:
#         return
#
#     org_model_ws = os.path.join('..', 'examples', 'freyberg_sfr_reaches')
#     tmp_model_ws = setup_tmp(org_model_ws, tmp_path)
#     bd = Path.cwd()
#     os.chdir(tmp_path)
#     nam_file = "freyberg.nam"
#     try:
#         tmp_model_ws = tmp_model_ws.relative_to(tmp_path)
#         m = flopy.modflow.Modflow.load(nam_file, model_ws=tmp_model_ws,
#                                        check=False, forgive=False,
#                                        exe_name=mf_exe_path)
#         flopy.modflow.ModflowRiv(m, stress_period_data={
#             0: [[0, 0, 0, m.dis.top.array[0, 0], 1.0, m.dis.botm.array[0, 0, 0]],
#                 [0, 0, 1, m.dis.top.array[0, 1], 1.0, m.dis.botm.array[0, 0, 1]],
#                 [0, 0, 1, m.dis.top.array[0, 1], 1.0, m.dis.botm.array[0, 0, 1]]]})
#
#         m.external_path = "."
#         m.write_input()
#         runstr = ("{0} {1}".format(mf_exe_path, m.name + ".nam"), tmp_model_ws)
#         print(runstr)
#         os_utils.run(*runstr)
#         template_ws = "template"
#         if os.path.exists(template_ws):
#             shutil.rmtree(template_ws)
#         sr = pyemu.helpers.SpatialReference.from_namfile(
#             os.path.join(m.model_ws, m.namefile),
#             delr=m.dis.delr, delc=m.dis.delc)
#         # set up PstFrom object
#         pf = PstFrom(original_d=tmp_model_ws, new_d=template_ws,
#                      remove_existing=True,
#                      longnames=True, spatial_reference=sr,
#                      zero_based=False, start_datetime="1-1-2018",
#                      chunk_len=1)
#         yield pf
#     except Exception as e:
#         os.chdir(bd)
#         raise e
#     os.chdir(bd)


def freyberg_test(tmp_path):
    import numpy as np
    import pandas as pd
    from pyemu import PyemuWarning
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)
    try:
        import flopy
    except:
        return

    org_model_ws = os.path.join("..", "examples", "freyberg_sfr_update")
    tmp_model_ws = setup_tmp(org_model_ws, tmp_path)

    bd = Path.cwd()
    os.chdir(tmp_path)

    nam_file = "freyberg.nam"
    try:
        org_model_ws = tmp_model_ws.relative_to(tmp_path)
        m = flopy.modflow.Modflow.load(nam_file, model_ws=org_model_ws,
                                       check=False, forgive=False,
                                       exe_name=mf_exe_path)
        flopy.modflow.ModflowRiv(m, stress_period_data={
            0: [[0, 0, 0, m.dis.top.array[0, 0], 1.0, m.dis.botm.array[0, 0, 0]],
                [0, 0, 1, m.dis.top.array[0, 1], 1.0, m.dis.botm.array[0, 0, 1]],
                [0, 0, 1, m.dis.top.array[0, 1], 1.0, m.dis.botm.array[0, 0, 1]]]})

        m.external_path = "."
        m.write_input()
        print("{0} {1}".format(mf_exe_path, m.name + ".nam"), org_model_ws)
        os_utils.run("{0} {1}".format(mf_exe_path, m.name + ".nam"),
                     cwd=org_model_ws)
        hds_kperk = []
        for k in range(m.nlay):
            for kper in range(m.nper):
                hds_kperk.append([kper, k])
        hds_runline, df = pyemu.gw_utils.setup_hds_obs(
            os.path.join(org_model_ws, f"{m.name}.hds"), kperk_pairs=None, skip=None,
            prefix="hds", include_path=False)
        pyemu.gw_utils.apply_hds_obs(os.path.join(org_model_ws, f"{m.name}.hds"))

        sfo = flopy.utils.SfrFile(os.path.join(m.model_ws, 'freyberg.sfr.out'))
        sfodf = sfo.get_dataframe()
        sfodf[['kstp', 'kper']] = pd.DataFrame(sfodf.kstpkper.to_list(),
                                               index=sfodf.index)
        sfodf = sfodf.drop('kstpkper', axis=1)
        # just adding a bit of header in for test purposes
        sfo_pp_file = os.path.join(m.model_ws, 'freyberg.sfo.dat')
        with open(sfo_pp_file, 'w') as fp:
            fp.writelines(["This is a post processed sfr output file\n",
                          "Processed into tabular form using the lines:\n",
                          "sfo = flopy.utils.SfrFile('freyberg.sfr.out')\n",
                          "sfo.get_dataframe().to_csv('freyberg.sfo.dat')\n"])
            sfodf.sort_index(axis=1).to_csv(fp, sep=' ', index_label='idx', lineterminator='\n')
        sfodf.sort_index(axis=1).to_csv(os.path.join(m.model_ws, 'freyberg.sfo.csv'),
                     index_label='idx',lineterminator='\n')
        template_ws = "new_temp"
        if os.path.exists(template_ws):
            shutil.rmtree(template_ws)

        # sr0 = m.sr
        sr = pyemu.helpers.SpatialReference.from_namfile(
            os.path.join(m.model_ws, m.namefile),
            delr=m.dis.delr, delc=m.dis.delc)
        # set up PstFrom object
        pf = PstFrom(original_d=org_model_ws, new_d=template_ws,
                     remove_existing=True,
                     longnames=True, spatial_reference=sr,
                     zero_based=False)
        # obs
        #   using tabular style model output
        #   (generated by pyemu.gw_utils.setup_hds_obs())
        f, fdf = _gen_dummy_obs_file(pf.new_d)
        pf.add_observations(f, index_cols='idx', use_cols='yes')
        pf.add_py_function(__file__, '_gen_dummy_obs_file()',
                           is_pre_cmd=False)
        pf.add_observations('freyberg.hds.dat', insfile='freyberg.hds.dat.ins2',
                            index_cols='obsnme', use_cols='obsval', prefix='hds')
        #   using the ins file generated by pyemu.gw_utils.setup_hds_obs()
        pf.add_observations_from_ins(ins_file='freyberg.hds.dat.ins')
        pf.post_py_cmds.append(hds_runline)
        pf.tmp_files.append(f"{m.name}.hds")
        # sfr outputs to obs
        sfr_idx = ['segment', 'reach', 'kstp', 'kper']
        sfr_use = ["Qaquifer", "Qout", 'width']
        with pytest.warns(PyemuWarning):
            pf.add_py_function(__file__, '_gen_dummy_obs_file()',
                               is_pre_cmd=False)
        pf.add_observations('freyberg.sfo.dat', insfile=None,
                            index_cols=sfr_idx,
                            use_cols=sfr_use, prefix='sfr',
                            ofile_skip=4, ofile_sep=' ', use_rows=np.arange(0, 50))
        # check obs set up
        sfrobs = pf.obs_dfs[-1].copy()
        sfrobs[["oname","otype",'usecol'] + sfr_idx] = sfrobs.obsnme.apply(
            lambda x: pd.Series(
                dict([s.split(':') for s in x.split('_') if ':' in s])))
        sfrobs.pop("oname")
        sfrobs.pop("otype")
        sfrobs.loc[:, sfr_idx] = sfrobs.loc[:, sfr_idx].astype(int)
        sfrobs_p = sfrobs.pivot_table(index=sfr_idx,
                                      columns=['usecol'], values='obsval')
        sfodf_c = sfodf.set_index(sfr_idx).sort_index()
        sfodf_c.columns = sfodf_c.columns.str.lower()
        assert (sfrobs_p == sfodf_c.loc[sfrobs_p.index,
                                        sfrobs_p.columns]).all().all(), (
            "Mis-match between expected and processed obs values\n",
            sfrobs_p.head(),
            sfodf_c.loc[sfrobs_p.index, sfrobs_p.columns].head())

        pf.tmp_files.append(f"{m.name}.sfr.out")
        pf.extra_py_imports.append('flopy')
        pf.post_py_cmds.extend(
            ["sfo_pp_file = 'freyberg.sfo.dat'",
             "sfo = flopy.utils.SfrFile('freyberg.sfr.out')",
             "sfodf = sfo.get_dataframe()",
             "sfodf[['kstp', 'kper']] = pd.DataFrame(sfodf.kstpkper.to_list(), index=sfodf.index)",
             "sfodf = sfodf.drop('kstpkper', axis=1)",
             "with open(sfo_pp_file, 'w') as fp:",
             "    fp.writelines(['This is a post processed sfr output file\\n', "
             "'Processed into tabular form using the lines:\\n', "
             "'sfo = flopy.utils.SfrFile(`freyberg.sfr.out`)\\n', "
             "'sfo.get_dataframe().to_csv(`freyberg.sfo.dat`)\\n'])",
             "    sfodf.sort_index(axis=1).to_csv(fp, sep=' ', index_label='idx',lineterminator='\\n')"])
        # csv version of sfr obs
        # sfr outputs to obs
        pf.add_observations('freyberg.sfo.csv', insfile=None,
                            index_cols=['segment', 'reach', 'kstp', 'kper'],
                            use_cols=["Qaquifer", "Qout", "width"], prefix='sfr2',
                            ofile_sep=',', obsgp=['qaquifer', 'qout', "width"],
                            use_rows=np.arange(50, 101))
        # check obs set up
        sfrobs = pf.obs_dfs[-1].copy()
        sfrobs[['oname','otype','usecol'] + sfr_idx] = sfrobs.obsnme.apply(
            lambda x: pd.Series(
                dict([s.split(':') for s in x.split('_') if ':' in s])))
        sfrobs.pop("oname")
        sfrobs.pop("otype")
        sfrobs.loc[:, sfr_idx] = sfrobs.loc[:, sfr_idx].astype(int)
        sfrobs_p = sfrobs.pivot_table(index=sfr_idx,
                                      columns=['usecol'], values='obsval')
        sfodf_c = sfodf.set_index(sfr_idx).sort_index()
        sfodf_c.columns = sfodf_c.columns.str.lower()
        assert (sfrobs_p == sfodf_c.loc[sfrobs_p.index,
                                        sfrobs_p.columns]).all().all(), (
            "Mis-match between expected and processed obs values")
        obsnmes = pd.concat([df.obgnme for df in pf.obs_dfs]).unique()
        assert all([gp in obsnmes for gp in ['qaquifer', 'qout']])
        pf.post_py_cmds.append(
            "sfodf.sort_index(axis=1).to_csv('freyberg.sfo.csv', sep=',', index_label='idx')")
        zone_array = np.arange(m.nlay*m.nrow*m.ncol)
        s = lambda x: "zval_"+str(x)
        zone_array = np.array([s(x) for x in zone_array]).reshape(m.nlay,m.nrow,m.ncol)
        # pars
        pf.add_parameters(filenames="RIV_0000.dat", par_type="grid",
                          index_cols=[0, 1, 2], use_cols=[3, 5],
                          par_name_base=["rivstage_grid", "rivbot_grid"],
                          mfile_fmt='%10d%10d%10d %15.8F %15.8F %15.8F',
                          pargp='rivbot')
        pf.add_parameters(filenames="RIV_0000.dat", par_type="grid",
                          index_cols=[0, 1, 2], use_cols=4)
        pf.add_parameters(filenames=["WEL_0000.dat", "WEL_0001.dat"],
                          par_type="grid", index_cols=[0, 1, 2], use_cols=3,
                          par_name_base="welflux_grid",
                          zone_array=zone_array)
        pf.add_parameters(filenames="WEL_0000.dat",
                          par_type="grid", index_cols=[0, 1, 2], use_cols=3,
                          par_name_base="welflux_grid_direct",
                          zone_array=zone_array,par_style="direct",transform="none")
        pf.add_parameters(filenames=["WEL_0000.dat"], par_type="constant",
                          index_cols=[0, 1, 2], use_cols=3,
                          par_name_base=["flux_const"])
        pf.add_parameters(filenames="rech_1.ref", par_type="grid",
                          zone_array=m.bas6.ibound[0].array,
                          par_name_base="rch_datetime:1-1-1970")
        pf.add_parameters(filenames=["rech_1.ref", "rech_2.ref"],
                          par_type="zone", zone_array=m.bas6.ibound[0].array)
        pf.add_parameters(filenames="rech_1.ref", par_type="pilot_point",
                          zone_array=m.bas6.ibound[0].array,
                          par_name_base="rch_datetime:1-1-1970", pp_space=4)
        pf.add_parameters(filenames="rech_1.ref", par_type="pilot_point",
                          zone_array=m.bas6.ibound[0].array,
                          par_name_base="rch_datetime:1-1-1970", pp_space=1,
                          ult_ubound=100, ult_lbound=0.0)
        pf.add_parameters(filenames="rech_1.ref", par_type="pilot_point",
                          par_name_base="rch_datetime:1-1-1970", pp_space=1,
                          ult_ubound=100, ult_lbound=0.0)


        # add model run command
        pf.mod_sys_cmds.append("{0} {1}".format(mf_exe_name, m.name + ".nam"))
        print(pf.mult_files)
        print(pf.org_files)


        # build pest
        pst = pf.build_pst('freyberg.pst')

        # check mult files are in pst input files
        csv = os.path.join(template_ws, "mult2model_info.csv")
        df = pd.read_csv(csv, index_col=0)
        df = df.loc[pd.notna(df.mlt_file),:]
        pst_input_files = {str(f) for f in pst.input_files}
        mults_not_linked_to_pst = ((set(df.mlt_file.unique()) -
                                    pst_input_files) -
                                   set(df.loc[df.pp_file.notna()].mlt_file))
        assert len(mults_not_linked_to_pst) == 0, print(mults_not_linked_to_pst)
        check_apply(pf)
        pst.control_data.noptmax = 0
        pst.write(os.path.join(pf.new_d, "freyberg.pst"))
        pyemu.os_utils.run("{0} freyberg.pst".format(ies_exe_path), cwd=pf.new_d)

        res_file = os.path.join(pf.new_d, "freyberg.base.rei")
        assert os.path.exists(res_file), res_file
        pst.set_res(res_file)
        print(pst.phi)
        assert np.isclose(pst.phi, 0.), pst.phi
    except Exception as e:
        os.chdir(bd)
        raise e
    os.chdir(bd)


def freyberg_prior_build_test(tmp_path):
    import numpy as np
    import pandas as pd
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)
    try:
        import flopy
    except:
        return

    org_model_ws = os.path.join("..", "examples", "freyberg_sfr_update")
    tmp_model_ws = setup_tmp(org_model_ws, tmp_path)
    bd = Path.cwd()
    os.chdir(tmp_path)

    nam_file = "freyberg.nam"
    try:
        org_model_ws = tmp_model_ws.relative_to(tmp_path)
        m = flopy.modflow.Modflow.load(nam_file, model_ws=org_model_ws,
                                       check=False, forgive=False,
                                       exe_name=mf_exe_path)
        flopy.modflow.ModflowRiv(m, stress_period_data={
            0: [[0, 0, 0, m.dis.top.array[0, 0], 1.0, m.dis.botm.array[0, 0, 0]],
                [0, 0, 1, m.dis.top.array[0, 1], 1.0, m.dis.botm.array[0, 0, 1]],
                [0, 0, 1, m.dis.top.array[0, 1], 1.0, m.dis.botm.array[0, 0, 1]]]})

        welsp = m.wel.stress_period_data.data.copy()
        addwell = welsp[0].copy()
        addwell['k'] = 1
        welsp[0] = np.rec.array(np.concatenate([welsp[0], addwell]))
        samewell = welsp[1].copy()
        samewell['flux'] *= 10
        welsp[1] = np.rec.array(np.concatenate([welsp[1], samewell]))
        m.wel.stress_period_data = welsp

        m.external_path = "."
        m.write_input()

        # for exe in [mf_exe_path, mt_exe_path, ies_exe_path]:
        #     shutil.copy(os.path.relpath(exe, '..'), org_model_ws)

        print("{0} {1}".format(mf_exe_path, m.name + ".nam"), org_model_ws)
        os_utils.run("{0} {1}".format(mf_exe_path, m.name + ".nam"),
                     cwd=org_model_ws)
        hds_kperk = []
        for k in range(m.nlay):
            for kper in range(m.nper):
                hds_kperk.append([kper, k])
        hds_runline, df = pyemu.gw_utils.setup_hds_obs(
            os.path.join(m.model_ws, f"{m.name}.hds"), kperk_pairs=None, skip=None,
            prefix="hds", include_path=False)
        pyemu.gw_utils.apply_hds_obs(os.path.join(m.model_ws, f"{m.name}.hds"))

        template_ws = Path(tmp_path, "new_temp")
        if os.path.exists(template_ws):
            shutil.rmtree(template_ws)
        # sr0 = m.sr
        sr = pyemu.helpers.SpatialReference.from_namfile(
            os.path.join(m.model_ws, m.namefile),
            delr=m.dis.delr, delc=m.dis.delc)
        # set up PstFrom object
        pf = PstFrom(original_d=org_model_ws, new_d=template_ws,
                     remove_existing=True,
                     longnames=True, spatial_reference=sr,
                     zero_based=False)
        pf.extra_py_imports.append('flopy')
        if "linux" in platform.platform().lower():
            pf.mod_sys_cmds.append("which python")
        # obs
        #   using tabular style model output
        #   (generated by pyemu.gw_utils.setup_hds_obs())
        pf.add_observations('freyberg.hds.dat', insfile='freyberg.hds.dat.ins2',
                            index_cols='obsnme', use_cols='obsval', prefix='hds')
        pf.post_py_cmds.append(hds_runline)
        pf.tmp_files.append(f"{m.name}.hds")

        # pars
        v = pyemu.geostats.ExpVario(contribution=1.0, a=2500)
        geostruct = pyemu.geostats.GeoStruct(variograms=v, transform='log')
        # Pars for river list style model file, every entry in columns 3 and 4
        # specifying formatted model file and passing a geostruct  # TODO method for appending specific ult bounds
        # pf.add_parameters(filenames="RIV_0000.dat", par_type="grid",
        #                   index_cols=[0, 1, 2], use_cols=[3, 4],
        #                   par_name_base=["rivstage_grid", "rivcond_grid"],
        #                   mfile_fmt='%10d%10d%10d %15.8F %15.8F %15.8F',
        #                   geostruct=geostruct, lower_bound=[0.9, 0.01],
        #                   upper_bound=[1.1, 100.], ult_lbound=[0.3, None])
        # # 2 constant pars applied to columns 3 and 4
        # # this time specifying free formatted model file
        # pf.add_parameters(filenames="RIV_0000.dat", par_type="constant",
        #                   index_cols=[0, 1, 2], use_cols=[3, 4],
        #                   par_name_base=["rivstage", "rivcond"],
        #                   mfile_fmt='free', lower_bound=[0.9, 0.01],
        #                   upper_bound=[1.1, 100.], ult_lbound=[None, 0.01])
        # Pars for river list style model file, every entry in column 4
        pf.add_parameters(filenames="RIV_0000.dat", par_type="grid",
                          index_cols=[0, 1, 2], use_cols=[4],
                          par_name_base=["rivcond_grid"],
                          mfile_fmt='%10d%10d%10d %15.8F %15.8F %15.8F',
                          geostruct=geostruct, lower_bound=[0.01],
                          upper_bound=[100.], ult_lbound=[None])
        # constant par applied to column 4
        # this time specifying free formatted model file
        pf.add_parameters(filenames="RIV_0000.dat", par_type="constant",
                          index_cols=[0, 1, 2], use_cols=[4],
                          par_name_base=["rivcond"],
                          mfile_fmt='free', lower_bound=[0.01],
                          upper_bound=[100.], ult_lbound=[0.01])
        # pf.add_parameters(filenames="RIV_0000.dat", par_type="constant",
        #                   index_cols=[0, 1, 2], use_cols=5,
        #                   par_name_base="rivbot",
        #                   mfile_fmt='free', lower_bound=0.9,
        #                   upper_bound=1.1, ult_ubound=100.,
        #                   ult_lbound=0.001)
        # setting up temporal variogram for correlating temporal pars
        date = m.dis.start_datetime
        v = pyemu.geostats.ExpVario(contribution=1.0, a=180.0)  # 180 correlation length
        t_geostruct = pyemu.geostats.GeoStruct(variograms=v, transform='log')
        # looping over temporal list style input files
        # setting up constant parameters for col 3 for each temporal file
        # making sure all are set up with same pargp and geostruct (to ensure correlation)
        # Parameters for wel list style
        well_mfiles = ["WEL_0000.dat", "WEL_0001.dat", "WEL_0002.dat"]
        for t, well_file in enumerate(well_mfiles):
            # passing same temporal geostruct and pargp,
            # date is incremented and will be used for correlation with
            pf.add_parameters(filenames=well_file, par_type="constant",
                              index_cols=[0, 1, 2], use_cols=3,
                              par_name_base="flux", alt_inst_str='kper',
                              datetime=date, geostruct=t_geostruct,
                              pargp='wellflux_t', lower_bound=0.25,
                              upper_bound=1.75)
            date = (pd.to_datetime(date) +
                    pd.DateOffset(m.dis.perlen.array[t], 'day'))
        # par for each well (same par through time)
        pf.add_parameters(filenames=well_mfiles,
                          par_type="grid", index_cols=[0, 1, 2], use_cols=3,
                          par_name_base="welflux_grid",
                          zone_array=m.bas6.ibound.array,
                          geostruct=geostruct, lower_bound=0.25, upper_bound=1.75)
        pf.add_parameters(filenames=well_mfiles,
                          par_type="grid", index_cols=[0, 1, 2], use_cols=3,
                          par_name_base="welflux_grid",
                          zone_array=m.bas6.ibound.array,
                          use_rows=(1, 3, 4),
                          geostruct=geostruct, lower_bound=0.25, upper_bound=1.75)
        # global constant across all files
        pf.add_parameters(filenames=well_mfiles,
                          par_type="constant",
                          index_cols=[0, 1, 2], use_cols=3,
                          par_name_base=["flux_global"],
                          lower_bound=0.25, upper_bound=1.75)

        # Spatial array style pars - cell-by-cell
        hk_files = ["hk_Layer_{0:d}.ref".format(i) for i in range(1, 4)]
        for hk in hk_files:
            pf.add_parameters(filenames=hk, par_type="grid",
                              zone_array=m.bas6.ibound[0].array,
                              par_name_base="hk", alt_inst_str='lay',
                              geostruct=geostruct,
                              lower_bound=0.01, upper_bound=100.)

        # Pars for temporal array style model files
        date = m.dis.start_datetime  # reset date
        rch_mfiles = ["rech_0.ref", "rech_1.ref", "rech_2.ref"]
        for t, rch_file in enumerate(rch_mfiles):
            # constant par for each file but linked by geostruct and pargp
            pf.add_parameters(filenames=rch_file, par_type="constant",
                              zone_array=m.bas6.ibound[0].array,
                              par_name_base="rch", alt_inst_str='kper',
                              datetime=date, geostruct=t_geostruct,
                              pargp='rch_t', lower_bound=0.9, upper_bound=1.1)
            date = (pd.to_datetime(date) +
                    pd.DateOffset(m.dis.perlen.array[t], 'day'))
        # spatially distributed array style pars - cell-by-cell
        # pf.add_parameters(filenames=rch_mfiles, par_type="grid",
        #                   zone_array=m.bas6.ibound[0].array,
        #                   par_name_base="rch",
        #                   geostruct=geostruct)
        pf.add_parameters(filenames=rch_mfiles, par_type="pilot_point",
                          zone_array=m.bas6.ibound[0].array,
                          par_name_base="rch", pp_space=1,
                          ult_ubound=None, ult_lbound=None,
                          geostruct=geostruct, lower_bound=0.9, upper_bound=1.1)
        # global constant recharge par
        pf.add_parameters(filenames=rch_mfiles, par_type="constant",
                          zone_array=m.bas6.ibound[0].array,
                          par_name_base="rch_global", lower_bound=0.9,
                          upper_bound=1.1)
        # zonal recharge pars
        pf.add_parameters(filenames=rch_mfiles,
                          par_type="zone", par_name_base='rch_zone',
                          lower_bound=0.9, upper_bound=1.1, ult_lbound=1.e-6,
                          ult_ubound=100.)


        # add model run command
        pf.mod_sys_cmds.append("{0} {1}".format(mf_exe_name, m.name + ".nam"))
        print(pf.mult_files)
        print(pf.org_files)


        # build pest
        pst = pf.build_pst('freyberg.pst')
        cov = pf.build_prior(fmt="ascii")
        pe = pf.draw(100, use_specsim=True)
        # check mult files are in pst input files
        csv = os.path.join(template_ws, "mult2model_info.csv")
        df = pd.read_csv(csv, index_col=0)
        pst_input_files = {str(f) for f in pst.input_files}
        mults_not_linked_to_pst = ((set(df.mlt_file.unique()) -
                                    pst_input_files) -
                                   set(df.loc[df.pp_file.notna()].mlt_file))
        assert len(mults_not_linked_to_pst) == 0, print(mults_not_linked_to_pst)

        pst.write_input_files(pst_path=pf.new_d)
        # test par mults are working
        os.chdir(pf.new_d)
        pyemu.helpers.apply_list_and_array_pars(
            arr_par_file="mult2model_info.csv")
        os.chdir(tmp_path)

        pst.control_data.noptmax = 0
        pst.write(os.path.join(pf.new_d, "freyberg.pst"))
        pyemu.os_utils.run("{0} freyberg.pst".format(ies_exe_path), cwd=pf.new_d)

        res_file = os.path.join(pf.new_d, "freyberg.base.rei")
        assert os.path.exists(res_file), res_file
        pst.set_res(res_file)
        print(pst.phi)
        assert np.isclose(pst.phi, 0), pst.phi

        pe.to_binary(os.path.join(pf.new_d, 'par.jcb'))

        # quick sweep test?
        pst.pestpp_options["ies_par_en"] = 'par.jcb'
        pst.pestpp_options["ies_num_reals"] = 10
        pst.control_data.noptmax = -1
        # par = pst.parameter_data
        # par.loc[:, 'parval1'] = pe.iloc[0].T
        pst.write(os.path.join(pf.new_d, "freyberg.pst"))
        pyemu.os_utils.run("{0} freyberg.pst".format(ies_exe_path), cwd=pf.new_d)
        # pyemu.os_utils.start_workers(pf.new_d,
        #                              exe_rel_path="pestpp-ies",
        #                              pst_rel_path="freyberg.pst",
        #                              num_workers=20, master_dir="master",
        #                              cleanup=False, port=4005)
    except Exception as e:
        os.chdir(bd)
        raise Exception(str(e))
    os.chdir(bd)


def generic_function(wd='.'):
    import pandas as pd
    import numpy as np
    #onames = ["generic_obs_{0}".format(i) for i in range(100)]
    onames = pd.date_range("1-1-2020",periods=100,freq='d')
    df = pd.DataFrame({"index_2":np.arange(100),"simval1":1,"simval2":2,"datetime":onames})
    df.index = df.pop("datetime")
    df.to_csv(os.path.join(wd, "generic.csv"), date_format="%d-%m-%Y %H:%M:%S")
    return df


def another_generic_function(some_arg):
    import pandas as pd
    import numpy as np
    print(some_arg)


def mf6_freyberg_test(setup_freyberg_mf6):
    pf, sim = setup_freyberg_mf6
    m = sim.get_model()
    mg = m.modelgrid
    template_ws = pf.new_d

    # SETUP pest stuff...
    # os_utils.run("{0} ".format(mf6_exe_path), cwd=tmp_model_ws)
    # doctor some of the list par files to add a comment string
    with open(
            Path(template_ws,
                 "freyberg6.wel_stress_period_data_1.txt"), 'r') as fr:
        lines = [line for line in fr]
    with open(
            Path(template_ws,
                 "freyberg6.wel_stress_period_data_1.txt"), 'w') as fw:
        fw.write("# comment line explaining this external file\n")
        for line in lines:
            fw.write(line)

    with open(
            Path(template_ws,
                 "freyberg6.wel_stress_period_data_2.txt"), 'r') as fr:
        lines = [line for line in fr]
    with open(
            Path(template_ws,
                 "freyberg6.wel_stress_period_data_2.txt"), 'w') as fw:
        fw.write("# comment line explaining this external file\n")
        for line in lines[0:3] + ["# comment mid table \n"] + lines[3:]:
            fw.write(line)

    with open(
            Path(template_ws,
                 "freyberg6.wel_stress_period_data_3.txt"), 'r') as fr:
        lines = [line for line in fr]
    with open(
            Path(template_ws,
                 "freyberg6.wel_stress_period_data_3.txt"), 'w') as fw:
        fw.write("#k i j flux \n")
        for line in lines:
            fw.write(line)

    with open(
            Path(template_ws,
                 "freyberg6.wel_stress_period_data_4.txt"), 'r') as fr:
        lines = [line for line in fr]
    with open(
            Path(template_ws,
                 "freyberg6.wel_stress_period_data_4.txt"), 'w') as fw:
        fw.write("# comment line explaining this external file\n"
                 "#k i j flux\n")
        for line in lines:
            fw.write(line)

    # generate a test with headers and non spatial idex
    sfr_pkgdf = pd.DataFrame.from_records(m.sfr.packagedata.array).rename(columns={'ifno':'rno'})
    l = sfr_pkgdf.columns.to_list()
    l = ['#rno', 'k', 'i', 'j'] + l[2:]
    with open(
            Path(template_ws,
                 "freyberg6.sfr_packagedata.txt"), 'r') as fr:
        lines = [line for line in fr]
    with open(Path(template_ws,
                   "freyberg6.sfr_packagedata_test.txt"), 'w') as fw:
        fw.write(' '.join(l))
        fw.write('\n')
        for line in lines:
            fw.write(line)

    # call generic once so that the output file exists
    df = generic_function(template_ws)
    # add the values in generic to the ctl file
    f, fdf = _gen_dummy_obs_file(template_ws, sep=' ')
    pf.add_observations(f, index_cols='idx', use_cols='yes')
    pf.add_py_function(__file__, "_gen_dummy_obs_file(sep=' ')",
                       is_pre_cmd=False)
    pf.add_observations("generic.csv", insfile="generic.csv.ins",
                        index_cols=["datetime", "index_2"],
                        use_cols=["simval1", "simval2"])
    # add the function call to make generic to the forward run script
    pf.add_py_function(__file__, "generic_function()", is_pre_cmd=False)

    # add a function that isnt going to be called directly
    pf.add_py_function(__file__, "another_generic_function(some_arg)",
                       is_pre_cmd=None)

    #pf.post_py_cmds.append("generic_function()")
    # df = pd.read_csv(Path(template_ws, "sfr.csv"), index_col=0)
    # pf.add_observations("sfr.csv", insfile="sfr.csv.ins",
    #                     index_cols="time", use_cols=list(df.columns.values))
    v = pyemu.geostats.ExpVario(contribution=1.0,a=1000)
    gr_gs = pyemu.geostats.GeoStruct(variograms=v)
    rch_temporal_gs = pyemu.geostats.GeoStruct(variograms=pyemu.geostats.ExpVario(contribution=1.0,a=60))
    pf.extra_py_imports.append('flopy')
    ib = m.dis.idomain[0].array
    with open(Path(template_ws, "inflow1.txt"), 'w') as fp:
        fp.write("# rid type rate idx0 idx1\n")
        fp.write("205 666 500000.0 1 1")
    pf.add_parameters(filenames='inflow1.txt',
                      pargp='inflow1',
                      comment_char='#',
                      use_cols=2,
                      index_cols=0,
                      upper_bound=10,
                      lower_bound=0.1,
                      par_type="grid",
                      )

    with open(Path(template_ws, "inflow2.txt"), 'w') as fp:
        fp.write("# rid type rate idx0 idx1\n")
        fp.write("205 infl 500000.3 1 1\n")
        fp.write("205 div 1 500000.7 1\n")
        fp.write("206 infl 600000.7 1 1\n")
        fp.write("206 div 1 500000.7 1")
    inflow2_pre = pd.read_csv(Path(pf.new_d, "inflow2.txt"),
                              header=None, sep=' ', skiprows=1)
    with open(Path(template_ws, "inflow3.txt"), 'w') as fp:
        fp.write("# rid type rate idx0 idx1\n")
        fp.write("205 infl 700000.3 1 1\n")
        fp.write("205 div 1 500000.7 1\n")
        fp.write("206 infl 800000.7 1 1\n")
        fp.write("206 div 1 500000.7 1")
    inflow3_pre = pd.read_csv(Path(pf.new_d, "inflow3.txt"),
                              header=None, sep=' ', skiprows=1)
    pf.add_parameters(filenames=['inflow2.txt', "inflow3.txt"],
                      pargp='inflow',
                      comment_char='#',
                      use_cols=2,
                      index_cols=[0, 1],
                      upper_bound=10,
                      lower_bound=0.1,
                      par_type="grid",
                      use_rows=[[205, 'infl'], [206, 'infl']],
                      )
    pf.add_parameters(filenames=['inflow2.txt', "inflow3.txt"],
                      pargp='inflow2',
                      comment_char='#',
                      use_cols=3,
                      index_cols=[0, 1],
                      upper_bound=5,
                      lower_bound=-5,
                      par_type="grid",
                      use_rows=[[205, 'div'], [206, 'div']],
                      par_style='a',
                      transform='none'
                      )
    with open(Path(template_ws, "inflow4.txt"), 'w') as fp:
        fp.write("# rid type rate idx0 idx1\n")
        fp.write("204_1 infl 700000.3 1 1\n")
        fp.write("205_1 div 1 500000.7 1\n")
        fp.write("206_1 infl 800000.7 1 1\n")
        fp.write("207_1 div 1 500000.7 1")

    inflow4_pre = pd.read_csv(Path(pf.new_d, "inflow4.txt"),
                              header=None, sep=' ', skiprows=1)
    pf.add_parameters(filenames="inflow4.txt",
                      pargp='inflow4',
                      comment_char='#',
                      use_cols=2,
                      index_cols=[0, 1],
                      upper_bound=10,
                      lower_bound=0.1,
                      par_type="grid",
                      use_rows=[("204_1", "infl")],
                      )
    pf.add_parameters(filenames="inflow4.txt",
                      pargp='inflow5',
                      comment_char='#',
                      use_cols=3,
                      index_cols=[0],
                      upper_bound=10,
                      lower_bound=0.1,
                      par_type="grid",
                      use_rows=(1, 3),
                      )
    # pf.add_parameters(filenames=['inflow2.txt'],
    #                   pargp='inflow3',
    #                   comment_char='#',
    #                   use_cols=2,
    #                   index_cols=[0, 1],
    #                   upper_bound=10,
    #                   lower_bound=0.1,
    #                   par_type="grid",
    #                   use_rows=[0, 2],
    #                   )
    ft, ftd = _gen_dummy_obs_file(pf.new_d, sep=',', ext='txt')
    pf.add_parameters(filenames=f, par_type="grid", mfile_skip=1, index_cols=0,
                      use_cols=[2], par_name_base="tmp",
                      pargp="tmp")
    pf.add_parameters(filenames=ft, par_type="grid", mfile_skip=1, index_cols=0,
                      use_cols=[1, 2], par_name_base=["tmp2_1", "tmp2_2"],
                      pargp="tmp2", mfile_sep=',', par_style='direct')
    tags = {"npf_k_":[0.1,10.],"npf_k33_":[.1,10],"sto_ss":[.1,10],"sto_sy":[.9,1.1],"rch_recharge":[.5,1.5]}
    dts = pd.to_datetime("1-1-2018") + pd.to_timedelta(np.cumsum(sim.tdis.perioddata.array["perlen"]),unit="d")
    print(dts)
    for tag, bnd in tags.items():
        lb, ub = bnd[0], bnd[1]
        arr_files = [f for f in os.listdir(template_ws) if tag in f and f.endswith(".txt")]
        if "rch" in tag:
            pf.add_parameters(filenames=arr_files, par_type="grid", par_name_base="rch_gr",
                              pargp="rch_gr", zone_array=ib, upper_bound=ub, lower_bound=lb,
                              geostruct=gr_gs)
            for arr_file in arr_files:
                kper = int(arr_file.split('.')[1].split('_')[-1]) - 1
                pf.add_parameters(filenames=arr_file,par_type="constant",par_name_base=arr_file.split('.')[1]+"_cn",
                                  pargp="rch_const",zone_array=ib,upper_bound=ub,lower_bound=lb,geostruct=rch_temporal_gs,
                                  datetime=dts[kper])
        else:
            for arr_file in arr_files:
                # these ult bounds are used later in an assert
                # and also are used so that the initial input array files
                # are preserved
                ult_lb = None
                ult_ub = None
                if "npf_k_" in arr_file:
                   ult_ub = 31.0
                   ult_lb = -1.3
                pf.add_parameters(filenames=arr_file,par_type="grid",par_name_base=arr_file.split('.')[1]+"_gr",
                                  pargp=arr_file.split('.')[1]+"_gr",zone_array=ib,upper_bound=ub,lower_bound=lb,
                                  geostruct=gr_gs,ult_ubound=None if ult_ub is None else ult_ub + 1,
                                  ult_lbound=None if ult_lb is None else ult_lb + 1)
                # use a slightly lower ult bound here
                pf.add_parameters(filenames=arr_file, par_type="pilotpoints", par_name_base=arr_file.split('.')[1]+"_pp",
                                  pargp=arr_file.split('.')[1]+"_pp", zone_array=ib,upper_bound=ub,lower_bound=lb,
                                  ult_ubound=None if ult_ub is None else ult_ub - 1,
                                  ult_lbound=None if ult_lb is None else ult_lb - 1,geostruct=gr_gs)

                # use a slightly lower ult bound here
                pf.add_parameters(filenames=arr_file, par_type="constant",
                                  par_name_base=arr_file.split('.')[1] + "_cn",
                                  pargp=arr_file.split('.')[1] + "_cn", zone_array=ib,
                                  upper_bound=ub, lower_bound=lb,geostruct=gr_gs)

    # arr = np.loadtxt(Path(template_ws, 'freyberg6.npf_k_layer1.txt'))
    # onecolf = Path(template_ws, '1col.txt')
    # np.savetxt(onecolf, arr.ravel()[:,None])
    # pdf = pf.add_parameters(filenames=onecolf.relative_to(template_ws),
    #                   par_type="grid", par_name_base="onecol-gr",
    #                   pargp="onecol-gr", zone_array=ib.ravel()[:,None],
    #                   upper_bound=10, lower_bound=0.1)

    # add SP1 spatially constant, but temporally correlated wel flux pars
    kper = 0
    list_file = "freyberg6.wel_stress_period_data_{0}.txt".format(kper+1)
    pf.add_parameters(filenames=list_file, par_type="constant",
                      par_name_base="twel_mlt_{0}".format(kper),
                      pargp="twel_mlt".format(kper), index_cols=[0, 1, 2],
                      use_cols=[3], upper_bound=1.5, lower_bound=0.5,
                      datetime=dts[kper], geostruct=rch_temporal_gs,
                      mfile_skip=1)

    # add temporally indep, but spatially correlated wel flux pars
    pf.add_parameters(filenames=list_file, par_type="grid",
                      par_name_base="wel_grid_{0}".format(kper),
                      pargp="wel_{0}".format(kper), index_cols=[0, 1, 2],
                      use_cols=[3], upper_bound=1.5, lower_bound=0.5,
                      geostruct=gr_gs, mfile_skip=1)
    kper = 1
    list_file = "freyberg6.wel_stress_period_data_{0}.txt".format(kper+1)
    pf.add_parameters(filenames=list_file, par_type="constant",
                      par_name_base="twel_mlt_{0}".format(kper),
                      pargp="twel_mlt".format(kper), index_cols=[0, 1, 2],
                      use_cols=[3], upper_bound=1.5, lower_bound=0.5,
                      datetime=dts[kper], geostruct=rch_temporal_gs,
                      mfile_skip='#')
    # add temporally indep, but spatially correlated wel flux pars
    pf.add_parameters(filenames=list_file, par_type="grid",
                      par_name_base="wel_grid_{0}".format(kper),
                      pargp="wel_{0}".format(kper), index_cols=[0, 1, 2],
                      use_cols=[3], upper_bound=1.5, lower_bound=0.5,
                      geostruct=gr_gs, mfile_skip='#')
    kper = 2
    list_file = "freyberg6.wel_stress_period_data_{0}.txt".format(kper+1)
    pf.add_parameters(filenames=list_file, par_type="constant",
                      par_name_base="twel_mlt_{0}".format(kper),
                      pargp="twel_mlt".format(kper), index_cols=['#k', 'i', 'j'],
                      use_cols=['flux'], upper_bound=1.5, lower_bound=0.5,
                      datetime=dts[kper], geostruct=rch_temporal_gs)
    # add temporally indep, but spatially correlated wel flux pars
    pf.add_parameters(filenames=list_file, par_type="grid",
                      par_name_base="wel_grid_{0}".format(kper),
                      pargp="wel_{0}".format(kper), index_cols=['#k', 'i', 'j'],
                      use_cols=['flux'], upper_bound=1.5, lower_bound=0.5,
                      geostruct=gr_gs)
    kper = 3
    list_file = "freyberg6.wel_stress_period_data_{0}.txt".format(kper+1)
    pf.add_parameters(filenames=list_file, par_type="constant",
                      par_name_base="twel_mlt_{0}".format(kper),
                      pargp="twel_mlt".format(kper), index_cols=['#k', 'i', 'j'],
                      use_cols=['flux'], upper_bound=1.5, lower_bound=0.5,
                      datetime=dts[kper], geostruct=rch_temporal_gs,
                      mfile_skip=1)
    # add temporally indep, but spatially correlated wel flux pars
    pf.add_parameters(filenames=list_file, par_type="grid",
                      par_name_base="wel_grid_{0}".format(kper),
                      pargp="wel_{0}".format(kper), index_cols=['#k', 'i', 'j'],
                      use_cols=['flux'], upper_bound=1.5, lower_bound=0.5,
                      geostruct=gr_gs, mfile_skip=1)
    list_files = ["freyberg6.wel_stress_period_data_{0}.txt".format(t)
                  for t in range(5, m.nper+1)]
    for list_file in list_files:
        kper = int(list_file.split(".")[1].split('_')[-1]) - 1
        # add spatially constant, but temporally correlated wel flux pars
        pf.add_parameters(filenames=list_file,par_type="constant",par_name_base="twel_mlt_{0}".format(kper),
                          pargp="twel_mlt".format(kper),index_cols=[0,1,2],use_cols=[3],
                          upper_bound=1.5,lower_bound=0.5, datetime=dts[kper], geostruct=rch_temporal_gs)

        # add temporally indep, but spatially correlated wel flux pars
        pf.add_parameters(filenames=list_file, par_type="grid", par_name_base="wel_grid_{0}".format(kper),
                          pargp="wel_{0}".format(kper), index_cols=[0, 1, 2], use_cols=[3],
                          upper_bound=1.5, lower_bound=0.5, geostruct=gr_gs)
    pf.add_parameters(filenames=list_file, par_type="grid", par_name_base=f"wel_grid_{kper}",
                      pargp=f"wel_{kper}_v2", index_cols=[0, 1, 2], use_cols=[3], use_rows=[1],
                      upper_bound=1.5, lower_bound=0.5, geostruct=gr_gs)
    # test non spatial idx in list like
    pf.add_parameters(filenames="freyberg6.sfr_packagedata_test.txt", par_name_base="sfr_rhk",
                      pargp="sfr_rhk", index_cols=['#rno'], use_cols=['rhk'], upper_bound=10.,
                      lower_bound=0.1,
                      par_type="grid")

    # SFR inflow
    files = [f for f in os.listdir(pf.new_d) if "sfr_perioddata" in f and f.endswith(".txt")]
    sp = [int(f.split(".")[1].split('_')[-1]) for f in files]
    d = {s: f for s, f in zip(sp, files)}
    sp.sort()
    files = [d[s] for s in sp]
    print(files)
    for f in files:
        # get the stress period number from the file name
        kper = int(f.split('.')[1].split('_')[-1]) - 1
        if kper != 0:
            continue  # only want 1 kper in test
        # add the parameters
        pf.add_parameters(filenames=f,
                          index_cols=[0],  # reach number
                          use_cols=[2],  # columns with parameter values
                          par_type="grid",
                          par_name_base="grsfr",
                          pargp="grsfr",
                          upper_bound=10, lower_bound=0.1,
                          # don't need ult_bounds because it is a single multiplier
                          datetime=dts[kper],  # this places the parameter value on the "time axis"
                          geostruct=rch_temporal_gs)

    # add model run command
    pf.mod_sys_cmds.append("mf6")
    print(pf.mult_files)
    print(pf.org_files)

    # build pest
    pst = pf.build_pst('freyberg.pst')

    # # quick check of write and apply method
    pars = pst.parameter_data
    # set reach 1 hk to 100
    sfr_pars = pars.loc[pars.parnme.str.startswith('pname:sfr')].index
    pars.loc[sfr_pars, 'parval1'] = np.random.random(len(sfr_pars)) * 10

    sfr_pars = pars.loc[sfr_pars].copy()
    print(sfr_pars)
    sfr_pars[["name",'inst',"ptype", 'usecol',"pstyle", '#rno']] = sfr_pars.parnme.apply(
        lambda x: pd.DataFrame([s.split(':') for s in x.split('_')
                                if ':' in s]).set_index(0)[1])

    sfr_pars['#rno'] = sfr_pars['#rno'].astype(int)

    dummymult = 4.
    pars = pst.parameter_data
    pst.parameter_data.loc[pars.index.str.contains('_pp'), 'parval1'] = dummymult
    check_apply(pf)
    # os.chdir(pf.new_d)
    # pst.write_input_files()
    # pyemu.helpers.apply_list_and_array_pars()
    # os.chdir(tmp_path)
    # verify apply
    inflow2_df = pd.read_csv(Path(pf.new_d, "inflow2.txt"),
                             header=None, sep=' ', skiprows=1)
    inflow3_df = pd.read_csv(Path(pf.new_d, "inflow3.txt"),
                             header=None, sep=' ', skiprows=1)
    inflow4_df = pd.read_csv(Path(pf.new_d, "inflow4.txt"),
                             header=None, sep=' ', skiprows=1)
    assert (inflow2_df == inflow2_pre).all().all()
    assert (inflow3_df == inflow3_pre).all().all()
    assert (inflow4_df == inflow4_pre).all().all()
    multinfo = pd.read_csv(Path(pf.new_d, "mult2model_info.csv"),
                           index_col=0)
    ppmultinfo = multinfo.dropna(subset=['pp_file'])
    for mfile in ppmultinfo.model_file.unique():
        subinfo = ppmultinfo.loc[ppmultinfo.model_file == mfile]
        assert subinfo.org_file.nunique() == 1
        org = np.loadtxt(Path(pf.new_d, subinfo.org_file.values[0]))
        m = dummymult ** len(subinfo)
        check = org * m
        check[ib == 0] = org[ib == 0]
        ult_u = subinfo.upper_bound.astype(float).values[0]
        ult_l = subinfo.lower_bound.astype(float).values[0]
        check[check < ult_l] = ult_l
        check[check > ult_u] = ult_u
        result = np.loadtxt(Path(pf.new_d, mfile))
        assert np.isclose(check, result).all(), (f"Problem with par apply for "
                                                 f"{mfile}")
    df = pd.read_csv(Path(pf.new_d, "freyberg6.sfr_packagedata_test.txt"),
                     sep=r'\s+', index_col=0)
    df.index = df.index - 1
    print(df.rhk)
    print((sfr_pkgdf.set_index('rno').loc[df.index, 'rhk'] *
                 sfr_pars.set_index('#rno').loc[df.index, 'parval1']))
    assert np.isclose(
        df.rhk, (sfr_pkgdf.set_index('rno').loc[df.index, 'rhk'] *
                 sfr_pars.set_index('#rno').loc[df.index, 'parval1'])).all()
    pars.loc[sfr_pars.index, 'parval1'] = 1.0
    pars.loc[pars.index.str.contains('_pp'), 'parval1'] = 1.0
    # add more:
    pf.add_parameters(filenames="freyberg6.sfr_packagedata.txt", par_name_base="sfr_rhk",
                      pargp="sfr_rhk", index_cols={'k': 1, 'i': 2, 'j': 3}, use_cols=[9], upper_bound=10.,
                      lower_bound=0.1,
                      par_type="grid", rebuild_pst=True)

    df = pd.read_csv(Path(template_ws, "heads.csv"), index_col=0)
    pf.add_observations("heads.csv", insfile="heads.csv.ins",
                        index_cols="time", use_cols=list(df.columns.values),
                        prefix="hds", rebuild_pst=True)

    # test par mults are working
    check_apply(pf)

    # cov build
    cov = pf.build_prior(fmt="none").to_dataframe()
    twel_pars = [p for p in pst.par_names if "twel_mlt" in p]
    twcov = cov.loc[twel_pars,twel_pars]
    dsum = np.diag(twcov.values).sum()
    assert twcov.sum().sum() > dsum

    rch_cn = [p for p in pst.par_names if "_cn" in p]
    print(rch_cn)
    rcov = cov.loc[rch_cn,rch_cn]
    dsum = np.diag(rcov.values).sum()
    assert rcov.sum().sum() > dsum

    num_reals = 100
    pe = pf.draw(num_reals, use_specsim=False)
    #pe = pe.copy()
    pe.enforce()
    lbnd = pst.parameter_data.parlbnd.to_dict()
    for pname,lb in lbnd.items():
        diff = pe.loc[:,pname].values - lb
        print(pname,lb,diff.min())
        assert diff.min() >= 0
    pe.to_binary(Path(template_ws, "prior.jcb"))
    assert pe.shape[1] == pst.npar_adj, "{0} vs {1}".format(pe.shape[1], pst.npar_adj)
    assert pe.shape[0] == num_reals

    pst.control_data.noptmax = 0
    pst.pestpp_options["additional_ins_delimiters"] = ","

    pst.write(os.path.join(pf.new_d, "freyberg.pst"))
    pyemu.os_utils.run("{0} freyberg.pst".format(ies_exe_path), cwd=pf.new_d)

    res_file = os.path.join(pf.new_d, "freyberg.base.rei")
    assert os.path.exists(res_file), res_file
    pst.set_res(res_file)
    print(pst.phi)
    assert np.isclose(pst.phi, 0)

    # check mult files are in pst input files
    csv = os.path.join(template_ws, "mult2model_info.csv")
    df = pd.read_csv(csv, index_col=0)
    pst_input_files = {str(f) for f in pst.input_files}
    mults_not_linked_to_pst = ((set(df.mlt_file.dropna().unique()) -
                                pst_input_files) -
                               set(df.loc[df.pp_file.notna()].mlt_file))
    assert len(mults_not_linked_to_pst) == 0, print(mults_not_linked_to_pst)

    # make sure the appropriate ult bounds have made it thru
    df = pd.read_csv(os.path.join(template_ws, "mult2model_info.csv"))
    # print(df.columns)
    df = df.loc[df.model_file.apply(lambda x: "npf_k_" in x),:]
    # print(df)
    # print(df.upper_bound)
    # print(df.lower_bound)
    assert np.isclose(np.abs(float(df.upper_bound.min()) - 30.), 0.), df.upper_bound.min()
    assert np.isclose(np.abs(float(df.lower_bound.max()) - -0.3), 0.), df.lower_bound.max()


def mf6_freyberg_shortnames_test(setup_freyberg_mf6):
    import numpy as np
    import pandas as pd
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)
    # set up PstFrom object
    # obs
    #   using tabular style model output
    #   (generated by pyemu.gw_utils.setup_hds_obs())
    # pf.add_observations('freyberg.hds.dat', insfile='freyberg.hds.dat.ins2',
    #                     index_cols='obsnme', use_cols='obsval', prefix='hds')
    pf, sim = setup_freyberg_mf6
    m = sim.get_model()
    tmp_model_ws = m.model_ws
    template_ws = pf.new_d
    pf.longnames = False

    v = pyemu.geostats.ExpVario(contribution=1.0, a=1000)
    gr_gs = pyemu.geostats.GeoStruct(variograms=v)
    rch_temporal_gs = pyemu.geostats.GeoStruct(variograms=pyemu.geostats.ExpVario(contribution=1.0,a=60))
    pf.extra_py_imports.append('flopy')
    ib = m.dis.idomain[0].array
    tags = {"npf_k_":[0.1,10.],"npf_k33_":[.1,10],"sto_ss":[.1,10],"sto_sy":[.9,1.1],"rch_recharge":[.5,1.5]}
    dts = pd.to_datetime("1-1-2018") + pd.to_timedelta(np.cumsum(sim.tdis.perioddata.array["perlen"]),unit="d")
    print(dts)
    for tag,bnd in tags.items():
        lb,ub = bnd[0],bnd[1]
        arr_files = [f for f in os.listdir(tmp_model_ws)
                     if tag in f and f.endswith(".txt")]
        if "rch" in tag:
            pf.add_parameters(filenames=arr_files, par_type="grid", par_name_base="rg",
                              pargp="rg", zone_array=ib, upper_bound=ub, lower_bound=lb,
                              geostruct=gr_gs)
            for arr_file in arr_files:
                kper = int(arr_file.split('.')[1].split('_')[-1]) - 1
                pf.add_parameters(filenames=arr_file,par_type="constant",par_name_base="rc{0}_".format(kper),
                                  pargp="rc",zone_array=ib,upper_bound=ub,lower_bound=lb,geostruct=rch_temporal_gs,
                                  datetime=dts[kper])
        else:

            for arr_file in arr_files:
                pb = tag.split('_')[1] + arr_file.split('.')[1][-1]
                pf.add_parameters(filenames=arr_file,par_type="grid",par_name_base=pb+"g",
                                  pargp=pb+"g",zone_array=ib,upper_bound=ub,lower_bound=lb,
                                  geostruct=gr_gs)
                pf.add_parameters(filenames=arr_file, par_type="pilotpoints", par_name_base=pb+"p",
                                  pargp=pb+"p", zone_array=ib,upper_bound=ub,lower_bound=lb,)
        for arr_file in arr_files:
            pf.add_observations(arr_file)
    list_files = [f for f in os.listdir(tmp_model_ws) if "wel_stress_period_data" in f]
    for list_file in list_files:
        kper = list_file.split(".")[1].split('_')[-1]
        pf.add_parameters(filenames=list_file,par_type="constant",par_name_base="w{0}".format(kper),
                          pargp="wel_{0}".format(kper),index_cols=[0,1,2],use_cols=[3],
                          upper_bound=1.5,lower_bound=0.5)
    f = list_files[-1] if 'dup' not in list_files[-1] else list_files[-2]
    kper = f.split('.')[1].split('_')[-1]
    za = np.ones((3, 40, 20))
    df = pd.read_csv(os.path.join(m.model_ws, f),
                     sep=r'\s+', header=None) - 1
    za[tuple(df.loc[0:2, [0, 1, 2]].values.T)] = [2,3,4]
    pdf = pf.add_parameters(filenames=f, par_type="zone",
                            par_name_base="w{0}".format(kper),
                            pargp="wz_{0}".format(kper), index_cols=[0, 1, 2],
                            use_cols=[3],
                            upper_bound=1.5, lower_bound=0.5,
                            zone_array=za)
    assert len(pdf) == 4

    # add model run command
    pf.mod_sys_cmds.append("mf6")
    print(pf.mult_files)
    print(pf.org_files)

    # build pest
    pst = pf.build_pst('freyberg.pst')
    obs = set(pst.observation_data.obsnme)
    obsin = set()
    for ins in pst.instruction_files:
        with open(os.path.join(pf.new_d, ins), "rt") as f:
            text = f.read()
            for ob in obs:
                if f"!{ob}!" in text:
                    obsin.add(ob)
        obs = obs - obsin
    assert len(obs) == 0, f"{len(obs)} obs not found in insfiles: {obs}"

    par = set(pst.parameter_data.parnme)
    parin = set()
    for tpl in pst.template_files:
        with open(os.path.join(pf.new_d, tpl), "rt") as f:
            text = f.read()
            for p in par:
                if f"{p} " in text:
                    parin.add(p)
        par = par - parin
    assert len(par) == 0, f"{len(par)} pars not found in tplfiles: {par}"
    # test update/rebuild
    pf.add_parameters(filenames="freyberg6.sfr_packagedata.txt",
                      par_name_base="rhk",
                      pargp="sfr_rhk", index_cols=[0, 1, 2, 3], use_cols=[9],
                      upper_bound=10., lower_bound=0.1,
                      par_type="grid", rebuild_pst=True)
    pf.add_parameters(filenames=arr_file, par_type="grid", par_name_base=pb + "g2",
                      pargp=pb + "g2", zone_array=ib, upper_bound=ub, lower_bound=lb,
                      geostruct=gr_gs, rebuild_pst=True)
    df = pd.read_csv(os.path.join(tmp_model_ws, "heads.csv"), index_col=0)
    pf.add_observations("heads.csv", insfile="heads.csv.ins", index_cols="time",
                        use_cols=list(df.columns.values), prefix="hds",
                        rebuild_pst=True)
    obs = set(pst.observation_data.obsnme)
    obsin = set()
    for ins in pst.instruction_files:
        with open(os.path.join(pf.new_d, ins), "rt") as f:
            text = f.read()
            for ob in obs:
                if f"!{ob}!" in text:
                    obsin.add(ob)
            obs = obs - obsin
    assert len(obs) == 0, f"{len(obs)} obs not found in insfiles: {obs}"

    par = set(pst.parameter_data.parnme)
    parin = set()
    for tpl in pst.template_files:
        with open(os.path.join(pf.new_d, tpl), "rt") as f:
            text = f.read()
            for p in par:
                if f"{p} " in text:
                    parin.add(p)
        par = par - parin
    assert len(par) == 0, f"{len(par)} pars not found in tplfiles: {par}"

    assert pst.parameter_data.parnme.apply(lambda x: len(x)).max() <= 12
    assert pst.observation_data.obsnme.apply(lambda x: len(x)).max() <= 20

    num_reals = 100
    pe = pf.draw(num_reals, use_specsim=True)
    pe.to_binary(os.path.join(template_ws, "prior.jcb"))
    assert pe.shape[1] == pst.npar_adj, "{0} vs {1}".format(pe.shape[0], pst.npar_adj)
    assert pe.shape[0] == num_reals

    # test par mults are working
    check_apply(pf)
    pst.try_parse_name_metadata()
    pst.control_data.noptmax = 0
    pst.pestpp_options["additional_ins_delimiters"] = ","

    pst.write(os.path.join(pf.new_d, "freyberg.pst"))
    pyemu.os_utils.run("{0} freyberg.pst".format(ies_exe_path), cwd=pf.new_d)

    pst = pyemu.Pst(os.path.join(pf.new_d, "freyberg.pst"))
    res_file = os.path.join(pf.new_d, "freyberg.base.rei")
    assert os.path.exists(res_file), res_file
    pst.set_res(res_file)
    print(pst.phi)
    assert np.isclose(pst.phi, 0.0), pst.phi

    # check mult files are in pst input files
    csv = os.path.join(template_ws, "mult2model_info.csv")
    df = pd.read_csv(csv, index_col=0)
    pst_input_files = {str(f) for f in pst.input_files}
    mults_not_linked_to_pst = ((set(df.mlt_file.unique()) -
                                pst_input_files) -
                               set(df.loc[df.pp_file.notna()].mlt_file))
    assert len(mults_not_linked_to_pst) == 0, print(mults_not_linked_to_pst)


def mf6_freyberg_da_test(tmp_path):
    import numpy as np
    import pandas as pd
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)
    try:
        import flopy
    except:
        return

    org_model_ws = os.path.join('..', 'examples', 'freyberg_mf6_da')
    tmp_model_ws = setup_tmp(org_model_ws, tmp_path)
    bd = Path.cwd()
    os.chdir(tmp_path)
    try:
        tmp_model_ws = tmp_model_ws.relative_to(tmp_path)
        sim = flopy.mf6.MFSimulation.load(sim_ws=str(tmp_model_ws))
        m = sim.get_model("freyberg6")

        # SETUP pest stuff...
        os_utils.run("{0} ".format("mf6"), cwd=tmp_model_ws)

        template_ws = "new_temp_da"
        if os.path.exists(template_ws):
             shutil.rmtree(template_ws)
        # sr0 = m.sr
        sr = pyemu.helpers.SpatialReference.from_namfile(
            os.path.join(tmp_model_ws, "freyberg6.nam"),
            delr=m.dis.delr.array, delc=m.dis.delc.array)
        # set up PstFrom object
        pf = PstFrom(original_d=tmp_model_ws, new_d=template_ws,
                     remove_existing=True,
                     longnames=True, spatial_reference=sr,
                     zero_based=False,start_datetime="1-1-2018")
        # obs
        #   using tabular style model output
        #   (generated by pyemu.gw_utils.setup_hds_obs())
        # pf.add_observations('freyberg.hds.dat', insfile='freyberg.hds.dat.ins2',
        #                     index_cols='obsnme', use_cols='obsval', prefix='hds')

        df = pd.read_csv(os.path.join(tmp_model_ws,"heads.csv"),index_col=0)
        pf.add_observations("heads.csv",insfile="heads.csv.ins",index_cols="time",use_cols=list(df.columns.values),prefix="hds")
        df = pd.read_csv(os.path.join(tmp_model_ws, "sfr.csv"), index_col=0)
        pf.add_observations("sfr.csv", insfile="sfr.csv.ins", index_cols="time", use_cols=list(df.columns.values))
        v = pyemu.geostats.ExpVario(contribution=1.0,a=1000)
        gr_gs = pyemu.geostats.GeoStruct(variograms=v)
        rch_temporal_gs = pyemu.geostats.GeoStruct(variograms=pyemu.geostats.ExpVario(contribution=1.0,a=60))
        pf.extra_py_imports.append('flopy')
        ib = m.dis.idomain[0].array
        tags = {"npf_k_":[0.1,10.],"npf_k33_":[.1,10],"sto_ss":[.1,10],"sto_sy":[.9,1.1],"rch_recharge":[.5,1.5]}
        dts = pd.to_datetime("1-1-2018") + pd.to_timedelta(np.cumsum(sim.tdis.perioddata.array["perlen"]),unit="d")
        print(dts)
        for tag,bnd in tags.items():
            lb,ub = bnd[0],bnd[1]
            arr_files = [f for f in os.listdir(tmp_model_ws) if tag in f and f.endswith(".txt")]
            if "rch" in tag:
                pf.add_parameters(filenames=arr_files, par_type="grid", par_name_base="rch_gr",
                                  pargp="rch_gr", zone_array=ib, upper_bound=ub, lower_bound=lb,
                                  geostruct=gr_gs)
                for arr_file in arr_files:
                    kper = int(arr_file.split('.')[1].split('_')[-1]) - 1
                    pf.add_parameters(filenames=arr_file,par_type="constant",par_name_base=arr_file.split('.')[1]+"_cn",
                                      pargp="rch_const",zone_array=ib,upper_bound=ub,lower_bound=lb,geostruct=rch_temporal_gs,
                                      datetime=dts[kper])
            else:
                for arr_file in arr_files:
                    pf.add_parameters(filenames=arr_file,par_type="grid",par_name_base=arr_file.split('.')[1]+"_gr",
                                      pargp=arr_file.split('.')[1]+"_gr",zone_array=ib,upper_bound=ub,lower_bound=lb,
                                      geostruct=gr_gs)
                    pf.add_parameters(filenames=arr_file, par_type="pilotpoints", par_name_base=arr_file.split('.')[1]+"_pp",
                                      pargp=arr_file.split('.')[1]+"_pp", zone_array=ib,upper_bound=ub,lower_bound=lb,)


        list_files = [f for f in os.listdir(tmp_model_ws) if "wel_stress_period_data" in f]
        for list_file in list_files:
            kper = int(list_file.split(".")[1].split('_')[-1]) - 1
            # add spatially constant, but temporally correlated wel flux pars
            pf.add_parameters(filenames=list_file,par_type="constant",par_name_base="twel_mlt_{0}".format(kper),
                              pargp="twel_mlt".format(kper),index_cols=[0,1,2],use_cols=[3],
                              upper_bound=1.5,lower_bound=0.5, datetime=dts[kper], geostruct=rch_temporal_gs)

            # add temporally indep, but spatially correlated wel flux pars
            pf.add_parameters(filenames=list_file, par_type="grid", par_name_base="wel_grid_{0}".format(kper),
                              pargp="wel_{0}".format(kper), index_cols=[0, 1, 2], use_cols=[3],
                              upper_bound=1.5, lower_bound=0.5, geostruct=gr_gs)

        pf.add_parameters(filenames="freyberg6.sfr_packagedata.txt",par_name_base="sfr_rhk",
                          pargp="sfr_rhk",index_cols={'k':1,'i':2,'j':3},use_cols=[9],upper_bound=10.,lower_bound=0.1,
                          par_type="grid")

        # add model run command
        pf.mod_sys_cmds.append("mf6")
        print(pf.mult_files)
        print(pf.org_files)

        # build pest
        pst = pf.build_pst('freyberg.pst', version=2)
        pst.write(os.path.join(template_ws,"freyberg6_da.pst"),version=2)


        # setup direct (non mult) pars on the IC files with par names that match the obs names
        obs = pst.observation_data
        hobs = obs.loc[obs.obsnme.str.startswith("hds"),:].copy()
        hobs.loc[:,"k"] = hobs.obsnme.apply(lambda x: int(x.split(':')[1].split("_")[1]))
        hobs.loc[:, "i"] = hobs.obsnme.apply(lambda x: int(x.split(':')[1].split("_")[2]))
        hobs.loc[:, "j"] = hobs.obsnme.apply(lambda x: int(x.split(':')[1].split("_")[3]))
        hobs_set = set(hobs.obsnme.to_list())
        ic_files = [f for f in os.listdir(template_ws) if "ic_strt" in f and f.endswith(".txt")]
        print(ic_files)
        ib = m.dis.idomain[0].array
        tpl_files = []
        for ic_file in ic_files:
            tpl_file = os.path.join(template_ws,ic_file+".tpl")
            vals,names = [],[]
            with open(tpl_file,'w') as f:
                f.write("ptf ~\n")
                k = int(ic_file.split('.')[1][-1]) - 1
                org_arr = np.loadtxt(os.path.join(template_ws,ic_file))
                for i in range(org_arr.shape[0]):
                    for j in range(org_arr.shape[1]):
                        if ib[i,j] < 1:
                            f.write(" -1.0e+30 ")
                        else:
                            pname = "hds_usecol:trgw_{0}_{1}_{2}_time:31.0".format(k,i,j)
                            if pname not in hobs_set and ib[i,j] > 0:
                                print(k,i,j,pname,ib[i,j])
                            f.write(" ~  {0}   ~".format(pname))
                            vals.append(org_arr[i,j])
                            names.append(pname)
                    f.write("\n")
            df = pf.pst.add_parameters(tpl_file,pst_path=".")
            pf.pst.parameter_data.loc[df.parnme,"partrans"] = "fixed"
            pf.pst.parameter_data.loc[names,"parval1"] = vals

        pf.pst.write(os.path.join(template_ws,"freyberg6_da.pst"),version=2)

        num_reals = 100
        pe = pf.draw(num_reals, use_specsim=True)
        pe.to_binary(os.path.join(template_ws, "prior.jcb"))
        assert pe.shape[1] == pst.npar_adj, "{0} vs {1}".format(pe.shape[0], pst.npar_adj)
        assert pe.shape[0] == num_reals

        # test par mults are working
        os.chdir(pf.new_d)
        pst.write_input_files()
        pyemu.helpers.apply_list_and_array_pars(
            arr_par_file="mult2model_info.csv")
        os.chdir(tmp_path)

        pst.control_data.noptmax = 0
        pst.pestpp_options["additional_ins_delimiters"] = ","

        pst.write(os.path.join(pf.new_d, "freyberg.pst"))
        pyemu.os_utils.run("{0} freyberg.pst".format(ies_exe_path), cwd=pf.new_d)

        res_file = os.path.join(pf.new_d, "freyberg.base.rei")
        assert os.path.exists(res_file), res_file
        pst.set_res(res_file)
        print(pst.phi)
        assert np.isclose(pst.phi, 0), pst.phi

        # check mult files are in pst input files
        csv = os.path.join(template_ws, "mult2model_info.csv")
        df = pd.read_csv(csv, index_col=0)
        pst_input_files = {str(f) for f in pst.input_files}
        mults_not_linked_to_pst = ((set(df.mlt_file.unique()) -
                                    pst_input_files) -
                                   set(df.loc[df.pp_file.notna()].mlt_file))
        assert len(mults_not_linked_to_pst) == 0, print(mults_not_linked_to_pst)
    except Exception as e:
        os.chdir(bd)
        raise e
    os.chdir(bd)


@pytest.fixture
def setup_freyberg_mf6(tmp_path, request):
    try:
        import flopy
    except:
        return
    try:
        model = request.param
    except AttributeError:
        model = "freyberg_mf6"
    org_model_ws = os.path.join('..', 'examples', model)
    tmp_model_ws = setup_tmp(org_model_ws, tmp_path)
    # print(tmp_model_ws)
    dup_file = "freyberg6.wel_stress_period_data_with_dups.txt"
    shutil.copy2(os.path.join("utils", dup_file), tmp_model_ws)
    bd = Path.cwd()
    os.chdir(tmp_path)
    try:
        tmp_model_ws = tmp_model_ws.relative_to(tmp_path)
        sim = flopy.mf6.MFSimulation.load(sim_ws=str(tmp_model_ws))
        m = sim.get_model()
        sim.set_all_data_external()
        sim.write_simulation()

        # SETUP pest stuff...
        os_utils.run("{0} ".format("mf6"), cwd=tmp_model_ws)

        template_ws = Path(tmp_path, "template")
        if os.path.exists(template_ws):
            shutil.rmtree(template_ws)
        sr = m.modelgrid
        # set up PstFrom object
        pf = PstFrom(original_d=tmp_model_ws,
                     new_d=template_ws,
                     remove_existing=True,
                     longnames=True,
                     spatial_reference=sr,
                     zero_based=False,
                     start_datetime="1-1-2018")
        if model == 'freyberg_mf6':
            pf.add_observations("sfr.csv", insfile="sfr.csv.ins",
                                index_cols="time",
                                use_cols=["GAGE_1", "HEADWATER", "TAILWATER"],
                                ofile_sep=",")
        yield pf, sim
    except Exception as e:
        os.chdir(bd)
        raise e
    os.chdir(bd)


def build_direct(pf):
    pf.mod_sys_cmds.append("mf6")
    print(pf.mult_files)
    print(pf.org_files)

    # build pest
    pst = pf.build_pst('freyberg.pst')
    # cov = pf.build_prior(fmt="non")
    # cov.to_coo(os.path.join(template_ws, "prior.jcb"))
    pst.try_parse_name_metadata()
    # df = pd.read_csv(os.path.join(pf.original_d, "heads.csv"), index_col=0)
    # pf.add_observations("heads.csv", insfile="heads.csv.ins", index_cols="time",
    #                     use_cols=list(df.columns.values),
    #                     prefix="hds", rebuild_pst=True)
    return pf, pst


def check_apply(pf):
    # test par mults are working
    bd = Path.cwd()
    os.chdir(pf.new_d)
    try:
        pf.pst.write_input_files()
        pyemu.helpers.apply_list_and_array_pars(
            arr_par_file="mult2model_info.csv", chunk_len=100)
    except Exception as e:
        os.chdir(bd)
        raise e
    os.chdir(bd)


def direct_quickfull_test(setup_freyberg_mf6):
    pf, sim = setup_freyberg_mf6
    m = sim.get_model()
    mg = m.modelgrid
    # Setup geostruct for spatial pars
    gr_v = pyemu.geostats.ExpVario(contribution=1.0, a=1000)
    gr_gs = pyemu.geostats.GeoStruct(variograms=gr_v, transform="log")
    pp_v = pyemu.geostats.ExpVario(contribution=1.0, a=5000)
    pp_gs = pyemu.geostats.GeoStruct(variograms=pp_v, transform="log")
    rch_temporal_gs = pyemu.geostats.GeoStruct(variograms=pyemu.geostats.ExpVario(contribution=1.0, a=60))
    pf.extra_py_imports.append('flopy')
    ib = mg.idomain[0]

    dts = pd.to_datetime("1-1-2018") + pd.to_timedelta(np.cumsum(m.modeltime.perlen), unit="d")
    print(dts)
    # ib = m.dis.idomain.array[0,:,:]
    # setup from array style pars
    tag, bnd = ("rch_recharge", [.5, 1.5])
    lb, ub = bnd[0], bnd[1]
    arr_files = Path(pf.original_d).glob(f"*{tag}_[1-5].txt")
    for arr_file in arr_files:
        # indy direct grid pars for each array type file
        kper = int(arr_file.stem.split('_')[-1]) - 1
        pf.add_parameters(filenames=arr_file.name, par_type="grid",
                          par_name_base="rch_gr",
                          pargp=f"rch_gr_{kper}", zone_array=ib,
                          upper_bound=1.0e-3, lower_bound=1.0e-7,
                          par_style="direct",
                          geostruct=gr_gs)
        # additional constant mults
        pf.add_parameters(filenames=arr_file.name, par_type="constant",
                          par_name_base=arr_file.stem.split('.')[-1] + "_cn",
                          pargp="rch_const", zone_array=ib,
                          upper_bound=ub, lower_bound=lb,
                          geostruct=rch_temporal_gs,
                          datetime=dts[kper])
    # Add a variety of list style pars
    list_files = Path(pf.original_d).glob(
        "*wel_stress_period_data_[1-5].txt"
    )
    for list_file in list_files:
        kper = int(list_file.stem.split('_')[-1]) - 1
        # add spatially constant, but temporally correlated wel flux pars
        pf.add_parameters(filenames=list_file.name,
                          par_type="constant",
                          par_name_base="twel_mlt_{0}".format(kper),
                          pargp="twel_mlt_{0}".format(kper),
                          index_cols=[0, 1, 2], use_cols=[3],
                          upper_bound=1.5, lower_bound=0.5,
                          datetime=dts[kper], geostruct=rch_temporal_gs)

        # add temporally indep, but spatially correlated wel flux pars
        pf.add_parameters(filenames=list_file.name,
                          par_type="grid",
                          par_name_base="wel_grid_{0}".format(kper),
                          pargp="wel_{0}".format(kper),
                          index_cols=[0, 1, 2], use_cols=[3],
                          upper_bound=0.0, lower_bound=-1000,
                          geostruct=gr_gs, par_style="direct",
                          transform="none")
    pf, pst = build_direct(pf)
    # check that cov build works -- this is mem intensive
    cov = pf.build_prior(fmt="non")
    cov.to_coo(os.path.join(pf.new_d, "prior.jcb"))
    del cov
    # check adding obs after initial build
    df = pd.read_csv(os.path.join(pf.new_d, "heads.csv"), index_col=0)
    pf.add_observations("heads.csv", insfile="heads.csv.ins", index_cols="time",
                        use_cols=list(df.columns.values),
                        prefix="hds", rebuild_pst=True)
    # check applying pars
    check_apply(pf)
    # check prior ensemble build
    num_reals = 100
    pe = pf.draw(num_reals, use_specsim=True)
    pe.to_binary(os.path.join(pf.new_d, "prior.jcb"))
    assert pe.shape[1] == pst.npar_adj, "{0} vs {1}".format(pe.shape[0], pst.npar_adj)
    assert pe.shape[0] == num_reals
    pst.pestpp_options['ies_par_en'] = "prior.jcb"
    pst.pestpp_options['ies_num_reals'] = 5
    pst.control_data.noptmax=-1
    # check run and results -- phi should be small...
    pst.write(os.path.join(pf.new_d, "freyberg.pst"))
    pyemu.os_utils.run("{0} freyberg.pst".format(ies_exe_path), cwd=pf.new_d)
    res_file = os.path.join(pf.new_d, "freyberg.base.rei")
    assert os.path.exists(res_file), res_file
    pst.set_res(res_file)
    print(pst.phi)
    assert np.isclose(pst.phi ,0), pst.phi

    df = pd.read_csv(Path(pf.new_d, "freyberg.0.obs.csv"))
    ens, qs = pyemu.helpers.calc_observation_ensemble_quantiles(
        df, pst, [0.05,0.25,0.5,0.75,0.95]
    )


def direct_multadd_combo_test(setup_freyberg_mf6):
    dup_file = "freyberg6.wel_stress_period_data_with_dups.txt"
    pf, _ = setup_freyberg_mf6
    pf.longnames = False
    ghb_files = [f for f in os.listdir(pf.new_d) if ".ghb_stress" in f and f.endswith("txt")]
    pf.add_parameters(ghb_files, par_type="grid", par_style="add", use_cols=3, par_name_base="ghbstage",
                      pargp="ghbstage", index_cols=[0, 1, 2], transform="none", lower_bound=-5, upper_bound=5)

    pf.add_parameters(ghb_files, par_type="grid", par_style="multiplier", use_cols=3, par_name_base="mltstage",
                      pargp="ghbstage", index_cols=[0, 1, 2], transform="log", lower_bound=0.5,
                      upper_bound=1.5)
    list_file = "freyberg6.ghb_stress_period_data_1.txt"
    pf.add_parameters(filenames=list_file, par_type="constant", par_name_base=["ghb_stage", "ghb_cond"],
                      pargp=["ghb_stage", "ghb_cond"], index_cols=[0, 1, 2], use_cols=[3, 4],
                      upper_bound=[35, 150], lower_bound=[32, 50], par_style="direct",
                      transform="none")
    pf.add_parameters(filenames=dup_file, par_type="grid", par_name_base="dups",
                      pargp="dups", index_cols=[0, 1, 2], use_cols=[3],
                      upper_bound=0.0, lower_bound=-500, par_style="direct",
                      transform="none")
    pf, pst = build_direct(pf)
    check_apply(pf)

    # check ghb files
    org_ghb = pd.read_csv(os.path.join(pf.new_d, "org", "freyberg6.ghb_stress_period_data_1.txt"),
                          header=None, names=["l", "r", "c", "stage", "cond"])
    new_ghb = pd.read_csv(os.path.join(pf.new_d, "freyberg6.ghb_stress_period_data_1.txt"),
                          sep=r'\s+',
                          header=None, names=["l", "r", "c", "stage", "cond"])
    d = org_ghb.stage - new_ghb.stage
    print(d)
    assert d.sum() == 0, d.sum()

    # test the additive ghb stage pars
    par = pst.parameter_data
    par.loc[par.longname.str.contains("ghbstage_inst:0"), "parval1"] = 3.0
    pst.write(os.path.join(pf.new_d, "freyberg.pst"))
    pyemu.os_utils.run("{0} freyberg.pst".format(ies_exe_path), cwd=pf.new_d)

    org_ghb = pd.read_csv(
        os.path.join(pf.new_d, "org", "freyberg6.ghb_stress_period_data_1.txt"),
        header=None, names=["l", "r", "c", "stage", "cond"]
    )
    new_ghb = pd.read_csv(
        os.path.join(pf.new_d, "freyberg6.ghb_stress_period_data_1.txt"),
        sep=r'\s+',
        header=None, names=["l", "r", "c", "stage", "cond"]
    )
    d = (org_ghb.stage - new_ghb.stage).apply(np.abs)
    print(d)
    assert d.mean() == 3.0, d.mean()

    # check that the interaction between the direct ghb stage par and the additive ghb stage pars
    # is working
    par.loc[par.longname.str.contains("ghb_stage"), "parval1"] -= 3.0
    pst.write(os.path.join(pf.new_d, "freyberg.pst"))
    pyemu.os_utils.run("{0} freyberg.pst".format(ies_exe_path), cwd=pf.new_d)
    org_ghb = pd.read_csv(
        os.path.join(pf.original_d, "freyberg6.ghb_stress_period_data_1.txt"),
        header=None, names=["l", "r", "c", "stage", "cond"],
        sep=r'\s+'
    )
    new_ghb = pd.read_csv(
        os.path.join(pf.new_d, "freyberg6.ghb_stress_period_data_1.txt"),
        sep=r'\s+',
        header=None, names=["l", "r", "c", "stage", "cond"]
    )
    d = org_ghb.stage - new_ghb.stage
    print(new_ghb.stage)
    print(org_ghb.stage)
    print(d)
    assert d.sum() == 0.0, d.sum()

    # check the interaction with multiplicative ghb stage, direct ghb stage and additive ghb stage
    par.loc[par.longname.str.contains("mltstage"), "parval1"] = 1.1
    # par.loc[par.parnme.str.contains("ghbstage_inst:0"), "parval1"] = 0.0
    # par.loc[par.parnme.str.contains("ghb_stage"), "parval1"] += 3.0
    pst.write(os.path.join(pf.new_d, "freyberg.pst"))
    pyemu.os_utils.run("{0} freyberg.pst".format(ies_exe_path), cwd=pf.new_d)
    org_ghb = pd.read_csv(
        os.path.join(pf.original_d, "freyberg6.ghb_stress_period_data_1.txt"),
        header=None, names=["l", "r", "c", "stage", "cond"], sep=r'\s+')
    new_ghb = pd.read_csv(
        os.path.join(pf.new_d, "freyberg6.ghb_stress_period_data_1.txt"),
        sep=r'\s+',
        header=None, names=["l", "r", "c", "stage", "cond"])
    d = (org_ghb.stage * 1.1) - new_ghb.stage
    print(new_ghb.stage)
    print(org_ghb.stage)
    print(d)
    assert d.sum() == 0.0, d.sum()


def direct_arraypars_test(setup_freyberg_mf6):
    pf, sim = setup_freyberg_mf6
    m = sim.get_model()
    mg = m.modelgrid
    # Setup geostruct for spatial pars
    gr_v = pyemu.geostats.ExpVario(contribution=1.0, a=1000)
    gr_gs = pyemu.geostats.GeoStruct(variograms=gr_v, transform="log")
    pp_v = pyemu.geostats.ExpVario(contribution=1.0, a=5000)
    pp_gs = pyemu.geostats.GeoStruct(variograms=pp_v, transform="log")
    rch_temporal_gs = pyemu.geostats.GeoStruct(variograms=pyemu.geostats.ExpVario(contribution=1.0, a=60))
    pf.extra_py_imports.append('flopy')
    ib = mg.idomain[0]
    tags = {"npf_k_": [0.1, 10.], "npf_k33_": [.1, 10], "sto_ss": [.1, 10], "sto_sy": [.9, 1.1],
            "rch_recharge": [.5, 1.5]}
    dts = pd.to_datetime("1-1-2018") + pd.to_timedelta(np.cumsum(m.modeltime.perlen), unit="d")
    print(dts)
    # ib = m.dis.idomain.array[0,:,:]
    # setup from array style pars
    for tag, bnd in tags.items():
        lb, ub = bnd[0], bnd[1]
        arr_files = [f for f in os.listdir(pf.original_d)
                     if tag in f and f.endswith(".txt")]
        if "rch" in tag:
            for arr_file in arr_files:
                # indy direct grid pars for each array type file
                recharge_files = ["recharge_1.txt", "recharge_2.txt", "recharge_3.txt"]
                pf.add_parameters(filenames=arr_file, par_type="grid", par_name_base="rch_gr",
                                  pargp="rch_gr", zone_array=ib, upper_bound=1.0e-3, lower_bound=1.0e-7,
                                  par_style="direct")
                # additional constant mults
                kper = int(arr_file.split('.')[1].split('_')[-1]) - 1
                pf.add_parameters(filenames=arr_file, par_type="constant",
                                  par_name_base=arr_file.split('.')[1] + "_cn",
                                  pargp="rch_const", zone_array=ib, upper_bound=ub, lower_bound=lb,
                                  geostruct=rch_temporal_gs,
                                  datetime=dts[kper])
        else:
            for arr_file in arr_files:
                # grid mults pure and simple
                pf.add_parameters(filenames=arr_file, par_type="grid", par_name_base=arr_file.split('.')[1] + "_gr",
                                  pargp=arr_file.split('.')[1] + "_gr", zone_array=ib, upper_bound=ub,
                                  lower_bound=lb,
                                  geostruct=gr_gs)
    pf, pst = build_direct(pf)
    check_apply(pf)

    # turn direct recharge to min and direct wel to min and
    # check that the model results are consistent
    par = pst.parameter_data
    rch_par = par.loc[par.parnme.apply(
        lambda x: "pname:rch_gr" in x and "ptype:gr_pstyle:d" in x), "parnme"]
    par.loc[rch_par, "parval1"] = par.loc[rch_par, "parlbnd"]

    pst.write(os.path.join(pf.new_d, "freyberg.pst"))
    pyemu.os_utils.run("{0} freyberg.pst".format(ies_exe_path), cwd=pf.new_d)

    rch_files = [f for f in os.listdir(pf.new_d)
                 if ".rch_recharge" in f and f.endswith(".txt")]
    rch_val = par.loc[rch_par, "parval1"].iloc[0]
    i, j = par.loc[rch_par, ["i", 'j']].astype(int).values.T
    for rch_file in rch_files:
        arr = np.loadtxt(os.path.join(pf.new_d, rch_file))[i, j]
        print(rch_file, rch_val, arr.mean(), arr.max(), arr.min())
        if np.abs(arr.max() - rch_val) > 1.0e-6 or np.abs(arr.min() - rch_val) > 1.0e-6:
            raise Exception("recharge too diff")


def direct_listpars_test(setup_freyberg_mf6):
    import flopy
    pf, sim = setup_freyberg_mf6
    m = sim.get_model()
    template_ws = pf.new_d
    # Setup geostruct for spatial pars
    gr_v = pyemu.geostats.ExpVario(contribution=1.0, a=1000)
    gr_gs = pyemu.geostats.GeoStruct(variograms=gr_v, transform="log")
    pp_v = pyemu.geostats.ExpVario(contribution=1.0, a=5000)
    rch_temporal_gs = pyemu.geostats.GeoStruct(variograms=pyemu.geostats.ExpVario(contribution=1.0, a=60))
    pf.extra_py_imports.append('flopy')
    dts = pd.to_datetime("1-1-2018") + pd.to_timedelta(np.cumsum(m.modeltime.perlen), unit="d")

    # Add a variety of list style pars
    list_files = ["freyberg6.wel_stress_period_data_{0}.txt".format(t)
                  for t in range(1, m.nper + 1)]
    # make dummy versions with headers
    for fl in list_files[0:2]:  # this adds a header to well file
        with open(os.path.join(template_ws, fl), 'r') as fr:
            lines = [line for line in fr]
        with open(os.path.join(template_ws, f"new_{fl}"), 'w') as fw:
            fw.write("k i j flux \n")
            for line in lines:
                fw.write(line)

    # fl = "freyberg6.wel_stress_period_data_3.txt" # Add extra string col_id
    for fl in list_files[2:7]:
        with open(os.path.join(template_ws, fl), 'r') as fr:
            lines = [line for line in fr]
        with open(os.path.join(template_ws, f"new_{fl}"), 'w') as fw:
            fw.write("well k i j flux \n")
            for i, line in enumerate(lines):
                fw.write(f"well{i}" + line)

    list_files.sort()
    for list_file in list_files:
        kper = int(list_file.split(".")[1].split('_')[-1]) - 1
        # add spatially constant, but temporally correlated wel flux pars
        pf.add_parameters(filenames=list_file, par_type="constant", par_name_base="twel_mlt_{0}".format(kper),
                          pargp="twel_mlt_{0}".format(kper), index_cols=[0, 1, 2], use_cols=[3],
                          upper_bound=1.5, lower_bound=0.5, datetime=dts[kper], geostruct=rch_temporal_gs)

        # add temporally indep, but spatially correlated wel flux pars
        pf.add_parameters(filenames=list_file, par_type="grid", par_name_base="wel_grid_{0}".format(kper),
                          pargp="wel_{0}".format(kper), index_cols=[0, 1, 2], use_cols=[3],
                          upper_bound=0.0, lower_bound=-1000, geostruct=gr_gs, par_style="direct",
                          transform="none")
    # Adding dummy list pars with different file structures
    list_file = "new_freyberg6.wel_stress_period_data_1.txt"  # with a header
    pf.add_parameters(filenames=list_file, par_type="grid", par_name_base='nwell_mlt',
                      pargp='nwell_mult', index_cols=['k', 'i', 'j'], use_cols='flux',
                      upper_bound=10, lower_bound=-10, geostruct=gr_gs,
                      transform="none")
    pf.add_parameters(filenames=list_file, par_type="grid", par_name_base='nwell_grid',
                      pargp='nwell', index_cols=['k', 'i', 'j'], use_cols='flux',
                      upper_bound=10, lower_bound=-10, geostruct=gr_gs, par_style="direct",
                      transform="none")
    # with skip instead
    list_file = "new_freyberg6.wel_stress_period_data_2.txt"
    pf.add_parameters(filenames=list_file, par_type="grid", par_name_base='nwell_grid',
                      pargp='nwell', index_cols=[0, 1, 2], use_cols=3,
                      upper_bound=10, lower_bound=-10, geostruct=gr_gs, par_style="direct",
                      transform="none", mfile_skip=1)

    list_file = "new_freyberg6.wel_stress_period_data_3.txt"
    pf.add_parameters(filenames=list_file, par_type="grid", par_name_base='nwell_mlt',
                      pargp='nwell_mult', index_cols=['well', 'k', 'i', 'j'], use_cols='flux',
                      upper_bound=10, lower_bound=-10, geostruct=gr_gs,
                      transform="none")
    pf.add_parameters(filenames=list_file, par_type="grid", par_name_base='nwell_grid',
                      pargp='nwell', index_cols=['well', 'k', 'i', 'j'], use_cols='flux',
                      upper_bound=10, lower_bound=-10, geostruct=gr_gs, par_style="direct",
                      transform="none")
    # with skip instead
    list_file = "new_freyberg6.wel_stress_period_data_4.txt"
    pf.add_parameters(filenames=list_file, par_type="grid",
                      par_name_base='nwell_grid', pargp='nwell',
                      index_cols=[0, 1, 2, 3],  # or... {'well': 0, 'k': 1, 'i': 2, 'j': 3},
                      use_cols=4, upper_bound=10, lower_bound=-10,
                      geostruct=gr_gs, par_style="direct", transform="none",
                      mfile_skip=1)

    list_file = "new_freyberg6.wel_stress_period_data_5.txt"
    pf.add_parameters(filenames=list_file, par_type="grid",
                      par_name_base=['nwell5_k', 'nwell5_q'],
                      pargp='nwell5',
                      index_cols=['well', 'i', 'j'],
                      use_cols=['k', 'flux'], upper_bound=10, lower_bound=-10,
                      geostruct=gr_gs, par_style="direct", transform="none",
                      mfile_skip=0, use_rows=[3, 4])

    list_file = "new_freyberg6.wel_stress_period_data_6.txt"
    pf.add_parameters(filenames=list_file, par_type="grid",
                      par_name_base=['nwell6_k', 'nwell6_q'],
                      pargp='nwell6',
                      index_cols=['well', 'i', 'j'],
                      use_cols=['k', 'flux'], upper_bound=10, lower_bound=-10,
                      geostruct=gr_gs, par_style="direct", transform="none",
                      mfile_skip=0, use_rows=[(3, 21, 15), (3, 30, 7)])
    # use_rows should match so all should be setup 2 cols 6 rows
    assert len(pf.par_dfs[-1]) == 2 * 6  # should be
    list_file = "new_freyberg6.wel_stress_period_data_7.txt"
    pf.add_parameters(filenames=list_file, par_type="grid",
                      par_name_base=['nwell6_k', 'nwell6_q'],
                      pargp='nwell6',
                      index_cols=['well', 'i', 'j'],
                      use_cols=['k', 'flux'], upper_bound=10, lower_bound=-10,
                      geostruct=gr_gs, par_style="direct", transform="none",
                      mfile_skip=0,
                      use_rows=[('well2', 21, 15), ('well4', 30, 7)])
    assert len(pf.par_dfs[-1]) == 2 * 2  # should be

    pf, pst = build_direct(pf)
    check_apply(pf)

    # check on that those dummy pars compare to the model versions.
    for f in Path(pf.new_d).glob("new_*txt"):
        n_df = pd.read_csv(f, sep=r"\s+")
        o_df = pd.read_csv(f.with_name(f.name.strip('new_')),
                           sep=r"\s+", header=None)
        o_df.columns = ['k', 'i', 'j', 'flux']
        assert np.isclose(n_df.loc[:, o_df.columns], o_df).all(), (
            f"Something broke with alternative style model files ({f})"
        )

    # turn direct recharge to min and direct wel to min and
    # check that the model results are consistent
    par = pst.parameter_data
    rch_par = par.loc[par.parnme.apply(
        lambda x: "pname:rch_gr" in x and "ptype:gr_pstyle:d" in x), "parnme"]
    wel_par = par.loc[par.parnme.apply(
        lambda x: "pname:wel_grid" in x and "ptype:gr_usecol:3_pstyle:d" in x), "parnme"]
    par.loc[rch_par, "parval1"] = par.loc[rch_par, "parlbnd"]
    # this should set wells to zero since they are negative values in the control file
    par.loc[wel_par, "parval1"] = par.loc[wel_par, "parubnd"]

    pst.write(os.path.join(pf.new_d, "freyberg.pst"))
    pyemu.os_utils.run("{0} freyberg.pst".format(ies_exe_path), cwd=pf.new_d)

    lst = flopy.utils.Mf6ListBudget(os.path.join(pf.new_d, "freyberg6.lst"))
    flx, cum = lst.get_dataframes(diff=True)
    wel_tot = flx.wel.apply(np.abs).sum()
    print(flx.wel)
    assert wel_tot < 1.0e-6, wel_tot


@pytest.mark.skip("now broken down into parts to avoid some mem issues")
def mf6_freyberg_direct_test(tmp_path):

    import numpy as np
    import pandas as pd
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)
    try:
        import flopy
    except:
        return

    org_model_ws = os.path.join('..', 'examples', 'freyberg_mf6')
    tmp_model_ws = setup_tmp(org_model_ws, tmp_path)
    dup_file = "freyberg6.wel_stress_period_data_with_dups.txt"
    shutil.copy2(os.path.join("utils", dup_file), tmp_model_ws)
    bd = Path.cwd()
    os.chdir(tmp_path)
    try:
        tmp_model_ws = tmp_model_ws.relative_to(tmp_path)
        sim = flopy.mf6.MFSimulation.load(sim_ws=str(tmp_model_ws))
        m = sim.get_model("freyberg6")
        sim.set_all_data_external()
        sim.write_simulation()

        # SETUP pest stuff...
        os_utils.run("{0} ".format("mf6"), cwd=tmp_model_ws)

        template_ws = Path(tmp_path, "new_temp_direct")
        if os.path.exists(template_ws):
            shutil.rmtree(template_ws)
        sr = m.modelgrid
        # set up PstFrom object
        pf = PstFrom(original_d=tmp_model_ws, new_d=template_ws,
                     remove_existing=True,
                     longnames=True, spatial_reference=sr,
                     zero_based=False, start_datetime="1-1-2018")
        # obs
        #   using tabular style model output
        #   (generated by pyemu.gw_utils.setup_hds_obs())
        # pf.add_observations('freyberg.hds.dat', insfile='freyberg.hds.dat.ins2',
        #                     index_cols='obsnme', use_cols='obsval', prefix='hds')


        ghb_files = [f for f in os.listdir(template_ws) if ".ghb_stress" in f and f.endswith("txt")]
        pf.add_parameters(ghb_files,par_type="grid",par_style="add",use_cols=3,par_name_base="ghbstage",
                          pargp="ghbstage",index_cols=[0,1,2],transform="none",lower_bound=-5,upper_bound=5)

        pf.add_parameters(ghb_files, par_type="grid", par_style="multiplier", use_cols=3, par_name_base="mltstage",
                          pargp="ghbstage", index_cols=[0, 1, 2], transform="log", lower_bound=0.5,
                          upper_bound=1.5)

        # Add stream flow observation
        # df = pd.read_csv(os.path.join(tmp_model_ws, "sfr.csv"), index_col=0)
        pf.add_observations("sfr.csv", insfile="sfr.csv.ins", index_cols="time",
                            use_cols=["GAGE_1","HEADWATER","TAILWATER"],ofile_sep=",")
        # Setup geostruct for spatial pars
        gr_v = pyemu.geostats.ExpVario(contribution=1.0, a=1000)
        gr_gs = pyemu.geostats.GeoStruct(variograms=gr_v, transform="log")
        pp_v = pyemu.geostats.ExpVario(contribution=1.0, a=5000)
        pp_gs = pyemu.geostats.GeoStruct(variograms=pp_v, transform="log")
        rch_temporal_gs = pyemu.geostats.GeoStruct(variograms=pyemu.geostats.ExpVario(contribution=1.0, a=60))
        pf.extra_py_imports.append('flopy')
        ib = m.dis.idomain[0].array
        tags = {"npf_k_": [0.1, 10.], "npf_k33_": [.1, 10], "sto_ss": [.1, 10], "sto_sy": [.9, 1.1],
                "rch_recharge": [.5, 1.5]}
        dts = pd.to_datetime("1-1-2018") + pd.to_timedelta(np.cumsum(sim.tdis.perioddata.array["perlen"]), unit="d")
        print(dts)
        #ib = m.dis.idomain.array[0,:,:]
        # setup from array style pars
        for tag, bnd in tags.items():
            lb, ub = bnd[0], bnd[1]
            arr_files = [f for f in os.listdir(tmp_model_ws) if tag in f and f.endswith(".txt")]
            if "rch" in tag:
                for arr_file in arr_files:
                    # indy direct grid pars for each array type file
                    recharge_files = ["recharge_1.txt","recharge_2.txt","recharge_3.txt"]
                    pf.add_parameters(filenames=arr_file, par_type="grid", par_name_base="rch_gr",
                                      pargp="rch_gr", zone_array=ib, upper_bound=1.0e-3, lower_bound=1.0e-7,
                                      par_style="direct")
                    # additional constant mults
                    kper = int(arr_file.split('.')[1].split('_')[-1]) - 1
                    pf.add_parameters(filenames=arr_file, par_type="constant",
                                      par_name_base=arr_file.split('.')[1] + "_cn",
                                      pargp="rch_const", zone_array=ib, upper_bound=ub, lower_bound=lb,
                                      geostruct=rch_temporal_gs,
                                      datetime=dts[kper])
            else:
                for arr_file in arr_files:
                    # grid mults pure and simple
                    pf.add_parameters(filenames=arr_file, par_type="grid", par_name_base=arr_file.split('.')[1] + "_gr",
                                      pargp=arr_file.split('.')[1] + "_gr", zone_array=ib, upper_bound=ub,
                                      lower_bound=lb,
                                      geostruct=gr_gs)

        # Add a variety of list style pars
        list_files = ["freyberg6.wel_stress_period_data_{0}.txt".format(t)
                      for t in range(1, m.nper + 1)]
        # make dummy versions with headers
        for fl in list_files[0:2]: # this adds a header to well file
            with open(os.path.join(template_ws, fl), 'r') as fr:
                lines = [line for line in fr]
            with open(os.path.join(template_ws, f"new_{fl}"), 'w') as fw:
                fw.write("k i j flux \n")
                for line in lines:
                    fw.write(line)

        # fl = "freyberg6.wel_stress_period_data_3.txt" # Add extra string col_id
        for fl in list_files[2:7]:
            with open(os.path.join(template_ws, fl), 'r') as fr:
                lines = [line for line in fr]
            with open(os.path.join(template_ws, f"new_{fl}"), 'w') as fw:
                fw.write("well k i j flux \n")
                for i, line in enumerate(lines):
                    fw.write(f"well{i}" + line)

        list_files.sort()
        for list_file in list_files:
            kper = int(list_file.split(".")[1].split('_')[-1]) - 1
            #add spatially constant, but temporally correlated wel flux pars
            pf.add_parameters(filenames=list_file, par_type="constant", par_name_base="twel_mlt_{0}".format(kper),
                              pargp="twel_mlt_{0}".format(kper), index_cols=[0, 1, 2], use_cols=[3],
                              upper_bound=1.5, lower_bound=0.5, datetime=dts[kper], geostruct=rch_temporal_gs)

            # add temporally indep, but spatially correlated wel flux pars
            pf.add_parameters(filenames=list_file, par_type="grid", par_name_base="wel_grid_{0}".format(kper),
                              pargp="wel_{0}".format(kper), index_cols=[0, 1, 2], use_cols=[3],
                              upper_bound=0.0, lower_bound=-1000, geostruct=gr_gs, par_style="direct",
                              transform="none")
        # Adding dummy list pars with different file structures
        list_file = "new_freyberg6.wel_stress_period_data_1.txt"  # with a header
        pf.add_parameters(filenames=list_file, par_type="grid", par_name_base='nwell_mlt',
                          pargp='nwell_mult', index_cols=['k', 'i', 'j'], use_cols='flux',
                          upper_bound=10, lower_bound=-10, geostruct=gr_gs,
                          transform="none")
        pf.add_parameters(filenames=list_file, par_type="grid", par_name_base='nwell_grid',
                          pargp='nwell', index_cols=['k', 'i', 'j'], use_cols='flux',
                          upper_bound=10, lower_bound=-10, geostruct=gr_gs, par_style="direct",
                          transform="none")
        # with skip instead
        list_file = "new_freyberg6.wel_stress_period_data_2.txt"
        pf.add_parameters(filenames=list_file, par_type="grid", par_name_base='nwell_grid',
                          pargp='nwell', index_cols=[0, 1, 2], use_cols=3,
                          upper_bound=10, lower_bound=-10, geostruct=gr_gs, par_style="direct",
                          transform="none", mfile_skip=1)

        list_file = "new_freyberg6.wel_stress_period_data_3.txt"
        pf.add_parameters(filenames=list_file, par_type="grid", par_name_base='nwell_mlt',
                          pargp='nwell_mult', index_cols=['well', 'k', 'i', 'j'], use_cols='flux',
                          upper_bound=10, lower_bound=-10, geostruct=gr_gs,
                          transform="none")
        pf.add_parameters(filenames=list_file, par_type="grid", par_name_base='nwell_grid',
                          pargp='nwell', index_cols=['well', 'k', 'i', 'j'], use_cols='flux',
                          upper_bound=10, lower_bound=-10, geostruct=gr_gs, par_style="direct",
                          transform="none")
        # with skip instead
        list_file = "new_freyberg6.wel_stress_period_data_4.txt"
        pf.add_parameters(filenames=list_file, par_type="grid",
                          par_name_base='nwell_grid', pargp='nwell',
                          index_cols=[0, 1, 2, 3],  # or... {'well': 0, 'k': 1, 'i': 2, 'j': 3},
                          use_cols=4, upper_bound=10, lower_bound=-10,
                          geostruct=gr_gs, par_style="direct", transform="none",
                          mfile_skip=1)

        list_file = "freyberg6.ghb_stress_period_data_1.txt"
        pf.add_parameters(filenames=list_file, par_type="constant", par_name_base=["ghb_stage","ghb_cond"],
                          pargp=["ghb_stage","ghb_cond"], index_cols=[0, 1, 2], use_cols=[3,4],
                          upper_bound=[35,150], lower_bound=[32,50], par_style="direct",
                          transform="none")
        pf.add_parameters(filenames=dup_file, par_type="grid", par_name_base="dups",
                          pargp="dups", index_cols=[0, 1, 2], use_cols=[3],
                          upper_bound=0.0, lower_bound=-500,par_style="direct",
                          transform="none")

        list_file = "new_freyberg6.wel_stress_period_data_5.txt"
        pf.add_parameters(filenames=list_file, par_type="grid",
                          par_name_base=['nwell5_k', 'nwell5_q'],
                          pargp='nwell5',
                          index_cols=['well', 'i',  'j'],
                          use_cols=['k', 'flux'], upper_bound=10, lower_bound=-10,
                          geostruct=gr_gs, par_style="direct", transform="none",
                          mfile_skip=0, use_rows=[3, 4])

        list_file = "new_freyberg6.wel_stress_period_data_6.txt"
        pf.add_parameters(filenames=list_file, par_type="grid",
                          par_name_base=['nwell6_k', 'nwell6_q'],
                          pargp='nwell6',
                          index_cols=['well', 'i',  'j'],
                          use_cols=['k', 'flux'], upper_bound=10, lower_bound=-10,
                          geostruct=gr_gs, par_style="direct", transform="none",
                          mfile_skip=0, use_rows=[(3, 21, 15), (3, 30, 7)])
        # use_rows should match so all should be setup 2 cols 6 rows
        assert len(pf.par_dfs[-1]) == 2*6 # should be
        list_file = "new_freyberg6.wel_stress_period_data_7.txt"
        pf.add_parameters(filenames=list_file, par_type="grid",
                          par_name_base=['nwell6_k', 'nwell6_q'],
                          pargp='nwell6',
                          index_cols=['well', 'i',  'j'],
                          use_cols=['k', 'flux'], upper_bound=10, lower_bound=-10,
                          geostruct=gr_gs, par_style="direct", transform="none",
                          mfile_skip=0,
                          use_rows=[('well2', 21, 15), ('well4', 30, 7)])
        assert len(pf.par_dfs[-1]) == 2 * 2  # should be
        # add model run command
        pf.mod_sys_cmds.append("mf6")
        print(pf.mult_files)
        print(pf.org_files)

        # build pest
        pst = pf.build_pst('freyberg.pst')
        cov = pf.build_prior(fmt="non")
        cov.to_coo(os.path.join(template_ws, "prior.jcb"))
        pst.try_parse_name_metadata()
        df = pd.read_csv(os.path.join(tmp_model_ws, "heads.csv"), index_col=0)
        pf.add_observations("heads.csv", insfile="heads.csv.ins", index_cols="time",
                            use_cols=list(df.columns.values),
                            prefix="hds", rebuild_pst=True)

        # test par mults are working

        os.chdir(pf.new_d)
        pst.write_input_files()
        pyemu.helpers.apply_list_and_array_pars(
            arr_par_file="mult2model_info.csv", chunk_len=1)
        # TODO Some checks on resultant par files...
        list_files = [f for f in os.listdir('.')
                      if f.startswith('new_') and f.endswith('txt')]
        # check on that those dummy pars compare to the model versions.
        for f in list_files:
            n_df = pd.read_csv(f, sep=r"\s+")
            o_df = pd.read_csv(f.strip('new_'), sep=r"\s+", header=None)
            o_df.columns = ['k', 'i', 'j', 'flux']
            assert np.isclose(n_df.loc[:, o_df.columns], o_df).all(), (
                "Something broke with alternative style model files"
            )
        os.chdir(tmp_path)

        num_reals = 100
        pe = pf.draw(num_reals, use_specsim=True)
        pe.to_binary(os.path.join(template_ws, "prior.jcb"))
        assert pe.shape[1] == pst.npar_adj, "{0} vs {1}".format(pe.shape[0], pst.npar_adj)
        assert pe.shape[0] == num_reals

        pst.control_data.noptmax = 0
        pst.pestpp_options["additional_ins_delimiters"] = ","

        pst.write(os.path.join(pf.new_d, "freyberg.pst"))
        pyemu.os_utils.run("{0} freyberg.pst".format(ies_exe_path), cwd=pf.new_d)

        res_file = os.path.join(pf.new_d, "freyberg.base.rei")
        assert os.path.exists(res_file), res_file
        pst.set_res(res_file)
        print(pst.phi)
        assert pst.phi < 0.1, pst.phi

        org_ghb = pd.read_csv(os.path.join(pf.new_d,"org","freyberg6.ghb_stress_period_data_1.txt"),
                              header=None,names=["l","r","c","stage","cond"])
        new_ghb = pd.read_csv(os.path.join(pf.new_d, "freyberg6.ghb_stress_period_data_1.txt"),
                              sep=r"\s+",
                              header=None, names=["l", "r", "c", "stage", "cond"])
        d = org_ghb.stage - new_ghb.stage
        print(d)
        assert d.sum() == 0, d.sum()


        # test the additive ghb stage pars
        par = pst.parameter_data
        par.loc[par.parnme.str.contains("ghbstage_inst:0"),"parval1"] = 3.0
        pst.write(os.path.join(pf.new_d, "freyberg.pst"))
        pyemu.os_utils.run("{0} freyberg.pst".format(ies_exe_path), cwd=pf.new_d)
        org_ghb = pd.read_csv(os.path.join(pf.new_d, "org", "freyberg6.ghb_stress_period_data_1.txt"),
                              header=None, names=["l", "r", "c", "stage", "cond"])
        new_ghb = pd.read_csv(os.path.join(pf.new_d, "freyberg6.ghb_stress_period_data_1.txt"),
                              sep=r"\s+",
                              header=None, names=["l", "r", "c", "stage", "cond"])
        d = (org_ghb.stage - new_ghb.stage).apply(np.abs)
        print(d)
        assert d.mean() == 3.0, d.mean()

        # check that the interaction between the direct ghb stage par and the additive ghb stage pars
        # is working
        par.loc[par.parnme.str.contains("ghb_stage"),"parval1"] -= 3.0
        pst.write(os.path.join(pf.new_d, "freyberg.pst"))
        pyemu.os_utils.run("{0} freyberg.pst".format(ies_exe_path), cwd=pf.new_d)
        org_ghb = pd.read_csv(os.path.join(tmp_model_ws,"freyberg6.ghb_stress_period_data_1.txt"),
                              header=None, names=["l", "r", "c", "stage", "cond"],sep=r"\s+")
        new_ghb = pd.read_csv(os.path.join(pf.new_d, "freyberg6.ghb_stress_period_data_1.txt"),
                              sep=r"\s+",
                              header=None, names=["l", "r", "c", "stage", "cond"])
        d = org_ghb.stage - new_ghb.stage
        print(new_ghb.stage)
        print(org_ghb.stage)
        print(d)
        assert d.sum() == 0.0, d.sum()

        # check the interaction with multiplicative ghb stage, direct ghb stage and additive ghb stage
        par.loc[par.parnme.str.contains("mltstage"), "parval1"] = 1.1
        #par.loc[par.parnme.str.contains("ghbstage_inst:0"), "parval1"] = 0.0
        #par.loc[par.parnme.str.contains("ghb_stage"), "parval1"] += 3.0
        pst.write(os.path.join(pf.new_d, "freyberg.pst"))
        pyemu.os_utils.run("{0} freyberg.pst".format(ies_exe_path), cwd=pf.new_d)
        org_ghb = pd.read_csv(os.path.join(tmp_model_ws, "freyberg6.ghb_stress_period_data_1.txt"),
                              header=None, names=["l", "r", "c", "stage", "cond"], sep=r"\s+")
        new_ghb = pd.read_csv(os.path.join(pf.new_d, "freyberg6.ghb_stress_period_data_1.txt"),
                              sep=r"\s+",
                              header=None, names=["l", "r", "c", "stage", "cond"])
        d = (org_ghb.stage * 1.1) - new_ghb.stage
        print(new_ghb.stage)
        print(org_ghb.stage)
        print(d)
        assert d.sum() == 0.0, d.sum()

        # turn direct recharge to min and direct wel to min and
        # check that the model results are consistent
        par = pst.parameter_data
        rch_par = par.loc[par.parnme.apply(
            lambda x: "pname:rch_gr" in x and "ptype:gr_pstyle:d" in x ), "parnme"]
        wel_par = par.loc[par.parnme.apply(
            lambda x: "pname:wel_grid" in x and "ptype:gr_usecol:3_pstyle:d" in x), "parnme"]
        par.loc[rch_par,"parval1"] = par.loc[rch_par, "parlbnd"]
        # this should set wells to zero since they are negative values in the control file
        par.loc[wel_par,"parval1"] = par.loc[wel_par, "parubnd"]
        pst.write(os.path.join(pf.new_d, "freyberg.pst"))
        pyemu.os_utils.run("{0} freyberg.pst".format(ies_exe_path), cwd=pf.new_d)
        lst = flopy.utils.Mf6ListBudget(os.path.join(pf.new_d, "freyberg6.lst"))
        flx, cum = lst.get_dataframes(diff=True)
        wel_tot = flx.wel.apply(np.abs).sum()
        print(flx.wel)
        assert wel_tot < 1.0e-6, wel_tot

        rch_files = [f for f in os.listdir(pf.new_d)
                     if ".rch_recharge" in f and f.endswith(".txt")]
        rch_val = par.loc[rch_par,"parval1"][0]
        i, j = par.loc[rch_par, ["i", 'j']].astype(int).values.T
        for rch_file in rch_files:
            arr = np.loadtxt(os.path.join(pf.new_d, rch_file))[i, j]
            print(rch_file, rch_val, arr.mean(), arr.max(), arr.min())
            if np.abs(arr.max() - rch_val) > 1.0e-6 or np.abs(arr.min() - rch_val) > 1.0e-6:
                raise Exception("recharge too diff")
    except Exception as e:
        os.chdir(bd)
        raise e
    os.chdir(bd)


def mf6_freyberg_varying_idomain(tmp_path):
    import numpy as np
    import pandas as pd
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)
    try:
        import flopy
    except:
        return

    org_model_ws = os.path.join('..', 'examples', 'freyberg_mf6')
    tmp_model_ws = setup_tmp(org_model_ws, tmp_path)
    bd = Path.cwd()
    os.chdir(tmp_path)
    try:
        tmp_model_ws = tmp_model_ws.relative_to(tmp_path)
        sim = flopy.mf6.MFSimulation.load(sim_ws=str(tmp_model_ws))
        # sim.set_all_data_external()
        sim.set_sim_path(str(tmp_model_ws))
        # sim.set_all_data_external()
        m = sim.get_model("freyberg6")
        sim.set_all_data_external(check_data=False)
        sim.write_simulation()

        #sim = None
        ib_file = os.path.join(tmp_model_ws,"freyberg6.dis_idomain_layer1.txt")
        arr = np.loadtxt(ib_file,dtype=np.int64)

        arr[:2,:14] = 0
        np.savetxt(ib_file,arr,fmt="%2d")
        print(arr)

        sim = flopy.mf6.MFSimulation.load(sim_ws=str(tmp_model_ws))
        m = sim.get_model("freyberg6")

        # SETUP pest stuff...
        os_utils.run("{0} ".format(mf6_exe_path), cwd=str(tmp_model_ws))



        template_ws = "new_temp"
        if os.path.exists(template_ws):
            shutil.rmtree(template_ws)

        # if os.path.exists(template_ws):
        #     shutil.rmtree(template_ws)
        # shutil.copytree(tmp_model_ws,template_ws)
        # hk_file = os.path.join(template_ws, "freyberg6.npf_k_layer1.txt")
        # hk = np.loadtxt(hk_file)
        #
        # hk[arr == 0] = 1.0e+30
        # np.savetxt(hk_file,hk,fmt="%50.45f")
        # os_utils.run("{0} ".format(mf6_exe_path), cwd=template_ws)
        # import matplotlib.pyplot as plt
        # hds1 = flopy.utils.HeadFile(os.path.join(tmp_model_ws, "freyberg6_freyberg.hds"))
        # hds2 = flopy.utils.HeadFile(os.path.join(template_ws, "freyberg6_freyberg.hds"))
        #
        # d = hds1.get_data() - hds2.get_data()
        # for dd in d:
        #     cb = plt.imshow(dd)
        #     plt.colorbar(cb)
        #     plt.show()
        # return

        # sr0 = m.sr
        # sr = pyemu.helpers.SpatialReference.from_namfile(
        #     os.path.join(tmp_model_ws, "freyberg6.nam"),
        #     delr=m.dis.delr.array, delc=m.dis.delc.array)

        sr = m.modelgrid
        # set up PstFrom object
        pf = PstFrom(original_d=tmp_model_ws, new_d=template_ws,
                     remove_existing=True,
                     longnames=True, spatial_reference=sr,
                     zero_based=False, start_datetime="1-1-2018")


        # pf.post_py_cmds.append("generic_function()")
        df = pd.read_csv(os.path.join(tmp_model_ws, "sfr.csv"), index_col=0)
        pf.add_observations("sfr.csv", insfile="sfr.csv.ins", index_cols="time", use_cols=list(df.columns.values),
                            ofile_sep=",")
        v = pyemu.geostats.ExpVario(contribution=1.0, a=1000)
        gr_gs = pyemu.geostats.GeoStruct(variograms=v)
        rch_temporal_gs = pyemu.geostats.GeoStruct(variograms=pyemu.geostats.ExpVario(contribution=1.0, a=60))
        pf.extra_py_imports.append('flopy')

        ib = {}
        for k in range(m.dis.nlay.data):
            a = m.dis.idomain.array[k,:,:].copy()
            print(a)
            ib[k] = a

        tags = {"npf_k_": [0.1, 10.,0.003,35]}#, "npf_k33_": [.1, 10], "sto_ss": [.1, 10], "sto_sy": [.9, 1.1]}
        dts = pd.to_datetime("1-1-2018") + pd.to_timedelta(np.cumsum(sim.tdis.perioddata.array["perlen"]), unit="d")
        print(dts)
        for tag, bnd in tags.items():
            lb, ub = bnd[0], bnd[1]
            ult_lb = bnd[2]
            ult_ub = bnd[3]
            arr_files = [f for f in os.listdir(tmp_model_ws) if tag in f and f.endswith(".txt")]

            for arr_file in arr_files:

                # these ult bounds are used later in an assert

                k = int(arr_file.split(".")[-2].split("layer")[1].split("_")[0]) - 1
                pf.add_parameters(filenames=arr_file, par_type="pilotpoints", par_name_base=arr_file.split('.')[1] + "_pp",
                                  pargp=arr_file.split('.')[1] + "_pp", upper_bound=ub, lower_bound=lb,
                                  geostruct=gr_gs, zone_array=ib[k],ult_lbound=ult_lb,ult_ubound=ult_ub)

        # add model run command
        pf.mod_sys_cmds.append("mf6")
        df = pd.read_csv(os.path.join(tmp_model_ws, "heads.csv"), index_col=0)
        df = pf.add_observations("heads.csv", insfile="heads.csv.ins", index_cols="time", use_cols=list(df.columns.values),
                            prefix="hds", ofile_sep=",")


        #pst = pf.build_pst('freyberg.pst')
        pf.parfile_relations.to_csv(os.path.join(pf.new_d, "mult2model_info.csv"))
        os.chdir(pf.new_d)
        df = pyemu.helpers.calc_array_par_summary_stats()
        os.chdir(tmp_path)
        pf.post_py_cmds.append("pyemu.helpers.calc_array_par_summary_stats()")
        pf.add_observations("arr_par_summary.csv",index_cols=["model_file"],use_cols=df.columns.tolist(),
                            obsgp=["arr_par_summary" for _ in df.columns],prefix=["arr_par_summary" for _ in df.columns])
        pst = pf.build_pst('freyberg.pst')
        pst.control_data.noptmax = 0
        pst.write(os.path.join(pf.new_d, "freyberg.pst"))
        pyemu.os_utils.run("{0} freyberg.pst".format(ies_exe_path), cwd=pf.new_d)

        res_file = os.path.join(pf.new_d, "freyberg.base.rei")
        assert os.path.exists(res_file), res_file
        pst.set_res(res_file)
        print(pst.phi)
        assert pst.phi < 1.0e-6

        pe = pf.draw(10,use_specsim=True)
        pe.enforce()
        pst.parameter_data.loc[:,"parval1"] = pe.loc[pe.index[0],pst.par_names]
        pst.write(os.path.join(pf.new_d, "freyberg.pst"))
        pyemu.os_utils.run("{0} freyberg.pst".format(ies_exe_path), cwd=pf.new_d)

        res_file = os.path.join(pf.new_d, "freyberg.base.rei")
        assert os.path.exists(res_file), res_file
        pst.set_res(res_file)
        print(pst.phi)

        df = pd.read_csv(os.path.join(pf.new_d,"mult2model_info.csv"), index_col=0)
        arr_pars = df.loc[df.index_cols.isna()].copy()
        model_files = arr_pars.model_file.unique()
        pst.try_parse_name_metadata()
        for model_file in model_files:
            arr = np.loadtxt(os.path.join(pf.new_d,model_file))
            clean_name = model_file.replace(".","_").replace("\\","_").replace("/","_")
            sim_val = pst.res.loc[pst.res.name.apply(lambda x: clean_name in x ),"modelled"]
            sim_val = sim_val.loc[sim_val.index.map(lambda x: "mean_model_file" in x)]
            print(model_file,sim_val,arr.mean())
    except Exception as e:
        os.chdir(bd)
        raise e
    os.chdir(bd)


def xsec_test(tmp_path):
    import numpy as np
    import pandas as pd
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)
    try:
        import flopy
    except:
        return

    org_model_ws = os.path.join('..', 'examples', 'xsec')
    tmp_model_ws = setup_tmp(org_model_ws, tmp_path)
    # SETUP pest stuff...
    nam_file = "10par_xsec.nam"
    os_utils.run("{0} {1}".format(mf_exe_path,nam_file), cwd=tmp_model_ws)
    bd = Path.cwd()
    os.chdir(tmp_path)
    try:
        tmp_model_ws = tmp_model_ws.relative_to(tmp_path)
        m = flopy.modflow.Modflow.load(nam_file,model_ws=tmp_model_ws,version="mfnwt")
        sr = m.modelgrid
        t_d = "template_xsec"
        pf = pyemu.utils.PstFrom(tmp_model_ws,t_d,remove_existing=True,spatial_reference=sr)
        pf.add_parameters("hk_Layer_1.ref",par_type="grid",par_style="direct",upper_bound=25,
                          lower_bound=0.25)
        pf.add_parameters("hk_Layer_1.ref", par_type="grid", par_style="multiplier", upper_bound=10.0,
                          lower_bound=0.1)

        hds_arr = np.loadtxt(os.path.join(t_d,"10par_xsec.hds"))
        with open(os.path.join(t_d,"10par_xsec.hds.ins"),'w')  as f:
            f.write("pif ~\n")
            for kper in range(hds_arr.shape[0]):
                f.write("l1 ")
                for j in range(hds_arr.shape[1]):
                    oname = "hds_{0}_{1}".format(kper,j)
                    f.write(" !{0}! ".format(oname))
                f.write("\n")
        pf.add_observations_from_ins(os.path.join(t_d,"10par_xsec.hds.ins"),pst_path=".")

        pf.mod_sys_cmds.append("mfnwt {0}".format(nam_file))

        pf.build_pst(os.path.join(t_d,"pest.pst"))

        pyemu.os_utils.run("{0} {1}".format(ies_exe_path,"pest.pst"),cwd=t_d)
        pst = pyemu.Pst(os.path.join(t_d,"pest.pst"))
        print(pst.phi)
        assert np.isclose(pst.phi, 0), pst.phi
    except Exception as e:
        os.chdir(bd)
        raise e
    os.chdir(bd)


def mf6_freyberg_short_direct_test(tmp_path):

    import numpy as np
    import pandas as pd
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)
    try:
        import flopy
    except:
        return

    org_model_ws = os.path.join('..', 'examples', 'freyberg_mf6')
    tmp_model_ws = setup_tmp(org_model_ws, tmp_path)

    bd = Path.cwd()
    os.chdir(tmp_path)
    try:
        tmp_model_ws = tmp_model_ws.relative_to(tmp_path)
        sim = flopy.mf6.MFSimulation.load(sim_ws=str(tmp_model_ws))
        m = sim.get_model("freyberg6")
        sim.set_all_data_external()
        sim.write_simulation()

        # SETUP pest stuff...
        os_utils.run("{0} ".format("mf6"), cwd=tmp_model_ws)

        template_ws = "new_temp_direct"
        if os.path.exists(template_ws):
            shutil.rmtree(template_ws)
        sr = m.modelgrid
        # set up PstFrom object
        pf = PstFrom(original_d=tmp_model_ws, new_d=template_ws,
                     remove_existing=True,
                     longnames=False, spatial_reference=sr,
                     zero_based=False, start_datetime="1-1-2018")
        # obs
        #   using tabular style model output
        #   (generated by pyemu.gw_utils.setup_hds_obs())
        # pf.add_observations('freyberg.hds.dat', insfile='freyberg.hds.dat.ins2',
        #                     index_cols='obsnme', use_cols='obsval', prefix='hds')

        # Add stream flow observation
        # df = pd.read_csv(os.path.join(tmp_model_ws, "sfr.csv"), index_col=0)
        pf.add_observations("sfr.csv", insfile="sfr.csv.ins", index_cols="time",
                            use_cols=["GAGE_1","HEADWATER","TAILWATER"],ofile_sep=",")
        # Setup geostruct for spatial pars
        v = pyemu.geostats.ExpVario(contribution=1.0, a=1000)
        gr_gs = pyemu.geostats.GeoStruct(variograms=v, transform="log")
        rch_temporal_gs = pyemu.geostats.GeoStruct(variograms=pyemu.geostats.ExpVario(contribution=1.0, a=60))
        pf.extra_py_imports.append('flopy')
        ib = m.dis.idomain[0].array
        tags = {
            "npf_k_": [0.1, 10.],
            "npf_k33_": [.1, 10],
            "sto_ss": [.1, 10],
            "sto_sy": [.9, 1.1],
            "rch_recharge": [.5, 1.5]
        }
        dts = pd.to_datetime("1-1-2018") + pd.to_timedelta(np.cumsum(sim.tdis.perioddata.array["perlen"]), unit="d")
        print(dts)
        # setup from array style pars
        for tag, bnd in tags.items():
            lb, ub = bnd[0], bnd[1]
            arr_files = [f for f in os.listdir(tmp_model_ws) if tag in f and f.endswith(".txt")]
            if "rch" in tag:
                for arr_file in arr_files:
                    nmebase = arr_file.split('.')[1].replace('layer','').replace('_','').replace("npf",'').replace("sto",'').replace("recharge",'')
                    # indy direct grid pars for each array type file
                    recharge_files = ["recharge_1.txt","recharge_2.txt","recharge_3.txt"]
                    pf.add_parameters(filenames=arr_file, par_type="grid", par_name_base="rchg",
                                      pargp="rch_gr", zone_array=ib, upper_bound=1.0e-3, lower_bound=1.0e-7,
                                      par_style="direct")
                    # additional constant mults
                    kper = int(arr_file.split('.')[1].split('_')[-1]) - 1
                    pf.add_parameters(filenames=arr_file, par_type="constant",
                                      par_name_base=nmebase + "cn",
                                      pargp="rch_const", zone_array=ib, upper_bound=ub, lower_bound=lb,
                                      geostruct=rch_temporal_gs,
                                      datetime=dts[kper])
            else:
                for arr_file in arr_files:
                    nmebase = arr_file.split('.')[1].replace(
                        'layer', '').replace('_','').replace("npf",'').replace(
                        "sto",'').replace("recharge",'')
                    # grid mults pure and simple
                    pf.add_parameters(filenames=arr_file, par_type="grid", par_name_base=nmebase,
                                      pargp=arr_file.split('.')[1] + "_gr", zone_array=ib, upper_bound=ub,
                                      lower_bound=lb,
                                      geostruct=gr_gs)

        # Add a variety of list style pars
        list_files = ["freyberg6.wel_stress_period_data_{0}.txt".format(t)
                      for t in range(1, m.nper + 1)]
        # make dummy versions with headers
        for fl in list_files[0:2]: # this adds a header to well file
            with open(os.path.join(template_ws, fl), 'r') as fr:
                lines = [line for line in fr]
            with open(os.path.join(template_ws, f"new_{fl}"), 'w') as fw:
                fw.write("k i j flx \n")
                for line in lines:
                    fw.write(line)

        # fl = "freyberg6.wel_stress_period_data_3.txt" # Add extra string col_id
        for fl in list_files[2:4]:
            with open(os.path.join(template_ws, fl), 'r') as fr:
                lines = [line for line in fr]
            with open(os.path.join(template_ws, f"new_{fl}"), 'w') as fw:
                fw.write("well k i j flx \n")
                for i, line in enumerate(lines):
                    fw.write(f"w{i}" + line)

        list_files.sort()
        for list_file in list_files:
            kper = int(list_file.split(".")[1].split('_')[-1]) - 1
            #add spatially constant, but temporally correlated wel flux pars
            pf.add_parameters(filenames=list_file, par_type="constant", par_name_base="wel{0}".format(kper),
                              pargp="twel_mlt_{0}".format(kper), index_cols=[0, 1, 2], use_cols=[3],
                              upper_bound=1.5, lower_bound=0.5, datetime=dts[kper], geostruct=rch_temporal_gs)

            # add temporally indep, but spatially correlated wel flux pars
            pf.add_parameters(filenames=list_file, par_type="grid", par_name_base="wel{0}".format(kper),
                              pargp="wel_{0}_direct".format(kper), index_cols=[0, 1, 2], use_cols=[3],
                              upper_bound=0.0, lower_bound=-1000, geostruct=gr_gs, par_style="direct",
                              transform="none")
        # # Adding dummy list pars with different file structures
        list_file = "new_freyberg6.wel_stress_period_data_1.txt"  # with a header
        pf.add_parameters(filenames=list_file, par_type="grid", par_name_base='nw',
                          pargp='nwell_mult', index_cols=['k', 'i', 'j'], use_cols='flx',
                          upper_bound=10, lower_bound=-10, geostruct=gr_gs,
                          transform="none")
        pf.add_parameters(filenames=list_file, par_type="grid", par_name_base='nw',
                          pargp='nwell', index_cols=['k', 'i', 'j'], use_cols='flx',
                          upper_bound=10, lower_bound=-10, geostruct=gr_gs, par_style="direct",
                          transform="none")
        # with skip instead
        list_file = "new_freyberg6.wel_stress_period_data_2.txt"
        pf.add_parameters(filenames=list_file, par_type="grid", par_name_base='nw',
                          pargp='nwell', index_cols=[0, 1, 2], use_cols=3,
                          upper_bound=10, lower_bound=-10, geostruct=gr_gs, par_style="direct",
                          transform="none", mfile_skip=1)

        list_file = "new_freyberg6.wel_stress_period_data_3.txt"
        pf.add_parameters(filenames=list_file, par_type="grid",
                          pargp='nwell_mult', index_cols=['well', 'k', 'i', 'j'], use_cols='flx',
                          upper_bound=10, lower_bound=-10, geostruct=gr_gs,
                          transform="none")
        pf.add_parameters(filenames=list_file, par_type="grid",
                          pargp='nwell', index_cols=['well', 'k', 'i', 'j'], use_cols='flx',
                          upper_bound=10, lower_bound=-10, geostruct=gr_gs, par_style="direct",
                          transform="none")
        # with skip instead
        list_file = "new_freyberg6.wel_stress_period_data_4.txt"
        pf.add_parameters(filenames=list_file, par_type="grid",
                          par_name_base='nw', pargp='nwell',
                          index_cols=[0, 1, 2, 3],  # or... {'well': 0, 'k': 1, 'i': 2, 'j': 3},
                          use_cols=4, upper_bound=10, lower_bound=-10,
                          geostruct=gr_gs, par_style="direct", transform="none",
                          mfile_skip=1)

        list_file = "freyberg6.ghb_stress_period_data_1.txt"
        pf.add_parameters(filenames=list_file, par_type="constant", par_name_base=["ghbst","ghbc"],
                          pargp=["ghb_stage","ghb_cond"], index_cols=[0, 1, 2], use_cols=[3,4],
                          upper_bound=[35,150], lower_bound=[32,50], par_style="direct",
                          transform="none")

        # add model run command
        pf.mod_sys_cmds.append("mf6")
        print(pf.mult_files)
        print(pf.org_files)

        # build pest
        pst = pf.build_pst('freyberg.pst')
        #cov = pf.build_prior(fmt="non")
        #cov.to_coo("prior.jcb")
        pst.try_parse_name_metadata()
        df = pd.read_csv(os.path.join(tmp_model_ws, "heads.csv"), index_col=0)
        pf.add_observations("heads.csv", insfile="heads.csv.ins", index_cols="time",
                            use_cols=list(df.columns.values),
                            prefix="hds", rebuild_pst=True)

        # test par mults are working

        os.chdir(pf.new_d)
        pst.write_input_files()
        pyemu.helpers.apply_list_and_array_pars(
            arr_par_file="mult2model_info.csv", chunk_len=1)

        # TODO Some checks on resultant par files...
        list_files = [f for f in os.listdir('.')
                      if f.startswith('new_') and f.endswith('txt')]
        # check on that those dummy pars compare to the model versions.
        for f in list_files:
            n_df = pd.read_csv(f, sep=r"\s+")
            o_df = pd.read_csv(f.strip('new_'), sep=r"\s+", header=None)
            o_df.columns = ['k', 'i', 'j', 'flx']
            assert np.allclose(o_df.values,
                               n_df.loc[:, o_df.columns].values,
                               rtol=1e-4), (
                f"Something broke with alternative style model file: {f}"
            )
        os.chdir(tmp_path)

        num_reals = 100
        pe = pf.draw(num_reals, use_specsim=True)
        pe.to_binary(os.path.join(template_ws, "prior.jcb"))
        assert pe.shape[1] == pst.npar_adj, "{0} vs {1}".format(pe.shape[0], pst.npar_adj)
        assert pe.shape[0] == num_reals

        pst.control_data.noptmax = 0
        pst.pestpp_options["additional_ins_delimiters"] = ","

        pst.write(os.path.join(pf.new_d, "freyberg.pst"))
        pyemu.os_utils.run("{0} freyberg.pst".format(ies_exe_path), cwd=pf.new_d)

        res_file = os.path.join(pf.new_d, "freyberg.base.rei")
        assert os.path.exists(res_file), res_file
        pst.set_res(res_file)
        print(pst.phi)
        assert np.isclose(pst.phi, 0), pst.phi


        # turn direct recharge to min and direct wel to min and
        # check that the model results are consistent
        par = pst.parameter_data
        rch_par = par.loc[(par.pname == 'rch') &
                          (par.ptype == 'gr') &
                          (par.pstyle == 'd'),
                          "parnme"]
        wel_par = par.loc[(par.pname.str.contains('wel')) &
                          (par.pstyle == 'd'),
                          "parnme"]
        par.loc[rch_par, "parval1"] = par.loc[rch_par, "parlbnd"]
        # this should set wells to zero since they are negative values in the control file
        par.loc[wel_par, "parval1"] = par.loc[wel_par, "parubnd"]
        pst.write(os.path.join(pf.new_d, "freyberg.pst"))
        pyemu.os_utils.run("{0} freyberg.pst".format(ies_exe_path), cwd=pf.new_d)
        lst = flopy.utils.Mf6ListBudget(os.path.join(pf.new_d, "freyberg6.lst"))
        flx, cum = lst.get_dataframes(diff=True)
        wel_tot = flx.wel.apply(np.abs).sum()
        print(flx.wel)
        assert np.isclose(wel_tot, 0), wel_tot

        # shortpars so not going to be able to get ij easily
        # rch_files = [f for f in os.listdir(pf.new_d)
        #              if ".rch_recharge" in f and f.endswith(".txt")]
        # rch_val = par.loc[rch_par,"parval1"][0]
        # i, j = par.loc[rch_par, ["i", 'j']].astype(int).values.T
        # for rch_file in rch_files:
        #     arr = np.loadtxt(os.path.join(pf.new_d, rch_file))[i, j]
        #     print(rch_file, rch_val, arr.mean(), arr.max(), arr.min())
        #     if np.abs(arr.max() - rch_val) > 1.0e-6 or np.abs(arr.min() - rch_val) > 1.0e-6:
        #         raise Exception("recharge too diff")
    except Exception as e:
        os.chdir(bd)
        raise e
    os.chdir(bd)


class TestPstFrom():
    """Test class for some PstFrom functionality
    """
    @classmethod
    @pytest.fixture(autouse=True)
    def setup(cls, tmp_path):
        # record the original wd
        cls.original_wd = Path().cwd()

        cls.sim_ws = Path(tmp_path, 'pst-from-small')
        external_files_folders = [cls.sim_ws / 'external',
                                  cls.sim_ws / '../external_files']
        for folder in external_files_folders:
            folder.mkdir(parents=True, exist_ok=True)

        cls.dest_ws = Path(tmp_path, 'pst-from-small-template')

        cls.sr = pyemu.helpers.SpatialReference(delr=np.ones(3),
                                            delc=np.ones(3),
                                            rotation=0,
                                            epsg=3070,
                                            xul=0.,
                                            yul=0.,
                                            units='meters',  # gis units of meters?
                                            lenuni=2  # model units of meters
                                            )
        # make some fake external data
        # array data
        cls.array_file = cls.sim_ws / 'hk.dat'
        cls.array_data = np.ones((3, 3))
        np.savetxt(cls.array_file, cls.array_data)
        # list data
        cls.list_file = cls.sim_ws / 'wel.dat'
        cls.list_data = pd.DataFrame({'#k': [1, 1, 1],
                                      'i': [2, 3, 3],
                                      'j': [2, 2, 1],
                                      'flux': [1., 10., 100.]
                                      }, columns=['#k', 'i', 'j', 'flux'])
        cls.list_data.to_csv(cls.list_file, sep=' ', index=False)

        # set up the zones
        zone_array = np.ones((3, 3))  # default of zone 1
        zone_array[2:, 2:] = 0  # position 3, 3 is not parametrized (no zone)
        #zone_array[0, :2] = 2  # 0, 0 and 0, 1 are in zone 2
        zone_array[1, 1] = 2  # 1, 1 is in zone 2
        cls.zone_array = zone_array

        # "geostatistical structure(s)"
        v = pyemu.geostats.ExpVario(contribution=1.0, a=1000)
        cls.grid_gs = pyemu.geostats.GeoStruct(variograms=v, transform='log')
        cls.tmp_path = tmp_path
        os.chdir(tmp_path)
        cls.sim_ws = cls.sim_ws.relative_to(tmp_path)
        cls.dest_ws = cls.dest_ws.relative_to(tmp_path)
        cls.pf = pyemu.utils.PstFrom(original_d=cls.sim_ws, new_d=cls.dest_ws,
                                 remove_existing=True,
                                 longnames=True, spatial_reference=cls.sr,
                                 zero_based=False, tpl_subfolder='tpl')
        yield
        os.chdir(cls.original_wd)

    def test_add_array_parameters(self):
        """test setting up array parameters with different external file
        configurations and path formats.
        """
        try:
            tag = 'hk'
            # test with different array input configurations
            array_file_input = [
                Path('hk0.dat'),  # sim_ws; just file name as Path instance
                'hk1.dat',  # sim_ws; just file name as string
                Path(self.sim_ws, 'hk2.dat'),  # sim_ws; full path as Path instance
                'external/hk3.dat',  # subfolder; relative file path as string
                Path('external/hk4.dat'),  # subfolder; relative path as Path instance
                '../external_files/hk5.dat',  # subfolder up one level
                                ]
            for i, array_file in enumerate(array_file_input):
                par_name_base = f'{tag}_{i:d}'

                # create the file
                # dest_file is the data file relative to the sim or dest ws
                dest_file = Path(array_file)
                if self.sim_ws in dest_file.parents:
                    dest_file = dest_file.relative_to(self.sim_ws)
                shutil.copy(self.array_file, Path(self.dest_ws, dest_file))

                self.pf.add_parameters(filenames=array_file, par_type='zone',
                                       zone_array=self.zone_array,
                                       par_name_base=par_name_base,  # basename for parameters that are set up
                                       pargp=f'{tag}_zone',  # Parameter group to assign pars to.
                                       )

                assert (self.dest_ws / dest_file).exists()
                assert (self.dest_ws / f'org/{dest_file.name}').exists()
                # mult file name is par_name_base + `instance` identifier + part_type
                mult_filename = f'{par_name_base}_inst0_zone.csv'
                assert (self.dest_ws / f'mult/{mult_filename}').exists()
                # for now, assume tpl file should be in main folder
                template_file = (self.pf.tpl_d / f'{mult_filename}.tpl')
                assert template_file.exists()

                # make the PEST control file
                pst = self.pf.build_pst()
                assert pst.filename == Path(self.dest_ws, 'pst-from-small.pst')
                assert pst.filename.exists()
                rel_tpl = pyemu.utils.pst_from.get_relative_filepath(self.pf.new_d, template_file)
                assert rel_tpl in pst.template_files

                # make the PEST control file (just filename)
                pst = self.pf.build_pst('junk.pst')
                assert pst.filename == Path(self.dest_ws, 'junk.pst')
                assert pst.filename.exists()

                # make the PEST control file (file path)
                pst = self.pf.build_pst(str(Path(self.dest_ws, 'junk2.pst')))
                assert pst.filename == Path(self.dest_ws, 'junk2.pst')
                assert pst.filename.exists()

                # check the mult2model info
                df = pd.read_csv(self.dest_ws / 'mult2model_info.csv')
                # org data file relative to dest_ws
                org_file = Path(df['org_file'].values[i])
                assert org_file == Path(f'org/{dest_file.name}')
                # model file relative to dest_ws
                model_file = Path(df['model_file'].values[i])
                assert model_file == dest_file
                # mult file
                mult_file = Path(df['mlt_file'].values[i])
                assert mult_file == Path(f'mult/{mult_filename}')

                # check applying the parameters (in the dest or template ws)
                os.chdir(self.dest_ws)
                # first delete the model file in the template ws
                model_file.unlink()
                # manually apply a multipler
                mult = 4
                mult_values = np.loadtxt(mult_file)
                mult_values[:] = mult
                np.savetxt(mult_file, mult_values)
                # apply the multiplier
                pyemu.helpers.apply_list_and_array_pars(arr_par_file='mult2model_info.csv')
                # model file should have been remade by apply_list_and_array_pars
                assert model_file.exists()
                result = np.loadtxt(model_file)
                # results should be the same with default multipliers of 1
                # assume details of parameterization are handled by other tests
                assert np.allclose(result, self.array_data * mult)
                # revert to original wd
                os.chdir(self.tmp_path)
        except Exception as e:
            os.chdir(self.original_wd)
            raise e

    def test_add_list_parameters(self):
        """test setting up list parameters with different external file
        configurations and path formats.
        """
        try:
            tag = 'wel'
            # test with different array input configurations
            list_file_input = [
                Path('wel0.dat'),  # sim_ws; just file name as Path instance
                'wel1.dat',  # sim_ws; just file name as string
                Path(self.sim_ws, 'wel2.dat'),  # sim_ws; full path as Path instance
                'external/wel3.dat',  # subfolder; relative file path as string
                Path('external/wel4.dat'),  # subfolder; relative path as Path instance
                '../external_files/wel5.dat',  # subfolder up one level
                                ]
            par_type = 'constant'
            for i, list_file in enumerate(list_file_input):
                par_name_base = f'{tag}_{i:d}'

                # create the file
                # dest_file is the data file relative to the sim or dest ws
                dest_file = Path(list_file)
                if self.sim_ws in dest_file.parents:
                    dest_file = dest_file.relative_to(self.sim_ws)
                shutil.copy(self.list_file, Path(self.dest_ws, dest_file))

                self.pf.add_parameters(filenames=list_file, par_type=par_type,
                                       par_name_base=par_name_base,
                                       index_cols=[0, 1, 2], use_cols=[3],
                                       pargp=f'{tag}_{i}',
                                       comment_char='#',
                                       )

                assert (self.dest_ws / dest_file).exists()
                assert (self.dest_ws / f'org/{dest_file.name}').exists()
                # mult file name is par_name_base + `instance` identifier + part_type
                mult_filename = f'{par_name_base}_inst0_{par_type}.csv'
                assert (self.dest_ws / f'mult/{mult_filename}').exists()
                # for now, assume tpl file should be in main folder
                template_file = (self.pf.tpl_d / f'{mult_filename}.tpl')
                assert template_file.exists()

                # make the PEST control file
                pst = self.pf.build_pst()
                rel_tpl = pyemu.utils.pst_from.get_relative_filepath(self.pf.new_d, template_file)
                assert rel_tpl in pst.template_files

                # check the mult2model info
                df = pd.read_csv(self.dest_ws / 'mult2model_info.csv')
                # org data file relative to dest_ws
                org_file = Path(df['org_file'].values[i])
                assert org_file == Path(f'org/{dest_file.name}')
                # model file relative to dest_ws
                model_file = Path(df['model_file'].values[i])
                assert model_file == dest_file
                # mult file
                mult_file = Path(df['mlt_file'].values[i])
                assert mult_file == Path(f'mult/{mult_filename}')

                # check applying the parameters (in the dest or template ws)
                os.chdir(self.dest_ws)
                # first delete the model file in the template ws
                model_file.unlink()
                # manually apply a multipler
                mult = 4
                mult_df = pd.read_csv(mult_file)
                # no idea why '3' is the column with multipliers and 'parval1_3' isn't
                # what is the purpose of 'parval1_3'?
                parval_col = '3'
                mult_df[parval_col] = mult
                mult_df.to_csv(mult_file, index=False)
                # apply the multiplier
                pyemu.helpers.apply_list_and_array_pars(arr_par_file='mult2model_info.csv')
                # model file should have been remade by apply_list_and_array_pars
                assert model_file.exists()
                result = pd.read_csv(model_file, sep=r'\s+')
                # results should be the same with default multipliers of 1
                # assume details of parameterization are handled by other tests
                assert np.allclose(result['flux'], self.list_data['flux'] * mult)

                # revert to original wd
                os.chdir(self.tmp_path)
            new_list = self.list_data.copy()
            new_list[['x', 'y']] = new_list[['i', 'j']].apply(
                lambda r: pd.Series(
                    [self.pf._spatial_reference.ycentergrid[r.i - 1, r.j - 1],
                     self.pf._spatial_reference.xcentergrid[r.i - 1, r.j - 1]]
                ), axis=1)
            new_list.to_csv(Path(self.dest_ws, "xylist.csv"), index=False, header=False)
            self.pf.add_parameters(filenames="xylist.csv", par_type='grid',
                                   par_name_base="xywel",
                                   index_cols={'lay': 0, 'x': 4, 'y': 5},
                                   use_cols=[3],
                                   pargp=f'xywel',
                                   geostruct=self.grid_gs,
                                   rebuild_pst=True
                                   )
            cov = self.pf.build_prior()
            x = cov.as_2d[-3:, -3:]
            assert np.count_nonzero(x - np.diag(np.diagonal(x))) == 6
            assert np.sum(x < np.diag(x)) == 6
        except Exception as e:
            os.chdir(self.original_wd)
            raise e

    def test_add_array_parameters_pps_grid(self):
        """test setting up array parameters with a list of array text
        files in a subfolder.
        """
        try:
            tag = 'hk'
            par_styles = ['multiplier', #'direct'
                          ]
            array_files = ['hk_{}_{}.dat', 'external/hk_{}_{}.dat']
            for par_style in par_styles:
                mult2model_row = 0
                for j, array_file in enumerate(array_files):

                    par_types = {'pilotpoints': 'pp',
                                 'grid': 'gr'}
                    for i, (par_type, suffix) in enumerate(par_types.items()):
                        # (re)create the file
                        dest_file = array_file.format(mult2model_row, suffix)
                        shutil.copy(self.array_file, Path(self.dest_ws, dest_file))
                        # add the parameters
                        par_name_base = f'{tag}_{suffix}'
                        self.pf.add_parameters(filenames=dest_file, par_type=par_type,
                                               zone_array=self.zone_array,
                                               par_name_base=par_name_base,
                                               pargp=f'{tag}_zone',
                                               pp_space=1, geostruct=self.grid_gs,
                                               par_style=par_style
                                               )
                        if par_type != 'pilotpoints':
                            template_file = (self.pf.tpl_d / f'{par_name_base}_inst0_grid.csv.tpl')
                            assert template_file.exists()
                        else:
                            template_file = (self.pf.tpl_d / f'{par_name_base}_inst0pp.dat.tpl')
                            assert template_file.exists()

                        # make the PEST control file
                        pst = self.pf.build_pst()
                        rel_tpl = pyemu.utils.pst_from.get_relative_filepath(self.pf.new_d, template_file)
                        assert rel_tpl in pst.template_files

                        # check the mult2model info
                        df = pd.read_csv(self.dest_ws / 'mult2model_info.csv')
                        mult_file = Path(df['mlt_file'].values[mult2model_row])

                        # check applying the parameters (in the dest or template ws)
                        os.chdir(self.dest_ws)
                        # first delete the model file in the template ws
                        model_file = df['model_file'].values[mult2model_row]
                        os.remove(model_file)
                        # manually apply a multipler
                        mult = 4
                        if par_type != "pilotpoints":
                            mult_values = np.loadtxt(mult_file)
                            mult_values[:] = mult
                            np.savetxt(mult_file, mult_values)
                        else:
                            ppdata = pp_file_to_dataframe(df['pp_file'].values[mult2model_row])
                            ppdata['parval1'] = mult
                            write_pp_file(df['pp_file'].values[mult2model_row], ppdata)
                        # apply the multiplier
                        pyemu.helpers.apply_list_and_array_pars(arr_par_file='mult2model_info.csv')
                        # model files should have been remade by apply_list_and_array_pars
                        for model_file in df['model_file']:
                            assert os.path.exists(model_file)
                            result = np.loadtxt(model_file)
                            # results should be the same with default multipliers of 1
                            # assume details of parameterization are handled by other tests

                            # not sure why zone 2 is coming back as invalid (1e30)
                            zone1 = self.zone_array == 1
                            assert np.allclose(result[zone1], self.array_data[zone1] * mult)

                        # revert to original wd
                        os.chdir(self.tmp_path)
                        mult2model_row += 1
        except Exception as e:
            os.chdir(self.original_wd)
            raise e

    def test_add_direct_array_parameters(self):
        """test setting up array parameters with a list of array text
        files in a subfolder.
        """
        try:
            tag = 'hk'
            par_styles = ['direct', #'direct'
                          ]
            array_files = ['hk_{}_{}.dat', 'external/hk_{}_{}.dat']
            for par_style in par_styles:
                mult2model_row = 0
                for j, array_file in enumerate(array_files):

                    par_types = {#'constant': 'cn',
                                 'zone': 'zn',
                                 'grid': 'gr'}
                    for i, (par_type, suffix) in enumerate(par_types.items()):
                        # (re)create the file
                        dest_file = array_file.format(mult2model_row, suffix)

                        # make a new input array file with initial values
                        arr = np.loadtxt(self.array_file)
                        parval = 8
                        arr[:] = parval
                        np.savetxt(Path(self.dest_ws, dest_file), arr)

                        # add the parameters
                        par_name_base = f'{tag}_{suffix}'
                        self.pf.add_parameters(filenames=dest_file, par_type=par_type,
                                               zone_array=self.zone_array,
                                               par_name_base=par_name_base,
                                               pargp=f'{tag}_zone',
                                               par_style=par_style
                                               )
                        template_file = (self.pf.tpl_d / f'{Path(dest_file).name}.tpl')
                        assert template_file.exists()

                        # make the PEST control file
                        pst = self.pf.build_pst()
                        rel_tpl = pyemu.utils.pst_from.get_relative_filepath(self.pf.new_d, template_file)
                        assert rel_tpl in pst.template_files

                        # check the mult2model info
                        df = pd.read_csv(self.dest_ws / 'mult2model_info.csv')

                        # check applying the parameters (in the dest or template ws)
                        os.chdir(self.dest_ws)

                        # first delete the model file that was in the template ws
                        model_file = df['model_file'].values[mult2model_row]
                        assert Path(model_file) == Path(dest_file), (f"model_file: {model_file} "
                                                         f"differs from dest_file {dest_file}")
                        os.remove(model_file)

                        # pretend that PEST created the input files
                        # values from dest_file above formed basis for parval in PEST control data
                        # PEST input file is set up as the org/ version
                        # apply_list_and_array_pars then takes the org/ version and writes model_file
                        np.savetxt(pst.input_files[mult2model_row], arr)

                        pyemu.helpers.apply_list_and_array_pars(arr_par_file='mult2model_info.csv')
                        # model files should have been remade by apply_list_and_array_pars
                        for model_file in df['model_file']:
                            assert os.path.exists(model_file)
                            result = np.loadtxt(model_file)
                            # results should be the same with default multipliers of 1
                            # assume details of parameterization are handled by other tests

                            # not sure why zone 2 is coming back as invalid (1e30)
                            zone1 = self.zone_array == 1
                            assert np.allclose(result[zone1], parval)

                        # revert to original wd
                        os.chdir(self.tmp_path)
                        mult2model_row += 1
        except Exception as e:
            os.chdir(self.original_wd)
            raise e

    def test_add_array_parameters_to_file_list(self):
        """test setting up array parameters with a list of array text
        files in a subfolder.
        """
        try:
            tag = 'r'
            array_file_input = ['external/r0.dat',
                                'external/r1.dat',
                                'external/r2.dat']
            for file in array_file_input:
                shutil.copy(self.array_file, Path(self.dest_ws, file))

            # single 2D zone array applied to each file in filesnames
            self.pf.add_parameters(filenames=array_file_input, par_type='zone',
                                   zone_array=self.zone_array,
                                   par_name_base=tag,  # basename for parameters that are set up
                                   pargp=f'{tag}_zone',  # Parameter group to assign pars to.
                                   )
            # make the PEST control file
            pst = self.pf.build_pst()
            # check the mult2model info
            df = pd.read_csv(self.dest_ws / 'mult2model_info.csv')
            mult_file = Path(df['mlt_file'].values[0])

            # check applying the parameters (in the dest or template ws)
            os.chdir(self.dest_ws)
            # first delete the model file in the template ws
            for model_file in df['model_file']:
                os.remove(model_file)
            # manually apply a multipler
            mult = 4
            mult_values = np.loadtxt(mult_file)
            mult_values[:] = mult
            np.savetxt(mult_file, mult_values)
            # apply the multiplier
            pyemu.helpers.apply_list_and_array_pars(arr_par_file='mult2model_info.csv')
            # model files should have been remade by apply_list_and_array_pars
            for model_file in df['model_file']:
                assert os.path.exists(model_file)
                result = np.loadtxt(model_file)
                # results should be the same with default multipliers of 1
                # assume details of parameterization are handled by other tests
                assert np.allclose(result, self.array_data * mult)

            # revert to original wd
            os.chdir(self.tmp_path)
        except Exception as e:
            os.chdir(self.original_wd)
            raise e

    def test_add_array_parameters_alt_inst_str_none_m(self):
        """Given a list of text file arrays, test setting up
        array parameters that can extend across multiple files,
        but have a different multiplier file for each text array.
        For example, if the same zones are present in each layer of a model, 
        but have different configurations in each layer
        (such that a different zone array is needed for each layer).
        
        Test alt_inst_str=None and par_style="multiplier"
        
        TODO: switch to pytest so that we could simply use one function 
        for this with multiple parameters
        """
        try:
            tag = 'r'
            array_file_input = ['external/r0.dat',
                                'external/r1.dat',
                                'external/r2.dat']
            for file in array_file_input:
                shutil.copy(self.array_file, Path(self.dest_ws, file))
            for array_file in array_file_input:
                self.pf.add_parameters(filenames=array_file, par_type='zone',
                                        par_style="multiplier",
                                        zone_array=self.zone_array,
                                        par_name_base=tag,  # basename for parameters that are set up
                                        pargp=f'{tag}_zone',  # Parameter group to assign pars to.
                                        alt_inst_str=None
                                        )
            pst = self.pf.build_pst()
            # the parameter data section should have
            # only 2 parameters, for zones 1 and 2
            parzones = sorted(pst.parameter_data.zone.astype(float).astype(int).tolist())
            assert parzones == [1, 2]
            assert len(pst.template_files) == 3
            assert len(self.pf.mult_files) == 3
        except Exception as e:
            os.chdir(self.original_wd)
            raise e

    def test_add_array_parameters_alt_inst_str_0_d(self):
        """Given a list of text file arrays, test setting up
        array parameters that can extend across multiple files,
        but have a different multiplier file for each text array.
        For example, if the same zones are present in each layer of a model, 
        but have different configurations in each layer
        (such that a different zone array is needed for each layer).
        
        Test alt_inst_str="" and par_style="direct"
        """
        try:
            tag = 'r'
            array_file_input = ['external/r0.dat',
                                'external/r1.dat',
                                'external/r2.dat']
            for file in array_file_input:
                shutil.copy(self.array_file, Path(self.dest_ws, file))
            # test both None and "" for alt_inst_str
            for array_file in array_file_input:
                self.pf.add_parameters(filenames=array_file, par_type='zone',
                                        par_style="direct",
                                        zone_array=self.zone_array,
                                        par_name_base=tag,  # basename for parameters that are set up
                                        pargp=f'{tag}_zone',  # Parameter group to assign pars to.
                                        alt_inst_str=""
                                        )
            pst = self.pf.build_pst()
            # the parameter data section should have
            # only 2 parameters, for zones 1 and 2
            parzones = sorted(pst.parameter_data.zone.astype(float).astype(int).tolist())
            assert parzones == [1, 2]
            assert len(pst.template_files) == 3
        except Exception as e:
            os.chdir(self.original_wd)
            raise e


def test_get_filepath():
    from pyemu.utils.pst_from import get_filepath

    input_expected = [(('folder', 'file.txt'), Path('folder/file.txt')),
                      ((Path('folder'), 'file.txt'), Path('folder/file.txt')),
                      (('folder', Path('file.txt')), Path('folder/file.txt')),
                      ((Path('folder'), Path('file.txt')), Path('folder/file.txt')),
                      ]
    for input, expected in input_expected:
        result = get_filepath(*input)
        assert result == expected


def invest():
    import os
    import pyemu

    i = pyemu.pst_utils.InstructionFile(os.path.join("new_temp","freyberg.sfo.dat.ins"))
    i.read_output_file(os.path.join("new_temp","freyberg.sfo.dat"))


def pstfrom_profile():
    import cProfile
    import numpy as np
    import pandas as pd
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)
    try:
        import flopy
    except:
        return

    org_model_ws = os.path.join("..", "examples", "freyberg_sfr_update")
    nam_file = "freyberg.nam"
    m = flopy.modflow.Modflow.load(nam_file, model_ws=org_model_ws,
                                   check=False, forgive=False,
                                   exe_name=mf_exe_path)
    flopy.modflow.ModflowRiv(m, stress_period_data={
        0: [[0, 0, 0, m.dis.top.array[0, 0], 1.0, m.dis.botm.array[0, 0, 0]],
            [0, 0, 1, m.dis.top.array[0, 1], 1.0, m.dis.botm.array[0, 0, 1]],
            [0, 0, 1, m.dis.top.array[0, 1], 1.0, m.dis.botm.array[0, 0, 1]]]})

    tmp_model_ws = "temp_pst_from"
    if os.path.exists(tmp_model_ws):
        shutil.rmtree(tmp_model_ws)
    m.external_path = "."
    m.change_model_ws(tmp_model_ws)
    m.write_input()
    print("{0} {1}".format(mf_exe_path, m.name + ".nam"), tmp_model_ws)
    os_utils.run("{0} {1}".format(mf_exe_path, m.name + ".nam"),
                 cwd=tmp_model_ws)

    template_ws = "new_temp"
    if os.path.exists(template_ws):
        shutil.rmtree(template_ws)
    # sr0 = m.sr
    sr = pyemu.helpers.SpatialReference.from_namfile(
        os.path.join(m.model_ws, m.namefile),
        delr=m.dis.delr, delc=m.dis.delc)
    # set up PstFrom object
    shutil.copy(os.path.join(org_model_ws, 'ucn.csv'),
                os.path.join(tmp_model_ws, 'ucn.csv'))
    pf = PstFrom(original_d=tmp_model_ws, new_d=template_ws,
                 remove_existing=True,
                 longnames=True, spatial_reference=sr,
                 zero_based=False)
    # obs
    pr = cProfile.Profile()
    pr.enable()
    pf.add_observations('ucn.csv', insfile=None,
                        index_cols=['t', 'k', 'i', 'j'],
                        use_cols=["ucn"], prefix=['ucn'],
                        ofile_sep=',', obsgp=['ucn'])
    pr.disable()

    # pars
    pf.add_parameters(filenames="RIV_0000.dat", par_type="grid",
                      index_cols=[0, 1, 2], use_cols=[3, 5],
                      par_name_base=["rivstage_grid", "rivbot_grid"],
                      mfile_fmt='%10d%10d%10d %15.8F %15.8F %15.8F',
                      pargp='rivbot')
    # pf.add_parameters(filenames="RIV_0000.dat", par_type="grid",
    #                   index_cols=[0, 1, 2], use_cols=4)
    # pf.add_parameters(filenames=["WEL_0000.dat", "WEL_0001.dat"],
    #                   par_type="grid", index_cols=[0, 1, 2], use_cols=3,
    #                   par_name_base="welflux_grid",
    #                   zone_array=m.bas6.ibound.array)
    # pf.add_parameters(filenames=["WEL_0000.dat"], par_type="constant",
    #                   index_cols=[0, 1, 2], use_cols=3,
    #                   par_name_base=["flux_const"])
    # pf.add_parameters(filenames="rech_1.ref", par_type="grid",
    #                   zone_array=m.bas6.ibound[0].array,
    #                   par_name_base="rch_datetime:1-1-1970")
    # pf.add_parameters(filenames=["rech_1.ref", "rech_2.ref"],
    #                   par_type="zone", zone_array=m.bas6.ibound[0].array)
    # pf.add_parameters(filenames="rech_1.ref", par_type="pilot_point",
    #                   zone_array=m.bas6.ibound[0].array,
    #                   par_name_base="rch_datetime:1-1-1970", pp_space=4)
    # pf.add_parameters(filenames="rech_1.ref", par_type="pilot_point",
    #                   zone_array=m.bas6.ibound[0].array,
    #                   par_name_base="rch_datetime:1-1-1970", pp_space=1,
    #                   ult_ubound=100, ult_lbound=0.0)
    #
    # # add model run command
    # pf.mod_sys_cmds.append("{0} {1}".format(mf_exe_name, m.name + ".nam"))
    # print(pf.mult_files)
    # print(pf.org_files)
    #
    # # build pest
    # pst = pf.build_pst('freyberg.pst')
    #
    # # check mult files are in pst input files
    # csv = os.path.join(template_ws, "mult2model_info.csv")
    # df = pd.read_csv(csv, index_col=0)
    # pst_input_files = {str(f) for f in pst.input_files}
    # mults_not_linked_to_pst = ((set(df.mlt_file.unique()) -
    #                             pst_input_files) -
    #                            set(df.loc[df.pp_file.notna()].mlt_file))
    # assert len(mults_not_linked_to_pst) == 0, print(mults_not_linked_to_pst)
    #
    # pst.write_input_files(pst_path=pf.new_d)
    # # test par mults are working
    # b_d = os.getcwd()
    # os.chdir(pf.new_d)
    # try:
    #     pyemu.helpers.apply_list_and_array_pars(
    #         arr_par_file="mult2model_info.csv")
    # except Exception as e:
    #     os.chdir(b_d)
    #     raise Exception(str(e))
    # os.chdir(b_d)
    #
    # pst.control_data.noptmax = 0
    # pst.write(os.path.join(pf.new_d, "freyberg.pst"))
    # pyemu.os_utils.run("{0} freyberg.pst".format(ies_exe_path), cwd=pf.new_d)
    #
    # res_file = os.path.join(pf.new_d, "freyberg.base.rei")
    # assert os.path.exists(res_file), res_file
    # pst.set_res(res_file)
    # print(pst.phi)
    # assert pst.phi < 1.0e-5, pst.phi
    pr.print_stats(sort="cumtime")


def mf6_freyberg_arr_obs_and_headerless_test(tmp_path):
    import numpy as np
    import pandas as pd
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)
    try:
        import flopy
    except:
        return

    org_model_ws = os.path.join('..', 'examples', 'freyberg_mf6')
    tmp_model_ws = setup_tmp(org_model_ws, tmp_path)
    bd = Path.cwd()
    os.chdir(tmp_path)
    try:
        tmp_model_ws = tmp_model_ws.relative_to(tmp_path)
        sim = flopy.mf6.MFSimulation.load(sim_ws=str(tmp_model_ws))
        m = sim.get_model("freyberg6")
        sim.set_all_data_external(check_data=False)
        sim.write_simulation()

        # to by pass the issues with flopy
        # shutil.copytree(org_model_ws,tmp_model_ws)
        # sim = flopy.mf6.MFSimulation.load(sim_ws=org_model_ws)
        # m = sim.get_model("freyberg6")

        # SETUP pest stuff...
        os_utils.run("{0} ".format(mf6_exe_path), cwd=tmp_model_ws)

        template_ws = "new_temp"
        if os.path.exists(template_ws):
            shutil.rmtree(template_ws)
        # sr0 = m.sr
        # sr = pyemu.helpers.SpatialReference.from_namfile(
        #     os.path.join(tmp_model_ws, "freyberg6.nam"),
        #     delr=m.dis.delr.array, delc=m.dis.delc.array)
        sr = m.modelgrid
        # set up PstFrom object
        pf = PstFrom(original_d=tmp_model_ws, new_d=template_ws,
                     remove_existing=True,
                     longnames=True, spatial_reference=sr,
                     zero_based=False, start_datetime="1-1-2018")

        list_file = "freyberg6.wel_stress_period_data_1.txt"
        df = pd.read_csv(os.path.join(template_ws, list_file), header=None, sep=r'\s+')
        df.loc[:,4] = 4
        df.loc[:,5] = 5
        df.to_csv(os.path.join(template_ws,list_file),sep=" ",index=False,header=False)
        pf.add_observations(list_file, index_cols=[0, 1, 2], use_cols=[3,5], ofile_skip=0, includes_header=False,
                            prefix="welobs")

        with open(os.path.join(template_ws,"badlistcall.txt"), "w") as fp:
            fp.write("this is actually a header\n")
            fp.write("entry 0 10 100.4 2\n")
            fp.write("entry 1 10 2.4 5.0")
        pf.add_observations("badlistcall.txt", index_cols=[0, 1], use_cols=[3, 4],
                            ofile_skip=0, includes_header=False,
                            prefix="badlistcall")


        v = pyemu.geostats.ExpVario(contribution=1.0, a=1000)
        gr_gs = pyemu.geostats.GeoStruct(variograms=v)
        rch_temporal_gs = pyemu.geostats.GeoStruct(variograms=pyemu.geostats.ExpVario(contribution=1.0, a=60))
        pf.extra_py_imports.append('flopy')
        ib = m.dis.idomain[0].array
        tags = {"npf_k_": [0.1, 10.]}
        dts = pd.to_datetime("1-1-2018") + pd.to_timedelta(np.cumsum(sim.tdis.perioddata.array["perlen"]), unit="d")
        print(dts)
        arr_dict = {}
        for tag, bnd in tags.items():
            lb, ub = bnd[0], bnd[1]
            arr_files = [f for f in os.listdir(tmp_model_ws) if tag in f and f.endswith(".txt")]
            for arr_file in arr_files:
                #pf.add_parameters(filenames=arr_file, par_type="grid", par_name_base=arr_file.split('.')[1] + "_gr",
                #                  pargp=arr_file.split('.')[1] + "_gr", zone_array=ib, upper_bound=ub, lower_bound=lb,
                #                  geostruct=gr_gs)
                pf.add_parameters(filenames=arr_file, par_type="constant", par_name_base=arr_file.split('.')[1] + "_cn",
                                  pargp=arr_file.split('.')[1] + "_cn", zone_array=ib, upper_bound=ub, lower_bound=lb,
                                  transform="fixed")
                pf.add_parameters(filenames=arr_file, par_type="constant", par_name_base=arr_file.split('.')[1] + "_cn",
                                  pargp=arr_file.split('.')[1] + "_cn", zone_array=ib, upper_bound=ub, lower_bound=lb,
                                  transform="log")


                pf.add_observations(arr_file,zone_array=ib)
                arr_dict[arr_file] = np.loadtxt(pf.new_d / arr_file)

        # add model run command
        pf.mod_sys_cmds.append("mf6")
        print(pf.mult_files)
        print(pf.org_files)

        # build pest
        pst = pf.build_pst('freyberg.pst')
        pe = pf.draw(100,use_specsim=True)
        cov = pf.build_prior()
        scnames = set(cov.row_names)
        print(pst.npar_adj,pst.npar,pe.shape)
        par = pst.parameter_data
        fpar = set(par.loc[par.partrans=="fixed","parnme"].tolist())
        spe = set(list(pe.columns))
        assert len(fpar.intersection(spe)) == 0,str(fpar.intersection(spe))
        assert len(fpar.intersection(scnames)) == 0, str(fpar.intersection(scnames))
        pst.try_parse_name_metadata()
        obs = pst.observation_data
        for fname,arr in arr_dict.items():

            fobs = obs.loc[obs.obsnme.str.contains(Path(fname).stem), :]
            #print(fobs)
            fobs = fobs.astype({c: int for c in ['i', 'j']})

            pval = fobs.loc[fobs.apply(lambda x: x.i==3 and x.j==1,axis=1),"obsval"]
            assert len(pval) == 1
            pval = pval.iloc[0]
            aval = arr[3,1]
            print(fname,pval,aval)
            assert pval == aval,"{0},{1},{2}".format(fname,pval,aval)

        df = pd.read_csv(os.path.join(template_ws, list_file),
                         header=None, sep=r'\s+')
        print(df)
        wobs = obs.loc[obs.obsnme.str.contains("welobs"),:]
        print(wobs)
        fvals = df.iloc[:,3]
        pvals = wobs.loc[:,"obsval"].iloc[:df.shape[0]]
        d = fvals.values - pvals.values
        print(d)
        assert d.sum() == 0
        fvals = df.iloc[:, 5]
        pvals = wobs.loc[:, "obsval"].iloc[df.shape[0]:]
        d = fvals.values - pvals.values
        print(d)
        assert d.sum() == 0
    except Exception as e:
        os.chdir(bd)
        raise e
    os.chdir(bd)


def mf6_freyberg_pp_locs_test(tmp_path):
    import numpy as np
    import pandas as pd
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)
    try:
        import flopy
    except:
        return

    org_model_ws = os.path.join('..', 'examples', 'freyberg_mf6')
    tmp_model_ws = setup_tmp(org_model_ws, tmp_path)

    bd = Path.cwd()
    os.chdir(tmp_path)
    try:
        tmp_model_ws = tmp_model_ws.relative_to(tmp_path)
        sim = flopy.mf6.MFSimulation.load(sim_ws=str(tmp_model_ws))
        m = sim.get_model("freyberg6")
        sim.set_all_data_external(check_data=False)
        sim.write_simulation()

        # SETUP pest stuff...
        os_utils.run("{0} ".format(mf6_exe_path), cwd=tmp_model_ws)

        template_ws = "new_temp"
        if os.path.exists(template_ws):
            shutil.rmtree(template_ws)
        sr = m.modelgrid
        # set up PstFrom object
        pf = PstFrom(original_d=tmp_model_ws, new_d=template_ws,
                     remove_existing=True,
                     longnames=True, spatial_reference=sr,
                     zero_based=False, start_datetime="1-1-2018",
                     chunk_len=1)

        # pf.post_py_cmds.append("generic_function()")
        df = pd.read_csv(os.path.join(tmp_model_ws, "sfr.csv"), index_col=0)
        pf.add_observations("sfr.csv", insfile="sfr.csv.ins", index_cols="time", use_cols=list(df.columns.values))
        v = pyemu.geostats.ExpVario(contribution=1.0, a=5000)
        pp_gs = pyemu.geostats.GeoStruct(variograms=v)
        pf.extra_py_imports.append('flopy')
        ib = m.dis.idomain[0].array
        tags = {"npf_k_": [0.1, 10.]}#, "npf_k33_": [.1, 10], "sto_ss": [.1, 10], "sto_sy": [.9, 1.1],
        #         "rch_recharge": [.5, 1.5]}
        dts = pd.to_datetime("1-1-2018") + pd.to_timedelta(np.cumsum(sim.tdis.perioddata.array["perlen"]), unit="d")
        print(dts)

        xmn = m.modelgrid.xvertices.min()
        xmx = m.modelgrid.xvertices.max()
        ymn = m.modelgrid.yvertices.min()
        ymx = m.modelgrid.yvertices.max()

        numpp = 20
        xvals = np.random.uniform(xmn,xmx,numpp)
        yvals = np.random.uniform(ymn, ymx, numpp)
        pp_locs = pd.DataFrame({"x":xvals,"y":yvals})
        pp_locs.loc[:,"zone"] = 1
        pp_locs.loc[:,"name"] = ["pp_{0}".format(i) for i in range(numpp)]
        pp_locs.loc[:,"parval1"] = 1.0

        pyemu.pp_utils.write_pp_shapfile(pp_locs,os.path.join(template_ws,"pp_locs.shp"))
        df = pyemu.pp_utils.pilot_points_from_shapefile(os.path.join(template_ws,"pp_locs.shp"))

        #pp_locs = pyemu.pp_utils.setup_pilotpoints_grid(sr=sr,prefix_dict={0:"pps_1"})
        #pp_locs = pp_locs.loc[:,["name","x","y","zone","parval1"]]
        pp_locs.to_csv(os.path.join(template_ws,"pp.csv"))
        pyemu.pp_utils.write_pp_file(os.path.join(template_ws,"pp_file.dat"),pp_locs)
        pp_container = ["pp_file.dat","pp.csv","pp_locs.shp"]

        for tag, bnd in tags.items():
            lb, ub = bnd[0], bnd[1]
            arr_files = [f for f in os.listdir(tmp_model_ws) if tag in f and f.endswith(".txt")]
            if "rch" in tag:
                pass
                # pf.add_parameters(filenames=arr_files, par_type="grid", par_name_base="rch_gr",
                #                   pargp="rch_gr", zone_array=ib, upper_bound=ub, lower_bound=lb,
                #                   geostruct=gr_gs)
                # for arr_file in arr_files:
                #     kper = int(arr_file.split('.')[1].split('_')[-1]) - 1
                #     pf.add_parameters(filenames=arr_file, par_type="constant", par_name_base=arr_file.split('.')[1] + "_cn",
                #                       pargp="rch_const", zone_array=ib, upper_bound=ub, lower_bound=lb,
                #                       geostruct=rch_temporal_gs,
                #                       datetime=dts[kper])
            else:
                for i,arr_file in enumerate(arr_files):
                    if i < len(pp_container):
                        pp_opt = pp_container[i]
                    else:
                        pp_opt = pp_locs
                    pf.add_parameters(filenames=arr_file, par_type="pilotpoints",
                                      par_name_base=arr_file.split('.')[1] + "_pp",
                                      pargp=arr_file.split('.')[1] + "_pp", zone_array=ib,
                                      upper_bound=ub, lower_bound=lb,pp_space=pp_opt)

        # add model run command
        pf.mod_sys_cmds.append("mf6")
        print(pf.mult_files)
        print(pf.org_files)

        # build pest
        pst = pf.build_pst('freyberg.pst')

        num_reals = 10
        pe = pf.draw(num_reals, use_specsim=True)
        pe.to_binary(os.path.join(template_ws, "prior.jcb"))

        pst.parameter_data.loc[:,"partrans"] = "fixed"
        pst.parameter_data.loc[::10, "partrans"] = "log"
        pst.control_data.noptmax = -1
        pst.write(os.path.join(template_ws,"freyberg.pst"))

        #pyemu.os_utils.run("{0} freyberg.pst".format("pestpp-glm"),cwd=template_ws)
        m_d = "master_glm"
        port = _get_port()
        print(f"Running ies on port: {port}")
        pyemu.os_utils.start_workers(template_ws,pp_exe_path,"freyberg.pst",num_workers=5,
                                     worker_root=tmp_path,
                                     master_dir=m_d, port=port)

        sen_df = pd.read_csv(os.path.join(m_d,"freyberg.isen"),index_col=0).loc[:,pst.adj_par_names]
        print(sen_df.T)
        mn = sen_df.values.min()
        print(mn)
        assert mn > 0.0
    except Exception as e:
        os.chdir(bd)
        raise e
    os.chdir(bd)


@pytest.mark.xfail
def usg_freyberg_test(tmp_path):
    import numpy as np
    import pandas as pd
    import flopy
    import pyemu
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)

    #path to org model files
    org_model_ws = os.path.join('..', 'examples', 'freyberg_usg')
    tmp_model_ws = setup_tmp(org_model_ws, tmp_path)

    bd = Path.cwd()
    os.chdir(tmp_path)
    try:
        tmp_model_ws = tmp_model_ws.relative_to(tmp_path)
        # flopy is not liking the rch package in unstruct, so allow it to fail and keep going...
        m = flopy.mfusg.MfUsg.load("freyberg.usg.nam", model_ws=tmp_model_ws,
                                   verbose=True, forgive=True, check=False)

        #convert to all open/close
        m.external_path = "."
        m.write_input()

        nam_file = os.path.join(tmp_model_ws,"freyberg.usg.nam")

        #make sure the model runs in the new dir with all external formats
        pyemu.os_utils.run("mfusg freyberg.usg.nam", cwd=tmp_model_ws)

        # for usg, we need to do some trickery to support the unstructured by layers concept
        # this is just for array-based parameters, list-based pars are g2g because they have an index
        gsf = pyemu.gw_utils.GsfReader(os.path.join(tmp_model_ws,"freyberg.usg.gsf"))
        df = gsf.get_node_data()
        df.loc[:,"xy"] = df.apply(lambda x: (x.x, x.y),axis=1)
        # these need to be zero based since they are with zero-based array indices later...
        df.loc[:,"node"] -= 1
        # process each layer
        layers = df.layer.unique()
        layers.sort()
        sr_dict_by_layer = {}
        for layer in layers:
            df_lay = df.loc[df.layer==layer,:].copy()
            df_lay.sort_values(by="node")
            #substract off the min node number so that each layers node dict starts at zero
            df_lay.loc[:,"node"] = df_lay.node - df_lay.node.min()
            print(df_lay)
            srd = {n:xy for n,xy in zip(df_lay.node.values,df_lay.xy.values)}
            sr_dict_by_layer[layer] = srd

        # gen up a fake zone array
        zone_array_k0 = np.ones((1, len(sr_dict_by_layer[1])))
        zone_array_k0[:, 200:420] = 2
        zone_array_k0[:, 600:1000] = 3

        zone_array_k2 = np.ones((1, len(sr_dict_by_layer[3])))
        zone_array_k2[:, 200:420] = 2
        zone_array_k2[:, 500:1000:3] = 3
        zone_array_k2[:,:100] = 0

        #gen up some fake pp locs
        np.random.seed(pyemu.en.SEED)
        num_pp = 20
        data = {"name":[],"x":[],"y":[],"zone":[]}
        visited = set()
        for i in range(num_pp):
            while True:
                idx = np.random.randint(0,len(sr_dict_by_layer[1]))
                if idx  not in visited:
                    break
            x,y = sr_dict_by_layer[1][idx]
            data["name"].append("pp_{0}".format(i))
            data["x"].append(x)
            data["y"].append(y)
            data["zone"].append(zone_array_k2[0,idx])
            visited.add(idx)
        # harded coded to get a zone 3 pp
        idx = 500
        assert zone_array_k2[0,idx] == 3,zone_array_k2[0,idx]

        x, y = sr_dict_by_layer[1][idx]
        data["name"].append("pp_{0}".format(i+1))
        data["x"].append(x)
        data["y"].append(y)
        data["zone"].append(zone_array_k2[0, idx])
        pp_df = pd.DataFrame(data=data,index=data["name"])

        # a geostruct that describes spatial continuity for properties
        # this is used for all props and for both grid and pilot point
        # pars cause Im lazy...
        v = pyemu.geostats.ExpVario(contribution=1.0,a=500)
        gs = pyemu.geostats.GeoStruct(variograms=v)

        # we pass the full listing of node coord info to the constructor for use
        # with list-type parameters
        template_d = Path(tmp_path, "template")
        pf = pyemu.utils.PstFrom(tmp_model_ws,template_d,longnames=True,remove_existing=True,
                                 zero_based=False,spatial_reference=gsf.get_node_coordinates(zero_based=True))

        pf.add_parameters("hk_Layer_3.ref", par_type="pilotpoints",
                          par_name_base="hk3_pp", pp_space=pp_df,
                          geostruct=gs, spatial_reference=sr_dict_by_layer[3],
                          upper_bound=2.0, lower_bound=0.5,
                          zone_array=zone_array_k2)

        # we pass layer specific sr dict for each "array" type that is spatially distributed
        pf.add_parameters("hk_Layer_1.ref",par_type="grid",par_name_base="hk1_Gr",geostruct=gs,
                          spatial_reference=sr_dict_by_layer[1],
                          upper_bound=2.0,lower_bound=0.5)
        pf.add_parameters("sy_Layer_1.ref", par_type="zone", par_name_base="sy1_zn",zone_array=zone_array_k0,
                          upper_bound=1.5,lower_bound=0.5,ult_ubound=0.35)



        # add a multiplier par for each well for each stress period
        wel_files = [f for f in os.listdir(tmp_model_ws) if f.lower().startswith("wel_") and f.lower().endswith(".dat")]
        for wel_file in wel_files:
            pf.add_parameters(wel_file,par_type="grid",par_name_base=wel_file.lower().split('.')[0],index_cols=[0],use_cols=[1],
                              geostruct=gs,lower_bound=0.5,upper_bound=1.5)

        # add pest "observations" for each active node for each stress period
        hds_runline, df = pyemu.gw_utils.setup_hds_obs(
            os.path.join(pf.new_d, "freyberg.usg.hds"), kperk_pairs=None,
            prefix="hds", include_path=False, text="headu", skip=-1.0e+30)
        pyemu.gw_utils.apply_hds_obs(os.path.join(pf.new_d, "freyberg.usg.hds"),
                                     precision='single', text='headu')
        pf.add_observations_from_ins(os.path.join(pf.new_d, "freyberg.usg.hds.dat.ins"), pst_path=".")
        pf.post_py_cmds.append(hds_runline)

        # the command the run the model
        pf.mod_sys_cmds.append("mfusg freyberg.usg.nam")

        #build the control file and draw the prior par ensemble
        pf.build_pst()
        pst = pf.pst
        par = pst.parameter_data

        gr_hk_pars = par.loc[par.parnme.str.contains("hk1_gr"),"parnme"]
        pf.pst.parameter_data.loc[gr_hk_pars,"parubnd"] = np.random.random(gr_hk_pars.shape[0]) * 5
        pf.pst.parameter_data.loc[gr_hk_pars, "parlbnd"] = np.random.random(gr_hk_pars.shape[0]) * 0.2
        pe = pf.draw(num_reals=100)
        pe.enforce()
        pe.to_csv(os.path.join(pf.new_d,"prior.csv"))
        cov = pf.build_prior(filename=None)
        #make sure the prior cov has off diagonals
        cov = pf.build_prior(sigma_range=6)
        covx = cov.x.copy()
        covx[np.abs(covx)>1.0e-7] = 1.0
        assert covx.sum() > pf.pst.npar_adj + 1
        dcov = pyemu.Cov.from_parameter_data(pf.pst,sigma_range=6)
        dcov = dcov.get(cov.row_names)
        diag = np.diag(cov.x)
        diff = np.abs(diag.flatten() - dcov.x.flatten())
        print(diag)
        print(dcov.x)
        print(diff)
        print(diff.max())
        assert np.isclose(diff.max(), 0), diff.close()

        # test that the arr hds obs process is working
        pyemu.gw_utils.apply_hds_obs(os.path.join(pf.new_d, "freyberg.usg.hds"),
                                     precision='single', text='headu')

        # run the full process once using the initial par values in the control file
        # since we are using only multipliers, the initial values are all 1's so
        # the phi should be pretty close to zero
        pf.pst.control_data.noptmax = 0
        pf.pst.write(os.path.join(pf.new_d,"freyberg.usg.pst"),version=2)
        pyemu.os_utils.run("{0} freyberg.usg.pst".format(ies_exe_path),cwd=pf.new_d)
        pst = pyemu.Pst(os.path.join(pf.new_d,"freyberg.usg.pst"))
        assert np.isclose(pst.phi, 0), pst.phi

        #make sure the processed model input arrays are veru similar to the org arrays (again 1s for mults)
        for arr_file in ["hk_Layer_1.ref","hk_Layer_3.ref"]:
            in_arr = np.loadtxt(os.path.join(pf.new_d,arr_file))
            org_arr = np.loadtxt(os.path.join(pf.new_d,"org",arr_file))
            d = np.abs(in_arr - org_arr)
            print(d.sum())
            assert np.isclose(d.sum(), 0), arr_file


        # now run a random realization from the prior par en and make sure things have changed
        pst.parameter_data.loc[pe.columns,"parval1"] = pe.iloc[0,:].values
        pst.write(os.path.join(pf.new_d, "freyberg.usg.pst"), version=2)
        pyemu.os_utils.run("{0} freyberg.usg.pst".format(ies_exe_path), cwd=pf.new_d)

        pst = pyemu.Pst(os.path.join(pf.new_d, "freyberg.usg.pst"))
        assert pst.phi > 1.0e-3, pst.phi

        for arr_file in ["hk_Layer_1.ref", "hk_Layer_3.ref"]:
            in_arr = np.loadtxt(os.path.join(pf.new_d, arr_file))
            org_arr = np.loadtxt(os.path.join(pf.new_d, "org", arr_file))
            d = np.abs(in_arr - org_arr)
            print(d.sum())
            assert d.sum() > 1.0e-3, arr_file

        # check that the pilot point process is respecting the zone array
        par = pst.parameter_data
        pp_par = par.loc[par.parnme.str.contains("pp"),:]
        pst.parameter_data.loc[pp_par.parnme,"parval1"] = pp_par.zone.apply(np.float64)
        pst.control_data.noptmax = 0
        pst.write(os.path.join(pf.new_d,"freyberg.usg.pst"),version=2)
        #pst.write_input_files(pf.new_d)
        pyemu.os_utils.run("{0} freyberg.usg.pst".format(ies_exe_path), cwd=pf.new_d)
        arr = np.loadtxt(os.path.join(pf.new_d,"mult","hk3_pp_inst0_pilotpoints.csv"))
        arr[zone_array_k2[0,:]==0] = 0
        d = np.abs(arr - zone_array_k2)
        print(d)
        print(d.sum())
        assert d.sum() == 0.0,d.sum()
    except Exception as e:
        os.chdir(bd)
        raise e
    os.chdir(bd)


def mf6_add_various_obs_test(tmp_path):
    import flopy
    org_model_ws = os.path.join('..', 'examples', 'freyberg_mf6')
    tmp_model_ws = setup_tmp(org_model_ws, tmp_path)

    bd = Path.cwd()
    os.chdir(tmp_path)
    try:
        tmp_model_ws = tmp_model_ws.relative_to(tmp_path)

        sim = flopy.mf6.MFSimulation.load(sim_ws=str(tmp_model_ws))
        m = sim.get_model("freyberg6")
        sim.set_all_data_external(check_data=False)
        sim.write_simulation()

        # SETUP pest stuff...
        os_utils.run("{0} ".format(mf6_exe_path), cwd=tmp_model_ws)

        template_ws = "new_temp"
        if os.path.exists(template_ws):
            shutil.rmtree(template_ws)
        sr = m.modelgrid
        # set up PstFrom object
        pf = PstFrom(original_d=tmp_model_ws, new_d=template_ws,
                     remove_existing=True,
                     longnames=True, spatial_reference=sr,
                     zero_based=False, start_datetime="1-1-2018",
                     chunk_len=1)

        # blind obs add
        pf.add_observations("sfr.csv", insfile="sfr.csv.ins", index_cols='time',
                            ofile_sep=',')
        pf.add_observations("heads.csv", index_cols=0, obsgp='hds')
        pf.add_observations("freyberg6.npf_k_layer1.txt",
                            obsgp='hk1', zone_array=m.dis.idomain.array[0])
        pf.add_observations("freyberg6.npf_k_layer2.txt",
                            zone_array=m.dis.idomain.array[0],
                            prefix='lay2k')
        pf.add_observations("freyberg6.npf_k_layer3.txt",
                            zone_array=m.dis.idomain.array[0])

        linelen = 10000
        _add_big_obsffile(pf, profile=True, nchar=linelen)

        # TODO more variations on the theme
        # add single par so we can run
        pf.add_parameters(["freyberg6.npf_k_layer1.txt",
                           "freyberg6.npf_k_layer2.txt",
                           "freyberg6.npf_k_layer3.txt"],par_type='constant')
        pf.mod_sys_cmds.append("mf6")
        pf.add_py_function(
            __file__,
            f"_add_big_obsffile('.', profile=False, nchar={linelen})",
            is_pre_cmd=False)
        pst = pf.build_pst('freyberg.pst', version=2)
        # pst.write(os.path.join(pf.new_d, "freyberg.usg.pst"), version=2)
        pyemu.os_utils.run("{0} {1}".format(ies_exe_path, pst.filename.name),
                           cwd=pf.new_d)
    except Exception as e:
        os.chdir(bd)
        raise e
    os.chdir(bd)


def _add_big_obsffile(pf, profile=False, nchar=50000):
    if isinstance(pf, str):
        pstfrom_add = False
        wd = pf
    else:
        pstfrom_add = True
        wd = pf.new_d
    np.random.seed(314)
    df = pd.DataFrame(np.random.random([10, nchar]),
                      columns=[hex(c) for c in range(nchar)])
    df.index.name = 'time'
    df.to_csv(os.path.join(wd, 'bigobseg.csv'))

    if pstfrom_add:
        if profile:
            import cProfile
            pr = cProfile.Profile()
            pr.enable()
            pf.add_observations('bigobseg.csv', index_cols='time')
            pr.disable()
            pr.print_stats(sort="cumtime")
        else:
            pf.add_observations('bigobseg.csv', index_cols='time')


def mf6_subdir_test(tmp_path):
    import numpy as np
    import pandas as pd
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)
    try:
        import flopy
    except:
        return

    org_model_ws = os.path.join('..', 'examples', 'freyberg_mf6')
    sd = "sub_dir"
    tmp_model_ws = setup_tmp(org_model_ws, tmp_path, sub=sd)

    bd = Path.cwd()
    os.chdir(tmp_path)
    try:
        tmp2_ws = tmp_model_ws.relative_to(tmp_path)
        tmp_model_ws = tmp2_ws.parents[tmp2_ws.parts.index(sd) - 1]
        sim = flopy.mf6.MFSimulation.load(sim_ws=str(tmp2_ws))
        m = sim.get_model("freyberg6")
        sim.set_all_data_external(check_data=False)
        sim.write_simulation()

        # SETUP pest stuff...
        if bin_path == '':
            exe = mf6_exe_path  # bit of flexibility for local/server run
        else:
            exe = os.path.join('..', mf6_exe_path)
        os_utils.run("{0} ".format(exe), cwd=tmp2_ws)
        # call generic once so that the output file exists
        df = generic_function(tmp2_ws)
        template_ws = "new_temp"
        if os.path.exists(template_ws):
            shutil.rmtree(template_ws)
        # sr0 = m.sr
        # sr = pyemu.helpers.SpatialReference.from_namfile(
        #     os.path.join(tmp_model_ws, "freyberg6.nam"),
        #     delr=m.dis.delr.array, delc=m.dis.delc.array)
        sr = m.modelgrid
        # set up PstFrom object
        pf = PstFrom(original_d=tmp_model_ws, new_d=template_ws,
                     remove_existing=True,
                     longnames=True, spatial_reference=sr,
                     zero_based=False,start_datetime="1-1-2018",
                     chunk_len=1)
        # obs
        #   using tabular style model output
        #   (generated by pyemu.gw_utils.setup_hds_obs())
        # pf.add_observations('freyberg.hds.dat', insfile='freyberg.hds.dat.ins2',
        #                     index_cols='obsnme', use_cols='obsval', prefix='hds')

        # add the values in generic to the ctl file
        pf.add_observations(
            os.path.join(sd, "generic.csv"),
            insfile="generic.csv.ins",
            index_cols=["datetime", "index_2"],
            use_cols=["simval1", "simval2"]
        )
        # add the function call to make generic to the forward run script
        pf.add_py_function(__file__, f"generic_function('{sd}')",is_pre_cmd=False)

        # add a function that isnt going to be called directly
        pf.add_py_function(__file__, "another_generic_function(some_arg)",is_pre_cmd=None)

        # pf.post_py_cmds.append("generic_function()")
        df = pd.read_csv(os.path.join(template_ws, sd, "sfr.csv"), index_col=0)
        pf.add_observations(os.path.join(sd, "sfr.csv"), index_cols="time", use_cols=list(df.columns.values))
        pf.add_observations(os.path.join(sd, "freyberg6.npf_k_layer1.txt"),
                            zone_array=m.dis.idomain.array[0])

        v = pyemu.geostats.ExpVario(contribution=1.0, a=1000)
        gr_gs = pyemu.geostats.GeoStruct(variograms=v)
        rch_temporal_gs = pyemu.geostats.GeoStruct(variograms=pyemu.geostats.ExpVario(contribution=1.0,a=60))
        pf.extra_py_imports.append('flopy')
        ib = m.dis.idomain[0].array
        tags = {"npf_k_":[0.1,10.],"npf_k33_":[.1,10],"sto_ss":[.1,10],"sto_sy":[.9,1.1],"rch_recharge":[.5,1.5]}
        dts = pd.to_datetime("1-1-2018") + pd.to_timedelta(np.cumsum(sim.tdis.perioddata.array["perlen"]),unit="d")
        print(dts)
        for tag,bnd in tags.items():
            lb,ub = bnd[0],bnd[1]
            arr_files = [os.path.join(sd, f) for f in os.listdir(os.path.join(tmp_model_ws, sd)) if tag in f and f.endswith(".txt")]
            if "rch" in tag:
                pf.add_parameters(filenames=arr_files, par_type="grid", par_name_base="rch_gr",
                                  pargp="rch_gr", zone_array=ib, upper_bound=ub, lower_bound=lb,
                                  geostruct=gr_gs)
                for arr_file in arr_files:
                    kper = int(arr_file.split('.')[1].split('_')[-1]) - 1
                    pf.add_parameters(filenames=arr_file,par_type="constant",par_name_base=arr_file.split('.')[1]+"_cn",
                                      pargp="rch_const",zone_array=ib,upper_bound=ub,lower_bound=lb,geostruct=rch_temporal_gs,
                                      datetime=dts[kper])
            else:
                for arr_file in arr_files:

                    # these ult bounds are used later in an assert
                    # and also are used so that the initial input array files
                    # are preserved
                    ult_lb = None
                    ult_ub = None
                    if "npf_k_" in arr_file:
                       ult_ub = 31.0
                       ult_lb = -1.3
                    pf.add_parameters(filenames=arr_file,par_type="grid",par_name_base=arr_file.split('.')[1]+"_gr",
                                      pargp=arr_file.split('.')[1]+"_gr",zone_array=ib,upper_bound=ub,lower_bound=lb,
                                      geostruct=gr_gs,ult_ubound=None if ult_ub is None else ult_ub + 1,
                                      ult_lbound=None if ult_lb is None else ult_lb + 1)
                    # use a slightly lower ult bound here
                    pf.add_parameters(filenames=arr_file, par_type="pilotpoints", par_name_base=arr_file.split('.')[1]+"_pp",
                                      pargp=arr_file.split('.')[1]+"_pp", zone_array=ib,upper_bound=ub,lower_bound=lb,
                                      ult_ubound=None if ult_ub is None else ult_ub - 1,
                                      ult_lbound=None if ult_lb is None else ult_lb - 1)
        #
        #
        # add SP1 spatially constant, but temporally correlated wel flux pars
        kper = 0
        list_file = os.path.join(
            sd, "freyberg6.wel_stress_period_data_{0}.txt".format(kper+1)
        )
        pf.add_parameters(filenames=list_file, par_type="constant",
                          par_name_base="twel_mlt_{0}".format(kper),
                          pargp="twel_mlt".format(kper), index_cols=[0, 1, 2],
                          use_cols=[3], upper_bound=1.5, lower_bound=0.5,
                          datetime=dts[kper], geostruct=rch_temporal_gs,
                          mfile_skip=1)
        #
        # # add temporally indep, but spatially correlated wel flux pars
        # pf.add_parameters(filenames=list_file, par_type="grid",
        #                   par_name_base="wel_grid_{0}".format(kper),
        #                   pargp="wel_{0}".format(kper), index_cols=[0, 1, 2],
        #                   use_cols=[3], upper_bound=1.5, lower_bound=0.5,
        #                   geostruct=gr_gs, mfile_skip=1)
        # kper = 1
        # list_file = "freyberg6.wel_stress_period_data_{0}.txt".format(kper+1)
        # pf.add_parameters(filenames=list_file, par_type="constant",
        #                   par_name_base="twel_mlt_{0}".format(kper),
        #                   pargp="twel_mlt".format(kper), index_cols=[0, 1, 2],
        #                   use_cols=[3], upper_bound=1.5, lower_bound=0.5,
        #                   datetime=dts[kper], geostruct=rch_temporal_gs,
        #                   mfile_skip='#')
        # # add temporally indep, but spatially correlated wel flux pars
        # pf.add_parameters(filenames=list_file, par_type="grid",
        #                   par_name_base="wel_grid_{0}".format(kper),
        #                   pargp="wel_{0}".format(kper), index_cols=[0, 1, 2],
        #                   use_cols=[3], upper_bound=1.5, lower_bound=0.5,
        #                   geostruct=gr_gs, mfile_skip='#')
        # kper = 2
        # list_file = "freyberg6.wel_stress_period_data_{0}.txt".format(kper+1)
        # pf.add_parameters(filenames=list_file, par_type="constant",
        #                   par_name_base="twel_mlt_{0}".format(kper),
        #                   pargp="twel_mlt".format(kper), index_cols=['#k', 'i', 'j'],
        #                   use_cols=['flux'], upper_bound=1.5, lower_bound=0.5,
        #                   datetime=dts[kper], geostruct=rch_temporal_gs)
        # # add temporally indep, but spatially correlated wel flux pars
        # pf.add_parameters(filenames=list_file, par_type="grid",
        #                   par_name_base="wel_grid_{0}".format(kper),
        #                   pargp="wel_{0}".format(kper), index_cols=['#k', 'i', 'j'],
        #                   use_cols=['flux'], upper_bound=1.5, lower_bound=0.5,
        #                   geostruct=gr_gs)
        # kper = 3
        # list_file = "freyberg6.wel_stress_period_data_{0}.txt".format(kper+1)
        # pf.add_parameters(filenames=list_file, par_type="constant",
        #                   par_name_base="twel_mlt_{0}".format(kper),
        #                   pargp="twel_mlt".format(kper), index_cols=['#k', 'i', 'j'],
        #                   use_cols=['flux'], upper_bound=1.5, lower_bound=0.5,
        #                   datetime=dts[kper], geostruct=rch_temporal_gs,
        #                   mfile_skip=1)
        # # add temporally indep, but spatially correlated wel flux pars
        # pf.add_parameters(filenames=list_file, par_type="grid",
        #                   par_name_base="wel_grid_{0}".format(kper),
        #                   pargp="wel_{0}".format(kper), index_cols=['#k', 'i', 'j'],
        #                   use_cols=['flux'], upper_bound=1.5, lower_bound=0.5,
        #                   geostruct=gr_gs, mfile_skip=1)
        # list_files = ["freyberg6.wel_stress_period_data_{0}.txt".format(t)
        #               for t in range(5, m.nper+1)]
        # for list_file in list_files:
        #     kper = int(list_file.split(".")[1].split('_')[-1]) - 1
        #     # add spatially constant, but temporally correlated wel flux pars
        #     pf.add_parameters(filenames=list_file,par_type="constant",par_name_base="twel_mlt_{0}".format(kper),
        #                       pargp="twel_mlt".format(kper),index_cols=[0,1,2],use_cols=[3],
        #                       upper_bound=1.5,lower_bound=0.5, datetime=dts[kper], geostruct=rch_temporal_gs)
        #
        #     # add temporally indep, but spatially correlated wel flux pars
        #     pf.add_parameters(filenames=list_file, par_type="grid", par_name_base="wel_grid_{0}".format(kper),
        #                       pargp="wel_{0}".format(kper), index_cols=[0, 1, 2], use_cols=[3],
        #                       upper_bound=1.5, lower_bound=0.5, geostruct=gr_gs)
        #
        # # test non spatial idx in list like
        # pf.add_parameters(filenames="freyberg6.sfr_packagedata_test.txt", par_name_base="sfr_rhk",
        #                   pargp="sfr_rhk", index_cols=['#rno'], use_cols=['rhk'], upper_bound=10.,
        #                   lower_bound=0.1,
        #                   par_type="grid")
        #
        # # add model run command
        pf.pre_py_cmds.append(f"os.chdir('{sd}')")
        pf.mod_sys_cmds.append("mf6")
        pf.post_py_cmds.insert(0, "os.chdir('..')")
        print(pf.mult_files)
        print(pf.org_files)

        # build pest
        pst = pf.build_pst('freyberg.pst')

        # # quick check of write and apply method
        # pars = pst.parameter_data
        # # set reach 1 hk to 100
        # sfr_pars = pars.loc[pars.parnme.str.startswith('sfr')].index
        # pars.loc[sfr_pars, 'parval1'] = np.random.random(len(sfr_pars)) * 10
        #
        # sfr_pars = pars.loc[sfr_pars].copy()
        # sfr_pars[['inst', 'usecol', '#rno']] = sfr_pars.parnme.apply(
        #     lambda x: pd.DataFrame([s.split(':') for s in x.split('_')
        #                             if ':' in s]).set_index(0)[1])
        #
        # sfr_pars['#rno'] = sfr_pars['#rno'] .astype(int)
        # os.chdir(pf.new_d)
        # pst.write_input_files()
        # try:
        #     pyemu.helpers.apply_list_and_array_pars()
        # except Exception as e:
        #     os.chdir('..')
        #     raise e
        # os.chdir('..')
        # # verify apply
        # df = pd.read_csv(os.path.join(
        #     pf.new_d, "freyberg6.sfr_packagedata_test.txt"),
        #     delim_whitespace=True, index_col=0)
        # df.index = df.index - 1
        # print(df.rhk)
        # print((sfr_pkgdf.set_index('rno').loc[df.index, 'rhk'] *
        #              sfr_pars.set_index('#rno').loc[df.index, 'parval1']))
        # assert np.isclose(
        #     df.rhk, (sfr_pkgdf.set_index('rno').loc[df.index, 'rhk'] *
        #              sfr_pars.set_index('#rno').loc[df.index, 'parval1'])).all()
        # pars.loc[sfr_pars.index, 'parval1'] = 1.0
        #
        # # add more:
        # pf.add_parameters(filenames="freyberg6.sfr_packagedata.txt", par_name_base="sfr_rhk",
        #                   pargp="sfr_rhk", index_cols={'k': 1, 'i': 2, 'j': 3}, use_cols=[9], upper_bound=10.,
        #                   lower_bound=0.1,
        #                   par_type="grid", rebuild_pst=True)
        #
        # df = pd.read_csv(os.path.join(tmp_model_ws, "heads.csv"), index_col=0)
        pf.add_observations(os.path.join(sd, "heads.csv"),
                            insfile=os.path.join(sd, "heads.csv.ins"),
                            index_cols="time",
                            prefix="hds",
                            rebuild_pst=True)
        #
        # # test par mults are working
        os.chdir(pf.new_d)
        pst.write_input_files()
        pyemu.helpers.apply_list_and_array_pars(
            arr_par_file="mult2model_info.csv",chunk_len=1)
        os.chdir(tmp_path)
        #
        # cov = pf.build_prior(fmt="none").to_dataframe()
        # twel_pars = [p for p in pst.par_names if "twel_mlt" in p]
        # twcov = cov.loc[twel_pars,twel_pars]
        # dsum = np.diag(twcov.values).sum()
        # assert twcov.sum().sum() > dsum
        #
        # rch_cn = [p for p in pst.par_names if "_cn" in p]
        # print(rch_cn)
        # rcov = cov.loc[rch_cn,rch_cn]
        # dsum = np.diag(rcov.values).sum()
        # assert rcov.sum().sum() > dsum
        #
        # num_reals = 100
        # pe = pf.draw(num_reals, use_specsim=True)
        # pe.to_binary(os.path.join(template_ws, "prior.jcb"))
        # assert pe.shape[1] == pst.npar_adj, "{0} vs {1}".format(pe.shape[0], pst.npar_adj)
        # assert pe.shape[0] == num_reals
        #
        #
        pst.control_data.noptmax = 0
        pst.pestpp_options["additional_ins_delimiters"] = ","
        #
        pst.write(os.path.join(pf.new_d, "freyberg.pst"))
        pyemu.os_utils.run("{0} freyberg.pst".format(ies_exe_path), cwd=pf.new_d)
        #
        res_file = os.path.join(pf.new_d, "freyberg.base.rei")
        assert os.path.exists(res_file), res_file
        pst.set_res(res_file)
        # print(pst.phi)
        assert np.isclose(pst.phi, 0), pst.phi
        #
        # check mult files are in pst input files
        csv = os.path.join(template_ws, "mult2model_info.csv")
        df = pd.read_csv(csv, index_col=0)
        pst_input_files = {str(f) for f in pst.input_files}
        mults_not_linked_to_pst = ((set(df.mlt_file.unique()) -
                                    pst_input_files) -
                                   set(df.loc[df.pp_file.notna()].mlt_file))
        assert len(mults_not_linked_to_pst) == 0, print(mults_not_linked_to_pst)

        # make sure the appropriate ult bounds have made it thru
        # df = pd.read_csv(os.path.join(template_ws,"mult2model_info.csv"))
        # print(df.columns)
        # df = df.loc[df.model_file.apply(lambda x: "npf_k_" in x),:]
        # print(df)
        # print(df.upper_bound)
        # print(df.lower_bound)
        # assert np.abs(float(df.upper_bound.min()) - 30.) < 1.0e-6,df.upper_bound.min()
        # assert np.abs(float(df.lower_bound.max()) - -0.3) < 1.0e-6,df.lower_bound.max()
    except Exception as e:
        os.chdir(bd)
        raise Exception(str(e))
    os.chdir(bd)


def shortname_conversion_test(tmp_path):
    import numpy as np
    import pandas as pd
    import re
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)
    try:
        import flopy
    except:
        return

    tmp_model_ws = Path(tmp_path, "temp_pst_from")
    if os.path.exists(tmp_model_ws):
        shutil.rmtree(tmp_model_ws)
    os.mkdir(tmp_model_ws)
    bd = Path.cwd()
    os.chdir(tmp_path)
    try:
        tmp_model_ws = tmp_model_ws.relative_to(tmp_path)
        dims = (20,60)
        np.savetxt(os.path.join(tmp_model_ws,"parfile1"), np.ones(dims))
        np.savetxt(os.path.join(tmp_model_ws, "parfile2"), np.ones(dims))
        np.savetxt(os.path.join(tmp_model_ws, "parfile3"), np.ones(dims))
        np.savetxt(os.path.join(tmp_model_ws, "parfile4"), np.ones(dims))
        np.savetxt(os.path.join(tmp_model_ws, "parfile5"), np.ones(dims))
        np.savetxt(os.path.join(tmp_model_ws, "parfile6"), np.ones(dims))
        np.savetxt(os.path.join(tmp_model_ws, "obsfile1"), np.ones(dims))
        np.savetxt(os.path.join(tmp_model_ws, "obsfile2"), np.ones(dims))

        np.savetxt(os.path.join(tmp_model_ws, "parfile7"), np.ones(dims))
        np.savetxt(os.path.join(tmp_model_ws, "obsfile3"), np.ones(dims))
        # SETUP pest stuff...

        template_ws = "new_temp"
        if os.path.exists(template_ws):
            shutil.rmtree(template_ws)
        # set up PstFrom object
        # obs
        #   using tabular style model output
        #   (generated by pyemu.gw_utils.setup_hds_obs())
        # pf.add_observations('freyberg.hds.dat', insfile='freyberg.hds.dat.ins2',
        #                     index_cols='obsnme', use_cols='obsval', prefix='hds')
        sr = pyemu.helpers.SpatialReference(delr=[10]*dims[1],
                                            delc=[10]*dims[0],
                                            rotation=0,
                                            epsg=3070,
                                            xul=0.,
                                            yul=0.,
                                            units='meters',  # gis units of meters?
                                            lenuni=2  # model units of meters
                                            )
        pf = PstFrom(original_d=tmp_model_ws, new_d=template_ws,
                     remove_existing=True,
                     longnames=False,
                     zero_based=False,
                     spatial_reference=sr)

        v = pyemu.geostats.ExpVario(contribution=1.0, a=1000)
        gr_gs = pyemu.geostats.GeoStruct(variograms=v)
        c = 0
        parfiles = [f.name for f in Path(template_ws).glob("parfile*")][0:3]
        for f in parfiles:
            c += 1
            pf.add_parameters(filenames=f, par_type="grid",
                              par_name_base=f"par{c}",
                              pargp=f"pargp{c}", upper_bound=0.1, lower_bound=10.0,
                              geostruct=gr_gs)
        pf.add_parameters(filenames=parfiles,
                          par_type="constant", par_name_base="cpar",
                          pargp="cpargp", upper_bound=0.1, lower_bound=10.0,
                          geostruct=gr_gs)

        pf.add_observations(
            "obsfile1",
            prefix="longobservationname",
            rebuild_pst=False,
            obsgp="longobservationgroup",
            includes_header=False
        )
        pf.add_observations(
            "obsfile2",
            prefix="longobservationname2",
            rebuild_pst=False,
            obsgp="longobservationgroup2",
            includes_header=False
        )
        pf.add_observations(
            "obsfile3",
            prefix="longobservationname-lt",
            rebuild_pst=False,
            obsgp="less_longobservationgroup",
            includes_header=False,
            insfile="lt_obsfile3.ins"
        )
        pf.add_observations(
            "obsfile3",
            prefix="longobservationname-gt",
            rebuild_pst=False,
            obsgp="greater_longobservationgroup",
            includes_header=False,
            insfile="gt_obsfile3.ins"
        )
        pst = pf.build_pst()
        obs = set(pst.observation_data.obsnme)
        trie = pyemu.helpers.Trie()
        [trie.add(ob) for ob in obs]
        rex = re.compile(trie.pattern())
        for ins in pst.instruction_files:
            with open(os.path.join(pf.new_d, ins), "rt") as f:
                obsin = set(rex.findall(f.read()))
            obs = obs - obsin
        assert len(obs) == 0, f"{len(obs)} obs not found in insfiles: {obs[:100]}..."

        par = set(pst.parameter_data.parnme)
        trie = pyemu.helpers.Trie()
        [trie.add(p) for p in par]
        rex = re.compile(trie.pattern())
        for tpl in pst.template_files:
            with open(os.path.join(pf.new_d, tpl), "rt") as f:
                parin = set(rex.findall(f.read()))
            par = par - parin
        assert len(par) == 0, f"{len(par)} pars not found in tplfiles: {par[:100]}..."
        # test update/rebuild
        pf.add_observations(
            "obsfile3",
            prefix="longobservationname3",
            rebuild_pst=True,
            obsgp="longobservationgroup3",
            includes_header=False
        )
        pf.add_parameters(filenames="parfile7",
                          par_type="grid", par_name_base="par7",
                          pargp="par7", upper_bound=0.1, lower_bound=10.0,
                          geostruct=gr_gs,
                          rebuild_pst=True)

        obs = set(pst.observation_data.obsnme)
        trie = pyemu.helpers.Trie()
        [trie.add(ob) for ob in obs]
        rex = re.compile(trie.pattern())
        for ins in pst.instruction_files:
            with open(os.path.join(pf.new_d, ins), "rt") as f:
                obsin = set(rex.findall(f.read()))
            obs = obs - obsin
        assert len(obs) == 0, f"{len(obs)} obs not found in insfiles: {obs[:100]}..."

        par = set(pst.parameter_data.parnme)
        parin = set()
        trie = pyemu.helpers.Trie()
        [trie.add(p) for p in par]
        rex = re.compile(trie.pattern())
        for tpl in pst.template_files:
            with open(os.path.join(pf.new_d, tpl), "rt") as f:
                parin = set(rex.findall(f.read()))
            par = par - parin
        assert len(par) == 0, f"{len(par)} pars not found in tplfiles: {par[:100]}..."
    except Exception as e:
        os.chdir(bd)
        raise Exception(str(e))
    os.chdir(bd)


@pytest.mark.parametrize("setup_freyberg_mf6", ['freyberg_quadtree'], indirect=True)
def vertex_grid_test(setup_freyberg_mf6):
    pf, sim = setup_freyberg_mf6
    m = sim.get_model()
    mg = m.modelgrid
    # the model grid is vertex type
    assert mg.grid_type=='vertex'

    template_ws = pf.new_d  # "new_temp"

    # exponential variogram for spatially varying parameters
    v_space = pyemu.geostats.ExpVario(contribution=1.0,
                                      a=1000,
                                      anisotropy=1.0,
                                      bearing=0.0)
    # geostatistical structure for spatially varying parameters
    grid_gs = pyemu.geostats.GeoStruct(variograms=v_space, transform='log')

    tag = "npf_k_"
    files = [f for f in os.listdir(template_ws) if tag in f.lower() and f.endswith(".txt")]

    # get the IDOMAIN array to use as zone array
    ib = m.dis.idomain.get_data()

    # make sure array files are tidy
    for f in files:
        filename=os.path.join(template_ws, f)
        with open(filename, 'r') as f:
            a = f.read()
        a = [float(i) for i in a.split()]
        np.savetxt(fname=filename, X=a)
    # run model after input file change
    pyemu.os_utils.run('mf6', cwd=template_ws)

    for f in files:
        layer = int(f.split('_layer')[-1].split('.')[0]) - 1
        # grid (fine) scale parameters
        df_gr = pf.add_parameters(
            f,
            zone_array=ib[layer],
            par_type="grid",
            geostruct=grid_gs,
            par_name_base=f.split('.')[1].replace("_","")+"gr",
            pargp=f.split('.')[1].replace("_","")+"gr",
            lower_bound=0.2, upper_bound=5.0,
            ult_ubound=100, ult_lbound=0.01
        )
        # pilot point (medium) scale parameters
        df_pp = pf.add_parameters(f,
                            zone_array=ib[layer],
                            par_type="pilotpoints",
                            use_pp_zones=True,
                            geostruct=grid_gs,
                            par_name_base=f.split('.')[1].replace("_","")+"pp",
                            pargp=f.split('.')[1].replace("_","")+"pp",
                            lower_bound=0.2,upper_bound=5.0,
                            ult_ubound=100, ult_lbound=0.01,
                            pp_space=500) # `

    tag = "sfr_packagedata"
    files = [f for f in os.listdir(template_ws) if tag in f.lower() and f.endswith(".txt")]
    f = files[0]
    # constant and grid scale multiplier conductance parameters
    name = "sfrcond"
    df_list = pf.add_parameters(f,
                    par_type="grid",
                    geostruct=grid_gs,
                    par_name_base=name+"gr",
                    pargp=name+"gr",
                    index_cols=[0,1,2], # this assumes 1,2 are row,col
                    use_cols=[8],
                    lower_bound=0.1,upper_bound=10.0)

    assert df_list.x.max() != df_list.x.min()
    assert df_list.y.max() != df_list.y.min()

    # add the observations to pf
    df = pd.read_csv(os.path.join(template_ws, "sfr_obs.csv"), index_col=0)
    sfr_df = pf.add_observations("sfr_obs.csv",
                                insfile="sfr_obs.csv.ins",
                                index_cols="time",
                                use_cols=list(df.columns.values),
                                prefix="sfr")
    pf.mod_sys_cmds.append('mf6')
    pst = pf.build_pst()
    # check_apply(pf)
    # run once
    pst.control_data.noptmax=0
    pst.write(os.path.join(template_ws, 'test.pst'))
    os_utils.run("{0} test.pst".format(pp_exe_path), cwd=template_ws)
    pstchk = pyemu.Pst(os.path.join(template_ws,'test.pst'))
    assert np.isclose(pstchk.phi, 0), f"expected near zero phi: {pstchk.phi}"

    # check zone par bounds are respected
    par = pst.parameter_data
    assert 'zone' in par.columns

    par['zone'] = [int(i.split(':')[-1]) for i in par.parnme.values]
    par_org = par.copy()
    #check gr pars
    for zone in par.zone.unique():
        par.loc[(par.zone==zone) & (par.ptype=='gr'), 'parval1'] = float(zone)
        par.loc[(par.zone==zone) & (par.ptype=='gr'), 'parlbnd'] = float(zone)-0.1
        par.loc[(par.zone==zone) & (par.ptype=='gr'), 'parubnd'] = float(zone)+0.1

    # write model input files
    check_apply(pf)

    #check pp pars
    #reset par values
    par.loc[:,par_org.columns] = par_org.values
    for zone in par.zone.unique():
        par.loc[(par.zone==zone) & (par.ptype=='pp'), 'parval1'] = float(zone)
        par.loc[(par.zone==zone) & (par.ptype=='pp'), 'parlbnd'] = float(zone)-0.1
        par.loc[(par.zone==zone) & (par.ptype=='pp'), 'parubnd'] = float(zone)+0.1

    # write model input files
    check_apply(pf)
    #pyemu.os_utils.run(r'python forward_run.py', cwd=template_ws)

    ib = m.dis.idomain.get_data()
    tag = "npf_k_"
    files = [f for f in os.listdir(template_ws) if tag in f.lower() and f.endswith(".txt")]
    npfpar = par.loc[par.pname.str.contains('npf'), :]
    assert len(npfpar) > 0
    for f in files:
        k = int(f.split('_layer')[-1].split('.')[0]) - 1
        a = np.loadtxt(os.path.join(template_ws, f))
        a_org = np.loadtxt(os.path.join(template_ws,'org', f))
        # weak check
        for zone in npfpar.loc[npfpar.pname.str.contains(f'layer{k+1}')].zone.unique():
            assert np.isclose(abs((a/a_org)[ib[k]==int(zone)]-int(zone)).max(), 0)
    return

def test_defaults(tmp_path):
    import numpy as np
    import pandas as pd
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)
    try:
        import flopy
    except:
        return

    org_model_ws = os.path.join('..', 'examples', 'freyberg_mf6')
    tmp_model_ws = setup_tmp(org_model_ws, tmp_path)
    bd = Path.cwd()
    os.chdir(tmp_path)
    try:
        tmp_model_ws = tmp_model_ws.relative_to(tmp_path)
        sim = flopy.mf6.MFSimulation.load(sim_ws=str(tmp_model_ws))
        m = sim.get_model()
        sim.set_all_data_external(check_data=False)
        sim.write_simulation()

        # SETUP pest stuff...
        os_utils.run("{0} ".format(mf6_exe_path), cwd=tmp_model_ws)
        template_ws = "new_temp"
        if os.path.exists(template_ws):
            shutil.rmtree(template_ws)
        # sr0 = m.sr
        # sr = pyemu.helpers.SpatialReference.from_namfile(
        #     os.path.join(tmp_model_ws, "freyberg6.nam"),
        #     delr=m.dis.delr.array, delc=m.dis.delc.array)
        sr = m.modelgrid
        # set up PstFrom object
        pf = PstFrom(original_d=tmp_model_ws, new_d=template_ws)
        kper = 0
        list_file = "freyberg6.wel_stress_period_data_{0}.txt".format(kper + 1)
        pf.add_parameters(list_file, par_type="grid", index_cols=[0, 1, 2],
                      use_cols=[3])
        pf.add_observations("heads.csv", index_cols="time")
        pst = pf.build_pst()
    except Exception as e:
        os.chdir(bd)
        raise Exception(str(e))
    os.chdir(bd)


def list_float_int_index_test(tmp_path):
    org_d = tmp_path
    # if os.path.exists(org_d):
    #     shutil.rmtree(org_d)
    # os.makedirs(org_d)
    shutil.copy2(os.path.join("utils", "ppoints.faults.csv"),
                 os.path.join(org_d, "ppoints.faults.csv"))
    shutil.copy2(os.path.join("utils", "ghb_ppt_part1.dat"),
                 os.path.join(org_d, "ghb_ppt_part1.dat"))
    # print(os.getcwd())
    faultdf_o = pd.read_csv(os.path.join(org_d, "ppoints.faults.csv"))
    ghbdf_o = pd.read_csv(os.path.join(org_d, "ghb_ppt_part1.dat"), sep=r'\s+')
    ghbdf_o.loc[slice(5), 'ppt'] = ghbdf_o.loc[slice(5), 'ppt'].str.strip('pt')
    ghbdf_o.to_csv(os.path.join(org_d, "ghb_ppt_part1.dat"), index=False, sep=' ')
    new_d = Path(org_d, "list_temp_new")
    pf = pyemu.utils.PstFrom(original_d=org_d, new_d=new_d,
                             remove_existing=True, zero_based=False)
    faultidx = ["x", "y", "zone"]
    pf.add_parameters(filenames="ppoints.faults.csv",
                      par_type="grid",
                      par_name_base=["kh", "ss", "sy", "w", "a"],
                      pargp=["kh","ss","sy","w","a"],
                      index_cols=faultidx,
                      use_cols=["kh", "ss", "sy", "w", "a"],
                      lower_bound=[0.01,0.1,0.2,0.5],
                      upper_bound=[1.5,2,4,5])
    pf.add_observations(filename="ppoints.faults.csv",
                        index_cols=faultidx,
                        use_cols=["kh", "ss", "sy", "w", "a"])
    ghbidx = ["ppt", 'x', 'y']
    pf.add_parameters(filenames="ghb_ppt_part1.dat",
                      par_type="grid",
                      par_name_base=["n"],
                      index_cols=ghbidx,
                      use_cols=["ghbcondN"],
                      lower_bound=0.01,
                      upper_bound=100)
    pf.add_observations(filename="ghb_ppt_part1.dat",
                        index_cols=ghbidx,
                        use_cols="ghbcondN")
    pst = pf.build_pst()
    par = pst.parameter_data
    assert par.shape[0] == faultdf_o.shape[0] * 5 + len(ghbdf_o)
    obs = pst.observation_data
    assert obs.shape[0] == faultdf_o.shape[0] * 5 + len(ghbdf_o)
    # pf.parfile_relations.to_csv(os.path.join(pf.new_d,"mult2model_info.csv"))
    kpar = par.parnme.str.contains("kh")
    kparval1 = np.linspace(0.1, 10, sum(kpar))
    par.loc[kpar, "parval1"] = kparval1
    # print(par.loc[par.parnme.str.contains("kh"),"parval1"])
    bpar = par.parnme.str.contains("ghbcondN")
    bparval1 = np.linspace(0.1, 10, sum(bpar))
    par.loc[bpar, "parval1"] = bparval1
    pst.write_input_files(pf.new_d)
    bd = os.getcwd()
    os.chdir(pf.new_d)
    try:
        pyemu.helpers.apply_list_and_array_pars(chunk_len=1000)
    except Exception as e:
        os.chdir(bd)
        raise e
    os.chdir(bd)
    faultdf_n = pd.read_csv(os.path.join(pf.new_d, "ppoints.faults.csv"))
    idxcheck = faultdf_n.set_index(faultidx).index.difference(faultdf_o.set_index(faultidx).index)
    assert len(idxcheck) == 0, idxcheck
    diff = faultdf_n.set_index(faultidx).kh/faultdf_o.set_index(faultidx).kh
    assert np.isclose(diff,kparval1).all(), diff.loc[~np.isclose(diff,kparval1)]

    ghbdf_n = pd.read_csv(os.path.join(pf.new_d, "ghb_ppt_part1.dat"), sep=r'\s+')
    idxcheck = ghbdf_n.set_index(ghbidx).index.difference(ghbdf_o.set_index(ghbidx).index)
    assert len(idxcheck) == 0, idxcheck
    diff = (ghbdf_n.set_index(ghbidx).ghbcondN/ghbdf_o.set_index(ghbidx).ghbcondN).sort_index(level=0)
    bparval1 = par.loc[bpar].sort_values('ppt').parval1.values
    assert np.isclose(diff,bparval1).all(), diff.loc[~np.isclose(diff,bparval1)]


def mf6_freyberg_thresh_test(tmp_path):

    import numpy as np
    import pandas as pd
    # pd.set_option('display.max_rows', 500)
    # pd.set_option('display.max_columns', 500)
    # pd.set_option('display.width', 1000)
    # try:
    import flopy
    # except:
    #     return

    org_model_ws = os.path.join('..', 'examples', 'freyberg_mf6')
    tmp_model_ws = setup_tmp(org_model_ws, tmp_path)


    tmp_model_ws = tmp_model_ws.relative_to(tmp_path)
    bd = os.getcwd()
    os.chdir(tmp_path)
    try:
        sim = flopy.mf6.MFSimulation.load(sim_ws=str(tmp_model_ws))
        m = sim.get_model("freyberg6")
        sim.set_all_data_external()
        sim.write_simulation()



        # SETUP pest stuff...
        os_utils.run("{0} ".format("mf6"), cwd=tmp_model_ws)

        template_ws = Path(tmp_path, "new_temp_thresh")
        if os.path.exists(template_ws):
            shutil.rmtree(template_ws)
        sr = m.modelgrid
        # set up PstFrom object
        pf = PstFrom(original_d=tmp_model_ws, new_d=template_ws,
                     remove_existing=True,
                     longnames=True, spatial_reference=sr,
                     zero_based=False, start_datetime="1-1-2018")

        df = pd.read_csv(os.path.join(tmp_model_ws, "heads.csv"), index_col=0)
        pf.add_observations("heads.csv", insfile="heads.csv.ins", index_cols="time",
                            use_cols=list(df.columns.values),
                            prefix="hds", rebuild_pst=True)

        # Add stream flow observation
        # df = pd.read_csv(os.path.join(tmp_model_ws, "sfr.csv"), index_col=0)
        pf.add_observations("sfr.csv", insfile="sfr.csv.ins", index_cols="time",
                            use_cols=["GAGE_1", "HEADWATER", "TAILWATER"], ofile_sep=",")

        # Setup geostruct for spatial pars
        gr_v = pyemu.geostats.ExpVario(contribution=1.0, a=500)
        gr_gs = pyemu.geostats.GeoStruct(variograms=gr_v, transform="log")
        pp_v = pyemu.geostats.ExpVario(contribution=1.0, a=1000)
        pp_gs = pyemu.geostats.GeoStruct(variograms=pp_v, transform="log")
        rch_temporal_gs = pyemu.geostats.GeoStruct(variograms=pyemu.geostats.ExpVario(contribution=1.0, a=60))
        pf.extra_py_imports.append('flopy')
        ib = m.dis.idomain[0].array
        #tags = {"npf_k_": [0.1, 10.], "npf_k33_": [.1, 10], "sto_ss": [.1, 10], "sto_sy": [.9, 1.1],
        #        "rch_recharge": [.5, 1.5]}
        tags = {"npf_k_": [0.1, 10.],"rch_recharge": [.5, 1.5]}
        dts = pd.to_datetime("1-1-2018") + pd.to_timedelta(np.cumsum(sim.tdis.perioddata.array["perlen"]), unit="d")
        print(dts)
        # ib = m.dis.idomain.array[0,:,:]
        # setup from array style pars
        num_cat_arrays = 0
        for tag, bnd in tags.items():
            lb, ub = bnd[0], bnd[1]
            arr_files = [f for f in os.listdir(tmp_model_ws) if tag in f and f.endswith(".txt")]
            if "rch" in tag:
                for arr_file in arr_files:
                    # indy direct grid pars for each array type file
                    recharge_files = ["recharge_1.txt", "recharge_2.txt", "recharge_3.txt"]
                    pf.add_parameters(filenames=arr_file, par_type="grid", par_name_base="rch_gr",
                                      pargp="rch_gr", zone_array=ib, upper_bound=1.0e-3, lower_bound=1.0e-7,
                                      par_style="direct")
                    # additional constant mults
                    kper = int(arr_file.split('.')[1].split('_')[-1]) - 1
                    pf.add_parameters(filenames=arr_file, par_type="constant",
                                      par_name_base=arr_file.split('.')[1] + "_cn",
                                      pargp="rch_const", zone_array=ib, upper_bound=ub, lower_bound=lb,
                                      geostruct=rch_temporal_gs,
                                      datetime=dts[kper])
            else:
                for arr_file in arr_files:
                    print(arr_file)
                    k = int(arr_file.split(".")[1][-1]) - 1
                    if k == 1:
                        arr_file = arr_file.replace("_k_","_k33_")
                    pth_arr_file = os.path.join(pf.new_d,arr_file)
                    arr = np.loadtxt(pth_arr_file)
                    cat_dict = {1:[0.4,arr.mean()],2:[0.6,arr.mean()]}
                    thresharr,threshcsv = pyemu.helpers.setup_threshold_pars(pth_arr_file,cat_dict=cat_dict,
                                                                             testing_workspace=pf.new_d,inact_arr=ib)

                    pf.pre_py_cmds.append("pyemu.helpers.apply_threshold_pars('{0}')".format(os.path.split(threshcsv)[1]))
                    prefix = arr_file.split('.')[1].replace("_","-")
                    pf.add_parameters(filenames=os.path.split(thresharr)[1],par_type="grid",transform="none",
                                      par_name_base=prefix+"-threshgr_k:{0}".format(k),
                                      pargp=prefix + "-threshgr_k:{0}".format(k),
                                      lower_bound=0.0,upper_bound=1.0,geostruct=gr_gs,par_style="d")


                    pf.add_parameters(filenames=os.path.split(thresharr)[1],par_type="pilotpoints",transform="none",
                                      par_name_base=prefix+"-threshpp_k:{0}".format(k),
                                      pargp=prefix + "-threshpp_k:{0}".format(k),
                                      lower_bound=0.0,upper_bound=2.0,geostruct=pp_gs,par_style="m",
                                      pp_space=3
                                      )


                    pf.add_parameters(filenames=os.path.split(threshcsv)[1], par_type="grid",index_cols=["threshcat"],
                                      use_cols=["threshproportion","threshfill"],
                                      par_name_base=[prefix+"threshproportion_k:{0}".format(k),prefix+"threshfill_k:{0}".format(k)],
                                      pargp=[prefix+"threshproportion_k:{0}".format(k),prefix+"threshfill_k:{0}".format(k)],
                                      lower_bound=[0.1,0.1],upper_bound=[10.0,10.0],transform="none",par_style='d')

                    pf.add_observations(arr_file,prefix="hkarr-"+prefix+"_k:{0}".format(k),
                                        obsgp="hkarr-"+prefix+"_k:{0}".format(k),zone_array=ib)

                    pf.add_observations(arr_file+".threshcat.dat", prefix="tcatarr-" + prefix+"_k:{0}".format(k),
                                        obsgp="tcatarr-" + prefix+"_k:{0}".format(k),zone_array=ib)

                    pf.add_observations(arr_file + ".thresharr.dat",
                                        prefix="tarr-" +prefix+"_k:{0}".format(k),
                                        obsgp="tarr-" + prefix + "_k:{0}".format(k), zone_array=ib)

                    df = pd.read_csv(threshcsv.replace(".csv","_results.csv"),index_col=0)
                    pf.add_observations(os.path.split(threshcsv)[1].replace(".csv","_results.csv"),index_cols="threshcat",use_cols=df.columns.tolist(),prefix=prefix+"-results_k:{0}".format(k),
                                        obsgp=prefix+"-results_k:{0}".format(k),ofile_sep=",")
                    num_cat_arrays += 1

        # add model run command
        pf.mod_sys_cmds.append("mf6")
        print(pf.mult_files)
        print(pf.org_files)

        # build pest
        pst = pf.build_pst('freyberg.pst')
        #cov = pf.build_prior(fmt="none")
        #cov.to_coo(os.path.join(template_ws, "prior.jcb"))
        pst.try_parse_name_metadata()

        pst.control_data.noptmax = 0
        pst.pestpp_options["additional_ins_delimiters"] = ","

        pst.write(os.path.join(pf.new_d, "freyberg.pst"))
        pyemu.os_utils.run("{0} freyberg.pst".format(ies_exe_path), cwd=pf.new_d)

        res_file = os.path.join(pf.new_d, "freyberg.base.rei")
        assert os.path.exists(res_file), res_file
        pst.set_res(res_file)
        print(pst.phi)
        assert pst.phi < 0.1, pst.phi

        #set the initial and bounds for the fill values
        par = pst.parameter_data
        cat1par = par.loc[par.apply(lambda x: x.threshcat=="0" and x.usecol=="threshfill",axis=1),"parnme"]
        cat2par = par.loc[par.apply(lambda x: x.threshcat == "1" and x.usecol == "threshfill", axis=1), "parnme"]
        print(cat1par,cat2par)
        assert cat1par.shape[0] == num_cat_arrays
        assert cat2par.shape[0] == num_cat_arrays

        cat1parhk = [p for p in cat1par if "k:1" not in p]
        cat2parhk = [p for p in cat2par if "k:1" not in p]
        cat1parvk = [p for p in cat1par if "k:1" in p]
        cat2parvk = [p for p in cat2par if "k:1" in p]
        for lst in [cat2parvk,cat2parhk,cat1parhk,cat1parvk]:
            assert len(lst) > 0
        par.loc[cat1parhk,"parval1"] = 0.1
        par.loc[cat1parhk, "parubnd"] = 1.0
        par.loc[cat1parhk, "parlbnd"] = 0.01
        par.loc[cat1parhk, "partrans"] = "log"
        par.loc[cat2parhk, "parval1"] = 10
        par.loc[cat2parhk, "parubnd"] = 100
        par.loc[cat2parhk, "parlbnd"] = 1
        par.loc[cat2parhk, "partrans"] = "log"

        par.loc[cat1parvk, "parval1"] = 0.0001
        par.loc[cat1parvk, "parubnd"] = 0.01
        par.loc[cat1parvk, "parlbnd"] = 0.00001
        par.loc[cat1parvk, "partrans"] = "log"
        par.loc[cat2parvk, "parval1"] = 0.1
        par.loc[cat2parvk, "parubnd"] = 1
        par.loc[cat2parvk, "parlbnd"] = 0.01
        par.loc[cat2parvk, "partrans"] = "log"


        cat1par = par.loc[par.apply(lambda x: x.threshcat == "0" and x.usecol == "threshproportion", axis=1), "parnme"]
        cat2par = par.loc[par.apply(lambda x: x.threshcat == "1" and x.usecol == "threshproportion", axis=1), "parnme"]

        print(cat1par, cat2par)
        assert cat1par.shape[0] == num_cat_arrays
        assert cat2par.shape[0] == num_cat_arrays

        par.loc[cat1par, "parval1"] = 0.5
        par.loc[cat1par, "parubnd"] = 1.0
        par.loc[cat1par, "parlbnd"] = 0.0
        par.loc[cat1par,"partrans"] = "none"

        # since the apply method only looks that first proportion, we can just fix this one
        par.loc[cat2par, "parval1"] = 1
        par.loc[cat2par, "parubnd"] = 1
        par.loc[cat2par, "parlbnd"] = 1
        par.loc[cat2par,"partrans"] = "fixed"

        assert par.loc[par.parnme.str.contains("threshgr"),:].shape[0] > 0
        #par.loc[par.parnme.str.contains("threshgr"),"parval1"] = 0.5
        par.loc[par.parnme.str.contains("threshgr"),"partrans"] = "fixed"
        
        print(pst.adj_par_names)
        print(pst.npar,pst.npar_adj)

        org_par = par.copy()
        num_reals = 100
        pe = pf.draw(num_reals, use_specsim=False)
        pe.enforce()
        print(pe.shape)
        assert pe.shape[1] == pst.npar_adj, "{0} vs {1}".format(pe.shape[1], pst.npar_adj)
        assert pe.shape[0] == num_reals
        
        # cat1par = cat1par[0]
        # pe = pe.loc[pe.loc[:,cat1par].values>0.35,:]
        # pe = pe.loc[pe.loc[:, cat1par].values < 0.5, :]
        # cat2par = par.loc[par.apply(lambda x: x.threshcat == "1" and x.usecol == "threshfill", axis=1), "parnme"]
        # cat2par = cat2par[0]
        # pe = pe.loc[pe.loc[:, cat2par].values > 10, :]
        # pe = pe.loc[pe.loc[:, cat2par].values < 50, :]

        print(pe.shape)
        assert pe.shape[0] > 0
        #print(pe.loc[:,cat1par].describe())
        #print(pe.loc[:, cat2par].describe())
        #return
        truth_idx = pe.index[0]
        pe = pe.loc[pe.index.map(lambda x: x != truth_idx),:]
        pe.to_dense(os.path.join(template_ws, "prior.jcb"))


        # just use a real as the truth...
        pst.parameter_data.loc[pst.adj_par_names,"parval1"] = pe.loc[pe.index[0],pst.adj_par_names].values
        pst.control_data.noptmax = 0
        pst.write(os.path.join(pf.new_d,"truth.pst"),version=2)
        pyemu.os_utils.run("{0} truth.pst".format(ies_exe_path),cwd=pf.new_d)

        pst = pyemu.Pst(os.path.join(pf.new_d,"truth.pst"))

        obs = pst.observation_data
        obs.loc[:,"obsval"] = pst.res.loc[pst.obs_names,"modelled"].values
        obs.loc[:,"weight"] = 0.0
        obs.loc[:,"standard_deviation"] = np.nan
        onames = obs.loc[obs.obsnme.apply(lambda x: ("trgw" in x or "gage" in x) and ("hdstd" not in x and "sfrtd" not in x)),"obsnme"].values
        #obs.loc[obs.oname=="hds","weight"] = 1.0
        #obs.loc[obs.oname == "hds", "standard_deviation"] = 0.001
        snames = [o for o in onames if "gage" in o]
        obs.loc[onames,"weight"] = 1.0
        obs.loc[snames,"weight"] = 1./(obs.loc[snames,"obsval"] * 0.2).values
        #obs.loc[onames,"obsval"] = truth.values
        #obs.loc[onames,"obsval"] *= np.random.normal(1.0,0.01,onames.shape[0])

        pst.write(os.path.join(pf.new_d, "freyberg.pst"),version=2)
        pyemu.os_utils.run("{0} freyberg.pst".format(ies_exe_path), cwd=pf.new_d)
        pst = pyemu.Pst(os.path.join(pf.new_d,"freyberg.pst"))
        assert pst.phi < 0.01,str(pst.phi)

        # reset away from the truth...
        pst.parameter_data.loc[:,"parval1"] = org_par.parval1.values.copy()

        pst.control_data.noptmax = 2
        pst.pestpp_options["ies_par_en"] = "prior.jcb"
        pst.pestpp_options["ies_num_reals"] = 30
        pst.pestpp_options["ies_subset_size"] = -10
        pst.pestpp_options["ies_no_noise"] = True
        #pst.pestpp_options["ies_bad_phi_sigma"] = 2.0
        pst.pestpp_options["overdue_giveup_fac"] = 100.0
        #pst.pestpp_options["panther_agent_freeze_on_fail"] = True

        #pst.write(os.path.join(pf.new_d, "freyberg.pst"))
        #pyemu.os_utils.start_workers(pf.new_d,ies_exe_path,"freyberg.pst",worker_root=".",master_dir="master_thresh",num_workers=15)

        #num_reals = 100
        #pe = pf.draw(num_reals, use_specsim=False)
        #pe.enforce()
        #pe.to_dense(os.path.join(template_ws, "prior.jcb"))
        #pst.pestpp_options["ies_par_en"] = "prior.jcb"
        
        pst.write(os.path.join(pf.new_d, "freyberg.pst"), version=2)
        m_d = "master_thresh"
        pyemu.os_utils.start_workers(pf.new_d, ies_exe_path, "freyberg.pst", worker_root=".", master_dir=m_d,
                                     num_workers=10)
        phidf = pd.read_csv(os.path.join(m_d,"freyberg.phi.actual.csv"))
        print(phidf["mean"])

        assert phidf["mean"].min() < phidf["mean"].max()

        #pst.pestpp_options["ies_multimodal_alpha"] = 0.99
        
        #pst.pestpp_options["ies_num_threads"] = 6
        #pst.write(os.path.join(pf.new_d, "freyberg.pst"),version=2)

        #pyemu.os_utils.start_workers(pf.new_d, ies_exe_path, "freyberg.pst", worker_root=".", master_dir="master_thresh_mm",
        #                             num_workers=40)
    except Exception as e:
        os.chdir(bd)
        raise Exception(e)
    os.chdir(bd)

def plot_thresh(m_d):
    import flopy

    sim = flopy.mf6.MFSimulation.load(sim_ws=m_d)
    dis = sim.get_model().dis
    ib = dis.idomain.array[0, :, :]
    nlay,nrow, ncol = dis.nlay.data,dis.nrow.data, dis.ncol.data

    # tpst = pyemu.Pst(os.path.join(m_d, "truth.pst"))
    # tobs = tpst.observation_data
    # print(tobs.oname.unique())
    # tobs = tobs.loc[tobs.oname == "hkarr-npf-k-layer1", :].copy()
    # tobs = tobs.loc[tobs.obsval>-1e10,:]
    # tobs.loc[:, "i"] = tobs.pop("i").astype(int)
    # tobs.loc[:, "j"] = tobs.pop("j").astype(int)
    # tarray = np.zeros((nrow,ncol))
    # tarray[tobs.i.values,tobs.j.values] = tobs.obsval.values

    pst = pyemu.Pst(os.path.join(m_d,"freyberg.pst"))
    obs = pst.observation_data
    print(obs.oname.unique())
    

    #pst.control_data.noptmax = 10
    phidf = pd.read_csv(os.path.join(m_d,"freyberg.phi.actual.csv"))
    mxiter = phidf.iteration.max()
    #mxiter = 1
    #print(mxiter)
    pr_oe = pyemu.ObservationEnsemble.from_csv(pst=pst,filename=os.path.join(m_d,"freyberg.0.obs.csv"))
    pr_oe.index = pr_oe.index.map(str)
    pr_pv = pr_oe.phi_vector
    pr_pv.sort_values(inplace=True,ascending=False)
    pr_oe = pr_oe.loc[pr_pv.index,:]

    reals_to_plot = pr_pv.index[:19].tolist()
    if "base" in pr_pv.index:
        reals_to_plot.append("base")
    for iiter in range(1,mxiter+1):
        pt_oe = pyemu.ObservationEnsemble.from_csv(pst=pst,filename=os.path.join(m_d, "freyberg.{0}.obs.csv".format(iiter)))
        pt_oe.index = pt_oe.index.map(str)
        pv = pt_oe.phi_vector
        # pv.sort_values(inplace=True)
        # pr_pv = pr_oe.phi_vector
        # print(pv)
        #pt_oe = pt_oe._df.loc[pv.index,:]
        #pr_oe.index = pr_oe.index.map(str)
        #print(pr_oe.index)
        #print(pt_oe.index)


        #pt_oe.loc[:, obs.obsnme] = np.log10(pt_oe.loc[:, obs.obsnme].values)
        #pr_oe.loc[:, obs.obsnme] = np.log10(pr_oe.loc[:, obs.obsnme].values)

        # vals = pt_oe.loc[:, obs.obsnme].values
        # mx = np.nanmax(vals)
        # vals = pt_oe.loc[:, obs.obsnme].values
        # vals[vals < 0.0] = np.nan
        # mn = np.nanmin(vals)

        
        for k in range(nlay):
            if k == 1:
                kobs = obs.loc[obs.oname == "hkarr-npf-k33-layer{0}".format(k + 1), :].copy()
                kcobs = obs.loc[obs.oname == "tarr-npf-k33-layer{0}".format(k + 1), :].copy()
            else:
                kobs = obs.loc[obs.oname=="hkarr-npf-k-layer{0}".format(k+1),:].copy()
                kcobs = obs.loc[obs.oname == "tarr-npf-k-layer{0}".format(k + 1), :].copy()
            if kobs.shape[0] == 0:
                print("no obs for layer {0}".format(k+1))
                continue
            
            kobs.loc[:, "i"] = kobs.pop("i").astype(int)
            kobs.loc[:, "j"] = kobs.pop("j").astype(int)
            kobs = kobs.loc[kobs.obsval > -1e10, :]

            kcobs.loc[:, "i"] = kcobs.pop("i").astype(int)
            kcobs.loc[:, "j"] = kcobs.pop("j").astype(int)
            kcobs = kcobs.loc[kcobs.obsval > -1e10, :]
            tarray = np.zeros((nrow, ncol)) - 1e10
            tarray[kobs.i.values, kobs.j.values] = kobs.obsval.values
            tarray[tarray==-1e10] = np.nan
            tcarray = np.zeros((nrow, ncol)) - 1e10
            tcarray[kcobs.i.values, kcobs.j.values] = kcobs.obsval.values
            tcarray[tcarray==-1e10] = np.nan
            tarray = np.log10(tarray)
            mn = np.log10(kobs.obsval.values).min()
            mx = np.log10(kobs.obsval.values).max()
            cmn = kcobs.obsval.min()
            cmx = kcobs.obsval.max()

            print(mn, mx)


            import matplotlib.pyplot as plt
            from matplotlib.backends.backend_pdf import PdfPages

            with PdfPages(os.path.join(m_d,"results_{0}_hk_layer_{1}.pdf".format(iiter,k+1))) as pdf:
                ireal = 0
                #for real in pr_oe.index:
                for real in reals_to_plot:
                    if real not in pt_oe.index:
                        continue
                    prarr = np.zeros((nrow,ncol)) - 1
                    prarr[kobs.i,kobs.j] = pr_oe.loc[real,kobs.obsnme]
                    prarr[ib==0] = np.nan
                    ptarr = np.zeros((nrow, ncol)) - 1
                    ptarr[kobs.i, kobs.j] = pt_oe.loc[real, kobs.obsnme]
                    ptarr[ib == 0] = np.nan
                    #print(prarr)
                    #print(ptarr)
                    #mx = max(np.nanmax(prarr),np.nanmax(ptarr))
                    #mn = max(np.nanmin(prarr), np.nanmin(ptarr))
                    fig,axes = plt.subplots(2,3,figsize=(10,10))
                    cb = axes[0,2].imshow(tarray, vmin=mn, vmax=mx, cmap="plasma")
                    plt.colorbar(cb, ax=axes[0,0])
                    cb = axes[0,0].imshow(np.log10(prarr),vmin=mn,vmax=mx,cmap="plasma")
                    plt.colorbar(cb,ax=axes[0,1])
                    cb = axes[0,1].imshow(np.log10(ptarr), vmin=mn, vmax=mx,cmap="plasma")
                    plt.colorbar(cb,ax=axes[0,2])
                    axes[0,1].set_title("post real: {1}, phi: {0:4.1f}".format(pv[real], real), loc="left")
                    axes[0,0].set_title("prior real: {1}, phi: {0:4.1f}".format(pr_pv[real],real),loc="left")
                    axes[0,2].set_title("truth", loc="left")


                    prarr = np.zeros((nrow,ncol)) - 1
                    prarr[kcobs.i,kcobs.j] = pr_oe.loc[real,kcobs.obsnme]
                    prarr[ib==0] = np.nan
                    ptarr = np.zeros((nrow, ncol)) - 1
                    ptarr[kcobs.i, kcobs.j] = pt_oe.loc[real, kcobs.obsnme]
                    ptarr[ib == 0] = np.nan
                    cb = axes[1,0].imshow(prarr,vmin=cmn,vmax=cmx,cmap="plasma")
                    plt.colorbar(cb, ax=axes[1,0])
                    cb = axes[1,1].imshow(prarr,vmin=cmn,vmax=cmx,cmap="plasma")
                    plt.colorbar(cb,ax=axes[1,1])
                    cb = axes[1,2].imshow(tcarray, vmin=cmn,vmax=cmx,cmap="plasma")
                    plt.colorbar(cb,ax=axes[1,2])
                    axes[1,1].set_title("post real: {1}, phi: {0:4.1f}".format(pv[real], real), loc="left")
                    axes[1,0].set_title("prior real: {1}, phi: {0:4.1f}".format(pr_pv[real],real),loc="left")
                    axes[1,2].set_title("truth", loc="left")

                    plt.tight_layout()
                    pdf.savefig()
                    plt.close(fig)
                    #plt.show()
                    #break
                    ireal += 1
                    if ireal > 20:
                        break
                    print(ireal)


def test_array_fmt(tmp_path):
    from pyemu.utils.pst_from import _load_array_get_fmt
    # psuedo ff option
    with open(Path(tmp_path, "test.dat"), 'w') as fp:
        fp.write("       3.000      3.0000      03.000\n"
                 "         3.0      3.0000      03.000")
    # will be converted to Exp format -- only safe option
    arr, fmt = _load_array_get_fmt(Path(tmp_path, "test.dat"))
    assert fmt == ''.join([" %11.4F"] * 3)
    assert arr.sum(axis=1).sum() == 18
    # actually space delim but could be fixed (first col is 1 wider)
    with open(Path(tmp_path, "test.dat"), 'w') as fp:
        fp.write("3.000 3.00 03.0\n"
                 "  3.0  3.0  03.")
    arr, fmt = _load_array_get_fmt(Path(tmp_path, "test.dat"))
    assert fmt == ''.join([" %4.1F"] * 3)
    # actually space delim but could be fixed (first col is 1 wider)
    with open(Path(tmp_path, "test.dat"), 'w') as fp:
        fp.write(" 3.000000000        3.00        03.0\n"
                 "         3.0         3.0         03.")
    arr, fmt = _load_array_get_fmt(Path(tmp_path, "test.dat"))
    assert fmt == ''.join([" %11.8F"] * 3)
    assert arr.sum(axis=1).sum() == 18
    # tru space delim option -- sep passed
    with open(Path(tmp_path, "test.dat"), 'w') as fp:
        fp.write("3.000 3.00000 03.000\n"
                 "3.0 3.0000 03.000")
    arr, fmt = _load_array_get_fmt(Path(tmp_path, "test.dat"), sep=' ')
    assert fmt == "%7.5F"
    assert arr.sum(axis=1).sum() == 18
    # tru space delim option with sep None
    with open(Path(tmp_path, "test.dat"), 'w') as fp:
        fp.write("3.000 3.00000 03.000\n"
                 "3.0 3.0000 03.000")
    arr, fmt = _load_array_get_fmt(Path(tmp_path, "test.dat"))
    assert fmt == "%7.5F"
    assert arr.sum(axis=1).sum() == 18
    # comma delim option
    with open(Path(tmp_path, "test.dat"), 'w') as fp:
        fp.write("3.000, 3.00000, 03.000\n"
                 " 3.0, 3.0000,03.000")
    arr, fmt = _load_array_get_fmt(Path(tmp_path, "test.dat"), sep=',')
    assert fmt == "%8.5F"
    assert arr.sum(axis=1).sum() == 18
    # partial sci note option (fixed format) but short
    with open(Path(tmp_path, "test.dat"), 'w') as fp:
        fp.write(" 00.3E01 30.0E-1   03.00\n"
                 "     3.0    3.00  03.000")
    arr, fmt = _load_array_get_fmt(Path(tmp_path, "test.dat"))
    assert fmt == ''.join([" %7.0E"] * 3)
    assert arr.sum(axis=1).sum() == 18
    try:
        # partial sci note option (fixed format) but short
        with open(Path(tmp_path, "test.dat"), 'w') as fp:
            fp.write(" 0.3E01 3.0E-1  03.00\n"
                     "    3.0   3.00 03.000")
        arr, fmt = _load_array_get_fmt(Path(tmp_path, "test.dat"))
    except ValueError:
        # should fail
        pass
    # sci note option fixed
    with open(Path(tmp_path, "test.dat"), 'w') as fp:
        fp.write("      3.0E00  30.0000E-1       03.00\n"
                 "         3.0        3.00      03.000")
    arr, fmt = _load_array_get_fmt(Path(tmp_path, "test.dat"))
    assert fmt == ''.join([" %11.4E"] * 3)
    assert arr.sum(axis=1).sum() == 18
    # free but not passing delim
    with open(Path(tmp_path, "test.dat"), 'w') as fp:
        fp.write(" 0.3E01   30.0E-1 03.00\n"
                 "3.0 3.00  03.000")
    arr, fmt = _load_array_get_fmt(Path(tmp_path, "test.dat"),
                                   fullfile=True)
    assert fmt == "%9.3G"
    assert arr.sum(axis=1).sum() == 18

    with open(Path(tmp_path, "test.dat"), 'w') as fp:
        fp.write(" 00.3E01,30.0E-1, 03.00\n"
                 "3.0, 3.00,03.000")
    arr, fmt = _load_array_get_fmt(Path(tmp_path, "test.dat"),
                                   fullfile=True, sep=',')
    assert fmt == "%8.3G"
    assert arr.sum(axis=1).sum() == 18
    # 1 col option
    with open(Path(tmp_path, "test.dat"), 'w') as fp:
        fp.write("3.0000000000\n30.000000E-1\n03.00000\n3.0\n3.00\n03.000")
    arr, fmt = _load_array_get_fmt(Path(tmp_path, "test.dat"))
    assert arr.shape == (6,1)
    assert fmt == "%12.10G"
    assert arr.sum(axis=1).sum() == 18


def test_array_fmt_pst_from(tmp_path):
    pf = PstFrom(Path("utils",'weird_array'),
                 Path(tmp_path, "weird_tmp"),
                 remove_existing=True)
    arr = np.loadtxt(Path(tmp_path, "weird_tmp", "ar.arr"))
    # pf.add_parameters("ar.arr", 'grid', zone_array=~np.isnan(arr),
    #                   mfile_sep=' ')
    pf.add_parameters("ar.arr", 'grid', zone_array=~np.isnan(arr))
    np.savetxt(Path(tmp_path, "weird_tmp", "ar2.arr"), arr, fmt="%15.8f",
               delimiter='')
    pf.add_parameters("ar2.arr", 'grid', zone_array=~np.isnan(arr))
    np.savetxt(Path(tmp_path, "weird_tmp", "ar3.arr"), arr, fmt="%15.8e",
               delimiter='')
    pf.add_parameters("ar3.arr", 'grid', zone_array=~np.isnan(arr))
    pf.add_observations("ar.arr", zone_array=~np.isnan(arr))
    pf.add_observations("ar2.arr", zone_array=~np.isnan(arr))
    pst = pf.build_pst()
    par = pst.parameter_data
    par.loc[par.sample(10).index, 'parval1'] = -100
    check_apply(pf)
    arr1 = np.loadtxt(Path(tmp_path, "weird_tmp", "ar.arr"))
    arr2 = np.loadtxt(Path(tmp_path, "weird_tmp", "ar2.arr"))
    arr3 = np.loadtxt(Path(tmp_path, "weird_tmp", "ar3.arr"))


if __name__ == "__main__":
    #mf6_freyberg_pp_locs_test()
    # invest()
    #freyberg_test(os.path.abspath("."))
    # freyberg_prior_build_test()
    #mf6_freyberg_test(os.path.abspath("."))
    #$mf6_freyberg_da_test()
    #shortname_conversion_test()
    #mf6_freyberg_shortnames_test()
    #mf6_freyberg_direct_test()

    #mf6_freyberg_thresh_test(".")

    plot_thresh("master_thresh")
    #plot_thresh("master_thresh_mm")
    #mf6_freyberg_varying_idomain()
    # xsec_test()
    # mf6_freyberg_short_direct_test()
    # mf6_add_various_obs_test()
    # mf6_subdir_test()
    # tpf = TestPstFrom()
    # tpf.setup()
    #tpf.test_add_array_parameters_to_file_list()
    #tpf.test_add_array_parameters_alt_inst_str_none_m()
    #tpf.test_add_array_parameters_alt_inst_str_0_d()
    # tpf.test_add_array_parameters_pps_grid()
    # tpf.test_add_list_parameters()
    # # pstfrom_profile()
    # mf6_freyberg_arr_obs_and_headerless_test()
    #usg_freyberg_test(".")
    #vertex_grid_test()
    #direct_quickfull_test()
    #list_float_int_index_test()
    #freyberg_test()




