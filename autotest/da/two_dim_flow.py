import os, sys
import numpy as np
import flopy
import platform
import pyemu

# https://modflowpy.github.io/flopydoc/tutorial2.html
bin_folder = r"..\..\bin"

def model_setup():

    # Model domain and grid definition
    Lx = 1000.
    Ly = 1000.
    ztop = 10.
    zbot = -50.
    nlay = 1
    nrow = 10
    ncol = 10
    delr = Lx / ncol
    delc = Ly / nrow
    delv = (ztop - zbot) / nlay
    botm = np.linspace(ztop, zbot, nlay + 1)
    hk = 1.
    vka = 1.
    sy = 0.1
    ss = 1.e-4
    laytyp = 1

    # Variables for the BAS package
    # Note that changes from the previous tutorial!
    ibound = np.ones((nlay, nrow, ncol), dtype=np.int32)
    strt = 10. * np.ones((nlay, nrow, ncol), dtype=np.float32)

    # Time step parameters
    nper = 3
    perlen = [1, 100, 100]
    nstp = [1, 100, 100]
    steady = [True, False, False]

    # Flopy objects
    modelname = 'mf2d'
    if platform.system().lower() == "windows":
        mf_exe = os.path.join(r".\win\MF_NWT.exe")
    else:
        mf_exe = r"\linux\mfnwt"
        m2s_exe = None
    exe_name = os.path.join(bin_folder,mf_exe)
    mf = flopy.modflow.Modflow(modelname, exe_name=exe_name)
    dis = flopy.modflow.ModflowDis(mf, nlay, nrow, ncol, delr=delr, delc=delc,
                                   top=ztop, botm=botm[1:],
                                   nper=nper, perlen=perlen, nstp=nstp, steady=steady)
    bas = flopy.modflow.ModflowBas(mf, ibound=ibound, strt=strt)
    lpf = flopy.modflow.ModflowLpf(mf, hk=hk, vka=vka, sy=sy, ss=ss, laytyp=laytyp, ipakcb=53)
    pcg = flopy.modflow.ModflowPcg(mf)

    # Make list for stress period 1
    stageleft = 10.
    stageright = 10.
    bound_sp1 = []
    for il in range(nlay):
        condleft = hk * (stageleft - zbot) * delc
        condright = hk * (stageright - zbot) * delc
        for ir in range(nrow):
            bound_sp1.append([il, ir, 0, stageleft, condleft])
            bound_sp1.append([il, ir, ncol - 1, stageright, condright])
    print('Adding ', len(bound_sp1), 'GHBs for stress period 1.')

    # Make list for stress period 2
    stageleft = 10.
    stageright = 0.
    condleft = hk * (stageleft - zbot) * delc
    condright = hk * (stageright - zbot) * delc
    bound_sp2 = []
    for il in range(nlay):
        for ir in range(nrow):
            bound_sp2.append([il, ir, 0, stageleft, condleft])
            bound_sp2.append([il, ir, ncol - 1, stageright, condright])
    print('Adding ', len(bound_sp2), 'GHBs for stress period 2.')

    # We do not need to add a dictionary entry for stress period 3.
    # Flopy will automatically take the list from stress period 2 and apply it
    # to the end of the simulation, if necessary
    stress_period_data = {0: bound_sp1, 1: bound_sp2}

    # Create the flopy ghb object
    ghb = flopy.modflow.ModflowGhb(mf, stress_period_data=stress_period_data)

    # Create the well package
    # Remember to use zero-based layer, row, column indices!
    pumping_rate = -100.
    wel_sp1 = [[0, nrow / 2 - 1, ncol / 2 - 1, 0.]]
    wel_sp2 = [[0, nrow / 2 - 1, ncol / 2 - 1, 0.]]
    wel_sp3 = [[0, nrow / 2 - 1, ncol / 2 - 1, pumping_rate]]
    stress_period_data = {0: wel_sp1, 1: wel_sp2, 2: wel_sp3}
    wel = flopy.modflow.ModflowWel(mf, stress_period_data=stress_period_data)

    stress_period_data = {}
    for kper in range(nper):
        for kstp in range(nstp[kper]):
            stress_period_data[(kper, kstp)] = ['save head',
                                                'save drawdown',
                                                'save budget',
                                                'print head',
                                                'print budget']
    oc = flopy.modflow.ModflowOc(mf, stress_period_data=stress_period_data,
                                 compact=True)

    # Write the model input files
    mf.write_input()

    # Run the model
    success, mfoutput = mf.run_model(silent=True, pause=False, report=True)
    if not success:
        raise Exception('MODFLOW did not terminate normally.')


if __name__ == "__main__":
    model_setup()