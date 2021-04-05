import os
import numpy as np
import pandas as pd
import flopy
import pyemu
try:
   os.remove('freyberg_transient.list')
except Exception as e:
   print('error removing tmp file:freyberg_transient.list')
try:
   os.remove('freyberg_transient.hds')
except Exception as e:
   print('error removing tmp file:freyberg_transient.hds')
pyemu.helpers.apply_list_pars()

pyemu.helpers.apply_array_pars()

pyemu.os_utils.run('mfnwt freyberg_transient.nam 1>freyberg_transient.nam.stdout 2>freyberg_transient.nam.stderr')
pyemu.gw_utils.apply_mflist_budget_obs('freyberg_transient.list',flx_filename='flux.dat',vol_filename='vol.dat',start_datetime='1-1-1970')
pyemu.gw_utils.apply_hds_obs('freyberg_transient.hds')
pyemu.gw_utils.apply_sfr_obs()
