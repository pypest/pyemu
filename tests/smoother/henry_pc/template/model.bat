@echo off
del model\ref_coarse\hk.ref
del model\ref_coarse\vk.ref
del model\henry_coarse.wel
exe\plproc.exe misc\plproc_coarse.in >nul
exe\par2par.exe misc\par2par_coarse.in >nul
cd model
swt_v4x64.exe henry_coarse.nam_swt >nul
cd ..
exe\mod2obs.exe <misc\mod2obs_head_coarse.in >nul
exe\mod2obs.exe <misc\mod2obs_conc_coarse.in >nul
exe\get_dist_pred.exe <misc\pred_dist_half_henry_coarse.in >nul
exe\get_dist_pred.exe <misc\pred_dist_ten_henry_coarse.in >nul
exe\get_dist_pred.exe <misc\pred_dist_one_henry_coarse.in >nul