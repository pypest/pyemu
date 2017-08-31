import os
import pandas as pd
import pyemu

gs = pyemu.utils.geostats.read_struct_file("structure.dat")
#print(gs.variograms[0].a,gs.variograms[0].contribution)
#gs.variograms[0].a *= 10.0
#gs.variograms[0].contribution *= 10.0
gs.nugget = 0.0
print(gs.variograms[0].a,gs.variograms[0].contribution)
xy = pd.read_csv(os.path.join("..","freyberg.xy"))
full_parcov = gs.covariance_matrix(xy.x,xy.y,xy.name)
print(full_parcov.col_names)
#exit()
pst = pyemu.Pst("freyberg.pst")
zero_groups = pst.par_groups
zero_groups.remove("hk")
pyemu.helpers.zero_order_tikhonov(pst,par_groups=zero_groups)
pyemu.helpers.first_order_pearson_tikhonov(pst,cov=full_parcov,reset=False,abs_drop_tol=0.5)
print(pst.prior_information)
pst.control_data.pestmode = "regularization"
pst.control_data.noptmax = 10
pst.write("freyberg_reg.pst")

