import os
import numpy as np
import pyemu


num_reals = 100

# jco = os.path.join("verification","henry","pest_ord.jco")
# sc = pyemu.Schur(jco=jco,forecasts="pd_ten",verbose=True)
# pc1 = sc.posterior_parameter

jco = os.path.join("verification","henry","pest.jcb")
sc = pyemu.Schur(jco=jco,forecasts="pd_ten",verbose=True)
pc2 = sc.posterior_parameter

# diff = pc1 - pc2
# print(diff.x.max())


#print(sc.prior_forecast)
#print(sc.posterior_forecast)
print(sc.get_parameter_summary().loc["mult1",:])
#print(sc.get_contribution_dataframe({'mult1':"mult1"}))
#print(sc.get_importance_dataframe())

#ev = pyemu.ErrVar(jco=jco,forecasts="pd_ten")
#print(ev.get_errvar_dataframe(np.arange(0,20)))
#print(ev.get_identifiability_dataframe(10))

#ev = pyemu.ErrVar(jco=jco,forecasts="pd_ten",omitted_parameters="mult1")
#print(ev.get_errvar_dataframe(np.arange(0,20)))
#print(ev.get_identifiability_dataframe(10))

parcov = pyemu.Cov()
parcov.from_ascii(os.path.join("verification","henry","post.cov"))
mc = pyemu.MonteCarlo(pst=jco.replace(".jcb",".pst"),parcov=parcov)
m1 = mc.parcov.get("mult1","mult1").x
print("predunc7:",m1)
mc.draw(num_reals)
print("draws from predunc7",np.var(mc.parensemble.loc[:,"mult1"]))

import matplotlib.pyplot as plt
mc = pyemu.MonteCarlo(jco=jco)
mc.draw(num_reals)
#ax = mc.parensemble.loc[:,["kr01c01",'mult1']].plot(kind="scatter",x="mult1",y="kr01c01",color='r')
new_en = mc.project_parensemble(inplace=False)
print("projected:",np.var(mc.parensemble.loc[:,"mult1"]))
#ax = mc.parensemble.loc[:,["kr01c01",'mult1']].plot(ax=ax,kind="scatter",x="mult1",y="kr01c01",color='g')

# mc = pyemu.MonteCarlo(pst=jco.replace(".jcb",".pst"),parcov=sc.posterior_parameter)
# m1 = mc.parcov.get("mult1","mult1")
# print(m1)
# ax = mc.parensemble.loc[:,["kr01c01",'mult1']].plot(ax=ax,kind="scatter",x="mult1",y="kr01c01",color='b')

#parcov = pyemu.Cov()
#parcov.from_ascii(os.path.join("verification","henry","post.cov"))
parcov = sc.posterior_parameter
mc = pyemu.MonteCarlo(pst=jco.replace(".jcb",".pst"),parcov=parcov)
mc.draw(num_reals)
m1 = mc.parcov.get("mult1","mult1").x
print("sc:",m1)
print("draws from sc:",np.var(mc.parensemble.loc[:,"mult1"]))

#ax = mc.parensemble.loc[:,["kr01c01",'mult1']].plot(ax=ax,kind="scatter",x="mult1",y="kr01c01",color='b')
#plt.show()

