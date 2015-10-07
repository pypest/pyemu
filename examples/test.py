import os
import numpy as np
import pyemu



jco = os.path.join("verification","henry","pest.jcb")

sc = pyemu.Schur(jco=jco,forecasts="pd_ten",verbose=True)
print(sc.prior_forecast)
print(sc.posterior_forecast)
print(sc.get_contribution_dataframe({'mult1':"mult1"}))
print(sc.get_importance_dataframe())

ev = pyemu.ErrVar(jco=jco,forecasts="pd_ten")
print(ev.get_errvar_dataframe(np.arange(0,20)))
print(ev.get_identifiability_dataframe(10))

ev = pyemu.ErrVar(jco=jco,forecasts="pd_ten",omitted_parameters="mult1")
print(ev.get_errvar_dataframe(np.arange(0,20)))
print(ev.get_identifiability_dataframe(10))

import matplotlib.pyplot as plt
mc = pyemu.MonteCarlo(jco=jco)
mc.draw(2000)
ax = mc.parensemble.loc[:,["kr01c01",'mult1']].plot(kind="scatter",x="mult1",y="kr01c01",color='r')

mc.project_parensemble()
ax = mc.parensemble.loc[:,["kr01c01",'mult1']].plot(ax=ax,kind="scatter",x="mult1",y="kr01c01",color='g')

mc = pyemu.MonteCarlo(pst=jco.replace(".jcb",".pst"),parcov=sc.posterior_parameter)
mc.draw(2000)
ax = mc.parensemble.loc[:,["kr01c01",'mult1']].plot(ax=ax,kind="scatter",x="mult1",y="kr01c01",color='b')

plt.show()

