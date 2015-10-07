import os
import numpy as np
import pyemu

jco = os.path.join("examples","henry","pest.jcb")

sc = pyemu.sc.Schur(jco=jco,forecasts="pd_ten")
print(sc.prior_forecast)
print(sc.posterior_forecast)
print(sc.prior_parameter)
print(sc.posterior_parameter)

ev = pyemu.ev.ErrVar(jco=jco,forecasts="pd_ten",omitted_parameters="mult1")
print(ev.get_errvar_dataframe(np.arange(0,20)))
print(ev.get_identifiability_dataframe(10))

mc = pyemu.mc.MonteCarlo(jco=jco)
mc.draw(1000)
mc.project_parensemble()
import matplotlib.pyplot as plt
mc.parensemble.loc[:,'mult1'].plot(kind="hist")
plt.show()

mc = pyemu.mc.MonteCarlo(pst=jco.replace(".jcb",".pst"),parcov=sc.posterior_parameter)
mc.draw(1000)
mc.parensemble.loc[:,'mult1'].plot(kind="hist")
plt.show()

