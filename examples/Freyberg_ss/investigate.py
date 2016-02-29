import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyemu

#df = pd.read_csv('par_contrib.csv')
#print(df.sort(inplace=False,columns=["percent_reduce"],ascending=False))

pst = pyemu.Pst("freyberg.pst")
pst.observation_data.index = pst.observation_data.obsnme
pst.observation_data.loc[:,"weight"] = 0.0
pst.observation_data.loc["or08c03_1","weight"] = 1.0
la = pyemu.Schur(jco="freyberg.jcb",pst=pst,forecasts="sw_gw_0")
#par_sum = la.get_parameter_summary()
#par_sum.sort(columns=["percent_reduction"],inplace=True,ascending=False)
#print(par_sum)
jco_df = la.jco.to_dataframe()
ovec = jco_df.loc["or08c03_1",:].apply(np.abs).sort(inplace=False)
print(ovec)
y_df = la.forecasts[0].to_dataframe().sort(columns=["sw_gw_0"])
y_df = y_df.apply(np.abs).sort(axis=1,inplace=False,ascending=False)
#print(la.get_forecast_summary())
print(y_df)
#
# df_par = la.get_par_contribution()
# #df_par = df_par.sort(columns=["post"],inplace=False,ascending=False)
# df_par.to_csv("par_contrib.csv")
#
# print(df_par)

#jco_df.loc["or08c03_1",:].apply(np.abs).sort(inplace=False).plot(kind="bar")
#y_df.apply(np.abs).sort(axis=1,inplace=False).plot(kind="bar")
