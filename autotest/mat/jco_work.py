import pandas as pd
import matplotlib.pyplot as plt
import pyemu

new_obs_names = ["test1","test2"]
new_df = pd.DataFrame({"obsnme":new_obs_names,
                       "weight":0.0,
                       "obgnme":"swgwex",
                       "obsval":-999.0},index=new_obs_names)
pst = pyemu.Pst("base_pest.pst")
pst.observation_data = pst.observation_data.append(new_df)
pst.write("test.pst")
exit()
jco1 = pyemu.Jco.from_binary("base_pest.jco")

sc = pyemu.Schur(pst="base_pest.pst",jco=jco1,verbose=True)
par_sum = sc.get_parameter_summary()
par_sum.sort_values(by="percent_reduction",inplace=True)
print(par_sum)
sc.get_par_group_contribution()
dw_df = sc.get_removed_obs_importance()

# la = pyemu.ErrVar(pst="base_pest.pst",jco=jco1)
# css_df = la.get_par_css_dataframe()
# css_df.sort_values(by="pest_css",inplace=True,ascending=False)
# print(css_df.pest_css)
#css_df.pest_css.iloc[:10].plot(kind="bar")
#plt.show()

# ident_df = la.get_identifiability_dataframe(20)
# print(ident_df.columns)
# ident_df.sort_values(by="ident",inplace=True,ascending=False)
# ident_df.ident.iloc[:20].plot(kind="bar",legend=False)
# plt.show()


# print(jco1.shape)
# new_par_names = jco1.par_names[:10]
# old_par_names = jco1.par_names[10:]
#
# jco1 = jco1.get(col_names=old_par_names)
#
# jco2 = pyemu.Jco.from_binary("base_pest.jco").get(col_names=new_par_names)
#
# print(jco1.shape,jco2.shape)
# jco_full = pyemu.concat([jco1,jco2])
# print(jco_full.shape)
#
#
#
#jco_df = jco_full.to_dataframe()
#par_sum = jco_df.sum(axis=1)
#print(par_sum)
