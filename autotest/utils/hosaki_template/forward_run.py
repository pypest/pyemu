import pandas as pd
import numpy as np
pdf = pd.read_csv('par.csv')
pval1 = float(pdf.iloc[0,1])
pval2 = float(pdf.iloc[1,1])
term1 = (pval2**2)*np.e**(-pval2)
term2 = 1. - (8.*(pval1)) + (7.*(pval1**2))
term3 = (-(7./3.)*(pval1**3)) + (0.25*(pval1**4))
sim = (term2 + term3) * term1
with open('sim.csv','w') as f:
    f.write('obsnme,obsval\n')
    f.write('sim,'+str(sim)+'\n')
