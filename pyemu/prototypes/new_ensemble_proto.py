import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import pyemu


class Loc(object):
    def __init__(self,ensemble):
        self._ensemble = ensemble

    def __getitem__(self,item):
        return type(self._ensemble)(self._ensemble.pst, 
            self._ensemble._df.loc[item],
            istransformed=self._ensemble.istransformed)

class Iloc(object):
    def __init__(self,ensemble):
        self._ensemble = ensemble

    def __getitem__(self,item):
        return type(self._ensemble)(self._ensemble.pst, 
            self._ensemble._df.iloc[item],
            istransformed=self._ensemble.istransformed)



class Ensemble(object):
    def __init__(self,pst,df,istransformed=False):
        self._df = df
        self._istransformed = istransformed
        self.loc = Loc(self)
        self.iloc = Iloc(self)

    def __repr__(self):
        return self._df.__repr__()

    def __str__(self):
        return self._df.__str__()

    @property
    def transformed(self):
        return copy.deepcopy(self._transformed)
    

    def _transform(self):
        if self.transformed:
            return
        self._transformed = True
        return

    def _back_transform(self):
        if not self.transformed:
            return
        self._transformed = False
        return

    def __getattr__(self,item):
        if item == "loc":
            return self.loc[item]
        elif item == "iloc":
            return self.iloc[item]
        elif item in set(dir(self)):
            return getattr(self,item)
        elif item in set(dir(self._df)):
            lhs = self._df.__getattr__(item)
            print(item,type(lhs))
            if type(lhs) == type(self._df):
                return MyDf(lhs)
            else:
                return lhs
        else:
            raise AttributeError("Ensemble error: {0} not found in Ensemble or DataFrame attributes".format(item))
        return 

    def some_method(self,stuff):
        print("made it")
        return stuff

    def plot(self,*args,**kwargs):
        self._df.plot(*args,**kwargs)

    
    def _gaussian_draw(self,cov,mean_values,num_reals):
        if cov.isdiagonal

class ParameterEnsemble(Ensemble):
    def __init__(self,pst,df,istransformed=False):
        super(ParameterEnsemble,self).__init__(pst,df,istransformed)




df = pd.DataFrame(np.random.random(100))
pst = "1"
mdf = ParameterEnsemble(pst,df)
#print(df.loc[:,0],mdf.loc[:,0])
#print(type(df.loc[:,1:]),type(mdf.loc[:,1:]))
print(type(mdf.drop(0)))
print(mdf.some_method("i"))
mdf.to_csv("test")
#mdf.plot(kind="bar")
#plt.show()
#mdf.to_csv("test.csv")
#mdf.plot()
#plt.show()



