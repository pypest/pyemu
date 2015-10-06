from __future__ import print_function, division
from pyemu.la import linear_analysis

class influence(linear_analysis):

    def __init__(self,jco,**kwargs):
        if "forecasts" in kwargs.keys() or "predictions" in kwargs.keys():
            raise Exception("influence.__init__(): forecast\\predictions " +
                            "not  allowed in influence analyses")
        self.__hat = None
        super(influence, self).__init__(jco,**kwargs)

    @property
    def observation_influence(self):
        obs_inf = []
        for iobs, obs in enumerate(self.hat.row_names):
            hii = self.hat[iobs,iobs].x[0][0]
            obs_inf.append(hii/(1.0 - hii))
        return pandas.DataFrame({"obs_influence":obs_inf},index=self.hat.row_names)

    @property
    def dfbetas(self):
        return

    @property
    def cooks_d(self):
        return

    @property
    def studentized_res(self):
        return


    @property
    def scaled_res(self):
        if self.res is None:
            raise Exception("influence.scaled_res: no residuals loaded")
        return self.qhalf * self.res.loc[:,"residual"]

    @property
    def hat(self):
        if self.__hat is not None:
            return self.__hat
        self.__hat = self.qhalfx * (self.qhalfx.T * self.qhalfx).inv\
                     * self.qhalfx.T
        return self.__hat

    @staticmethod
    def test():
        #non-pest
        pnames = ["p1","p2","p3"]
        onames = ["o1","o2","o3","o4"]
        npar = len(pnames)
        nobs = len(onames)
        j_arr = np.random.random((nobs,npar))
        parcov = mhand.cov(x=np.eye(npar),names=pnames)
        obscov = mhand.cov(x=np.eye(nobs),names=onames)
        jco = mhand.matrix(x=j_arr,row_names=onames,col_names=pnames)

        s = influence(jco=jco,obscov=obscov)
        print(s.hat)
        print(s.observation_influence)

