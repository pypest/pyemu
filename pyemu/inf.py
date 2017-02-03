from __future__ import print_function, division
from pyemu.la import LinearAnalysis
import pandas
import numpy as np

class Influence(LinearAnalysis):

    def __init__(self,jco,**kwargs):
        if "forecasts" in kwargs.keys() or "predictions" in kwargs.keys():
            raise Exception("influence.__init__(): forecast\\predictions " +
                            "not  allowed in influence analyses")
        self.__hat = None
        self.__scaled_res = None
        self.__estimated_err_var = None
        self.__scaled_res = None
        self.__studentized_res = None
        self.__cooks_d = None
        self.__dfbetas = None

        super(Influence, self).__init__(jco,**kwargs)

    @property
    def observation_leverage(self):

        obs_inf = []
        for iobs, obs in enumerate(self.__hat.row_names):
            hii = self.hat.x[iobs,iobs]
            obs_inf.append(hii/(1.0 - hii))
        return pandas.DataFrame({"obs_influence":obs_inf},
                                index=self.hat.row_names)

    @property
    def scaled_res(self):
        # this is f in the equations in the PEST manual
        if self.pst.res is None:
            raise Exception("influence.scaled_res: no residuals loaded and \n"
                                "load from pst object failed")

        # read in the residuals dataframe
        res = self.pst.res.copy()
        # drop forecasts from the residuals vector
        if "forecasts" in self.pst.pestpp_options.keys() and\
                        self.pst.pestpp_options['forecasts'] is not None:
            forecasts = self.pst.pestpp_options['forecasts'].split(',')
            for cfor in forecasts:
                res = res.drop(cfor)

        return self.qhalf * np.atleast_2d(res.loc[:,"residual"].values).T

    @property
    def estimated_err_var(self):
        if self.__estimated_err_var is None:
            if self.pst.nobs < self.pst.npar:
                print('statistics valid only for overdetermined problems (npar<=nobs)')
                return np.inf
            else:
                self.__estimated_err_var = np.squeeze((self.scaled_res.T *
                                                       self.scaled_res).x)/ \
                                       (self.pst.nobs-self.pst.npar)

            return self.__estimated_err_var

    @property
    def hat(self):
        if self.__hat is not None:
            return self.__hat
        try:
            self.__hat = self.qhalfx * self.xtqx.inv * self.qhalfx.T
        except:
            print('Normal Equations results in Singular Matrix')
            return None
        return self.__hat

    @property
    def studentized_res(self):
        if self.pst.nobs < self.pst.npar:
            print('statistics valid only for overdetermined problems (npar<=nobs)')
            return None
        else:
            if self.__studentized_res is None:
                ev = self.estimated_err_var
                self.__studentized_res = []
                for i, scaled_res_i in enumerate(self.scaled_res.x):
                    h_ii = self.hat.x[i,i]
                    print(h_ii)
                    self.__studentized_res.append(scaled_res_i/(np.sqrt(ev * (1.0 - h_ii))))

            return self.__studentized_res
    @property
    def cooks_d(self):
        if self.pst.nobs < self.pst.npar:
            print('statistics valid only for overdetermined problems (npar<=nobs)')
            return None
        else:
            if self.__cooks_d is None:
                self.__cooks_d = []
                for i in range(self.pst.nobs):
                    h_ii = self.hat.x[i][i]
                    self.__cooks_d.append((1/self.pst.npar) * self.studentized_res[i]**2 * (h_ii/(1-h_ii)))

            return self.__cooks_d

    @property
    def dfbetas(self):
        return

        # NOTE!!!! This needs work. The documentation in PEST is a little incomplete
        # in that f_i should be f with observation i removed
        if self.__dfbetas is None:
            # form up the C matrix
            self.__C = self.__XtQXinv * self.qhalfx.T
            self.__dfbetas = np.zeros((self.pst.par, self.pst.nobs))
            for i in range(self.pst.nobs):
                h_ii = self.__hat.x[i][i]
                s_i = np.sqrt((1 / (self.pst.nobs - self.pst.npar - 1)) *
                              ((self.pst.nobs - self.pst.npar) * self.__estimated_err_var
                               - (self.__scaled_res[i] ** 2 / (1 - h_ii))))

                for j in range(self.pst.npar):
                    sumcjk2 = self.__C.x[:,j].T * self.__C[:,j]
                    self.__dfbetas[i,j] = self.C[i,j]/np.sqrt(sumjk2)

        else:
            return self.__dfbetas

if __name__ == '__main__':

    def test():
        #non-pest
        from pyemu.mat import mat_handler as mhand
        from pyemu.pst import Pst
        import numpy as np

        inpst = Pst('../verification/Freyberg/Freyberg_pp/freyberg_pp.pst')

        pnames = inpst.par_names
        onames = inpst.obs_names
        npar = inpst.npar
        nobs = inpst.nobs
        j_arr = np.random.random((nobs,npar))
        parcov = mhand.Cov(x=np.eye(npar),names=pnames)
        obscov = mhand.Cov(x=np.eye(nobs),names=onames)
        jco = mhand.Jco.from_binary('../verification/Freyberg/freyberg_pp/freyberg_pp.jcb')
        resf = '../verification/Freyberg/freyberg_pp/freyberg_pp.rei'
        s = influence(jco=jco,obscov=obscov, pst=inpst,resfile=resf)
        print(s.hat)
        print(s.observation_leverage)
        #v = s.studentized_res
        print(s.estimated_err_var)
        print(s.studentized_res)
    test()