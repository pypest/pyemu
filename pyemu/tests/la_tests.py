def schur_test_nonpest():
    import numpy as np
    from pyemu import Matrix, Cov, Schur
    #non-pest
    pnames = ["p1","p2","p3"]
    onames = ["o1","o2","o3","o4"]
    npar = len(pnames)
    nobs = len(onames)
    j_arr = np.random.random((nobs,npar))
    jco = Matrix(x=j_arr,row_names=onames,col_names=pnames)
    parcov = Cov(x=np.eye(npar),names=pnames)
    obscov = Cov(x=np.eye(nobs),names=onames)
    forecasts = "o2"

    s = Schur(jco=jco,parcov=parcov,obscov=obscov,forecasts=forecasts)
    print(s.get_parameter_summary())
    print(s.get_forecast_summary())

    #this should fail
    passed = False
    try:
        print(s.get_contribution_dataframe_groups())
        passed = True
    except Exception as e:
        print(str(e))
    if passed:
        raise Exception("should have failed")

    #this should fail
    passed = False
    try:
        print(s.get_importance_dataframe_groups())
        passed = True
    except Exception as e:
        print(str(e))
    if passed:
        raise Exception("should have failed")

    print(s.get_contribution_dataframe({"group1":["p1","p3"]}))

    print(s.get_importance_dataframe({"group1":["o1","o3"]}))


def schur_test():
    import os
    from pyemu import Schur
    w_dir = os.path.join("..","..","verification","henry")
    forecasts = ["pd_ten","c_obs10_2"]
    sc = Schur(jco=os.path.join(w_dir,"pest.jcb"),forecasts=forecasts)
    print(sc.prior_forecast)
    print(sc.posterior_forecast)
    print(sc.get_contribution_dataframe_groups())
    print(sc.get_importance_dataframe_groups())


def errvar_test_nonpest():
    import numpy as np
    from pyemu import ErrVar, Matrix, Cov
    #non-pest
    pnames = ["p1","p2","p3"]
    onames = ["o1","o2","o3","o4"]
    npar = len(pnames)
    nobs = len(onames)
    j_arr = np.random.random((nobs,npar))
    jco = Matrix(x=j_arr,row_names=onames,col_names=pnames)
    parcov = Cov(x=np.eye(npar),names=pnames)
    obscov = Cov(x=np.eye(nobs),names=onames)
    forecasts = "o2"

    omitted = "p3"

    e = ErrVar(jco=jco,parcov=parcov,obscov=obscov,forecasts=forecasts,
               omitted_parameters=omitted)
    svs = [0,1,2,3,4,5]
    print(e.get_errvar_dataframe(svs))

def errvar_test():
    import os
    from pyemu import ErrVar
    w_dir = os.path.join("..","..","verification","henry")
    forecasts = ["pd_ten","c_obs10_2"]
    ev = ErrVar(jco=os.path.join(w_dir,"pest.jcb"),forecasts=forecasts)
    print(ev.prior_forecast)
    print(ev.get_errvar_dataframe())

if __name__ == "__main__":
    schur_test_nonpest()
    schur_test()
    errvar_test_nonpest()
    errvar_test()