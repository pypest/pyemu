
def res_and_ens_test():
    import pandas as pd
    import numpy as np
    import pyemu

    # make some fake residuals
    np.random.seed(42)
    t = np.linspace(1,20, 200)
    obs = t/10 * np.sin(np.pi*t)
    mod = obs+np.random.randn(200)*.5    
    obsnames = ['ob_t_{:03d}'.format(i) for i in range(len(t))]
    obsgroups = ['start_grp' if i<80 else 'end_grp' for i in range(len(t))]
    res = pd.DataFrame({'name':obsnames,
                        'group':obsgroups,
                        'measured':obs,
                        'modelled':mod,
                        'residual':obs-mod,
                        'weight':np.ones(len(t))})
    res.set_index(res['name'], inplace=True)
    np.random.seed(98)
    res.weight =  [float(i>.5) for i in np.random.random(200)]

    # and an ensemble version
    ens = pd.DataFrame(np.tile(obs,(10,1))+np.random.randn(10,200)*.5, columns=obsnames)
    ens.loc['base'] = mod

    # cook up a PEST file for obs and weights
    pst = pyemu.pst_utils.generic_pst(obs_names=obsnames)
    pst.observation_data['obsval'] = obs
    pst.observation_data['obsname'] = obsnames    

    pst.observation_data['weight'] = res.weight.values
    pst.observation_data['obgnme'] = obsgroups

    all_met_res_groups = pyemu.metrics.calc_metric_res(res)
    all_met_res_nodrop = pyemu.metrics.calc_metric_res(res, drop_zero_weight=False)
    all_met_res = pyemu.metrics.calc_metric_res(res, drop_zero_weight=False,bygroups=False)

    all_met_ens_groups = pyemu.metrics.calc_metric_ensemble(ens,pst)
    all_met_ens_nodrop = pyemu.metrics.calc_metric_ensemble(ens,pst, drop_zero_weight=False)
    all_met_ens = pyemu.metrics.calc_metric_ensemble(ens,pst, drop_zero_weight=False,bygroups=False)

    # make sure all the metrics match since 'base' in ens is same as single real in res
    assert all(all_met_ens.loc['base'].values == all_met_res.loc['single_realization'].values)
    assert all(all_met_ens_nodrop[all_met_ens_nodrop.columns].loc['base'].values == 
        all_met_res_nodrop[all_met_ens_nodrop.columns].loc['single_realization'].values)    
    assert all(all_met_ens_groups[all_met_ens_groups.columns].loc['base'].values == 
        all_met_res_groups[all_met_ens_groups.columns].loc['single_realization'].values)

    # test the functions - first the athens test (do they run?)
    fcns = pyemu.metrics.ALLMETRICS
    results = [fcns[i](mod,obs) for i in fcns]
    assert len(results) == len(fcns)

    # now spot check a couple obvious values
    m=np.array(range(100))
    o=np.array(range(100))
    assert fcns['nse'](m,o) == 1.0
    assert fcns['rmse'](m,o) == 0.0
    assert fcns['nrmse_sd'](m,o) == 0.0
    assert fcns['nrmse_mean'](m,o) == 0.0
    assert fcns['nrmse_iq'](m,o) == 0.0
    assert fcns['nrmse_maxmin'](m,o) == 0.0
    assert fcns['mse'](m,o) == 0.0
    assert fcns['mae'](m,o) == 0.0
    assert fcns['pbias'](m,o) == 0.0
    assert fcns['bias'](m,o) == 0.0
    assert fcns['relative_bias'](m,o) == 0.0
    assert fcns['standard_error'](m,o) == 0.0
    assert fcns['volumetric_efficiency'](m,o) == 1.0
    assert fcns['kge'](m,o) == 1.0
    
if __name__ == "__main__":
    res_and_ens_test()
