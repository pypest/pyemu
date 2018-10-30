from graphviz import Digraph

g = Digraph('G', filename='cluster_edge.gv')
g.attr(compound='true')
g.node_attr.update(color='lightblue2', style='filled')

# c.edges(['ab', 'ac', 'bd', 'cd'])
packages = {'pyemu': ['mat', 'plot', 'prototypes', 'pst', 'utils']}
files = {'mat': ['mat_handler'],
         'plot': ['plot_utils'],
         'prototypes': ['moouu', 'ensemble_method', 'smoother'],
         'pst': ['pst_controldata', 'pst_handler', 'pst_utils'],
        'utils':['geostats', 'gw_utils', 'helpers', 'optimization', 'os_utils', 'pp_utils', 'smp_utils'],
                'pyemu': ['en', 'ev', 'la', 'ObservationEnsemble', 'logger', 'mc', 'pyemu_warnings', 'sc']}
classes = {'mat_handler': ['Matrix', 'Jco', 'Cov', 'SparseMatrix'],
           'ensemble_method': ['EnsembleMethod'],
           'smoother': ['Phi', 'EnsembleSmoother'],
           'geostats': ['GeoStruct', 'OrdinaryKrige', 'Vario2d', 'ExpVario', 'GauVario', 'SphVario'],
           'en': ['Ensemble', 'ObservationEnsemble', 'ParameterEnsemble'],
           'ev': ['ErrVar'],
           'la': ['LinearAnalysis'],
           'logger': ['Logger'],
           'mc': ['MonteCarlo'],
           'pyemu_warnings': ['PyemuWarning']
           }
imports = {'mat_handler': ['Pst'],
           'ensemble_method': ['ParameterEnsemble', 'ObservationEnsemble', 'Cov', 'Matrix', 'Pst', 'Logger'],
           'smoother': ['ParameterEnsemble', 'ObservationEnsemble', 'Cov','Matrix', 'Pst', 'Logger', 'EnsembleMethod'],
           'pst_controldata':['PyemuWarning'],
           'pst_handler' : ['PyemuWarning', 'ControlData', 'SvdData', 'RegData', 'pst_utils', 'plot_utils'],
           'pst_utils':['PyemuWarning', 'pyemu'],
           'geostats':['Cov', 'SparseMatrix','PyemuWarning' ],
           'gw_utils':['pst_utils', 'os_utils', 'helpers', 'PyemuWarning'],
           'helpers':['pyemu', 'PyemuWarning'],
           'optimization' : [],
           'os_utils':[],
           'pp_utils':[],


           }
