import os
import pyemu

pyemu.utils.start_workers('template',"pestpp","freyberg.pst",15,worker_root='.',port=4004)