import os
import pyemu

pyemu.utils.start_slaves('template',"pestpp","freyberg.pst",15,slave_root='.',port=4004)