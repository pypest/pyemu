
def freyberg_smoother_test():
    import os
    import pyemu
    pst = pyemu.Pst(os.path.join("..","..","verification","Freyberg","freyberg.pst"))
    num_reals = 10
    es = pyemu.LM_enRML(pst)
    es.initialize(num_reals)



if __name__ == "__main__":
    freyberg_smoother_test()