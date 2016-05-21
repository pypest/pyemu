
def freyberg_smoother_test():
    import os
    import pyemu
    pst = pyemu.Pst(os.path.join("smoother","freyberg.pst"))
    num_reals = 10
    es = pyemu.EnsembleSmoother(pst)
    es.initialize(num_reals)
    es.update()




if __name__ == "__main__":
    freyberg_smoother_test()