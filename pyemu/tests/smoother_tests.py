import os
if not os.path.exists("temp"):
    os.mkdir("temp")

def freyberg_smoother_test():
    import os
    import pyemu
    pst = pyemu.Pst(os.path.join("smoother","freyberg.pst"))
    #mc = pyemu.MonteCarlo(pst=pst)
    #mc.draw(2)
    #print(mc.parensemble)
    num_reals = 5
    es = pyemu.EnsembleSmoother(pst)
    es.initialize(num_reals)
    es.update()




if __name__ == "__main__":
    freyberg_smoother_test()