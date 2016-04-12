from pyemu import MonteCarlo

mc = MonteCarlo(pst="pest.pst")
mc.draw(50)
mc.parensemble.to_csv("sweep_in.csv")