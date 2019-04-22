import numpy as np
import matplotlib.pyplot as plt

x0 = 10.0
n = 100
std = 0.5
xx = []
for j in range(300):
    x1 = np.copy(x0)
    xx = []
    for i in range(n):
        xx.append(x1)
        x1 = 0.99 * x1 + std * np.random.randn()

    plt.plot(xx)

pass



