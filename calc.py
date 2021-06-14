import numpy as np
import matplotlib.pyplot as plt


def H():
    from scipy.special import gamma
    k = np.linspace(1, 5, 100)
    s_sum_x = np.arange(1, max(k) - 1)
    sum_s = np.sum(1 / s_sum_x - 0.577)
    res = 1 / np.log(2) * (k + np.log(2000 / 30) + gamma(k) + (k + 1) * sum_s)
    plt.plot(k, res)
    plt.grid()
    plt.xlabel('k')
    plt.ylabel('H(k)')
    plt.show()

H()