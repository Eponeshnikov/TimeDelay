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

#H()
import pywt
from pylab import *
from numpy import *
discrete_wavelets = ['db5', 'sym4', 'coif5', 'haar']
print('discrete_wavelets-%s'%discrete_wavelets )
st='db4'
wavelet = pywt.DiscreteContinuousWavelet(st)
print(wavelet)
i=10
phi, psi, x = wavelet.wavefun(level=i)

#title("График самой вейвлет - функции -%s"%st)
'''plot(x,psi,linewidth=2, label='level=%s'%i)
grid()'''
#legend(loc='best')

title(st)
plt.plot(x,phi,linewidth=2, label='level=%s'%i)
#legend(loc='best')
grid()
show()
