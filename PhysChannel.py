import numpy as np
from scipy.stats import expon, norm, lognorm


def add_delays(sig, ts, us, dt):
    max_shift = max(ts)
    zer = np.zeros(round(max_shift/dt) + 1)
    signal = sig.tolist()
    signal.extend(zer)
    del_sig_y = np.zeros(len(signal))
    for t in range(len(ts)):
        del_sig_y[int(ts[t]/dt):int(len(sig) + int(ts[t]/dt))] += sig * us[t]
    del_sig_x = np.linspace(start=0, stop=len(del_sig_y)*dt, num=len(del_sig_y), dtype=np.float64)
    return del_sig_x, np.array(del_sig_y, dtype=np.float64)


class PhysChannel:
    def __init__(self):
        self.sig = None
        self.sig_step = None
        self.noise_u = 0
        self.diff = 0
        self.scale = 1
        self.r0 = 10
        self.a0 = 10
        self.sigma = 9
        self.n = 2.7
        self.dist = 1000

    def gen_noise(self, size):
        noise = np.random.normal(0, self.noise_u, size=size)
        return noise

    def gen_delays(self, uniform=False):
        if uniform is False:
            ts = expon.rvs(size=self.diff, scale=self.scale)
            ts = np.sort(ts)
        else:
            ts = np.arange(start=self.scale, stop=self.scale * self.diff + self.scale, step=self.scale)
        return ts

    def gen_amps(self, ts, rand=True):
        r = self.r0 + self.dist
        us = []
        for y in range(len(ts)):
            if y == 0:
                r += 300 * ts[y]
            else:
                r += 300 * (ts[y] - ts[y - 1])
            u = self.a0 * (self.r0 / r) ** self.n
            us.append(u)
        '''from matplotlib import pyplot as plt
        plt.plot(ts, us)
        plt.show()'''
        if rand:
            for u in range(len(us)):
                y = np.random.normal(0, self.sigma ** 2, size=1)
                alpha = 10 ** (y / 20)
                us[u] *= alpha
        return us


