import numpy as np


class SigSource:
    def __init__(self):
        self.modulation = 'Rect'
        self.power_type = 'U'
        self.power_val = 0
        self.duration = 0
        self.periods = 0
        self.N = None
        self.state = None

    def rect(self):
        x, step = np.linspace(start=0, stop=5*self.duration, num=1000, retstep=True)
        y = np.zeros(len(x))
        y[0:int(self.duration/step)] = self.power_val
        return x, y

    def radioSig(self):
        T = self.duration/self.periods
        w = 2*np.pi/T
        x, step = np.linspace(start=0, stop=5*self.duration, num=1000, retstep=True)
        y = np.sin(w * x)*self.power_val
        y[int(self.duration/step):] = 0
        return x, y

    def generateSig(self):
        sig_X, sig_Y = 0, 0
        if self.modulation == 'Rect':
            sig_X, sig_Y = self.rect()
        elif self.modulation == 'Radio Sig':
            sig_X, sig_Y = self.radioSig()
        sig = (sig_X, sig_Y)
        return sig

