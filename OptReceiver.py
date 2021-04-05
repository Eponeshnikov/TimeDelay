import numpy as np


class OptReceiver:
    def __init__(self):
        self.h = None
        self.sig = None

    def matched_fil(self):
        return np.convolve(self.h, self.sig, mode='same')
