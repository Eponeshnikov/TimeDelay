import numpy as np


class PhysChannel:
    def __init__(self, sig):
        self.sig = sig
        self.noise_u = 0
        self.diff = 0

    def noise(self):
        noise = np.random.normal(0, self.noise_u, len(self.sig))
        result = self.sig + noise
        return result

    def delays(self):
        pass
