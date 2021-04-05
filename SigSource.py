import numpy as np
from numpy.fft import rfft, rfftfreq


def spectrum(sig, dt):
    result = rfft(sig)
    n = len(sig)
    freqs = rfftfreq(n, dt)
    return 2 * (np.abs(result / len(sig))), freqs


def add_zer(sig):
    zer = np.zeros(2 * len(sig))
    tmp = zer
    tmp = tmp.tolist()
    tmp.extend(sig)
    tmp.extend(zer)
    tmp = np.array(tmp)
    return tmp


class SigSource:
    def __init__(self):
        self.modulation = 'Rect'
        self.u = 0
        self.periods = 0
        self.freq = 0
        self.polys = []
        self.poly = []
        self.state = []

    def rect(self):
        x, step = np.linspace(0, self.periods, 1000, retstep=True)
        y = np.zeros(len(x))
        y[0:int(self.periods / step)] = self.u
        return x, y, step

    def radio_sig(self, mod=None):
        T = 2 * np.pi / self.freq
        if mod is not None:
            modulation = mod
            modulation *= np.deg2rad(180)
            self.periods = len(modulation)
        else:
            modulation = np.zeros(self.periods)

        duration = self.periods * T

        y = []
        for i in range(self.periods):
            tmp_x = np.linspace(T * i, T * (i + 1), 25, endpoint=False)
            if i == self.periods - 1:
                tmp_x = np.linspace(T * i, T * (i + 1), 25)
            tmp_y = np.sin(self.freq * tmp_x + modulation[i]) * self.u
            y.extend(tmp_y)
        x, step = np.linspace(0, duration/(2*np.pi), len(y), retstep=True)
        y = np.array(y)
        return x, y, step

    def gen_V(self):
        length = max(self.poly)
        row1 = np.zeros(length)
        for i in self.poly:
            row1[i - 1] = 1
        row1 = np.array([row1])
        tmp_mat = []
        for i in range(length - 1):
            tmp_row = np.zeros(length)
            tmp_row[i] = 1
            tmp_mat.append(tmp_row)
        tmp_mat = np.array(tmp_mat)
        mat = np.append(row1, tmp_mat, axis=0)
        return mat

    def gen_state(self):
        length = max(self.poly)
        state = np.empty(length)
        for s in range(length):
            state[s] = np.random.randint(2)
        return state

    def M_gen(self, state, L=None):
        res_state = []
        length = max(self.poly)
        if L is None:
            L = 2 ** length - 1
        curr_state = state
        while L > 0:
            length = max(self.poly)
            V = self.gen_V()
            state2xor = np.zeros(V.shape)
            for i in range(length):
                ids = np.where(V[i] == 1)
                tmp_st = np.zeros(length)
                for id in ids[0]:
                    tmp_st[id] = curr_state[id]
                state2xor[i] = tmp_st
            state2xor = np.transpose(state2xor)
            tmp_st = state2xor[0]
            next_state = []
            for state in range(len(state2xor) - 1):
                next_state = np.logical_xor(tmp_st, state2xor[state + 1])
                tmp_st = next_state
            curr_state = next_state
            res_state.append(curr_state[0])
            L -= 1
        res_state = res_state
        return res_state

    def gold_code(self):
        ms = []
        for i, poly in enumerate(self.polys):
            self.poly = poly
            if len(self.state) == 0:
                state = self.gen_state()
            else:
                state = self.state[i]
            ms.append(self.M_gen(state))
        code = np.logical_xor(ms[0], ms[1])
        code = code.astype(np.float)
        return code

    def generate_sig(self):
        sig_X, sig_Y, step = 0, 0, 0
        if self.modulation == 'Rect':
            sig_X, sig_Y, step = self.rect()
        elif self.modulation == 'Radio Sig':
            sig_X, sig_Y, step = self.radio_sig()
        elif self.modulation == 'Gold':
            sig_X, sig_Y, step = self.radio_sig(self.gold_code())
        sig = (sig_X, sig_Y, step)
        return sig
