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
        self.ch_freq = 1
        self.ch_n = 1
        self.polys = []
        self.poly = []
        self.state = []
        self.bc = 0

    def rect(self):
        x, step = np.linspace(0, self.periods, 1000, retstep=True)
        y = np.zeros(len(x))
        y[0:int(self.periods / step)] = self.u
        return x, y, step

    def radio_sig(self, mod=None):
        T1 = 2 * np.pi / (self.ch_freq * self.ch_n)
        T = 2 * np.pi / self.freq
        devide = T1 / T
        T *= devide

        if mod is not None:
            modulation = np.copy(mod)
            modulation *= np.deg2rad(180)
            self.periods = len(modulation)
        else:
            modulation = np.zeros(self.periods)

        duration = self.periods * T

        y = []
        for i in range(self.periods):
            tmp_x = np.linspace(T * i, T * (i + 1), 25 * int(devide), endpoint=False)
            if i == self.periods - 1:
                tmp_x = np.linspace(T * i, T * (i + 1), 25 * int(devide))
            tmp_y = np.sin(self.freq * tmp_x + modulation[i]) * self.u
            y.extend(tmp_y)
        x, step = np.linspace(0, duration / (2 * np.pi), len(y), retstep=True)
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
        if np.sum(state) == 0:
            state[0] = 1
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
        return res_state

    def gen_M_seq(self):
        ms = []
        bc = self.bc
        self.gen_balanced_state()
        for i, poly in enumerate(self.polys):
            self.poly = poly
            if len(self.state) == 0:
                state = self.gen_state()
                if bc:
                    if i == 0:
                        state = self.gen_balanced_state()
                    elif i == 1:
                        state[0] = 0
            else:
                state = self.state[i]
            ms.append(self.M_gen(state))
        return ms

    def gen_balanced_state(self):
        main_poly = self.polys[0]
        for poly in self.polys:
            if len(poly) < len(main_poly):
                main_poly = poly
        main_poly = np.flip(main_poly)
        f = np.zeros(max(main_poly) + 1)
        f[main_poly] = 1
        f[0] = 1

        g = np.zeros(len(f))
        for i in range(len(f)):
            g[i] = f[i] * (i + 1)
        g = np.mod(g, 2)

        seq_ids = []

        while len(f) > 1:
            xor = np.logical_xor(g, f)
            xor = xor.astype(np.int)
            a = 0
            while xor[a] == 0:
                a += 1
            seq_ids.append(a)
            g = g[:len(g) - a]
            f = f[a:]
        for i in range(1, len(seq_ids)):
            seq_ids[i] = seq_ids[i - 1] + seq_ids[i]
        seq_ids = np.array(seq_ids)
        seq_ids -= 1
        seq = np.zeros(max(main_poly))
        seq[seq_ids] = 1
        return np.flip(seq)

    def gold_code(self, ms):
        code = np.logical_xor(ms[0], ms[1])
        code = code.astype(np.float)
        return code

    def check_balanced(self, code):
        ones = len(np.where(code == 1)[0])
        zer = len(np.where(code == 0)[0])
        balanced = 0
        if np.abs(ones - zer) == 1:
            balanced = 1
        return balanced

    def generate_sig(self):
        sig_X, sig_Y, step = 0, 0, 0
        if self.modulation == 'Rect':
            sig_X, sig_Y, step = self.rect()
        elif self.modulation == 'Radio Sig':
            sig_X, sig_Y, step = self.radio_sig()
        elif self.modulation == 'Gold':
            sig_X, sig_Y, step = self.radio_sig(self.gold_code(self.gen_M_seq()))
        sig = (sig_X, sig_Y, step)
        return sig
