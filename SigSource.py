import numpy as np
import ast


class SigSource:
    def __init__(self):
        self.modulation = 'Rect'
        self.power_type = 'U'
        self.u = 0
        self.periods = 0
        self.freq = 0
        self.polys = None
        self.poly = None
        self.state = None
        self.m = []

    def rect(self):
        x, step = np.linspace(start=0, stop=5 * self.periods, num=1000, retstep=True)
        y = np.zeros(len(x))
        y[0:int(self.periods / step)] = self.u
        return x, y

    def radio_sig(self, mod=None):
        T = 2 * np.pi / self.freq
        if mod is not None:
            modulation = mod.astype(np.float)
            modulation *= np.deg2rad(180)
            self.periods = len(modulation)
            print(self.periods)
        else:
            modulation = np.zeros(self.periods)

        duration = self.periods * T

        # T = self.duration / self.freq
        # w = 2 * np.pi / T

        x, step = np.linspace(start=0, stop=10 * duration, num=1000 * self.periods, retstep=True)
        y = []
        for i in range(self.periods):
            tmp_x = np.linspace(start=T * i, stop=T * (i + 1), num=100)
            tmp_y = np.sin(self.freq * tmp_x + modulation[i]) * self.u
            y.extend(tmp_y)
        zer_y = np.zeros(len(x) - len(y))
        y.extend(zer_y)
        y = np.array(y)
        print(len(x), len(y))
        # y[int(duration / step):] = 0
        return x, y

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
        state = np.random.randint(2, size=length)
        return state

    def M_gen(self, state, L=None):
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
            self.m.append(curr_state[0])
            L -= 1

    def gold_code(self):
        self.polys = ast.literal_eval(self.polys)
        ms = []
        for i, poly in enumerate(self.polys):
            self.poly = poly
            if self.state is None:
                state = self.gen_state()
            else:
                state = ast.literal_eval(self.state)[i]
            self.m = []
            self.M_gen(state=state)
            ms.append(self.m)
        code = np.logical_xor(ms[0], ms[1])
        return code

    def generate_sig(self):
        sig_X, sig_Y = 0, 0
        if self.modulation == 'Rect':
            sig_X, sig_Y = self.rect()
        elif self.modulation == 'Radio Sig':
            sig_X, sig_Y = self.radio_sig()
        elif self.modulation == 'Gold':
            sig_X, sig_Y = self.radio_sig(self.gold_code())
        sig = (sig_X, sig_Y)
        return sig
