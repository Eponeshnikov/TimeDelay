import ast
import numpy as np
from numba import float64

p = '[[4,3],[5,4,3,2,1]]'

'''def genV(poly):
    length = max(poly)
    row1 = np.zeros(length)
    for i in poly:
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


def gen_state1(l):
    state = np.random.randint(2, size=l)
    return state


def Mgen(poly, state, L=None):
    length = max(poly)
    if L is None:
        L = 2 ** length - 1
    length = max(poly)
    V = genV(poly)
    # V = np.transpose(V)
    curr_state = state
    state2xor = np.zeros(V.shape)
    for i in range(length):
        ids = np.where(V[i] == 1)
        tmp_st = np.zeros(length)
        # print(ids)
        for id in ids[0]:
            tmp_st[id] = curr_state[id]
        state2xor[i] = tmp_st
    state2xor = np.transpose(state2xor)
    tmp_st = state2xor[0]
    next_state = []
    for state in range(len(state2xor) - 1):
        next_state = np.logical_xor(tmp_st, state2xor[state + 1])
        tmp_st = next_state
    # print(next_state)
    curr_state = next_state
    res.append(curr_state[0])
    L -= 1
    # print(L)
    # print(res)
    if L == 0:
        return res
    else:
        Mgen(poly, curr_state, L=L)


def gen_state(l):
    length = l
    state = np.empty(length, dtype=float)
    for s in range(length):
        state[s] = np.random.randint(2)
    return state

res = []
pol = ast.literal_eval(p)
# stat = gen_state(2)
stat = [1, 1, 1, 1]
# print(pol[0])
Mgen(pol[0], stat)'''

# print(gen_state(5))
from scipy.stats import expon, lognorm, norm
from matplotlib import pyplot as plt


def rect(per):
    x, step = np.linspace(0, per, 100, retstep=True)
    y = np.zeros(len(x))
    y[0:int(per / step)] = 1
    # plt.plot(x, y)
    return x, y, step


def radio_sig(freq, per):
    T = 2 * np.pi / freq

    duration = per * T

    x, step = np.linspace(0, duration, 250 * per, retstep=True)
    y = []
    for i in range(per):
        tmp_x = np.linspace(T * i, T * (i + 1), 25, endpoint=False)
        if i == per - 1:
            tmp_x = np.linspace(T * i, T * (i + 1), 25)
        tmp_y = np.sin(freq * tmp_x)
        y.extend(tmp_y)
    zer_y = np.zeros(len(x) - len(y))
    y.extend(zer_y)
    y = np.array(y)
    plt.plot(x, y)
    plt.show()
    return x, y, step


def noise(sig, u):
    noise = np.random.normal(0, u, len(sig))
    result = sig + noise
    return result


def delays(diff, sig, dt):
    # y = expon.cdf(x)
    ys = expon.rvs(size=diff, scale=1000)
    ys = np.sort(ys)
    max_shift = max(ys)
    print(max_shift, min(ys))
    zer = np.zeros(round(max_shift))
    signal = sig.tolist()
    signal.extend(zer)
    del_sig = np.zeros(len(signal))
    r = 100
    for y in range(len(ys)):
        r += ys[y]
        u = 10 * (100 / r) ** 2.7

        print(r, u)

        del_sig[int(ys[y]):int(len(sig) + int(ys[y]))] += sig * u
    signal_n = noise(del_sig, 1 / 10)
    acf = np.convolve(sig, signal_n)
    # print()
    # plt.hist(y)
    fig, axs = plt.subplots(3)
    axs[0].plot(del_sig)
    axs[1].plot(signal_n)
    axs[2].plot(acf)
    axs[0].grid()
    axs[1].grid()
    axs[2].grid()
    plt.show()


# sign_x, signy, dt = radio_sig(1600, 5)
# sign_x, signy, dt = rect(5)

# delays(5, signy, dt)
def gen_delays(diff, scale):
    ys = expon.rvs(size=diff, scale=scale)
    # ys = np.arange(start=scale, stop=scale * diff + scale, step=scale)
    ys = np.sort(ys)
    print(np.mean(ys))
    print(ys)
    # x = np.zeros(len(ys)) + 1
    # plt.scatter(ys, x)
    # plt.show()
    return ys


def gen_amps(r0, a0, n, ys, dist, sigm, rand=True):
    r = r0 + dist
    us = []
    for y in range(len(ys)):
        if y == 0:
            r += 300 * ys[y]
        else:
            r += 300 * (ys[y] - ys[y - 1])
        u = a0 * (r0 / r) ** n
        us.append(u)
    if rand:
        for u in range(len(us)):
            sigma = 10 ** (sigm / 10) * us[u]
            us[u] += lognorm.rvs(sigma)
    plt.stem(ys, us)
    plt.show()
    return us


def add_delays(sig, ts, us, dt):
    max_shift = max(ts)
    zer = np.zeros(round(max_shift / dt) + 1)
    signal = sig.tolist()
    signal.extend(zer)
    del_sig = np.zeros(len(signal))
    for t in range(len(ts)):
        del_sig[int(ts[t] / dt):int(len(sig) + int(ts[t] / dt))] += sig * us[t]
    x, xs = np.linspace(start=0, stop=len(del_sig) * dt, num=len(del_sig), retstep=True)
    print(dt - xs)
    plt.plot(x, del_sig)
    plt.show()
    return del_sig


'''sign_x, signy, dt = rect(0.2)
ts = gen_delays(4, 2)
us = gen_amps(100, 100, 2.7, ts, 2000, 9)
add_delays(signy, ts, us,dt)'''


def test():
    x = np.linspace(0, 60, 200)
    # res = lognorm.rvs(22, size=1000)
    res = np.random.normal(0, 81, size=1000)
    res1 = np.exp(res)
    plt.hist(res)
    plt.show()


#test()
def H():
    from scipy.special import gamma
    k = np.linspace(2, 5, 100)
    s_sum_x = np.arange(1, max(k)-1)
    sum_s = np.sum(1/s_sum_x - 0.577)
    res = 1/np.log(2)*(k + np.log(2000/30) + gamma(k) + (k+1)*sum_s)
    plt.plot(k, res)
    plt.grid()
    plt.xlabel('k')
    plt.ylabel('H(k)')
    plt.show()
H()
