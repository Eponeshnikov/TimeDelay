import numpy as np
import pandas as pd
import ast
import matplotlib.pyplot as plt
from scipy.special import gamma
from mpl_toolkits.mplot3d import axes3d


def H(k):
    s_sum_x = np.arange(1, k - 1)
    sum_s = np.sum(1 / s_sum_x - 0.577)
    res = 1 / np.log(2) * (k + np.log(2000 / 30) + gamma(k) + (k + 1) * sum_s)
    print(res)
    return res


def gen_list(arr):
    res = []
    for a in arr:
        l = ast.literal_eval(a)
        res.append(l)
    return res


def sort_df(dataframe, name, val):
    res = dataframe.loc[dataframe.index[dataframe[name] == val]]
    return res


def extract_peaks(df_sort):
    r_p_ = gen_list(df_sort['Right peaks'])
    l_p_ = gen_list(df_sort['Lost peaks'])
    f_p_ = gen_list(df_sort['False peaks'])
    real_p_ = gen_list(df_sort['Real peaks'])

    return r_p_, l_p_, f_p_, real_p_


def extract_peaks_ids(df_sort):
    r_p_ids_ = gen_list(df_sort['Right peaks ids'])
    l_p_ids_ = gen_list(df_sort['Lost peaks ids'])
    return r_p_ids_, l_p_ids_


def gen_real_right(real, right_ids):
    right_reals = []
    for i in range(len(real)):
        real_np = np.array(real[i])
        for k in range(len(right_ids[i])):
            right_ids[i][k] -= 1
        right_ids_np = np.array(right_ids[i])
        if len(right_ids_np) != 0:
            right_real = real_np[right_ids_np]
            right_reals.append(right_real)
        else:
            right_reals.append([])
    return right_reals


def merge_right_lost(r, l, f):
    total_res = []
    for i in range(len(r)):
        res = np.append(r[i], l[i])
        res = np.append(res, f[i])
        res = np.sort(res)
        total_res.append(res)
    return total_res


def append_peaks(rlf, real):
    for i in range(len(rlf)):
        if len(rlf[i]) > len(real[i]):
            add = np.ones(len(rlf[i]) - len(real[i])) * real[i][len(real[i]) - 1]
            real[i] = np.append(real[i], add)
        elif len(rlf[i]) < len(real[i]):
            add = np.ones(len(real[i]) - len(rlf[i])) * rlf[i][len(rlf[i]) - 1]
            rlf[i] = np.append(rlf[i], add)
    return rlf, real


def gen_diff(rlf, real):
    all_diff = []
    for i in range(len(rlf)):
        for_diff = np.vstack((rlf[i], np.array(real[i])))
        diff = np.diff(for_diff, axis=0)
        all_diff.append(np.abs(diff[0]))
    return all_diff


def append_diffs(diffs, max_len=None):
    length_max = 0
    length_min = 100
    for dif in diffs:
        if len(dif) > length_max:
            length_max = len(dif)
        if len(dif) < length_min:
            length_min = len(dif)
    if max_len is not None:
        length_max = max_len
    for dif in range(len(diffs)):
        if len(diffs[dif]) != 0:
            add = np.ones(length_max - len(diffs[dif])) * diffs[dif][len(diffs[dif]) - 1]
        else:
            add = np.ones(length_max - len(diffs[dif]))
        diffs[dif] = np.append(diffs[dif], add)
    diffs = np.array(diffs)
    for i in range(length_min, length_max):
        if max_len is None:
            diffs[:, i] = max(max(diffs[:, i]), max(diffs[:, i - 1]))
        '''else:
            diffs[:, i] = min(max(diffs[:, i]), max(diffs[:, i - 1]))'''
    return diffs, length_max


def gen_avgs(data):
    avgs = []
    for i in range(np.shape(data)[1]):
        avg = np.mean(data[:, i])
        avgs.append(avg)
    return avgs


def gen_f(real_avg, finded_avg, f_, l_):
    f = []
    x = []
    y = []
    for i in range(len(real_avg)):
        for j in range(len(finded_avg)):
            if j > i:
                real_avg_sum = real_avg[i] + real_avg[j]
                finded_avg_sum = finded_avg[i] + finded_avg[j]
                z = (j - i) / (finded_avg_sum + 10e-17)
                f.append(z)
                x.append(i + 1)
                y.append(j + 1)
    return x, y, f


def gen_res(file):
    results = []
    n = 0
    df = pd.read_csv('csv/' + file + '1.csv', index_col=0)
    # df = pd.read_csv('csv/gold1.csv', index_col=0)
    for i in range(2, 5):
        df1 = pd.read_csv('csv/' + file + str(i) + '.csv')
        # df1 = pd.read_csv('csv/gold' + str(i) + '.csv')
        df = pd.concat([df, df1])
    df = df.reset_index(drop=True)
    for dist in range(int(df['Distance'].min()), int(df['Distance'].max() + 1), 50):
        dist_sort = sort_df(df, 'Distance', dist)
        for ch in range(int(dist_sort['ChipN'].min()), int(dist_sort['ChipN'].max()) + 1, 5):
            ch_sort = sort_df(dist_sort, 'ChipN', ch)
            for d in range(int(ch_sort['Diffusers'].min()), int(ch_sort['Diffusers'].max()) + 1):
                # print('\r%s %s' % (' ', n), end='\r')
                n += 1
                d_sort = sort_df(ch_sort, 'Diffusers', d)
                r_p, l_p, f_p, real_p = extract_peaks(d_sort)
                spectrum = d_sort['Δω'].tolist()
                spectrum = np.round(np.mean(spectrum), 1)
                fa = 0
                tru = 0
                re = 0

                for p in r_p:
                    tru += len(p)
                for p in f_p:
                    fa += len(p)

                for p in real_p:
                    re += len(p)

                # print('Lp', tmp)
                l_ = 1 - tru / re

                f_ = fa / (fa + tru)
                r_p_ids, l_p_ids = extract_peaks_ids(d_sort)
                r_l_f_p = merge_right_lost(r_p, l_p, f_p)
                r_l_f_p, real_p = append_peaks(r_l_f_p, real_p)
                diffs_all = gen_diff(r_l_f_p, real_p)
                diffs_all, max_length = append_diffs(diffs_all)
                finded_avgs = gen_avgs(diffs_all)
                r_r = gen_real_right(real_p, r_p_ids)
                diffs_real = gen_diff(r_p, r_r)
                diffs_real, _ = append_diffs(diffs_real, max_len=max_length)
                real_avgs = gen_avgs(diffs_real)
                # print(real_avgs, finded_avgs)
                X, Y, F = gen_f(real_avgs, finded_avgs, f_, l_)

                '''print('Diffusers', d)
                print('Chip', ch)
                print('Distance', dist)
                print(X[np.argmax(F)])
                print(Y[np.argmax(F)])'''
                '''fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                ax.set_xlabel('First Ray')
                ax.set_ylabel('Second Ray')'''
                title = 'Scatterers: ' + str(d) + ' Distance: ' + str(dist) + 'm Opt. rays: (' \
                        + str(X[np.argmax(F)]) + '; ' + str(Y[np.argmax(F)]) + ') Δf = ' + str(spectrum) + 'MHz'
                name = str(d) + '_' + str(dist) + '_' + str(X[np.argmax(F)]) + '_' + \
                       str(Y[np.argmax(F)]) + '_' + str(spectrum) + '.png'
                '''ax.set_title(title)
                ax.scatter(X, Y, F, c=F)'''
                val = [l_, f_, spectrum, Y[np.argmax(F)] - X[np.argmax(F)], F[np.argmax(F)]]
                results.append(val)
                # plt.savefig('plots/opt_rays/' + name, dpi=300)
                # print('save')
                # results.append()
                # plt.show()
    return results


def analys_res(results_all):
    results1 = np.array(results_all[0])
    results2 = np.array(results_all[1])
    results = np.concatenate((results1, results2), axis=1)
    rows = [0, 1]  # , 7, 6]
    colors11 = ['black', 'b', 'darkviolet', 'lightsalmon']
    colors12 = ['deepskyblue', 'r', 'lightsalmon', 'chocolate']
    colors21 = ['r', 'magenta', 'lawngreen', 'limegreen']
    colors22 = ['teal', 'cyan', 'gold', 'stategrey']
    labels = ['β; ', 'α; ', 'Opt. func.; ', 'K; ']
    suffix = ['Gold ', 'No ']
    fig, ax = plt.subplots(constrained_layout=True, figsize=(16, 9))

    for row in range(len(rows)):
        y1 = results[:, rows[row]]  # gold
        y2 = results[:, rows[row] + 5]  # radio
        max_y = max(max(y1), max(y2))
        x1 = np.arange(0, len(y1))
        x2 = np.arange(8, len(y2) + 8)
        spec1 = results[:, 2]
        spec2 = results[:, 7]

        y_spl1 = np.split(y1, 8)
        y_spl2 = np.split(y2, 8)
        x_spl1 = np.split(x1, 8)
        x_spl2 = np.split(x2, 8)
        y11 = []  # gold wide
        y12 = []  # gold narrow
        y21 = []  # radio wide
        y22 = []  # radio narrow
        x11 = []
        x12 = []
        x21 = []
        x22 = []
        for i in range(len(y_spl1)):
            if i % 2 == 0:
                y11.append(y_spl1[i])
                x11.append(x_spl1[i])
                y21.append(y_spl2[i])
                x21.append(x_spl2[i])
            else:
                y12.append(y_spl1[i])
                x12.append(x_spl1[i])
                y22.append(y_spl2[i])
                x22.append(x_spl2[i])

        for i in range(len(x11)):
            if i == 0:
                ax.plot(x11[i], y11[i], color=colors11[row], label=labels[row] + suffix[0] + 'modulation')
                ax.plot(x21[i], y21[i], color=colors21[row], label=labels[row] + suffix[1] + 'modulation',
                        linestyle='dotted')
            else:
                ax.plot(x11[i], y11[i], color=colors11[row])
                ax.plot(x21[i], y21[i], color=colors21[row], linestyle='dotted')
            ax.plot(x11[i], y11[i], 'ob', color=colors11[row])
            ax.plot(x21[i], y21[i], 'ob', color=colors21[row])
            an11 = 'Δf = ' + str(spec1[i]) + ' MHz'
            an21 = 'Δf = ' + str(spec2[i]) + ' MHz'
            ax.annotate(an11, xy=(np.mean(x11[i]) - len(y1) * 0.04, max_y * 1.02),
                        xytext=(np.mean(x11[i]) - len(y1) * 0.04, max_y * 1.02))
            ax.annotate(an21, xy=(np.mean(x21[i]) - len(y1) * 0.04, max_y * 1.02),
                        xytext=(np.mean(x21[i]) - len(y1) * 0.04, max_y * 1.02))
            '''if i == 0:
                ax.plot(x12[i], y12[i], color=colors12[row], label=labels[row] + suffix[1] + 'modulation')
                ax.plot(x22[i], y22[i], color=colors22[row], label=labels[row] + suffix[1] + 'modulation',
                        linestyle='dotted')
            else:
                ax.plot(x12[i], y12[i], color=colors12[row])
                ax.plot(x22[i], y22[i], color=colors22[row], linestyle='dotted')
            ax.plot(x12[i], y12[i], 'ob', color=colors12[row])
            ax.plot(x22[i], y22[i], 'ob', color=colors22[row])
            an12 = 'Δf = ' + str(spec1[i + 8]) + ' MHz'
            an22 = 'Δf = ' + str(spec2[i + 8]) + ' MHz'
            ax.annotate(an12, xy=(np.mean(x12[i]) - len(y2) * 0.04, max_y * 1.02),
                        xytext=(np.mean(x12[i]) - len(y2) * 0.04, max_y * 1.02))
            ax.annotate(an22, xy=(np.mean(x22[i]) - len(y2) * 0.04, max_y * 1.02),
                        xytext=(np.mean(x22[i]) - len(y2) * 0.04, max_y * 1.02))'''

        xx_2 = np.arange(50 / 200 * len(y1) - 8, (200 + 1) / 200 * len(y1) - 8, 50 / 200 * len(y1))
        labl2 = ['50', '100', '150', '200']
        xx_1 = np.zeros(int(len(y1) / 4))
        for i in range(int(len(y1) / 4)):
            xx_1[i] = int((i + 1) // 2 * 8 - i % 2)
        labl1 = np.zeros(len(xx_1))
        labl1 = labl1.tolist()
        for i in range(len(xx_1)):
            labl1[i] = str(int(xx_1[i] % 8 + 3))

        ax.set_xticks(xx_1)
        ax.set_xticklabels(labl1)
        ax.set_xlabel('Scatterers', fontsize=16)
        secax = ax.secondary_xaxis('top')
        secax.set_xticks(xx_2)
        secax.set_xticklabels(labl2)
        secax.set_xlabel('Distance, m', fontsize=16)
        ax.grid()

        ax.set_ylim(0, max_y * 1.1)
        ax.set_xlim(-1, len(y1))

        vl = np.arange(16, 64, 16).astype(np.float64)
        vl -= 0.5
        ax.vlines(vl, ymin=-max_y, ymax=2 * max_y, color='grey', linestyle='dashdot')
        ax.legend(bbox_to_anchor=(1, 1), loc='upper left')

        plt.savefig('plots/analys_opt_rays/' + labels[row] + '_radiosig_gold.png', dpi=300)
        print('s')
        ax.clear()


res_all = (gen_res('gold'), gen_res('radiosig'))
analys_res(res_all)
