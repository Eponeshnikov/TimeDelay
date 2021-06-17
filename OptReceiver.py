import matplotlib.pyplot as plt
import numpy as np
import pywt
from scipy.signal import find_peaks
from scipy.interpolate import interp1d
import os



class OptReceiver:
    def __init__(self):
        self.h = None
        self.sig = None
        self.us = None
        self.noise_u = 0
        self.rect = False
        self.gold = False
        self.dir_name = ''
        self.dev = 0
        self.dt = 1
        self.save_fig = 0
        self.match_flag = 1
        self.border_flag = 1

    def matched_fil(self):
        res = np.convolve(np.flip(self.h), self.sig, mode='same') / len(self.sig)
        if self.dev and self.match_flag:
            plt.figure(figsize=(10, 5.6))
            x = np.linspace(0, len(res) * self.dt, len(res))
            plt.plot(x, res)
            plt.grid()
            plt.xlabel('t, us')
            plt.ylabel('h(t)')
            plt.title('Response of correlation receiver')
            self.match_flag = 0
            if self.save_fig:
                if not os.path.exists(self.dir_name):
                    os.mkdir(self.dir_name)
                plt.savefig(self.dir_name + '/conv.png', dpi=400)
            plt.clf()
            plt.plot(x, np.abs(res))
            plt.grid()
            plt.xlabel('t, us')
            plt.ylabel('h(t)')
            plt.title('Response of correlation receiver')
            if self.save_fig:
                plt.savefig(self.dir_name + '/conv_abs.png', dpi=400)
        return res

    def gen_min_height(self):
        size = len(self.matched_fil())
        noise = np.random.normal(0, self.noise_u, size=size)
        conv = np.convolve(np.flip(self.h), noise, mode='same') / len(noise)
        return np.std(conv, ddof=1)

    def denoise(self, data, threshold):
        # Create wavelet object and define parameters
        w = pywt.Wavelet('sym4')
        maxlev = pywt.dwt_max_level(len(data), w.dec_len)
        # Decompose into wavelet components, to the level selected:
        coeffs = pywt.wavedec(data, 'sym4', level=maxlev)
        if self.dev:
            plt.figure(figsize=(16, 9))
        for i in range(1, len(coeffs)):
            if self.dev:
                x = np.linspace(0, len(data) * self.dt, len(coeffs[i]))
                plt.subplot(maxlev, 1, i)
                plt.plot(x, coeffs[i], label='Before threshold')
                plt.grid()
                if i == 1:
                    plt.title('Wavelet coefficients')
                if i == len(coeffs) - 1:
                    plt.xlabel('t, us')
                else:
                    plt.gca().axes.get_xaxis().set_visible(False)
            coeffs[i] = pywt.threshold(coeffs[i], threshold * max(coeffs[i]))
            if self.dev:
                x = np.linspace(0, len(data) * self.dt, len(coeffs[i]))
                plt.plot(x, coeffs[i], label='After threshold')
        if self.dev:
            plt.legend(bbox_to_anchor=(0.98, 15), loc='upper left')

        if self.save_fig:
            plt.savefig(self.dir_name + '/wavelet_coefs.png')
        datarec = pywt.waverec(coeffs, 'sym4')
        if self.dev:
            plt.figure(figsize=(10, 5.6))
            x = np.linspace(0, len(datarec) * self.dt, len(datarec))
            plt.plot(x, datarec)
            plt.grid()
            plt.xlabel('t, us')
            plt.ylabel('U, μV')
            plt.title('Denoised response')
            if self.save_fig:
                plt.savefig(self.dir_name + '/conv_denoised.png', dpi=400)
            plt.clf()
            plt.plot(x, np.abs(datarec))
            plt.grid()
            plt.xlabel('t, us')
            plt.ylabel('U, μV')
            plt.title('Denoised response')
            if self.save_fig:
                plt.savefig(self.dir_name + '/conv_denoised_abs.png', dpi=400)
        return datarec

    def gen_border(self):
        zer = np.zeros(int(len(self.h) / 10))
        h = np.append(zer, self.h)
        h = np.append(h, zer)
        conv = np.correlate(h, h, mode='full') / len(self.h)
        border = np.zeros(int(len(conv)))
        if not self.rect:
            conv /= max(conv)
            peaks, _ = find_peaks(conv, distance=int(len(conv) / 20))
            index = np.arange(len(conv))
            u_p = interp1d(index[peaks], conv[peaks], kind='quadratic', bounds_error=False, fill_value=0.0)
            for k in range(0, len(conv)):
                border[k] = u_p(k)
            border += max(border) / 15
            border /= max(border)
            border = np.where(border > 0.1, border, 0.1)
        else:
            border = conv / max(conv)
        if self.dev and self.border_flag:
            plt.figure()
            x = np.linspace(0, len(border) * self.dt, len(border))
            plt.plot(x, border)
            plt.grid()
            plt.xlabel('t, us')
            plt.title('Border')
            self.border_flag = 0
            if self.save_fig:
                plt.savefig(self.dir_name + '/border.png', dpi=400)
        return border

    def find_peaks_(self):
        threshold = self.noise_u / (max(self.us) * 6)  # Threshold for filtering
        if threshold > 0.6:
            threshold = 0.6
        denoised = self.denoise(self.matched_fil(), threshold)
        denoised_copy = np.copy(denoised)
        index = np.arange(len(denoised))
        height = self.gen_min_height() * 2.5
        peaks, _ = find_peaks(denoised, height=height)
        peaks_y = denoised[peaks]
        peaks_y_copy = np.copy(peaks_y)
        peaks_x = index[peaks]
        peaks_zer = np.zeros(len(denoised))
        peaks_zer[peaks_x] = peaks_y
        all_peaks_x = []
        all_peaks_y = []
        borders_x = []
        borders_y = []

        while np.sum(peaks_y) != 0:
            border = self.gen_border()
            max_peak_x = peaks_x[np.argmax(peaks_y)]
            border_peak_x = np.argmax(border)
            left_x = max_peak_x - border_peak_x
            right_x = max_peak_x + len(border) - border_peak_x
            border_left_x = 0
            border_right_x = len(border)
            if max_peak_x - border_peak_x < 0:
                left_x = 0
                border_left_x = border_peak_x - max_peak_x
            if max_peak_x + len(border) - border_peak_x > len(denoised):
                right_x = len(denoised)
                border_right_x = len(denoised) - max_peak_x - len(border) + border_peak_x

            border /= max(border)
            border *= peaks_y[np.argmax(peaks_y)]
            border -= max(border) / 1000

            borders_y.append(border[border_left_x:border_right_x])
            tmp_denoised = peaks_zer[left_x:right_x]
            tmp_denoised = np.where(tmp_denoised > 0, tmp_denoised, 0)
            tmp_x = np.arange(left_x, right_x)
            borders_x.append(tmp_x)
            tmp_peaks_x, _ = find_peaks(tmp_denoised, height=border[border_left_x:border_right_x], distance=20)

            if len(tmp_peaks_x) >= 2:
                for_del = []
                for i, peak in enumerate(tmp_peaks_x):
                    condition = np.abs(tmp_denoised[peak] / max(tmp_denoised)
                                       - border[border_left_x:border_right_x][peak] / max(tmp_denoised))
                    if condition < 0.15 and tmp_denoised[peak] != max(tmp_denoised[tmp_peaks_x]) \
                            or tmp_denoised[peak] < height:
                        for_del.append(i)
                for_del = np.array(for_del)
                if len(for_del) > 0:
                    tmp_peaks_x = np.delete(tmp_peaks_x, for_del)

                if len(tmp_peaks_x) > 2:
                    tmp_peaks_x = np.array(tmp_peaks_x)
                    tmp_peaks_x = np.sort(tmp_peaks_x)

                    diffs_x = np.diff(tmp_peaks_x)
                    for_del_diff = []
                    max_point = np.argmax(tmp_denoised)
                    fs = []
                    for i in range(len(tmp_peaks_x)):

                        border_condition = np.abs(
                            tmp_denoised[tmp_peaks_x[i]] / max(tmp_denoised) - border[border_left_x:border_right_x][
                                tmp_peaks_x[i]] / max(tmp_denoised))
                        if i == 0:
                            diff_condition = 2 * diffs_x[i]
                            diff_y_condition = 2 * tmp_denoised[tmp_peaks_x[i]] / tmp_denoised[tmp_peaks_x[i + 1]]
                        elif i == len(tmp_peaks_x) - 1:
                            diff_condition = 2 * diffs_x[i - 1]
                            diff_y_condition = 2 * tmp_denoised[tmp_peaks_x[i]] / tmp_denoised[tmp_peaks_x[i - 1]]
                        else:
                            diff_condition = diffs_x[i - 1] + diffs_x[i]
                            diff_y_condition = tmp_denoised[tmp_peaks_x[i]] / tmp_denoised[tmp_peaks_x[i + 1]] + \
                                               tmp_denoised[tmp_peaks_x[i]] / tmp_denoised[tmp_peaks_x[i - 1]]

                        if self.gold:
                            diff_condition = 1
                        f_ = border_condition * diff_condition * diff_y_condition
                        fs.append(f_)
                    for i, f in enumerate(fs):
                        if f < max(fs)*0.9:
                            for_del_diff.append(i)

                    if len(for_del_diff) > 1:
                        for_del_diff = list(set(for_del_diff))
                        for_del_diff = np.array(for_del_diff)
                        tmp_peaks_x = np.delete(tmp_peaks_x, for_del_diff)
                        tmp_peaks_x = np.append(tmp_peaks_x, max_point)

            for peak in tmp_peaks_x:
                all_peaks_x.append(tmp_x[peak])
                all_peaks_y.append(tmp_denoised[peak])

            peaks_y[np.where((right_x > peaks_x) & (peaks_x > left_x))] = 0
            peaks_zer[left_x:right_x] = 0

        all_peaks_x = list(set(all_peaks_x))
        all_peaks_x = np.array(all_peaks_x)
        if self.dev:
            plt.figure(figsize=(10, 5.6))
            x = np.linspace(0, len(denoised_copy) * self.dt, len(denoised_copy))
            plt.plot(x, np.abs(denoised_copy))
            plt.xlabel('t, us')
            plt.ylabel('U, μV')
            plt.grid()
            plt.title('Finding all peaks')
            plt.plot(x[peaks_x], peaks_y_copy, 'D', color='green')
            if self.save_fig:
                plt.savefig(self.dir_name + '/all_peaks.png', dpi=400)
            plt.title('Response with borders')
            for i in range(len(borders_x)):
                plt.plot(borders_x[i] * self.dt, borders_y[i], color='orange', linestyle='dashdot')
                if i == 0:
                    plt.plot(borders_x[i] * self.dt, borders_y[i], color='orange', linestyle='dashdot',
                             label='Border')

            plt.hlines(y=height, xmin=0, xmax=max(x), color='grey', linestyle='dashed',
                       label='Threshold')

            if self.save_fig:
                plt.savefig(self.dir_name + '/all_peaks_borders.png', dpi=400)
            plt.clf()
            plt.plot(x, np.abs(denoised_copy))
            plt.xlabel('t, us')
            plt.ylabel('U, μV')
            plt.title('Peaks after filtering')
            plt.grid()
            for i in range(len(borders_x)):
                plt.plot(borders_x[i] * self.dt, borders_y[i], color='orange', linestyle='dashdot')
                if i == 0:
                    plt.plot(borders_x[i] * self.dt, borders_y[i], color='orange', linestyle='dashdot',
                             label='Border')

            plt.hlines(y=height, xmin=0, xmax=max(x), color='grey', linestyle='dashed',
                       label='Threshold')
            plt.plot(all_peaks_x * self.dt, denoised_copy[all_peaks_x], 'v', color='g')
            if self.save_fig:
                plt.savefig(self.dir_name + '/filtr_peaks_borders.png', dpi=400)

        if self.dev and not self.save_fig:
            plt.show()
        self.match_flag = True
        self.border_flag = True
        return all_peaks_x, borders_x, borders_y, height
