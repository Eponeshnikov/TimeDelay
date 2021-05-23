import matplotlib.pyplot as plt
import numpy as np
import pywt
from scipy.signal import find_peaks, argrelextrema
from scipy.interpolate import interp1d


def gauss(x, A, mu, sigma):
    return A * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))


def cauchy(x, A, mu, gamma):
    return A / np.pi * (gamma / ((x - mu) ** 2 + gamma ** 2))


def map(x, in_min, in_max, out_min, out_max):
    return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min


class OptReceiver:
    def __init__(self):
        self.h = None
        self.sig = None
        self.us = None
        self.noise_u = 0

    def matched_fil(self):
        return np.convolve(np.flip(self.h), self.sig, mode='same') / len(self.sig)

    def gen_min_height(self):
        size = len(self.matched_fil())
        noise = np.random.normal(0, self.noise_u, size=size)
        conv = np.convolve(np.flip(self.h), noise, mode='same') / len(noise)
        return np.std(conv, ddof=1)

    def denoise(self, data, threshold):
        # Create wavelet object and define parameters
        w = pywt.Wavelet('sym4')
        maxlev = pywt.dwt_max_level(len(data), w.dec_len)
        # maxlev = 2 # Override if desired
        # print("maximum level is " + str(maxlev))
        # Decompose into wavelet components, to the level selected:
        coeffs = pywt.wavedec(data, 'sym4', level=maxlev)
        # cA = pywt.threshold(cA, threshold*max(cA))
        for i in range(1, len(coeffs)):
            # plt.subplot(maxlev, 1, i)
            # plt.plot(coeffs[i])
            coeffs[i] = pywt.threshold(coeffs[i], threshold * max(coeffs[i]))
            # plt.plot(coeffs[i])
        datarec = pywt.waverec(coeffs, 'sym4')
        return datarec

    def gen_border(self):
        conv = np.correlate(self.h, self.h, mode='full') / len(self.h)
        border = np.zeros(int(len(conv)))
        conv /= max(conv)
        peaks, _ = find_peaks(conv, distance=int(len(conv) / 20))
        # sorted_peaks = np.flip(np.sort(conv[peaks]))
        # sid_ratio = sorted_peaks[0] / (np.average(sorted_peaks) - sorted_peaks[0] / len(sorted_peaks))
        index = np.arange(len(conv))
        u_p = interp1d(index[peaks], conv[peaks], kind='quadratic', bounds_error=False, fill_value=0.0)
        for k in range(0, len(conv)):
            border[k] = u_p(k)
        border += max(border)/15
        border /= max(border)
        border = np.where(border > 0.1, border, 0.1)
        # params = {'sid_ratio': sid_ratio, 'conv': conv}
        return border

    def find_peaks_(self):
        threshold = self.noise_u / (max(self.us) * 6)  # Threshold for filtering
        if threshold > 0.6:
            threshold = 0.6
        denoised = self.denoise(self.matched_fil(), threshold)
        denoised_copy = np.copy(denoised)
        index = np.arange(len(denoised))
        height = self.gen_min_height() * 3
        peaks, _ = find_peaks(denoised, height=height)
        peaks_y = denoised[peaks]
        peaks_x = index[peaks]
        peaks_zer = np.zeros(len(denoised))
        peaks_zer[peaks_x] = peaks_y
        all_peaks_x = []
        all_peaks_y = []
        borders_x = []
        borders_y = []

        while np.sum(peaks_y) != 0:
            den_sum = np.sum(np.abs(denoised))
            print(den_sum)
            border = self.gen_border()
            # print('sid_ratio:', params['sid_ratio'])
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
            tmp_denoised = denoised[left_x:right_x]
            tmp_denoised = np.where(tmp_denoised > 0, tmp_denoised, 0)
            tmp_x = np.arange(left_x, right_x)
            borders_x.append(tmp_x)
            tmp_peaks_x, _ = find_peaks(tmp_denoised, height=border[border_left_x:border_right_x], distance=20)

            if len(tmp_peaks_x) >= 2:
                print('first')
                for_del = []
                for i, peak in enumerate(tmp_peaks_x):
                    condition = np.abs(tmp_denoised[peak] / max(tmp_denoised)
                                       - border[border_left_x:border_right_x][peak] / max(tmp_denoised))
                    if condition < 0.2 and tmp_denoised[peak] != max(tmp_denoised[tmp_peaks_x]):
                        for_del.append(i)
                for_del = np.array(for_del)
                tmp_peaks_x = np.delete(tmp_peaks_x, for_del)

                if len(tmp_peaks_x) > 2:
                    print('second')
                    diffs_x = np.diff(tmp_peaks_x)
                    for_del_diff = []
                    max_point = np.argmax(tmp_denoised[tmp_peaks_x])
                    for i in range(len(diffs_x)):
                        if diffs_x[i] <= 200:
                            for_del_diff.append(i)
                            for_del_diff.append(i + 1)
                    if len(for_del_diff) > 1:
                        for_del_diff = list(set(for_del_diff))
                        for_del_diff = np.array(for_del_diff)
                        tmp_peaks_x = np.delete(tmp_peaks_x, for_del_diff)
                        tmp_peaks_x = np.append(tmp_peaks_x, max_point)

            for peak in tmp_peaks_x:
                all_peaks_x.append(tmp_x[peak])
                all_peaks_y.append(tmp_denoised[peak])

            '''plt.cla()
            plt.plot(peaks_x, peaks_y, 'x')
            plt.plot(tmp_x, border[border_left_x:border_right_x])
            plt.plot(tmp_x, tmp_denoised)
            plt.plot(tmp_x[tmp_peaks_x], tmp_denoised[tmp_peaks_x], 'x')'''
            peaks_y[np.where((right_x > peaks_x) & (peaks_x > left_x))] = 0
            denoised[left_x:right_x] = 0
            # plt.savefig('signals/' + str(len(all_peaks_x)) + '.png')
            print('-' * 10)
            if (den_sum - np.sum(np.abs(denoised)))/den_sum < 0.2:
                break

        all_peaks_x = np.array(all_peaks_x)
        '''plt.plot(index, denoised_copy)
        plt.plot(all_peaks_x, all_peaks_y, 'x')
        plt.show()'''
        return all_peaks_x, borders_x, borders_y
