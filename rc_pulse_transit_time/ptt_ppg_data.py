import pandas as pd
from scipy import signal, stats
import os
import glob
from utils import cheby2_filter, norm_x_corr
import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import IsolationForest  # FOR OUTLIER DETECTION
from typing import List, Optional, Dict


class PTTPPGData:
    """
        Handles and analyzes PPG (Photoplethysmography) data from a CSV file.

        Attributes:
            df (pd.DataFrame): Loaded data from the CSV.
            file_name (str): Name of the input file.
            subject (str): Subject ID inferred from file name.
            activity (str): Activity label inferred from file name.
            fs (int): Sampling frequency (Hz), default is 500.
            cleaned (bool): Flag indicating if data has been cleaned.
            peaks (list): List of PPGPeak objects after peak detection.
        """

    def __init__(self, path_to_csv: str) -> None:
        """
            Initializes the class by loading the CSV file and extracting metadata.

            Args:
                path_to_csv (str): Path to the CSV file.
        """
        self.df = pd.read_csv(path_to_csv)
        self.file_name = os.path.split(path_to_csv)[-1]
        self.subject = os.path.split(path_to_csv)[-1].split('_')[0]
        self.activity = os.path.split(path_to_csv)[-1].split('_')[1].split('.')[0]
        self.fs = 500
        self.cleaned = False
        self.peaks: Optional[List[PPGPeak]] = None

    def clean_data(self) -> None:
        """
            Applies z-score normalization and a bandpass Chebyshev filter to clean
            the PPG signals. Also inverts the signal to correct orientation.
        """
        ple_col = self.df.filter(like='pleth').columns
        sig = self.df.loc[:, ple_col].values
        sig = stats.zscore(sig, axis=0)
        sig = cheby2_filter(sig, cut=[0.75, 20], fs=self.fs, btype='bandpass')
        bs = pd.DataFrame(sig).rolling(window=6*self.fs, center=True, min_periods=1).mean().values
        # sig = sig - bs
        self.df[ple_col] = self.df[ple_col].astype(float)
        self.df.loc[:, ple_col] = -sig
        self.cleaned = True

    def get_phalanx_lag(self) -> float:
        """
           Computes the temporal lag between pleth_2 and pleth_5 channels. We assume pleth_2 peaks arrive before pleth_5
            Some subject seem to present much more frequent pleth_5 arriving before pleth_2. I believe the recordings
            were sometimes inverted in that dataset.
           Returns:
               float: Lag in milliseconds.
        """
        step = 1
        wl = 4
        ppg2 = self.df.pleth_2.values
        ppg5 = self.df.pleth_5.values
        ppg2 = cheby2_filter(ppg2, cut=8, fs=self.fs, btype='low')
        ppg5 = cheby2_filter(ppg5, cut=8, fs=self.fs, btype='low')

        epoch = np.arange(0, len(ppg2), self.fs * step)[:-(wl+1)]
        lags_neg = []
        lags_pos = []

        for i, e in enumerate(epoch):
            lag, cco = norm_x_corr(ppg2[e:e + self.fs * wl], ppg5[e:e + self.fs * wl], fs=self.fs)
            p, _ = signal.find_peaks(cco)
            plag = lag[p] * 1000
            pot_plag = plag[(plag > -30) & (plag < 0)]
            if len(pot_plag):
                lags_neg.append(pot_plag[-1])
            else:
                pot_plag = plag[(plag > 0) & (plag < 30)]
                if len(pot_plag):
                    lags_pos.append(pot_plag[0])
        if len(lags_neg) > len(lags_pos):
            lags = lags_neg
        else:
            lags = lags_pos

        mlag = np.abs(np.mean(lags))

        return mlag

    def get_hr(self) -> float:
        """
         Estimates heart rate from pleth_2 using peak detection.

         Returns:
             float: Heart rate in beats per minute (bpm).
        """
        peaks, prop = signal.find_peaks(self.df.pleth_2.values, height=0, distance=int(self.fs / 3.3), prominence=0.3)
        bpm = np.mean(np.diff(peaks) / self.fs * 60)

        return bpm

    def get_ecg_hr(self) -> float:
        """
         Estimates ECG-based heart rate from pre-computed peak data.

         Returns:
             float: ECG heart rate in bpm.
        """
        peaks, prop = signal.find_peaks(self.df.peaks.values, height=1, distance=int(self.fs / 3.3))

        return np.mean(np.diff(peaks) / self.fs * 60)

    def compute_ppg_peaks(self) -> None:
        """
         Detects valleys in pleth_2 to segment and validate individual PPG cycles,
         instantiating PPGPeak objects and keeping valid ones.
        """
        ppg = self.df.pleth_2.values
        valley, valley_prop = signal.find_peaks(-ppg, height=0, distance=int(self.fs / 3.3), prominence=0.1)

        peaks = []
        for i in range(len(valley) - 1):
            p = PPGPeak(ppg, valley[i], valley[i + 1], self.fs)
            if p.valid:
                peaks.append(p)

        self.peaks = peaks

    def get_average_peak_feature(self, remove_outlier: bool = True) -> Dict[str, float]:
        """
            Computes the average of various peak-based PPG features.

            Args:
                remove_outlier (bool): Whether to remove outliers based on isolation forest.

            Returns:
                dict: Dictionary of averaged features.
        """
        feature_list = [p.get_features() for p in self.peaks]
        df = pd.concat([pd.Series(f) for f in feature_list], axis=1).T
        if remove_outlier:
            model = IsolationForest(contamination=0.05, random_state=7)
            labels = model.fit_predict(df)
            df = df[labels == 1]

        fe = dict(df.mean(axis=0))

        return fe

    def get_frequency_features(self) -> Dict[str, float]:
        """
          Extracts frequency domain features using Welch’s method.

          Returns:
              dict: Dictionary with power spectral peaks and their frequencies.
        """
        sig = self.df.pleth_2.values
        hz, psd = signal.welch(sig, fs=self.fs, nperseg=10 * self.fs)

        p25 = np.sum(psd[(hz >= 4) & (hz <= 5)])
        p1 = np.argmax(psd[(hz >= 1) & (hz <= 2)]) + np.where(hz == 1)[0][0]
        p2 = np.argmax(psd[(hz >= 2) & (hz <= 3)]) + np.where(hz == 2)[0][0]
        p3 = np.argmax(psd[(hz >= 3) & (hz <= 5)]) + np.where(hz == 3)[0][0]

        fe = dict(
            p25=p25,
            freq_p1=hz[p1],
            freq_p2=hz[p2],
            freq_p3=hz[p3],
            amp_p1=psd[p1],
            amp_p2=psd[p2],
            amp_p3=psd[p3],
        )

        return fe


class PPGPeak:
    """
    Represents a single PPG cycle and computes morphological features
    from the PPG, its first derivative (VPG), and second derivative (APG).
    """

    def __init__(self, ppg_sig: np.ndarray, start: int, finish: int, fs: int) -> None:
        self.fs = fs
        self.ppg = ppg_sig[start:finish]
        vpg = cheby2_filter(np.diff(ppg_sig), cut=20, fs=self.fs, btype='lp')
        self.vpg = vpg[start:finish]
        self.apg = cheby2_filter(np.diff(vpg), cut=20, fs=self.fs, btype='lp')[start:finish]
        self.start = start
        self.finish = finish
        self.valid = False
        self.__validate()

    def __validate(self) -> None:
        ppg_peaks, ppg_peak_props = signal.find_peaks(self.ppg, height=-0.2, distance=20)
        highest = np.flip(np.argsort(ppg_peak_props['peak_heights']))
        if not len(highest) >= 2:
            return
        self.idx_S = ppg_peaks[highest[0]]
        self.idx_D = ppg_peaks[highest[1]]
        if self.idx_S > self.idx_D:
            return

        ppg_valley, _ = signal.find_peaks(-self.ppg, height=-0.2, distance=20)
        n = ppg_valley[(ppg_valley > self.idx_S) & (ppg_valley < self.idx_D)]
        if not len(n):
            return
        self.idx_N = n[0]

        vpg_peaks, _ = signal.find_peaks(self.vpg, height=-0.2, distance=20)
        w = vpg_peaks[vpg_peaks < self.idx_S]
        if not len(w):
            return
        self.idx_w = w[-1]
        z = vpg_peaks[vpg_peaks > self.idx_S]
        if not len(z):
            return
        self.idx_z = z[0]

        vpg_valley, _ = signal.find_peaks(-self.vpg, height=0, distance=20)
        y = vpg_valley[vpg_valley > self.idx_S]
        if not len(y):
            return
        self.idx_y = y[0]

        apg_peaks, _ = signal.find_peaks(self.apg, height=-0.2, distance=20)
        c = apg_peaks[(apg_peaks > self.idx_y) & (apg_peaks < self.idx_z)]
        if not len(c):
            return
        self.idx_c = c[0]

        self.valid = True

    def show(self) -> None:
        x_time = np.arange(0, len(self.ppg) / self.fs, 1 / self.fs)
        fig, axes = plt.subplots(3, 1)

        axes[0].set_title('PPG')
        axes[0].plot(x_time, self.ppg)
        for idx, lab in zip([self.idx_S, self.idx_D], ['S', 'D']):
            axes[0].axvline(x_time[idx], color='tab:green')
            axes[0].text(x_time[idx], self.ppg[idx], lab)

        axes[1].set_title('VPG')
        axes[1].plot(x_time, self.vpg)
        for idx, lab in zip([self.idx_w, self.idx_y, self.idx_z], ['w', 'y', 'z']):
            axes[1].axvline(x_time[idx], color='tab:green')
            axes[1].text(x_time[idx], self.vpg[idx], lab)

        axes[2].set_title('APG')
        axes[2].plot(x_time, self.apg)
        for idx, lab in zip([self.idx_c], ['c']):
            axes[2].axvline(x_time[idx], color='tab:green')
            axes[2].text(x_time[idx], self.apg[idx], lab)
        plt.show()

    def get_features(self) -> Optional[Dict[str, float]]:
        if not self.valid:
            return None

        ti_sd = (self.idx_D - self.idx_S) / self.fs
        s_amp = self.ppg[self.idx_S] - self.ppg[0]
        d_amp = self.ppg[self.idx_D] - self.ppg[0]
        ap = self.ppg[self.idx_w] - self.ppg[0]

        relevant_features = dict(ti_S=self.idx_S / self.fs,
                                 ti_D=self.idx_D / self.fs,
                                 ti_y=self.idx_y / self.fs,
                                 ti_sd=ti_sd,
                                 AI=s_amp / (s_amp - ap) * 100,
                                 RI=d_amp / s_amp * 100,
                                 S=s_amp,
                                 D=d_amp,
                                 y=self.vpg[self.idx_y] - self.ppg[0])

        return relevant_features


if __name__ == '__main__':
    files = glob.glob('../DATA/csv/*.csv')
    pdata = PTTPPGData('../DATA/csv/s15_sit.csv')
    pdata.clean_data()
    pdata.compute_ppg_peaks()
