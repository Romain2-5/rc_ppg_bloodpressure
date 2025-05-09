import pandas as pd
from scipy import signal, stats, fft
import os
import glob

from scipy.signal import find_peaks

from utils import cheby2_filter, norm_x_corr
import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt
import numpy as np
from math import prod
from sklearn.ensemble import IsolationForest  #FOR OUTLIER DETECTION


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

    def __init__(self, path_to_csv):
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
        self.peaks = None

    def clean_data(self):
        """
            Applies z-score normalization and a bandpass Chebyshev filter to clean
            the PPG signals. Also inverts the signal to correct orientation.
            """

        ple_col = self.df.filter(like='pleth').columns
        sig = self.df.loc[:, ple_col].values
        sig = stats.zscore(sig, axis=0)
        sig = cheby2_filter(sig, cut=[0.75, 8], fs=self.fs, btype='bandpass')
        self.df[ple_col] = self.df[ple_col].astype(float)
        # inverse signal as it seems flipped
        self.df.loc[:, ple_col] = -sig
        self.cleaned = True

    def get_phalanx_lag(self):
        """
           Computes the temporal lag between pleth_2 and pleth_5 channels. We assume pleth_2 peaks arrive before pleth_5

           Returns:
               float: Lag in milliseconds.
           """

        step = 1
        wl = 6
        ppg2 = self.df.pleth_2.values
        ppg5 = self.df.pleth_5.values
        epoch = np.arange(0, len(ppg2), self.fs*step)
        lags = []
        for i, e in enumerate(epoch):
            lag, cco = norm_x_corr(ppg2[e:e + self.fs * wl], ppg5[e:e + self.fs * wl], fs=self.fs)
            p, _ = signal.find_peaks(cco)
            plag = lag[p] * 1000
            plag = plag[(plag > -50) & (plag < 0)]
            if len(plag):
                lags.append(plag[-1])
        mlag = np.abs(np.mean(lags))

        return mlag

    def get_hr(self):
        """
         Estimates heart rate from pleth_2 using peak detection.

         Returns:
             float: Heart rate in beats per minute (bpm).
         """

        peaks, prop = signal.find_peaks(self.df.pleth_2.values, height=0, distance=int(self.fs/3.3), prominence=0.3)
        bpm = np.mean(np.diff(peaks)/self.fs*60)
        return bpm

    def get_ecg_hr(self):
        """
         Estimates ECG-based heart rate from pre-computed peak data.

         Returns:
             float: ECG heart rate in bpm.
         """

        peaks, prop = signal.find_peaks(self.df.peaks.values, height=1, distance=int(self.fs / 3.3))
        return np.mean(np.diff(peaks) / self.fs * 60)

    def compute_ppg_peaks(self):
        """
         Detects valleys in pleth_2 to segment and validate individual PPG cycles,
         instantiating PPGPeak objects and keeping valid ones.
         """

        ppg = self.df.pleth_2.values
        valley, valley_prop = signal.find_peaks(-ppg, height=0, distance=int(self.fs / 3.3), prominence=0.1)

        peaks = []
        for i in range(len(valley)-1):
            p = PPGPeak(ppg, valley[i], valley[i+1], self.fs)
            if p.valid:
                peaks.append(p)

        self.peaks = peaks

    def get_average_peak_feature(self, remove_outlier=True):
        """
            Computes the average of various peak-based PPG features.

            Args:
                remove_outlier (bool): Whether to remove outliers based on isolationforest.

            Returns:
                dict: Dictionary of averaged features.
            """

        feature_list = [p.get_features() for p in self.peaks]
        df = pd.concat([pd.Series(f) for f in feature_list], axis=1).T
        if remove_outlier:
            model = IsolationForest(contamination=0.05)
            labels = model.fit_predict(df)
            df = df[labels == 1]

        fe = dict(df.mean(axis=0))

        return fe

    def get_frequency_features(self):
        """
          Extracts frequency domain features using Welch’s method.

          Returns:
              dict: Dictionary with power spectral peaks and their frequencies.
          """

        sig = self.df.pleth_2.values
        hz, psd = signal.welch(sig, fs=self.fs, nperseg=10*self.fs)

        p25 = np.sum(psd[(hz>=4) & (hz <= 5)])
        p1 = np.argmax(psd[(hz>=1) & (hz <= 2)]) + np.where(hz==1)[0][0]
        p2 = np.argmax(psd[(hz >= 2) & (hz <= 3)]) + np.where(hz==2)[0][0]
        p3 = np.argmax(psd[(hz >= 3) & (hz <= 5)]) + np.where(hz==3)[0][0]

        fe = dict(
            p25=p25,
            freq_p1= hz[p1],
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

    Attributes:
        fs (int): Sampling frequency.
        ppg (np.ndarray): Segment of the PPG signal.
        vpg (np.ndarray): First derivative of PPG.
        apg (np.ndarray): Second derivative of PPG.
        start (int): Start index of segment.
        finish (int): End index of segment.
        valid (bool): Whether the peak has valid characteristic points.
    """

    def __init__(self, ppg_sig, start, finish, fs):
        """
        Initializes a PPG peak object with filtered derivatives and validates peak structure.

        Args:
            ppg_sig (np.ndarray): Full PPG signal.
            start (int): Start index of the segment.
            finish (int): End index of the segment.
            fs (int): Sampling frequency.
        """

        self.fs = fs
        self.ppg = ppg_sig[start:finish]
        vpg = cheby2_filter(np.diff(ppg_sig), cut=20, fs=self.fs, btype='lp')
        self.vpg = vpg[start:finish]
        self.apg = cheby2_filter(np.diff(vpg), cut=20, fs=self.fs, btype='lp')[start:finish]
        self.start = start
        self.finish = finish
        self.valid = False

        try:
            self.__validate()
        except IndexError:
            pass


    def __validate(self):
        """
        Detects key characteristic points in PPG, VPG, and APG waves.
        Sets internal indices for S, D, N, w, y, z, and c.
        """

        # Look for two peaks S and D in PPG
        ppg_peaks, ppg_peak_props = signal.find_peaks(self.ppg, height=-0.2, distance=20)

        highest = np.flip(np.argsort(ppg_peak_props['peak_heights']))

        self.idx_S = ppg_peaks[highest[0]]
        self.idx_D = ppg_peaks[highest[1]]

        if self.idx_S > self.idx_D:
            return

        # Find N in ppg
        ppg_valley, _ = signal.find_peaks(-self.ppg, height=-0.2, distance=20)
        self.idx_N = ppg_valley[(ppg_valley > self.idx_S) & (ppg_valley < self.idx_D)][0]

        # Look for peaks w and z in vpg
        vpg_peaks, _ = signal.find_peaks(self.vpg, height=-0.2, distance=20)

        self.idx_w = vpg_peaks[vpg_peaks < self.idx_S][-1]
        self.idx_z = vpg_peaks[vpg_peaks > self.idx_S][0]

        # Look for y in vpg
        vpg_valley, _ = signal.find_peaks(-self.vpg, height=0, distance=20)
        self.idx_y = vpg_valley[vpg_valley > self.idx_S][0]

        # Look for c in APG
        apg_peaks, _ = signal.find_peaks(self.apg, height=-0.2, distance=20)
        self.idx_c = apg_peaks[(apg_peaks > self.idx_y) & (apg_peaks < self.idx_z)][0]

        self.valid = True

    def show(self):
        """
        Plots the PPG, VPG, and APG signal with annotated characteristic points.
        """

        x_time = np.arange(0, len(self.ppg)/self.fs, 1/self.fs)
        fig, axes = plt.subplots(3, 1)

        # PPG plot
        axes[0].set_title('PPG')
        axes[0].plot(x_time, self.ppg)
        # Add S point
        for idx, lab in zip([self.idx_S, self.idx_D], ['S', 'D']):
            axes[0].axvline(x_time[idx], color='tab:green')
            axes[0].text(x_time[idx], self.ppg[idx], lab)

        # VPG plot
        axes[1].set_title('VPG')
        axes[1].plot(x_time, self.vpg)
        for idx, lab in zip([self.idx_w, self.idx_y, self.idx_z], ['w', 'y', 'z']):
            axes[1].axvline(x_time[idx], color='tab:green')
            axes[1].text(x_time[idx], self.vpg[idx], lab)

        # APG plot
        axes[2].set_title('APG')
        axes[2].plot(x_time, self.apg)
        for idx, lab in zip([self.idx_c], ['c']):
            axes[2].axvline(x_time[idx], color='tab:green')
            axes[2].text(x_time[idx], self.apg[idx], lab)

    def get_features(self):
        """
        Extracts time and amplitude features from the validated peak.

        Returns:
            dict: Dictionary of time-indexed and amplitude-based features,
                  or None if the peak is invalid.
        """

        if not self.valid:
            return None

        sum_sc = np.sum(self.vpg[self.idx_S:self.idx_c])**2 / np.sum(self.vpg)**2
        slope_sc = (self.idx_c - self.idx_S) / (self.ppg[self.idx_S] - self.ppg[self.idx_c])
        ti_sc = (self.idx_c - self.idx_S) / self.fs
        slope_os = (self.idx_S - 0) / (self.ppg[0] - self.ppg[self.idx_S])
        rat_cs = self.ppg[self.idx_S] / self.ppg[self.idx_c]
        sum_os = np.sum(self.ppg[0:self.idx_S])**2 / np.sum(self.ppg)**2
        rat_cw = self.vpg[self.idx_c] / self.vpg[self.idx_w]
        ti_sd = (self.idx_D - self.idx_S) / self.fs

        features = dict(sum_sc=sum_sc,
                        slope_sc=slope_sc,
                        ti_sc=ti_sc,
                        slope_os=slope_os,
                        rat_cs=rat_cs,
                        sum_os=sum_os,
                        rat_cw=rat_cw,
                        ti_sd=ti_sd,
                        )

        features2 = dict(ti_S=self.idx_S/self.fs,
                        ti_D=self.idx_D / self.fs,
                        ti_N=self.idx_N / self.fs,
                        ti_w=self.idx_w / self.fs,
                        ti_y=self.idx_y / self.fs,
                        ti_z=self.idx_z / self.fs,
                        ti_c=self.idx_c / self.fs,
                        S=self.ppg[self.idx_S],
                        D=self.ppg[self.idx_D],
                        N=self.ppg[self.idx_N],
                        O=self.ppg[0],
                        w=self.vpg[self.idx_w],
                        y=self.vpg[self.idx_y],
                        z = self.vpg[self.idx_z],
                        c=self.apg[self.idx_c]
                        )
        features.update(features2)

        return features


if __name__ == '__main__':

    files = glob.glob('../DATA/csv/*.csv')
    pdata = PPTPPGData('../DATA/csv/s15_sit.csv')
    pdata.clean_data()
    pdata.compute_ppg_peaks()
