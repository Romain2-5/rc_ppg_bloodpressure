import pandas as pd
from scipy import signal, stats
import os
import glob
from utils import cheby2_filter, norm_x_corr
import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt
import numpy as np


class PPTPPGData:

    def __init__(self, path_to_csv):

        self.df = pd.read_csv(path_to_csv)
        self.file_name = os.path.split(path_to_csv)[-1]
        self.subject = os.path.split(path_to_csv)[-1].split('_')[0]
        self.activity = os.path.split(path_to_csv)[-1].split('_')[1].split('.')[0]
        self.fs = 500
        self.cleaned = False

    def clean_data(self):
        ple_col = self.df.filter(like='pleth').columns
        sig = self.df.loc[:, ple_col].values
        sig = stats.zscore(sig, axis=0)
        # sig = sig - np.mean(sig, axis=0, keepdims=True)
        sig = cheby2_filter(sig, cut=[0.75, 5], fs=self.fs, btype='bandpass')
        self.df[ple_col] = self.df[ple_col].astype(float)
        self.df.loc[:, ple_col] = sig
        self.cleaned = True

    def get_phalanx_lag(self):
        # Returns the phalanx lag between the two pleth in millisecond
        lag, cco = norm_x_corr(self.df.pleth_1, self.df.pleth_4, fs=self.fs)
        sig_lag = lag[np.argmax(cco)] * 1000
        return sig_lag

    def get_hr(self):
        peaks, prop = signal.find_peaks(self.df.pleth_2.values, height=0, distance=int(self.fs/3.3), prominence=0.3)
        bpm = np.mean(np.diff(peaks)/self.fs*60)
        return bpm

    def get_ecg_hr(self):
        peaks, prop = signal.find_peaks(self.df.peaks.values, height=1, distance=int(self.fs / 3.3))
        return np.mean(np.diff(peaks) / self.fs * 60)

    def get_ppg_features(self):
        ppg = self.df.pleth_2.values
        vpg = np.diff(ppg)
        apg = np.diff(vpg)
        peaks, peak_prop = signal.find_peaks(ppg, height=0, distance=int(self.fs / 3.3), prominence=0.3)

        for p in peaks:
            idx_o = np.argwhere(vpg == 0)

        return


if __name__ == '__main__':

    files = glob.glob('../DATA/csv/*.csv')
    pdata = PPTPPGData(files[5])
    pdata.clean_data()
