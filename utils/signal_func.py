from scipy import signal
import numpy as np


def cheby2_filter(sig, cut, fs, btype, axis=0):
    # Optimal filter according to https://www.nature.com/articles/sdata201876
    if len(cut) > 1:
        cut = np.array(cut)
    sos = signal.cheby2(N=4, rs=20, Wn=cut, fs=fs, btype=btype, output='sos')
    fsig = signal.sosfiltfilt(sos=sos, x=sig, axis=axis)

    return fsig

def norm_x_corr(a, b, fs):

    an = a - np.mean(a, axis=0)
    bn = b - np.mean(b, axis=0)
    norm_a = np.linalg.norm(an, axis=0)
    an = an / norm_a
    norm_b = np.linalg.norm(bn, axis=0)
    bn = bn / norm_b
    cco = signal.correlate(an, bn)
    lags = signal.correlation_lags(len(b), len(a))
    lags = lags / fs

    return lags, cco

def extract_ppg_features(ppg, fs):
    vpg = np.diff(ppg)
    apg = np.diff(vpg)
    peaks, peak_prop = signal.find_peaks(ppg, height=0, distance=int(fs / 3.3), prominence=0.3)
