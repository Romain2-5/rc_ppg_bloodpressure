from scipy import signal
import numpy as np


def cheby2_filter(sig, cut, fs, btype, axis=0):
    # Optimal filter according to https://www.nature.com/articles/sdata201876
    sos = signal.cheby2(N=2, rs=20, Wn=cut, fs=fs, btype=btype, output='sos')  # order 2 because filtfilt goes twice
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
