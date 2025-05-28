from scipy import signal
import numpy as np
from numpy.typing import ArrayLike
from typing import Tuple

def cheby2_filter(sig: ArrayLike, cut: float | list[float], fs: int, btype: str, axis: int = 0) -> np.ndarray:
    """
    Applies a Chebyshev Type II bandpass/lowpass/highpass filter using second-order sections.
    Optimal filter according to https://www.nature.com/articles/sdata201876
    Args:
        sig (ArrayLike): Input signal.
        cut (float | list[float]): Cutoff frequency/frequencies.
        fs (int): Sampling frequency.
        btype (str): Type of filter ('low', 'high', 'bandpass', 'bandstop').
        axis (int): Axis to filter along.

    Returns:
        np.ndarray: Filtered signal.
    """

    sos = signal.cheby2(N=2, rs=20, Wn=cut, fs=fs, btype=btype, output='sos')
    fsig = signal.sosfiltfilt(sos=sos, x=sig, axis=axis)

    return fsig

def norm_x_corr(a: ArrayLike, b: ArrayLike, fs: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes the normalized cross-correlation between two signals.

    Args:
        a (ArrayLike): First signal.
        b (ArrayLike): Second signal.
        fs (int): Sampling frequency.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Tuple of (lags in seconds, cross-correlation values).
    """

    an = a - np.mean(a, axis=0)
    bn = b - np.mean(b, axis=0)
    an = an / np.linalg.norm(an, axis=0)
    bn = bn / np.linalg.norm(bn, axis=0)
    cco = signal.correlate(an, bn)
    lags = signal.correlation_lags(len(b), len(a)) / fs

    return lags, cco