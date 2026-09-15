## İstatistiksel, FFT, otokorelasyon ve sinüzoidal eğri uydurma işlemlerini yürütür.

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

def _sine_func(t, amp, freq, phase, offset):
    return amp * np.sin(2 * np.pi * freq * t + phase) + offset

def _autocorr(x: np.ndarray) -> np.ndarray:
    norm_x = x - np.mean(x)
    std_val = np.std(x)
    if std_val == 0:
        return np.zeros(len(x))
    result = np.correlate(norm_x, norm_x, mode="full")
    return result[result.size // 2:] / (std_val**2 * len(x))

def extract_features(X: np.ndarray, advanced: bool = True) -> pd.DataFrame:
    """Mikroarray zaman serilerinden frekans ve istatistik temelli öznitelikleri çıkarır."""
    features = {}
    n_samples, n_timepoints = X.shape
    t = np.arange(n_timepoints)

    # 1. Temel İstatistiksel Öznitelikler
    features["mean"] = np.mean(X, axis=1)
    features["std"] = np.std(X, axis=1)
    features["max"] = np.max(X, axis=1)
    features["min"] = np.min(X, axis=1)
    features["range"] = features["max"] - features["min"]
    
    if not advanced:
        features["skew"] = pd.DataFrame(X.T).skew().values

    # 2. Fourier Dönüşümü (FFT)
    fft_vals = np.abs(np.fft.fft(X, axis=1))
    half_n = n_timepoints // 2
    
    features["fft_power_low"] = np.sum(fft_vals[:, 1:min(5, half_n)], axis=1)
    features["fft_power_mid"] = np.sum(fft_vals[:, min(5, half_n):min(12, half_n)], axis=1)
    features["fft_total"] = np.sum(fft_vals[:, 1:half_n], axis=1)
    features["dominant_freq"] = np.argmax(fft_vals[:, 1:half_n], axis=1)

    if advanced:
        if half_n > 12:
            features["fft_power_high"] = np.sum(fft_vals[:, 12:half_n], axis=1)
        else:
            features["fft_power_high"] = np.zeros(n_samples)

        # 3. Otokorelasyon
        ac = np.array([_autocorr(row) for row in X])
        features["autocorr_lag1"] = ac[:, 1] if ac.shape[1] > 1 else np.zeros(n_samples)
        features["autocorr_lag2"] = ac[:, 2] if ac.shape[1] > 2 else np.zeros(n_samples)

        # 4. Tepe Noktası (Peak) Sayısı
        features["num_peaks"] = np.array([len(find_peaks(row, prominence=0.3)[0]) for row in X])

        # 5. Sinüzoidal Eğri Uydurma
        sine_params = []
        for row in X:
            try:
                popt, _ = curve_fit(_sine_func, t, row, p0=[1.0, 0.1, 0, 0], maxfev=5000)
                sine_params.append(popt)
            except Exception:
                sine_params.append([0.0, 0.1, 0.0, float(np.mean(row))])

        sine_params = np.array(sine_params)
        features["sine_amp"] = sine_params[:, 0]
        features["sine_freq"] = sine_params[:, 1]
        features["sine_phase"] = sine_params[:, 2]
        features["sine_offset"] = sine_params[:, 3]

    return pd.DataFrame(features)