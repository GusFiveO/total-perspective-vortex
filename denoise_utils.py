import numpy as np
import pywt


def filter_bandpass(raw, low_freq=1, high_freq=50):
    """
    Apply a bandpass filter to the raw data.
    """
    raw.filter(low_freq, high_freq, fir_design="firwin")
    return raw


def dwt(signal, wavelet_name="db4", level=4):
    """
    Denoise a signal using wavelet transform.
    """
    coeffs = pywt.wavedec(signal, wavelet_name, level=level)
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    uthresh = sigma * np.sqrt(2 * np.log(len(signal)))
    coeffs_thresh = [
        pywt.threshold(c, value=uthresh, mode="soft") for c in coeffs
    ]
    denoised = pywt.waverec(coeffs_thresh, wavelet_name)
    return denoised[: len(signal)]


def denoise_signal(eeg_data, wavelet_name="db4", level=4):
    """
    Denoise EEG data using wavelet transform.
    """
    denoised_signals = []
    for signal in eeg_data:
        denoised = dwt(signal, wavelet_name, level)
        denoised_signals.append(denoised)
    return np.array(denoised_signals)
