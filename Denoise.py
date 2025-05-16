import numpy as np
import pywt
from sklearn.base import BaseEstimator, TransformerMixin


class BandpassFilter(BaseEstimator, TransformerMixin):
    def __init__(self, l_freq=1.0, h_freq=40.0):
        self.l_freq = l_freq
        self.h_freq = h_freq

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.filter(
            l_freq=self.l_freq,
            h_freq=self.h_freq,
            fir_design="firwin",
            verbose=False,
        )


class WaveletDenoiser(BaseEstimator, TransformerMixin):
    def __init__(self, wavelet="db4", level=4):
        self.wavelet = wavelet
        self.level = level

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        raw_copy = X.copy()
        eeg_data = raw_copy.get_data()
        denoised_signal = self._denoise_signal(
            eeg_data=eeg_data, wavelet_name=self.wavelet, level=self.level
        )
        raw_copy._data[:] = denoised_signal  # Replace in-place
        return raw_copy

    def _denoise_signal(self, eeg_data, wavelet_name="db4", level=4):
        """
        Denoise EEG data using wavelet transform.
        """
        denoised_signals = []
        for signal in eeg_data:
            denoised = self._dwt(signal, wavelet_name, level)
            denoised_signals.append(denoised)
        return np.array(denoised_signals)

    def _dwt(self, signal, wavelet_name="db4", level=4):
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
