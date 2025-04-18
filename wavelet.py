import mne
import numpy as np
import pywt
import matplotlib.pyplot as plt
import argparse

args_parser = argparse.ArgumentParser()
args_parser.add_argument("file", type=str, help="Path to the EDF file")
args = args_parser.parse_args()


# 1) Load the EDF file (replace 'your_file.edf' with your path)
raw = mne.io.read_raw_edf(args.file, preload=True)

# 2) Pick one channel (e.g., the first) and extract its data
data, times = raw.get_data(picks=[0], return_times=True)
signal = data[0]  # 1D EEG signal array
fs = raw.info["sfreq"]  # sampling rate in Hz

# 3) Define window parameters
win_sec = 2.0  # window length in seconds
step_sec = 1.0  # step/stride between windows in seconds
win_size = int(win_sec * fs)  # samples per window
step_size = int(step_sec * fs)  # samples per step

# 4) Segment the long signal into overlapping windows
windows = []
for start in range(0, len(signal) - win_size + 1, step_size):
    windows.append(signal[start : start + win_size])
windows = np.array(windows)  # shape: (n_windows, win_size)

# 5) Choose the first window for demonstration
window = windows[0]
time_axis = np.arange(win_size) / fs

# 6) Compute Continuous Wavelet Transform (CWT) with Morlet wavelet
scales = np.arange(1, 64)
coeffs, freqs = pywt.cwt(window, scales, "morl", sampling_period=1 / fs)
abs_coeffs = np.abs(coeffs)

# 7) Plot the scalogram
plt.figure()
plt.imshow(
    abs_coeffs,
    aspect="auto",
    extent=[time_axis[0], time_axis[-1], scales[-1], scales[0]],
)
plt.ylabel("Scale")
plt.xlabel("Time (s)")
plt.title("Scalogram (|wavelet coefficients|)")
plt.show()

# 8) Compute and plot histogram of coefficient magnitudes
threshold = abs_coeffs.mean() + 3 * abs_coeffs.std()
plt.figure()
plt.hist(abs_coeffs.ravel(), bins=100)
plt.axvline(threshold)
plt.xlabel("Coefficient magnitude")
plt.ylabel("Count")
plt.title("Histogram of |wavelet coefficients|")
plt.show()
