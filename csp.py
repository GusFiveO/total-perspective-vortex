import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import mne
from mne.decoding import Scaler, Vectorizer
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="CSP Algorithm for EEG Data")
    parser.add_argument("data", type=str, help="Path to the EEG data file")
    parser.add_argument(
        "--num_components",
        type=int,
        default=4,
        help="Number of CSP components to compute",
    )
    args = parser.parse_args()
    return args


def load_data(file_path):
    raw = mne.io.read_raw_edf(file_path, preload=True)
    return raw


def print_raw_channel(raw):
    print("Channel names:")
    for ch in raw:
        print(ch)


def compute_cov(raw):
    cov = np.cov(raw._data, rowvar=False)
    return cov


def scaler(raw):
    scaler = StandardScaler()
    # n_epochs, n_channels, n_times = raw._data.shape
    n_channels, n_times = raw._data.shape
    data_scaled = scaler.fit_transform(raw._data.T)
    data_scaled = data_scaled.reshape(n_channels, n_times)
    return data_scaled


if __name__ == "__main__":
    args = parse_args()
    raw = load_data(args.data)
    print(f"Loaded data from {args.data}")
    print(f"Number of channels: {len(raw.ch_names)}")
    print(f"Number of samples: {raw.n_times}")
    print(f"Sampling frequency: {raw.info['sfreq']} Hz")
    print(f"EEG data shape: {raw._data.shape}")
    print_raw_channel(raw)
    cov = compute_cov(raw)
    print(f"Covariance matrix shape: {cov.shape}")
    print(f"Covariance matrix:\n{cov}")
    print(raw.info)
    scaled_data = scaler(raw)
    print(f"Scaled data shape: {scaled_data.shape}")
    print(scaled_data.shape)
    plt.plot(scaled_data[0, :])
    # raw.plot()
    plt.show()
