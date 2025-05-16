import argparse
import os
from matplotlib import pyplot as plt
import numpy as np
import mne
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import make_pipeline
from Denoise import BandpassFilter, WaveletDenoiser
from denoise_utils import denoise_signal, filter_bandpass
from plot_utils import plot_signals_with_events
from Csp import CustomCSP


def parse_args():
    parser = argparse.ArgumentParser(
        description="EEG Wavelet Denoising (All Channels)"
    )
    parser.add_argument("edf_file", type=str, help="Path to the .edf file")
    parser.add_argument(
        "--wavelet",
        type=str,
        default="db4",
        help="Discrete wavelet type (default: db4)",
    )
    parser.add_argument(
        "--level", type=int, default=4, help="Decomposition level (default: 4)"
    )
    parser.add_argument(
        "--tmin", type=float, default=0, help="Epochs starting time"
    )
    parser.add_argument(
        "--tmax", type=float, default=2, help="Epochs ending time"
    )
    return parser.parse_args()


def preprocessing(raw_signal, wavelet, level):
    pipeline = make_pipeline(
        BandpassFilter(l_freq=1.0, h_freq=40.0),
        WaveletDenoiser(wavelet=wavelet, level=level),
    )

    preprocessed_signal = pipeline.fit_transform(raw_signal)
    return preprocessed_signal


def split_epochs(signal, tmin, tmax):
    events, event_id = mne.events_from_annotations(
        signal,
    )

    epochs = mne.Epochs(
        raw,
        events,
        event_id=event_id,
        tmin=tmin,
        tmax=tmax,
        baseline=None,
        preload=True,
        verbose=False,
    )

    return epochs


def training(epochs, test_size=0.2, random_state=42):
    X = epochs.get_data()
    y = epochs.events[:, 2]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    print(f"Train events shape {y_train.shape}")
    print(f"Train data shape {X_train.shape}")
    print(f"Test events shape {y_test.shape}")
    print(f"Test data shape {X_test.shape}")

    pipeline = OneVsRestClassifier(
        make_pipeline(CustomCSP(n_components=4), LogisticRegression())
    )

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    score = accuracy_score(y_test, y_pred)
    print(f"score: {score}")


if __name__ == "__main__":
    args = parse_args()
    raw = mne.io.read_raw_edf(args.edf_file, preload=True, verbose=True)
    preprocessed_signal = preprocessing(raw, args.wavelet, args.level)
    print(preprocessed_signal.info)
    # preprocessed_signal.plot()
    # plt.show()

    epochs = split_epochs(preprocessed_signal, args.tmin, args.tmax)
    print(epochs.info)

    training(epochs)
