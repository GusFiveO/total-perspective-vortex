import argparse
import os
from matplotlib import pyplot as plt
import numpy as np
import mne

from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score

from Denoise import BandpassFilter, WaveletDenoiser
from denoise_utils import denoise_signal, filter_bandpass
from plot_utils import plot_signals_with_events
from Csp import CustomCSP

from mne.datasets import eegbci
from mne.io import concatenate_raws, read_raw_edf
from mne.decoding import CSP


def parse_args():
    parser = argparse.ArgumentParser(
        description="EEG Wavelet Denoising (All Channels)"
    )
    parser.add_argument(
        "--subjects", nargs="+", type=int, default=[1], help="Subject IDs"
    )
    parser.add_argument(
        "-r",
        "--runs",
        nargs="+",
        type=int,
        default=[3, 7, 11],
        help="Target Run IDs",
    )
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


def load_dataset(subjects, runs):
    subjects_raw_signals = dict()
    for subject in subjects:
        raw_fnames = eegbci.load_data(subject, runs)
        raw = concatenate_raws(
            [read_raw_edf(f, preload=True) for f in raw_fnames]
        )
        raw.annotations.rename(dict(T1="left", T2="right"), verbose=False)
        subjects_raw_signals[subject] = raw
    return subjects_raw_signals


def preprocessing(raw_signal, wavelet, level):
    pipeline = make_pipeline(
        BandpassFilter(l_freq=7.0, h_freq=30.0),
        WaveletDenoiser(wavelet=wavelet, level=level),
    )

    preprocessed_signal = pipeline.fit_transform(raw_signal)
    return preprocessed_signal


def split_epochs(signal, tmin, tmax):
    events, event_id = mne.events_from_annotations(
        signal,
    )

    epochs = mne.Epochs(
        signal,
        events,
        # event_id=event_id,
        event_id=[2, 3],
        tmin=tmin,
        tmax=tmax,
        baseline=None,
        preload=True,
        verbose=False,
    )

    return epochs


def training(epochs, test_size=0.2, random_state=None):
    X = epochs.get_data()
    y = epochs.events[:, 2] - 2

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    print(f"Train events shape {y_train.shape}")
    print(f"Train data shape {X_train.shape}")
    print(f"Test events shape {y_test.shape}")
    print(f"Test data shape {X_test.shape}")

    # pipeline = OneVsRestClassifier(
    #     # make_pipeline(CustomCSP(n_components=4), LogisticRegression())
    #     # make_pipeline(CSP(n_components=4), LogisticRegression())
    #     # make_pipeline(CSP(n_components=4), LinearDiscriminantAnalysis())
    #     make_pipeline(CSP(n_components=4), RandomForestClassifier())
    # )

    pipeline = make_pipeline(CSP(n_components=4), RandomForestClassifier())

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    score = accuracy_score(y_test, y_pred)
    print(f"score: {score}")
    print("true", y_test)
    print("pred", y_pred)


def cross_val_training(epochs, cv=5):
    X = epochs.get_data()
    y = epochs.events[:, 2] - 2

    # pipeline = make_pipeline(CSP(n_components=4), RandomForestClassifier())
    pipeline = make_pipeline(
        CustomCSP(n_components=4), RandomForestClassifier()
    )

    scores = cross_val_score(pipeline, X, y, cv=cv)

    print(scores)
    print(scores.mean())
    return scores.mean()


if __name__ == "__main__":
    args = parse_args()
    subjects_raw_signals = load_dataset(args.subjects, args.runs)
    # raw = mne.io.read_raw_edf(args.edf_file, preload=True, verbose=True)
    scores = np.array([])
    for subject, raw_signal in subjects_raw_signals.items():
        print(subject)
        preprocessed_signal = preprocessing(
            raw_signal, args.wavelet, args.level
        )
        # print(preprocessed_signal.info)

        epochs = split_epochs(preprocessed_signal, args.tmin, args.tmax)
        # print(epochs.info)

        # training(epochs)
        score = cross_val_training(epochs, cv=10)
        scores = np.append(scores, score)

    print(scores)
    print(scores.mean())
