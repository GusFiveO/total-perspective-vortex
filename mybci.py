import argparse
import numpy as np
from mybci.training import (
    cross_val_training,
)
from mybci.preprocessing import (
    preprocessing,
    split_epochs,
)

from mybci.io import load_dataset


def parse_args():
    parser = argparse.ArgumentParser(
        description="EEG Wavelet Denoising (All Channels)"
    )
    parser.add_argument(
        "--subjects",
        nargs="+",
        type=int,
        default=range(1, 109),
        help="Subject IDs",
    )
    parser.add_argument(
        "-r",
        "--runs",
        nargs="+",
        type=int,
        # default=[3, 7, 11],
        default=range(1, 14),
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


def mybci():
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


if __name__ == "__main__":
    mybci()
