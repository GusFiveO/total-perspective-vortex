import argparse
import matplotlib.pyplot as plt
from mybci.io import load_runs
from mybci.preprocessing import preprocessing


def plot_eeg(data, title, ch_idx=0):
    plt.figure(figsize=(12, 4))
    plt.plot(data[ch_idx], color="blue")
    plt.title(f"{title} (Channel {ch_idx})")
    plt.xlabel("Samples")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    plt.show()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualisation du signal EEG brut et prétraité"
    )
    parser.add_argument(
        "--subject",
        type=int,
        default=1,
        help="ID du sujet (default: 1)",
    )
    parser.add_argument(
        "--run",
        type=int,
        default=1,
        help="Numéro du run (default: 1)",
    )
    parser.add_argument(
        "--wavelet",
        type=str,
        default="db4",
        help="Type d'ondelette (default: db4)",
    )
    parser.add_argument(
        "--level",
        type=int,
        default=4,
        help="Niveau de décomposition (default: 4)",
    )
    parser.add_argument(
        "--ch_idx",
        type=int,
        default=0,
        help="Index du canal à afficher (default: 0)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    raw = load_runs(args.subject, args.run)
    print(raw.info)
    eeg_raw = raw.get_data()
    plot_eeg(eeg_raw, "Signal EEG brut", args.ch_idx)

    preprocessed = preprocessing(raw, args.wavelet, args.level)
    eeg_preprocessed = preprocessed.get_data()
    plot_eeg(
        eeg_preprocessed,
        "Signal EEG prétraité (filtre + ondelettes)",
        args.ch_idx,
    )


if __name__ == "__main__":
    main()
