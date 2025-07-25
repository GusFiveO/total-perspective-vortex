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
        choices=range(1, 110),
        type=int,
        default=1,
        help="ID du sujet (default: 1)",
    )
    parser.add_argument(
        "--run",
        choices=range(1, 15),
        type=int,
        default=1,
        help="Numéro du run (default: 1)",
    )
    parser.add_argument(
        "--ch_idx",
        choices=range(0, 64),
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

    preprocessed = preprocessing(raw, 'db4', 4)
    eeg_preprocessed = preprocessed.get_data()
    plot_eeg(
        eeg_preprocessed,
        "Signal EEG prétraité (filtre + ondelettes)",
        args.ch_idx,
    )


if __name__ == "__main__":
    main()
