import argparse
import os
import numpy as np
import pywt
import matplotlib.pyplot as plt
import mne


def wavelet_denoise(data, wavelet_name="db4", level=4):
    coeffs = pywt.wavedec(data, wavelet_name, level=level)
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    uthresh = sigma * np.sqrt(2 * np.log(len(data)))
    coeffs_thresh = [pywt.threshold(c, value=uthresh, mode="soft") for c in coeffs]
    denoised = pywt.waverec(coeffs_thresh, wavelet_name)
    return denoised[: len(data)]


def plot_signals_with_events(
    signals, channel_names, times, event_times_sec, event_labels, title, output_path
):
    n_channels = len(signals)
    fig, axes = plt.subplots(n_channels, 1, figsize=(14, 2 * n_channels), sharex=True)
    fig.suptitle(title, fontsize=16)

    for i, ax in enumerate(axes):
        ax.plot(times, signals[i], color="black", linewidth=0.8)
        ax.set_ylabel(channel_names[i], rotation=0, labelpad=40)
        ax.tick_params(left=False, labelleft=False)

        # Add vertical lines for events
        for t, label in zip(event_times_sec, event_labels):
            ax.axvline(x=t, color="red", linestyle="--", alpha=0.5)
            ax.text(
                t,
                ax.get_ylim()[1] * 0.9,
                label,
                color="red",
                fontsize=8,
                rotation=90,
                ha="right",
            )

    axes[-1].set_xlabel("Time (s)")
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(output_path)
    plt.close()


def filter_signal(raw, low_cut=0.1, hi_cut=30):
    return raw.copy().filter(low_cut, hi_cut)


def parse_args():
    parser = argparse.ArgumentParser(description="EEG DWT with Event Highlights")
    parser.add_argument("edf_file", type=str, help="Path to the .edf file")
    parser.add_argument(
        "--wavelet", type=str, default="db4", help="Wavelet type (default: db4)"
    )
    parser.add_argument(
        "--level", type=int, default=4, help="Decomposition level (default: 4)"
    )
    parser.add_argument(
        "--output_dir", type=str, default="output", help="Directory to save plots"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Load raw data
    raw = mne.io.read_raw_edf(args.edf_file, preload=True, verbose=False)
    raw = filter_signal(raw)
    ch_names = raw.ch_names
    eeg_data, times = raw.get_data(return_times=True)

    print("Applying DWT denoising...")
    denoised_signals = []
    for i in range(len(ch_names)):
        denoised = wavelet_denoise(
            eeg_data[i], wavelet_name=args.wavelet, level=args.level
        )
        denoised_signals.append(denoised)
    denoised_signals = np.array(denoised_signals)

    # Extract event times (from annotations)
    event_times_sec = []
    event_labels = []
    for annot in raw.annotations:
        event_times_sec.append(annot["onset"])
        event_labels.append(annot["description"])

    # Plot
    plot_signals_with_events(
        denoised_signals,
        ch_names,
        times,
        event_times_sec,
        event_labels,
        "Denoised EEG with Event Highlights",
        os.path.join(args.output_dir, "denoised_with_events.png"),
    )

    print(f"✅ Plot saved at: {args.output_dir}/denoised_with_events.png")


if __name__ == "__main__":
    main()
