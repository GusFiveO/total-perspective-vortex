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


def plot_all_signals(signals, channel_names, title, output_path):
    n_channels = len(signals)
    fig, axes = plt.subplots(n_channels, 1, figsize=(12, 2 * n_channels), sharex=True)
    fig.suptitle(title, fontsize=16)

    for i, ax in enumerate(axes):
        ax.plot(signals[i], color="black")
        ax.set_ylabel(channel_names[i], rotation=0, labelpad=40)
        ax.tick_params(left=False, labelleft=False)

    axes[-1].set_xlabel("Time (samples)")
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(output_path)
    plt.close()


def filter_signal(raw, low_cut=0.1, hi_cut=30):
    raw_filt = raw.copy().filter(low_cut, hi_cut)
    return raw_filt


def parse_args():
    parser = argparse.ArgumentParser(description="EEG Wavelet Denoising (All Channels)")
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
        "--output_dir",
        type=str,
        default="output",
        help="Directory to save plots (default: ./output)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load EDF file using MNE
    raw = mne.io.read_raw_edf(args.edf_file, preload=True, verbose=False)
    print(raw.info)
    eeg_data, times = raw.get_data(return_times=True)

    raw_filt = filter_signal(raw)
    eeg_data_filt, times = raw_filt.get_data(return_times=True)
    ch_names = raw.ch_names

    print("Applying wavelet denoising to all channels...")

    denoised_signals = []
    for i in range(len(ch_names)):
        # signal = eeg_data[i]
        signal = eeg_data_filt[i]
        denoised = wavelet_denoise(signal, wavelet_name=args.wavelet, level=args.level)
        denoised_signals.append(denoised)

    # Plot original and denoised signals
    plot_all_signals(
        eeg_data,
        ch_names,
        "Original EEG Signals",
        os.path.join(args.output_dir, "original_signals.png"),
    )
    plot_all_signals(
        denoised_signals,
        ch_names,
        "Denoised EEG Signals",
        os.path.join(args.output_dir, "denoised_signals.png"),
    )

    print(f"✅ Plots saved in: {args.output_dir}")


if __name__ == "__main__":
    main()
