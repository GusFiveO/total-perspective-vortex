import numpy as np
import matplotlib.pyplot as plt
import pywt
import mne
import argparse


def process_eeg_signal(
    edf_file, channel_name, wavelet="cmor1.5-1.0", scales=np.arange(1, 128)
):
    # Load the EDF file
    raw = mne.io.read_raw_edf(edf_file, preload=True)

    # Select the channel of interest
    if channel_name not in raw.ch_names:
        raise ValueError(f"Channel '{channel_name}' not found in the EDF file.")

    signal = raw.get_data(picks=channel_name)[0]
    sampling_rate = raw.info["sfreq"]
    time = np.arange(0, len(signal)) / sampling_rate

    # Normalize the signal
    signal = (signal - np.mean(signal)) / np.std(signal)

    # Apply the Continuous Wavelet Transform (CWT)
    coefficients, frequencies = pywt.cwt(
        signal, scales, wavelet, sampling_period=1 / sampling_rate
    )

    # Visualize the results
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    # Plot the original signal
    axes[0].plot(time, signal)
    axes[0].set_title("Original Signal")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Amplitude")

    # Plot the scaleogram
    power = np.abs(coefficients) ** 2
    im = axes[1].imshow(
        power,
        extent=[time.min(), time.max(), frequencies.min(), frequencies.max()],
        aspect="auto",
        cmap="viridis",
        origin="lower",
    )
    axes[1].set_title("Scaleogram")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Frequency (Hz)")
    fig.colorbar(im, ax=axes[1], orientation="vertical", label="Power")

    # Plot the power spectral density (PSD) of the original signal for comparison
    psd, freqs = plt.psd(signal, Fs=sampling_rate)
    axes[2].plot(freqs, psd)
    axes[2].set_title("Power Spectral Density")
    axes[2].set_xlabel("Frequency (Hz)")
    axes[2].set_ylabel("Power/Frequency (dB/Hz)")

    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Process EEG signal using wavelet transform and plot scaleogram."
    )
    parser.add_argument("edf_file", type=str, help="Path to the EDF file.")
    parser.add_argument(
        "channel_name", type=str, help="Name of the EEG channel to process."
    )
    parser.add_argument(
        "--wavelet",
        type=str,
        default="cmor1.5-1.0",
        help="Wavelet function to use (default: cmor1.5-1.0).",
    )
    parser.add_argument(
        "--scales",
        type=str,
        default="1-128",
        help="Range of scales for CWT (default: 1-128).",
    )

    args = parser.parse_args()

    # Parse the scales argument
    scale_range = args.scales.split("-")
    if len(scale_range) != 2:
        raise ValueError("Scales argument must be in the format 'start-end'.")
    scales = np.arange(int(scale_range[0]), int(scale_range[1]))

    process_eeg_signal(args.edf_file, args.channel_name, args.wavelet, scales)


if __name__ == "__main__":
    main()
