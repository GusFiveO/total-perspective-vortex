import argparse
import os
from matplotlib import pyplot as plt
import numpy as np
import mne
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
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    raw = mne.io.read_raw_edf(args.edf_file, preload=True, verbose=True)
    # print(f"Raw data shape: {raw._data.shape}")
    # events, event_id = mne.events_from_annotations(raw)
    # print("Data loaded successfully.")
    # print("Applying bandpass filter...")
    # raw_filt = filter_bandpass(raw)
    # print("Bandpass filter applied.")
    # print("Applying wavelet denoising to all channels...")
    # eeg_data, times = raw_filt.get_data(return_times=True)
    # ch_names = raw.ch_names
    # denoised_signals = denoise_signal(
    #     eeg_data, wavelet_name=args.wavelet, level=args.level
    # )
    # print("Wavelet denoising completed.")
    pipeline = make_pipeline(
        BandpassFilter(l_freq=1.0, h_freq=40.0),
        WaveletDenoiser(wavelet=args.wavelet, level=args.level),
        CustomCSP(
            n_components=4,
        ),
    )

    raw_filt = pipeline.fit_transform(raw)
    events, event_id = mne.events_from_annotations(raw)
    # plot_signals_with_events(
    #     raw_filt,
    #     raw.ch_names,
    #     raw.times,
    #     raw.info["sfreq"],
    #     events,
    #     event_id,
    #     title="EEG Signals with Events",
    #     output_path="output_signals_with_events.png",
    # )
    print(raw_filt.info)
    raw_filt.plot()
    plt.show()
