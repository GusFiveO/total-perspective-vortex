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


# def plot_epoch_signals_by_class(epochs, labels, channel_names, output_dir):
#     class_ids = np.unique(labels)
#     os.makedirs(output_dir, exist_ok=True)

#     for class_id in class_ids:
#         class_epochs = epochs[labels == class_id]
#         avg_epoch = class_epochs.mean(axis=0)

#         fig, axes = plt.subplots(
#             len(channel_names), 1, figsize=(12, 2 * len(channel_names)), sharex=True
#         )
#         fig.suptitle(f"Class {class_id} - Averaged Epoch", fontsize=16)

#         for i, ax in enumerate(axes):
#             ax.plot(avg_epoch[i], color="black")
#             ax.set_ylabel(channel_names[i], rotation=0, labelpad=40)
#             ax.tick_params(left=False, labelleft=False)

#         axes[-1].set_xlabel("Time (samples)")
#         plt.tight_layout(rect=[0, 0, 1, 0.97])
#         plt.savefig(os.path.join(output_dir, f"class_{class_id}_avg_epoch.png"))
#         plt.close()


def plot_epoch_signals_by_class(epochs, labels, channel_names, output_dir):
    class_ids = np.unique(labels)
    os.makedirs(output_dir, exist_ok=True)

    # Choose a color map for class lines
    colors = plt.cm.tab10.colors  # up to 10 distinct colors

    # Prepare figure
    fig, axes = plt.subplots(
        len(channel_names), 1, figsize=(14, 2 * len(channel_names)), sharex=True
    )
    fig.suptitle("Averaged Epochs by Class", fontsize=18)

    # Plot each class average
    for idx, class_id in enumerate(class_ids):
        class_epochs = epochs[labels == class_id]
        avg_epoch = class_epochs.mean(axis=0)
        color = colors[idx % len(colors)]

        for ch_idx, ax in enumerate(axes):
            ax.plot(
                avg_epoch[ch_idx], label=f"Class {class_id}", color=color, linewidth=1.5
            )

    # Customize axes
    for i, ax in enumerate(axes):
        ax.set_ylabel(channel_names[i], rotation=0, labelpad=40)
        ax.tick_params(left=False, labelleft=False)
        ax.legend(loc="upper right", fontsize=8)

    axes[-1].set_xlabel("Time (samples)")
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(os.path.join(output_dir, "all_classes_avg_epochs.png"))
    plt.close()


def filter_signal(raw, low_cut=0.1, hi_cut=30):
    return raw.copy().filter(low_cut, hi_cut)


def parse_args():
    parser = argparse.ArgumentParser(description="EEG Epoch DWT with Class Labels")
    parser.add_argument("edf_file", type=str, help="Path to the .edf file")
    parser.add_argument(
        "--wavelet", type=str, default="db4", help="Wavelet type (default: db4)"
    )
    parser.add_argument(
        "--level", type=int, default=4, help="Wavelet level (default: 4)"
    )
    parser.add_argument(
        "--tmin", type=float, default=0.0, help="Epoch start time (sec)"
    )
    parser.add_argument("--tmax", type=float, default=1.0, help="Epoch end time (sec)")
    parser.add_argument(
        "--output_dir", type=str, default="output", help="Output directory"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Load EDF and annotations
    raw = mne.io.read_raw_edf(args.edf_file, preload=True, verbose=False)
    raw = filter_signal(raw)
    ch_names = raw.ch_names

    # Handle annotations (convert to events)
    if raw.annotations:
        events, event_id = mne.events_from_annotations(raw)
        print(f"Found {len(events)} events in the annotations.")
        print(f"Event IDs: {event_id}")
    else:
        raise RuntimeError(
            "No annotations found in the EDF file for event-based epoching."
        )

    # Create epochs
    epochs = mne.Epochs(
        raw,
        events,
        event_id=event_id,
        tmin=args.tmin,
        tmax=args.tmax,
        baseline=None,
        preload=True,
        verbose=False,
    )

    print(f"Found {len(epochs)} epochs with classes: {list(event_id.keys())}")

    # Extract epoch data and labels
    epoch_data = epochs.get_data()  # shape: (n_epochs, n_channels, n_times)
    labels = epochs.events[:, -1]  # Extract class labels (event codes)

    # Apply DWT denoising on each epoch
    denoised_epochs = []
    for epoch in epoch_data:
        denoised_epoch = np.array(
            [
                wavelet_denoise(epoch[ch], wavelet_name=args.wavelet, level=args.level)
                for ch in range(epoch.shape[0])
            ]
        )
        denoised_epochs.append(denoised_epoch)
    denoised_epochs = np.array(denoised_epochs)

    # Plot average signals per class
    plot_epoch_signals_by_class(denoised_epochs, labels, ch_names, args.output_dir)

    print(f"✅ Denoised class-averaged plots saved in: {args.output_dir}")


if __name__ == "__main__":
    main()
