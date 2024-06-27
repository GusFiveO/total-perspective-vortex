#! /usr/bin/env python3

import mne
import matplotlib.pyplot as plt
import numpy as np
from autoreject import AutoReject
import sys

raw = None


def ICA(raw):
    ica_low_cut = 1.0
    hi_cut = 30
    raw_ica = raw.copy().filter(ica_low_cut, hi_cut)
    # Break raw data into 1 s epochs

    tstep = 1.0
    events_ica = mne.make_fixed_length_events(raw_ica, duration=tstep)
    epochs_ica = mne.Epochs(
        raw_ica, events_ica, tmin=0.0, tmax=tstep, baseline=None, preload=True
    )

    ar = AutoReject(
        n_interpolate=[1, 2, 4],
        random_state=42,
        picks=mne.pick_types(epochs_ica.info, eeg=True, eog=False),
        n_jobs=-1,
        verbose=False,
    )

    ar.fit(epochs_ica)

    reject_log = ar.get_reject_log(epochs_ica)
    fig, ax = plt.subplots(figsize=[15, 5])
    reject_log.plot("horizontal", ax=ax, aspect="auto")

    # ICA parameters

    random_state = 42  # ensures ICA is reproducible each time it's run
    ica_n_components = (
        0.99  # Specify n_components as a decimal to set % explained variance
    )

    # Fit ICA
    ica = mne.preprocessing.ICA(
        n_components=ica_n_components,
        random_state=random_state,
    )
    ica.fit(epochs_ica[~reject_log.bad_epochs], decim=3)
    ica.plot_components()
    plt.show()


try:
    data_file = sys.argv[1]

    raw = mne.io.read_raw_edf(data_file, preload=True)

    # new_ch_names = [s.replace(".", "") for s in raw.ch_names]
    new_ch_names = [
        s.replace(".", "")
        .replace("c", "C")
        .replace("Cp", "CP")
        .replace("f", "F")
        .replace("t", "T")
        .replace("Tp", "TP")
        .replace("o", "O")
        for s in raw.ch_names
    ]

    raw.rename_channels(
        {original: new for original, new in zip(raw.info["ch_names"], new_ch_names)}
    )

    # layout = mne.channels.read_layout("EEG1005")  # Use an appropriate layout file name
    # pos_2d = layout.pos[:, :2]  # Only take the 2D positions
    # pos_3d = np.hstack(
    #     [pos_2d, np.zeros((pos_2d.shape[0], 1))]
    # )  # Add a zero column for the z-coordinates
    # ch_names = layout.names
    # montage = mne.channels.make_dig_montage(
    #     ch_pos=dict(zip(ch_names, pos_3d)), coord_frame="head"
    # )
    montage = mne.channels.make_standard_montage("standard_1020")
    raw.set_montage(montage)
    print(raw.info)

    fig = raw.compute_psd(
        # tmax=np.inf
    ).plot(
        # average=True, amplitude=False, picks="data", exclude="bads"
    )
    plt.show()
    ICA(raw)
    exit()

    low_cut = 0.1
    hi_cut = 30

    raw_filt = raw.copy().filter(low_cut, hi_cut)

    raw.plot(start=15, duration=5, scalings="auto", title="Unfiltered")
    raw_filt.plot(start=15, duration=5, scalings="auto", title="Filtered")

    events, event_id = mne.events_from_annotations(raw_filt)
    tmin = -0.2
    tmax = 0.5
    epochs = mne.Epochs(
        raw_filt, events, event_id, tmin, tmax, baseline=(None, 0), preload=True
    )

    T0_epochs = epochs["T0"]
    # T0_epochs.plot(events=True, scalings="auto")
    T0_evoked = T0_epochs.average()
    # T0_evoked.plot_image()
    # T0_evoked.plot()

    print(f"epochs {epochs['T0']}")
    plt.show()
except Exception as e:
    print(e)

    # new_ch_names = [
    #     s.replace(".", "")
    #     .replace("c", "C")
    #     .replace("Cp", "CP")
    #     .replace("f", "F")
    #     .replace("t", "T")
    #     .replace("Tp", "TP")
    #     .replace("o", "O")
    #     for s in raw.ch_names
    # ]

    # layout = mne.channels.read_layout("EEG1005")
    # print(layout.names)
    # picks = []
    # for channel in new_ch_names:
    #     picks.append(layout.names.index(channel))
    # display = layout.plot(picks=picks)
