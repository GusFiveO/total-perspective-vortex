#! /usr/bin/env python3

import mne
import matplotlib.pyplot as plt
import sys

try:
    data_file = sys.argv[1]

    raw = mne.io.read_raw_edf(data_file, preload=True)

    new_ch_names = [s.replace(".", "") for s in raw.ch_names]

    raw.rename_channels(
        {original: new for original, new in zip(raw.info["ch_names"], new_ch_names)}
    )
    raw.filter(1, 50)

    events, event_id = mne.events_from_annotations(raw)
    tmin = -0.2
    tmax = 0.5
    epochs = mne.Epochs(
        raw, events, event_id, tmin, tmax, baseline=(None, 0), preload=True
    )

    T0_epochs = epochs["T0"]
    T0_epochs.plot(events=True, scalings="auto")
    T0_evoked = T0_epochs.average()
    T0_evoked.plot_image()
    T0_evoked.plot()

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
