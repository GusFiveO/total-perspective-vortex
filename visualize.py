#! /usr/bin/env python3

import mne
import matplotlib.pyplot as plt
import sys

try:
    data_file = sys.argv[1]

    raw = mne.io.read_raw_edf(data_file, preload=True)

    raw.compute_psd().plot(
        picks="data",
        exclude="bads",
        amplitude=False,
        average=True,
        spatial_colors=False,
    )
    raw.plot(duration=5, scalings="auto")
    print(raw.info)
    print(raw.ch_names)
    print(raw.get_data())
    raw.filter(1, 50)
    print(raw.info)
    raw.plot(duration=5, scalings="auto")
    plt.show()
except Exception as e:
    print(e)
