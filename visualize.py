#! /usr/bin/env python3

import mne
import matplotlib.pyplot as plt
import sys

try:
    data_file = sys.argv[1]

    raw = mne.io.read_raw_edf(data_file)

    raw.compute_psd(fmax=50).plot(picks="data", exclude="bads", amplitude=False)
    raw.plot(duration=5, n_channels=10)
    plt.show()
except Exception as e:
    print(e)
