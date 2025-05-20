from mne import concatenate_raws
from mne.datasets import eegbci
import mne


def load_dataset(subjects, runs):
    print(subjects)
    subjects_raw_signals = dict()
    for subject in subjects:
        raw_fnames = eegbci.load_data(subject, runs, path="~/goinfre/eegdata")
        raw = concatenate_raws(
            [mne.io.read_raw_edf(f, preload=True) for f in raw_fnames]
        )
        raw.annotations.rename(dict(T1="left", T2="right"), verbose=False)
        subjects_raw_signals[subject] = raw
    return subjects_raw_signals
