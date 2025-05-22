from mne import concatenate_raws
from mne.datasets import eegbci
import mne

EXPERIENCES = {
    # "Baseline, eyes open": {"runs": [1], "events": {"T0": "rest"}},
    # "Baseline, eyes closed": {"runs": [2], "events": {"T0": "rest"}},
    "Task 1 (real fist movement)": {
        "runs": [3, 7, 11],
        "events": {"T0": "rest", "T1": "left_fist", "T2": "right_fist"},
    },
    "Task 2 (imagined fist movement)": {
        "runs": [4, 8, 12],
        "events": {
            "T0": "rest",
            "T1": "left_fist_imagined",
            "T2": "right_fist_imagined",
        },
    },
    "Task 3 (real movement: fists/feet)": {
        "runs": [5, 9, 13],
        "events": {"T0": "rest", "T1": "both_fists", "T2": "both_feet"},
    },
    "Task 4 (imagined movement: fists/feet)": {
        "runs": [6, 10, 14],
        "events": {
            "T0": "rest",
            "T1": "both_fists_imagined",
            "T2": "both_feet_imagined",
        },
    },
}


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


def load_run(subject, run, events):
    raw_fnames = eegbci.load_data(subject, [run], path="~/goinfre/eegdata")
    raw = concatenate_raws(
        [mne.io.read_raw_edf(f, preload=True) for f in raw_fnames]
    )
    raw.annotations.rename(dict(T1="left", T2="right"), verbose=False)
    return raw


def load_experiments(subjects, runs):
    subjects_raw_signals = dict()
    for subject in subjects:
        for exp_name, exp in EXPERIENCES.items():
            runs_raw_signals = dict()
            for run in exp["runs"]:
                if runs is None or run in runs:
                    print(f"Loading {exp_name} (run {run})")
                    run_raw_signals = load_run(
                        subject, run, events=exp["events"]
                    )
                    runs_raw_signals[run] = run_raw_signals
            subjects_raw_signals[subject] = runs_raw_signals
    return subjects_raw_signals
