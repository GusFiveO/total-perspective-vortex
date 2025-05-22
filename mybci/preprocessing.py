import mne
from sklearn.pipeline import make_pipeline

from mybci.custom_transformer.Denoise import BandpassFilter, WaveletDenoiser


def preprocessing(raw_signal, wavelet, level):
    pipeline = make_pipeline(
        BandpassFilter(l_freq=7.0, h_freq=30.0),
        WaveletDenoiser(wavelet=wavelet, level=level),
    )

    preprocessed_signal = pipeline.fit_transform(raw_signal)
    return preprocessed_signal


def split_epochs(signal, tmin, tmax):
    events, event_id = mne.events_from_annotations(signal, verbose=False)

    epochs = mne.Epochs(
        signal,
        events,
        # event_id=event_id,
        event_id=[2, 3],
        tmin=tmin,
        tmax=tmax,
        baseline=None,
        preload=True,
        verbose=False,
    )

    return epochs
