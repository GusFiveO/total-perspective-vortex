import argparse
from matplotlib import pyplot as plt
import numpy as np
import mne

from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score

from plot_utils import plot_signals_with_events
from mybci.custom_transformer.Csp import CustomCSP

from mne.decoding import CSP


def cross_val_training(epochs, cv=5):
    X = epochs.get_data()
    y = epochs.events[:, 2] - 2

    # pipeline = make_pipeline(CSP(n_components=4), RandomForestClassifier())
    pipeline = make_pipeline(
        CustomCSP(n_components=4), RandomForestClassifier()
    )

    scores = cross_val_score(pipeline, X, y, cv=cv)

    print(scores)
    print(scores.mean())
    return scores.mean()
