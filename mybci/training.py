from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score

from mybci.custom_transformer.Csp import CustomCSP

from mne.decoding import CSP


def cross_val_training(epochs, cv=5):
    X = epochs.get_data()
    y = epochs.events[:, 2] - 2

    # pipeline = make_pipeline(CSP(n_components=4), RandomForestClassifier())
    pipeline = make_pipeline(CustomCSP(n_components=4), LogisticRegression())

    scores = cross_val_score(pipeline, X, y, cv=cv)

    return scores.mean()
