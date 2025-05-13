from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np


class CustomCSP(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=4):
        self.n_components = n_components

    def fit(self, X, y):
        # X shape: (n_trials, n_channels, n_times)
        # y shape: (n_trials,)

        class_0 = X[y == 0]
        class_1 = X[y == 1]

        cov_0 = np.mean([np.cov(trial) for trial in class_0], axis=0)
        cov_1 = np.mean([np.cov(trial) for trial in class_1], axis=0)

        composite_cov = cov_0 + cov_1
        eigvals, eigvecs = np.linalg.eigh(composite_cov)
        whitening = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T

        S0 = whitening @ cov_0 @ whitening.T
        _, B = np.linalg.eigh(S0)

        filters = B.T @ whitening
        self.filters_ = filters[: self.n_components]
        return self

    def transform(self, X):
        # Apply spatial filters and return log-variance features
        X_transformed = np.array([self.filters_ @ trial for trial in X])
        features = np.log(np.var(X_transformed, axis=2))
        return features
