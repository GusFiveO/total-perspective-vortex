import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import mne
from mne.decoding import Scaler, Vectorizer
import argparse
import numpy as np
from scipy.linalg import eigh


def parse_args():
    parser = argparse.ArgumentParser(description="CSP Algorithm for EEG Data")
    parser.add_argument("data", type=str, help="Path to the EEG data file")
    parser.add_argument(
        "--num_components",
        type=int,
        default=4,
        help="Number of CSP components to compute",
    )
    parser.add_argument(
        "--tmin", type=float, default=0, help="Start time for epochs"
    )
    parser.add_argument(
        "--tmax",
        type=float,
        default=2,
        help="End time for epochs (in seconds)",
    )
    args = parser.parse_args()
    return args


def load_data(file_path):
    raw = mne.io.read_raw_edf(file_path, preload=True)
    return raw


def compute_covariance(trial):
    """Return normalized covariance matrix of shape (C, C)."""
    cov = np.dot(trial, trial.T)
    return cov / np.trace(cov)


def average_covariance(trials):
    """Return average normalized covariance matrix over all trials."""
    return np.mean([compute_covariance(trial) for trial in trials], axis=0)


def csp(X1, X2, n_components=4):
    """
    Apply CSP on two classes of trials.

    Args:
        X1: Trials from class 1 (shape: N1 x C x T)
        X2: Trials from class 2 (shape: N2 x C x T)
        N1, N2: Number of trials in each class
        C: Number of channels
        T: Number of time points
        n_components: Number of CSP components to keep

    Returns:
        filters: CSP projection matrix (C x C)
        features1, features2: Transformed feature sets (N1 x n_components, N2 x n_components)
    """
    # Step 1: Compute average covariances
    cov1 = average_covariance(X1)
    cov2 = average_covariance(X2)

    # Step 2: Composite covariance matrix
    composite_cov = cov1 + cov2

    # Step 3: Solve the generalized eigenvalue problem
    eigvals, eigvecs = eigh(cov1, composite_cov)

    # Step 4: Sort eigenvectors by eigenvalues
    idx = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, idx]

    # Step 5: Get spatial filters
    filters = eigvecs.T  # shape: C x C

    # Step 6: Select top and bottom filters
    selected_filters = np.concatenate(
        [
            filters[: n_components // 2],  # max variance class 1
            filters[-n_components // 2 :],  # max variance class 2
        ],
        axis=0,
    )

    # Step 7: Feature extraction (log-variance of filtered signals)
    def extract_features(trials):
        feats = []
        for trial in trials:
            Z = selected_filters @ trial  # project trial
            log_var = np.log(
                np.var(Z, axis=1)
            )  # log variance of each component
            feats.append(log_var)
        return np.array(feats)

    features1 = extract_features(X1)
    features2 = extract_features(X2)

    return selected_filters, features1, features2


if __name__ == "__main__":
    args = parse_args()
    raw = load_data(args.data)

    # Preprocess the data
    raw.filter(7, 30, fir_design="firwin")
    events, events_id = mne.events_from_annotations(raw)
    epochs = mne.Epochs(
        raw,
        events,
        events_id,
        tmin=args.tmin,
        tmax=args.tmax,
        baseline=None,
        preload=True,
        verbose=False,
    )
    print(events_id)

    # Split epochs into two classes
    X1 = epochs["T1"].get_data()
    X2 = epochs["T2"].get_data()

    # Apply CSP
    filters, features1, features2 = csp(
        X1, X2, n_components=args.num_components
    )

    print("X1 shape:", X1.shape)
    print("X2 shape:", X2.shape)
    print("CSP Filters shape:", filters.shape)
    print("Class 1 features shape:", features1.shape)
    print("Class 2 features shape:", features2.shape)

    # Plot the CSP filters
    plt.figure(figsize=(10, 5))
    plt.imshow(filters.T, aspect="auto", cmap="jet")
    plt.colorbar(label="CSP Filter Coefficients")
    plt.title("CSP Filters")
    plt.xlabel("CSP Components")
    plt.ylabel("EEG Channels")

    plt.show()
