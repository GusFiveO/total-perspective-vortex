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


def extract_features(trials, selected_filters):
    feats = []
    projected_signals = []
    for trial in trials:
        Z = selected_filters @ trial  # project trial
        log_var = np.log(np.var(Z, axis=1))  # log variance of each component
        feats.append(log_var)
        projected_signals.append(Z)
    return np.array(feats), np.array(projected_signals)


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

    features1, Z1 = extract_features(X1, selected_filters)
    features2, Z2 = extract_features(X2, selected_filters)

    return selected_filters, features1, features2, Z1, Z2


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
    filters, features1, features2, Z1, Z2 = csp(
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

    sfreq = raw.info["sfreq"]  # sampling frequency
    n_times = Z1.shape[2]
    times_ms = np.linspace(
        0, n_times / sfreq * 1000, n_times
    )  # convert to milliseconds

    # Compute average over all trials
    mean_Z1 = Z1.mean(axis=0)  # shape: (n_components, n_times)
    mean_Z2 = Z2.mean(axis=0)

    # Plot each CSP component
    fig, axes = plt.subplots(
        Z1.shape[1], 1, figsize=(12, 3 * Z1.shape[1]), sharex=True
    )

    for comp in range(Z1.shape[1]):
        ax = axes[comp]
        ax.plot(times_ms, mean_Z1[comp], label="Class 1", color="blue")
        ax.plot(times_ms, mean_Z2[comp], label="Class 2", color="green")
        ax.set_title(f"Mean CSP Component {comp+1} Over Time")
        ax.set_ylabel("Amplitude")
        ax.grid(True)
        if comp == 0:
            ax.legend()
        if comp == Z1.shape[1] - 1:
            ax.set_xlabel("Time (ms)")

    # Compute raw variances
    var_Z1 = np.var(Z1, axis=2)
    var_Z2 = np.var(Z2, axis=2)

    # Compute log-variances
    logvar_Z1 = np.log(var_Z1)
    logvar_Z2 = np.log(var_Z2)

    # Plot histogram for selected CSP components (e.g., component 0 and 3)
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    components_to_plot = [0, 3]

    for i, comp in enumerate(components_to_plot):
        # Raw variance histograms
        axs[i, 0].hist(var_Z1[:, comp], bins=10, alpha=0.7, label="Class 1")
        axs[i, 0].hist(var_Z2[:, comp], bins=10, alpha=0.7, label="Class 2")
        axs[i, 0].set_title(f"CSP Component {comp+1} - Raw Variance")
        axs[i, 0].set_xlabel("Variance")
        axs[i, 0].set_ylabel("Frequency")
        axs[i, 0].legend()

        # Log-variance histograms
        axs[i, 1].hist(logvar_Z1[:, comp], bins=10, alpha=0.7, label="Class 1")
        axs[i, 1].hist(logvar_Z2[:, comp], bins=10, alpha=0.7, label="Class 2")
        axs[i, 1].set_title(f"CSP Component {comp+1} - Log Variance")
        axs[i, 1].set_xlabel("log(Variance)")
        axs[i, 1].set_ylabel("Frequency")
        axs[i, 1].legend()

    plt.tight_layout()

    plt.show()
