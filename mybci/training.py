import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score, train_test_split

from mybci.custom_transformer.Csp import CustomCSP

from mne.decoding import CSP

from mybci.io import EXPERIMENTS, load_experiment, load_runs, load_task
from mybci.preprocessing import preprocessing, raw_to_epochs


def cross_val_training(epochs, cv=5):
    X = epochs.get_data()
    y = epochs.events[:, 2] - 2

    X_train, X_test, y_train, y_test = train_test_split(
        epochs.get_data(), epochs.events[:, 2] - 2, test_size=0.2, random_state=42
    )

    # pipeline = make_pipeline(CSP(n_components=4), RandomForestClassifier())
    pipeline = make_pipeline(CustomCSP(n_components=4), LogisticRegression())

    scores = cross_val_score(pipeline, X, y, cv=cv)

    pipeline.fit(X_train, y_train)

    accuracy = pipeline.score(X_test, y_test)
    print(f"accuracy: {accuracy:.2f}")

    return scores


def training_accuracy(epochs):
    """Train a classifier and return the accuracy."""
    X_train, X_test, y_train, y_test = train_test_split(
        epochs.get_data(), epochs.events[:, 2] - 2, test_size=0.2, random_state=42
    )

    # pipeline = make_pipeline(CSP(n_components=4), RandomForestClassifier())
    pipeline = make_pipeline(CustomCSP(n_components=4), LogisticRegression())
    pipeline.fit(X_train, y_train)
    accuracy = pipeline.score(X_test, y_test)
    return accuracy


def train_all(subjects, tmin, tmax, wavelet="db4", level=4):
    tasks_scores = dict()
    for task_id, task in enumerate(EXPERIMENTS):
        task_scores = np.array([])
        raw_tasks = load_experiment(subjects, task_id)
        for subject, raw_signal in raw_tasks.items():
            preprocessed_signal = preprocessing(raw_signal, wavelet, level)

            epochs = raw_to_epochs(preprocessed_signal, tmin, tmax)

            # subject_score = cross_val_training(epochs, cv=10)
            subject_score = training_accuracy(epochs)
            print(
                f"Experiment: {task["name"]}; Subject: {subject}; Score: {subject_score:.2f}"
            )
            experiment_scores = np.append(experiment_scores, subject_score)
        tasks_scores[task["name"]] = experiment_scores.mean()
    for exp_name, score in tasks_scores.items():
        print(f"{exp_name}: {score:.2f}")
    print("Average score: ", np.mean(list(tasks_scores.values())))


# def train_all(subjects, runs, tmin, tmax, wavelet="db4", level=4):
#     experiments_scores = dict()
#     for exp_name, exp in EXPERIMENTS.items():

#         if runs and not any(run in exp["runs"] for run in runs):
#             continue

#         selected_runs = [run for run in exp["runs"] if (not runs or run in runs)]

#         experiment_scores = np.array([])
#         raw_experiments = load_experiment(
#             subjects, exp_name, selected_runs, exp["events"]
#         )
#         for subject, raw_signal in raw_experiments.items():
#             preprocessed_signal = preprocessing(raw_signal, wavelet, level)

#             epochs = raw_to_epochs(preprocessed_signal, tmin, tmax)

#             # subject_score = cross_val_training(epochs, cv=10)
#             subject_score = training_accuracy(epochs)
#             print(
#                 f"Experiment: {exp_name}; Subject: {subject}; Score: {subject_score:.2f}"
#             )
#             experiment_scores = np.append(experiment_scores, subject_score)
#         experiments_scores[exp_name] = experiment_scores.mean()
#     for exp_name, score in experiments_scores.items():
#         print(f"{exp_name}: {score:.2f}")
#     print("Average score: ", np.mean(list(experiments_scores.values())))


def train_one(subjects, task, tmin, tmax, wavelet="db4", level=4):
    for subject in subjects:
        raw_runs = load_task(subject, task)
        preprocessed_signal = preprocessing(raw_runs, wavelet, level)

        epochs = raw_to_epochs(preprocessed_signal, tmin, tmax)

        subject_scores = cross_val_training(epochs)

        print(subject_scores)
        print(f"Subject: {subject}; Score: {subject_scores.mean():.2f}")


# def train_one(subjects, runs, tmin, tmax, wavelet="db4", level=4):
#     for subject in subjects:
#         raw_runs = load_runs(subject, runs)
#         preprocessed_signal = preprocessing(raw_runs, wavelet, level)

#         epochs = raw_to_epochs(preprocessed_signal, tmin, tmax)

#         subject_scores = cross_val_training(epochs)

#         print(subject_scores)
#         print(f"Subject: {subject}; Score: {subject_scores.mean():.2f}")
