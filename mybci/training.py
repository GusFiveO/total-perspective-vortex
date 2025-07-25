import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score, train_test_split

from mybci.custom_transformer.Csp import CustomCSP

from mybci.io import TASKS, load_task, save_model
from mybci.preprocessing import preprocessing, raw_to_epochs


def cross_val_training(epochs, subject, task, cv=5):
    X = epochs.get_data()
    y = epochs.events[:, 2] - 2

    X_train, X_test, y_train, y_test = train_test_split(
        epochs.get_data(),
        epochs.events[:, 2] - 2,
        test_size=0.2,
        random_state=42,
    )

    # pipeline = make_pipeline(CSP(n_components=4), RandomForestClassifier())
    pipeline = make_pipeline(CustomCSP(n_components=4), LogisticRegression())

    scores = cross_val_score(pipeline, X, y, cv=cv)

    pipeline.fit(X_train, y_train)

    accuracy = pipeline.score(X_test, y_test)
    print(f"accuracy: {accuracy:.2f}")
    save_model(pipeline, subject, task, "models")

    return scores


def training_accuracy(epochs):
    """Train a classifier and return the accuracy."""
    X_train, X_test, y_train, y_test = train_test_split(
        epochs.get_data(),
        epochs.events[:, 2] - 2,
        test_size=0.2,
        random_state=42,
    )

    # pipeline = make_pipeline(CSP(n_components=4), RandomForestClassifier())
    pipeline = make_pipeline(CustomCSP(n_components=4), LogisticRegression())
    pipeline.fit(X_train, y_train)
    accuracy = pipeline.score(X_test, y_test)
    return accuracy


def train_all(subjects, tmin, tmax, wavelet="db4", level=4):
    tasks_scores = dict()
    for task_id, task in enumerate(TASKS):
        task_scores = np.array([])
        for subject in subjects:
            raw_signal = load_task(subject, task_id)
            preprocessed_signal = preprocessing(raw_signal, wavelet, level)

            epochs = raw_to_epochs(preprocessed_signal, tmin, tmax)

            subject_score = training_accuracy(epochs)
            print(
                f"{task['name']}; Subject: {subject}; Accuracy = {subject_score:.2f}"
            )
            task_scores = np.append(task_scores, subject_score)
        tasks_scores[task["name"]] = task_scores.mean()
    print("\nMean accuracy per task on all subjects:")
    for exp_name, score in tasks_scores.items():
        print(f"{exp_name}: {score:.2f}")
    print(
        "\nMean accuracy on all tasks:", np.mean(list(tasks_scores.values()))
    )


def train_one(subject, task, tmin, tmax, wavelet="db4", level=4):
    # for subject in subjects:
    raw_runs = load_task(subject, task)
    preprocessed_signal = preprocessing(raw_runs, wavelet, level)

    epochs = raw_to_epochs(preprocessed_signal, tmin, tmax)

    subject_scores = cross_val_training(epochs, subject, task)

    print(subject_scores)
    print(f"Subject: {subject}; Score: {subject_scores.mean():.2f}")
