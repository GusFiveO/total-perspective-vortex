from sklearn.model_selection import train_test_split
from mybci.io import load_model, load_task
from mybci.preprocessing import preprocessing, raw_to_epochs


def predict(subject, task_id, wavelet="db4", level=4, tmin=0, tmax=2):
    try:
        model = load_model(subject, task_id, "models")
    except FileNotFoundError:
        print(f"Model for subject {subject} and task {task_id} not found.")
        return
    raw_signal = load_task(subject, task_id)

    preprocecssed_signal = preprocessing(raw_signal, wavelet, level)

    epochs = raw_to_epochs(preprocecssed_signal, tmin, tmax)

    X_train, X_test, y_train, y_test = train_test_split(
        epochs.get_data(), epochs.events[:, 2] - 2, test_size=0.2, random_state=42
    )

    print(X_test.shape, y_test.shape)
    pred = model.predict(X_test)
    print(f"Predicted labels: {pred}")
    model.score(X_test, y_test)
    print(f"Accuracy: {model.score(X_test, y_test):.2f}")
