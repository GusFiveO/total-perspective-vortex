import time

from sklearn.model_selection import train_test_split
from mybci.io import load_model, load_task
from mybci.preprocessing import preprocessing, raw_to_epochs


def predict_stream(subject, task_id, wavelet="db4", level=4, tmin=0, tmax=2):
    try:
        model = load_model(subject, task_id, "models")
    except FileNotFoundError:
        print(f"Model for subject {subject} and task {task_id} not found.")
        return

    # Load and preprocess raw EEG signal
    raw = load_task(subject, task_id)
    preprocessed = preprocessing(raw, wavelet, level)
    epochs = raw_to_epochs(preprocessed, tmin, tmax)

    X_train, X_test, y_train, y_test = train_test_split(
        epochs.get_data(),
        epochs.events[:, 2] - 2,
        test_size=0.2,
        random_state=42,
    )

    correct = 0
    total = len(X_test)

    print(f"Starting simulated real-time prediction ({total} epochs)...")

    for i, (epoch, label) in enumerate(zip(X_test, y_test)):
        start_time = time.time()

        # Predict single epoch
        prediction = model.predict(epoch[None, ...])[0]  # Add batch dimension

        elapsed = time.time() - start_time
        success = prediction == label
        correct += int(success)

        print(
            f"[{i+1}/{total}] Prediction: {prediction}, True: {label}, "
            f"{'✔️' if success else '❌'}, Time: {elapsed:.3f}s"
        )

        if elapsed > 2.0:
            print("⚠️ WARNING: Prediction exceeded 2s deadline!")

        time.sleep(max(0, 2.0 - elapsed))  # simulate 2s pacing

    accuracy = correct / total if total > 0 else 0
    print(f"\n✅ Stream prediction complete. Accuracy: {accuracy:.2%}")
