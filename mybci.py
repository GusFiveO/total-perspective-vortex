import argparse
import numpy as np
from mybci.training import (
    cross_val_training,
    train_all,
    train_one,
    training_accuracy,
)

from mybci.io import (
    load_experiment,
    EXPERIMENTS,
)


def parse_args():
    parser = argparse.ArgumentParser(description="EEG Wavelet Denoising (All Channels)")
    parser.add_argument(
        "--subjects",
        nargs="+",
        type=int,
        # default=[1],
        # default=range(1, 109),
        help="Subject IDs",
    )
    parser.add_argument(
        "-r",
        "--runs",
        nargs="+",
        type=int,
        # default=[3, 7, 11],
        # default=range(1, 14),
        help="Target Run IDs",
    )
    parser.add_argument(
        "-t",
        "--task",
        choices=[1, 2, 3, 4],
        type=int,
        help="Task ID (1: real fist movement, 2: imagined fist movement, 3: real movement fists/feet, 4: imagined movement fists/feet)",
    )
    parser.add_argument(
        "--wavelet",
        type=str,
        default="db4",
        help="Discrete wavelet type (default: db4)",
    )
    parser.add_argument(
        "--level", type=int, default=4, help="Decomposition level (default: 4)"
    )
    parser.add_argument("--tmin", type=float, default=0, help="Epochs starting time")
    parser.add_argument("--tmax", type=float, default=2, help="Epochs ending time")
    return parser.parse_args()


def mybci():
    args = parse_args()
    if args.subjects is None:
        train_all(
            subjects=range(1, 109),
            tmin=args.tmin,
            tmax=args.tmax,
            wavelet=args.wavelet,
            level=args.level,
        )
    else:
        train_one(
            subjects=args.subjects,
            task=args.task,
            # runs=args.runs,
            tmin=args.tmin,
            tmax=args.tmax,
            wavelet=args.wavelet,
            level=args.level,
        )
    # experiments_scores = dict()
    # for exp_name, exp in EXPERIMENTS.items():

    #     if args.runs and not any(run in exp["runs"] for run in args.runs):
    #         continue

    #     selected_runs = [
    #         run for run in exp["runs"] if (not args.runs or run in args.runs)
    #     ]

    #     experiment_scores = np.array([])
    #     raw_experiments = load_experiment(
    #         args.subjects, exp_name, selected_runs, exp["events"]
    #     )
    #     for subject, raw_signal in raw_experiments.items():
    #         preprocessed_signal = preprocessing(raw_signal, args.wavelet, args.level)

    #         epochs = raw_to_epochs(preprocessed_signal, args.tmin, args.tmax)

    #         # subject_score = cross_val_training(epochs, cv=10)
    #         subject_score = training_accuracy(epochs)
    #         print(
    #             f"Experiment: {exp_name}; Subject: {subject}; Score: {subject_score:.2f}"
    #         )
    #         experiment_scores = np.append(experiment_scores, subject_score)
    #     experiments_scores[exp_name] = experiment_scores.mean()
    # for exp_name, score in experiments_scores.items():
    #     print(f"{exp_name}: {score:.2f}")
    # print("Average score: ", np.mean(list(experiments_scores.values())))


if __name__ == "__main__":
    mybci()
