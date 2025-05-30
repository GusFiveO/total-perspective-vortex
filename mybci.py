import argparse
import numpy as np
from mybci.predict import predict
from mybci.training import (
    train_all,
    train_one,
)


def parse_args():
    parser = argparse.ArgumentParser(description="EEG Wavelet Denoising (All Channels)")
    parser.add_argument(
        "--subject",
        choices=range(1, 110),
        type=int,
        help="Subject IDs",
    )
    parser.add_argument(
        "-t",
        "--task",
        choices=[1, 2, 3, 4],
        type=int,
        help="Task ID (1: real fist movement, 2: imagined fist movement, 3: real movement fists/feet, 4: imagined movement fists/feet)",
    )
    parser.add_argument(
        "--mode",
        choices=["train", "predict"],
        type=str,
        default="train",
        help="Mode of operation (default: train)",
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
    if args.mode == "predict":
        if args.subject is None or args.task is None:
            raise ValueError("Subject and task must be specified for prediction.")
        predict(
            subject=args.subject,
            task_id=args.task,
            wavelet=args.wavelet,
            level=args.level,
            tmin=args.tmin,
            tmax=args.tmax,
        )
        return
    elif args.subject is None:
        train_all(
            subjects=range(1, 110),
            tmin=args.tmin,
            tmax=args.tmax,
            wavelet=args.wavelet,
            level=args.level,
        )
    else:
        train_one(
            subject=args.subject,
            task=args.task,
            # runs=args.runs,
            tmin=args.tmin,
            tmax=args.tmax,
            wavelet=args.wavelet,
            level=args.level,
        )


if __name__ == "__main__":
    mybci()
