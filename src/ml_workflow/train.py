import argparse
import pickle
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression

from ml_workflow.utils.logging import get_logger


def train_model(data_path: Path, model_output: Path, logger):
    train_file = data_path / "train.csv"

    logger.info(f"Loading training data from {train_file}")
    df = pd.read_csv(train_file)

    X = df.drop(columns=["target"])
    y = df["target"]

    logger.info("Training Logistic Regression model")
    model = LogisticRegression(max_iter=200)
    model.fit(X, y)

    model_output.mkdir(parents=True, exist_ok=True)
    model_path = model_output / "model.pkl"

    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    logger.info(f"Model saved to {model_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train ML model")

    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("data/processed"),
        help="Input dataset folder",
    )

    parser.add_argument(
        "--model-output",
        type=Path,
        default=Path("artifacts/models"),
        help="Output folder for trained models",
    )

    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--log-path", default=None)
    parser.add_argument("--no-console-log", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()

    logger = get_logger(
        name="train",
        log_level=args.log_level,
        log_path=args.log_path,
        console_log=not args.no_console_log,
    )

    train_model(args.data_path, args.model_output, logger)


if __name__ == "__main__":
    main()