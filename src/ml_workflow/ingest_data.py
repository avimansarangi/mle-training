import argparse
from pathlib import Path

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from ml_workflow.utils.logging import get_logger


def ingest_data(output_path: Path, logger):
    logger.info("Loading dataset")
    data = load_iris(as_frame=True)
    df = data.frame

    logger.info("Splitting into train and validation sets")
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

    output_path.mkdir(parents=True, exist_ok=True)

    train_path = output_path / "train.csv"
    val_path = output_path / "validation.csv"

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)

    logger.info(f"Training data saved to {train_path}")
    logger.info(f"Validation data saved to {val_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Ingest and prepare datasets")

    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("data/processed"),
        help="Output folder for processed datasets",
    )

    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--log-path", default=None)
    parser.add_argument(
        "--no-console-log",
        action="store_true",
        help="Disable console logging",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    logger = get_logger(
        name="ingest_data",
        log_level=args.log_level,
        log_path=args.log_path,
        console_log=not args.no_console_log,
    )

    ingest_data(args.output_path, logger)


if __name__ == "__main__":
    main()