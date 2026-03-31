import argparse
import pickle
from pathlib import Path

import pandas as pd

from ml_workflow.utils.logging import get_logger


def score_model(model_path: Path, data_path: Path, output_path: Path, logger):
    model_file = model_path / "model.pkl"
    val_file = data_path / "validation.csv"

    logger.info("Loading model")
    with open(model_file, "rb") as f:
        model = pickle.load(f)

    logger.info("Loading validation data")
    df = pd.read_csv(val_file)

    X = df.drop(columns=["target"])
    predictions = model.predict(X)

    output_path.mkdir(parents=True, exist_ok=True)
    output_file = output_path / "predictions.csv"

    pd.DataFrame({"prediction": predictions}).to_csv(output_file, index=False)

    logger.info(f"Predictions saved to {output_file}")


def parse_args():
    parser = argparse.ArgumentParser(description="Score ML model")

    parser.add_argument(
        "--model-path", type=Path, default=Path("artifacts/models")
    )
    parser.add_argument(
        "--data-path", type=Path, default=Path("data/processed")
    )
    parser.add_argument(
        "--output-path", type=Path, default=Path("artifacts/scores")
    )

    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--log-path", default=None)
    parser.add_argument("--no-console-log", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()

    logger = get_logger(
        name="score",
        log_level=args.log_level,
        log_path=args.log_path,
        console_log=not args.no_console_log,
    )

    score_model(
        args.model_path,
        args.data_path,
        args.output_path,
        logger,
    )


if __name__ == "__main__":
    main()