from pathlib import Path
import pandas as pd

from ml_workflow.train import train_model
from ml_workflow.utils.logging import get_logger


def test_train_creates_model(tmp_path):
    # Create dummy training data
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    df = pd.DataFrame({
        "feature1": [1, 2, 3],
        "feature2": [4, 5, 6],
        "target": [0, 1, 0],
    })
    df.to_csv(data_dir / "train.csv", index=False)

    model_dir = tmp_path / "models"
    logger = get_logger("test_train", console_log=False)

    train_model(data_path=data_dir, model_output=model_dir, logger=logger)

    model_file = model_dir / "model.pkl"
    assert model_file.exists()