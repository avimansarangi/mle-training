from pathlib import Path

from ml_workflow.ingest_data import ingest_data
from ml_workflow.train import train_model
from ml_workflow.score import score_model
from ml_workflow.utils.logging import get_logger


def test_full_ml_pipeline(tmp_path):
    logger = get_logger("e2e_test", console_log=False)

    data_dir = tmp_path / "data"
    model_dir = tmp_path / "models"
    output_dir = tmp_path / "scores"

    ingest_data(data_dir, logger)
    train_model(data_dir, model_dir, logger)
    score_model(model_dir, data_dir, output_dir, logger)

    assert (data_dir / "train.csv").exists()
    assert (model_dir / "model.pkl").exists()
    assert (output_dir / "predictions.csv").exists()