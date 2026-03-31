from pathlib import Path

from ml_workflow.ingest_data import ingest_data
from ml_workflow.utils.logging import get_logger


def test_ingest_data_creates_files(tmp_path):
    logger = get_logger("test_ingest", console_log=False)

    ingest_data(output_path=tmp_path, logger=logger)

    train_file = tmp_path / "train.csv"
    val_file = tmp_path / "validation.csv"

    assert train_file.exists()
    assert val_file.exists()