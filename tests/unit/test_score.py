import pickle
from pathlib import Path
import pandas as pd
from sklearn.linear_model import LogisticRegression

from ml_workflow.score import score_model
from ml_workflow.utils.logging import get_logger


def test_score_creates_predictions(tmp_path):
    # Create dummy model
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    X = pd.DataFrame({"feature": [0, 1]})
    y = [0, 1]
    model = LogisticRegression()
    model.fit(X, y)


    with open(model_dir / "model.pkl", "wb") as f:
        pickle.dump(model, f)

    # Create validation data
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    df = pd.DataFrame({
        "feature": [0, 1],
        "target": [0, 1],
    })
    df.to_csv(data_dir / "validation.csv", index=False)

    output_dir = tmp_path / "output"
    logger = get_logger("test_score", console_log=False)

    score_model(
        model_path=model_dir,
        data_path=data_dir,
        output_path=output_dir,
        logger=logger,
    )

    preds_file = output_dir / "predictions.csv"
    assert preds_file.exists()