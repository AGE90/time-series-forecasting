import pandas as pd

from tsforecasting.data.make_dataset import make_dataset
from tsforecasting.features.build_features import build_features
from tsforecasting.models.predict_model import predict
from tsforecasting.models.train_model import train


def test_pipeline_runs_end_to_end() -> None:
    raw = pd.DataFrame(
        {
            "size": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 10.0, None],
            "color": ["r", "g", "b", "r", "g", "b", "r", "g", "b", "r", "r", "g"],
            "target": [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0],
        }
    )

    clean = make_dataset(raw)
    assert len(clean) == 10  # one duplicate and one missing-value row dropped

    features = build_features(clean)
    assert "color" not in features.columns
    assert "target" in features.columns

    model, metrics = train(features)
    assert 0.0 <= metrics["accuracy"] <= 1.0

    assert "prediction" in predict(model, features).columns
