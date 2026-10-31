import numpy as np

from tripeta import model


def test_cleaning_removes_impossible_trips(frames):
    train, _ = frames
    assert (train.duration_min >= 1).all() and (train.duration_min <= 240).all()
    assert (train.distance_miles >= 0.1).all()
    assert train.pickup_zone.between(1, 265).all()


def test_model_beats_the_naive_baseline_on_unseen_month(frames, tiny_model):
    train, valid = frames
    booster, _ = model.load(tiny_model)
    base = model.Baseline.fit(train)
    m = model.mae(valid.duration_min, model.predict(booster, valid))
    b = model.mae(valid.duration_min, base.predict(valid))
    assert m < b, f"model {m:.2f} should beat baseline {b:.2f}"


def test_predictions_are_positive_and_finite(frames, tiny_model):
    booster, _ = model.load(tiny_model)
    p = model.predict(booster, frames[1])
    assert np.isfinite(p).all() and (p > 0).all()


def test_longer_trips_take_longer(frames, tiny_model):
    import pandas as pd
    booster, _ = model.load(tiny_model)
    base = dict(pickup_zone=161, dropoff_zone=237, hour=12, dow=2, passengers=1)
    short = model.predict(booster, pd.DataFrame([{**base, "distance_miles": 1.0}]))[0]
    long_ = model.predict(booster, pd.DataFrame([{**base, "distance_miles": 9.0}]))[0]
    assert long_ > short * 2


def test_baseline_is_distance_over_speed():
    import pandas as pd
    df = pd.DataFrame({"hour": [8] * 3, "distance_miles": [2.0, 4.0, 6.0], "duration_min": [12.0, 24.0, 36.0]})
    b = model.Baseline.fit(df)
    assert abs(b.predict(df.iloc[[0]])[0] - 12.0) < 1e-6
