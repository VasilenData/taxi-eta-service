"""Load and clean TLC yellow-taxi files into a training frame. Same validity rules as nyc-taxi-data-platform."""
from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

FEATURES = ["pickup_zone", "dropoff_zone", "hour", "dow", "distance_miles", "passengers"]
CATEGORICAL = ["pickup_zone", "dropoff_zone"]
TARGET = "duration_min"

# One rule set, used for training and for validating requests, so the model never sees what the API refuses.
LIMITS = {"distance_miles": (0.1, 100.0), "passengers": (1, 6), "zone": (1, 265)}


def load_month(path: str | Path, sample: int | None = None, seed: int = 0) -> pd.DataFrame:
    """Clean rows of one parquet file. Trips are kept only if they could be real."""
    lo, hi = LIMITS["distance_miles"]
    q = f"""
    select
        PULocationID::int as pickup_zone, DOLocationID::int as dropoff_zone,
        extract('hour' from tpep_pickup_datetime)::int as hour,
        (isodow(tpep_pickup_datetime) - 1)::int as dow,
        trip_distance::double as distance_miles,
        least(greatest(coalesce(passenger_count, 1), 1), 6)::int as passengers,
        date_diff('second', tpep_pickup_datetime, tpep_dropoff_datetime) / 60.0 as duration_min,
        strftime(tpep_pickup_datetime, '%Y-%m') as month
    from read_parquet('{path}')
    where tpep_dropoff_datetime > tpep_pickup_datetime
      and date_diff('second', tpep_pickup_datetime, tpep_dropoff_datetime) between 60 and 4 * 3600
      and trip_distance between {lo} and {hi}
      and PULocationID between 1 and 265 and DOLocationID between 1 and 265
      and fare_amount > 0
    """
    if sample:
        q = f"select * from ({q}) using sample {int(sample)} rows (reservoir, {int(seed)})"
    return duckdb.sql(q).df()


def load_months(folder: str | Path, months: list[str], sample_per_month: int | None = None) -> pd.DataFrame:
    frames = [load_month(Path(folder) / f"yellow_tripdata_{m}.parquet", sample_per_month, seed=i)
              for i, m in enumerate(months)]
    return pd.concat(frames, ignore_index=True)
