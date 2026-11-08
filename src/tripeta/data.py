"""Load and clean TLC yellow-taxi files into a training frame. Same validity rules as nyc-taxi-data-platform."""
from __future__ import annotations

from pathlib import Path
import re

import duckdb
import pandas as pd

FEATURES = ["pickup_zone", "dropoff_zone", "hour", "dow", "distance_miles", "passengers"]
CATEGORICAL = ["pickup_zone", "dropoff_zone"]
TARGET = "duration_min"

# One rule set, used for training and for validating requests, so the model never sees what the API refuses.
LIMITS = {"distance_miles": (0.1, 100.0), "passengers": (1, 6), "zone": (1, 265)}


def validate_month(month: str) -> str:
    if not re.fullmatch(r"[0-9]{4}-(?:0[1-9]|1[0-2])", month) or int(month[:4]) < 1:
        raise ValueError("month must use YYYY-MM with a valid year and month")
    return month


def load_month(path: str | Path, sample: int | None = None, seed: int = 0,
               expected_month: str | None = None) -> pd.DataFrame:
    """Clean rows of one parquet file. Trips are kept only if they could be real."""
    lo, hi = LIMITS["distance_miles"]
    if sample is not None and (isinstance(sample, bool) or not isinstance(sample, int) or sample < 1):
        raise ValueError("sample must be a positive integer or None")
    q = f"""
    select
        PULocationID::int as pickup_zone, DOLocationID::int as dropoff_zone,
        extract('hour' from tpep_pickup_datetime)::int as hour,
        (isodow(tpep_pickup_datetime) - 1)::int as dow,
        trip_distance::double as distance_miles,
        least(greatest(coalesce(passenger_count, 1), 1), 6)::int as passengers,
        date_diff('second', tpep_pickup_datetime, tpep_dropoff_datetime) / 60.0 as duration_min,
        strftime(tpep_pickup_datetime, '%Y-%m') as month
    from read_parquet(?)
    where tpep_dropoff_datetime > tpep_pickup_datetime
      and date_diff('second', tpep_pickup_datetime, tpep_dropoff_datetime) between 60 and 4 * 3600
      and trip_distance between {lo} and {hi}
      and PULocationID between 1 and 265 and DOLocationID between 1 and 265
      and fare_amount > 0
    """
    parameters = [str(path)]
    if expected_month is not None:
        validate_month(expected_month)
        q += " and strftime(tpep_pickup_datetime, '%Y-%m') = ?"
        parameters.append(expected_month)
    if sample is not None:
        q = f"select * from ({q}) using sample {int(sample)} rows (reservoir, {int(seed)})"
    return duckdb.sql(q, params=parameters).df()


def load_months(folder: str | Path, months: list[str], sample_per_month: int | None = None) -> pd.DataFrame:
    if not months:
        raise ValueError("at least one month is required")
    for month in months:
        validate_month(month)
    frames = [load_month(Path(folder) / f"yellow_tripdata_{m}.parquet", sample_per_month, seed=i, expected_month=m)
              for i, m in enumerate(months)]
    if any(frame.empty for frame in frames):
        raise ValueError("each requested month must contain valid in-period trips")
    return pd.concat(frames, ignore_index=True)
