import shutil

import duckdb
import pytest

from tripeta.data import load_month, load_months
from conftest import SAMPLE


def test_source_paths_with_apostrophes_are_bound_as_data(tmp_path):
    path = tmp_path / "owner's-trip.parquet"
    shutil.copy2(SAMPLE / "yellow_tripdata_2023-01.parquet", path)
    assert not load_month(path, sample=10).empty


def test_out_of_period_rows_cannot_cross_the_time_split(tmp_path):
    source = SAMPLE / "yellow_tripdata_2023-01.parquet"
    destination = tmp_path / "yellow_tripdata_2023-01.parquet"
    with duckdb.connect() as con:
        con.execute("""copy (
            select * from read_parquet(?)
            union all
            select * replace (tpep_pickup_datetime + interval '1 year' as tpep_pickup_datetime,
                              tpep_dropoff_datetime + interval '1 year' as tpep_dropoff_datetime)
            from read_parquet(?)
        ) to ? (format parquet)""", [str(source), str(source), str(destination)])
    loaded = load_months(tmp_path, ["2023-01"])
    assert set(loaded.month) == {"2023-01"}


@pytest.mark.parametrize("sample", [0, -1, True, 1.5])
def test_invalid_sample_size_fails_before_reading(sample):
    with pytest.raises(ValueError, match="positive"):
        load_month("does-not-exist.parquet", sample=sample)
