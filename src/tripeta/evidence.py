"""Content identities for model inputs, separate from configuration labels."""
import hashlib
import json
from pathlib import Path

import pandas as pd


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def frame_fingerprint(frame: pd.DataFrame) -> str:
    """Hash ordered columns, dtypes and actual values in bounded CSV chunks.

    This is a versioned serialization for this numeric training-data contract,
    not a cross-language universal dataframe hashing standard.
    """
    digest = hashlib.sha256(json.dumps({"format": "training-csv-v1", "columns": list(frame.columns),
                                      "dtypes": [str(t) for t in frame.dtypes]}, sort_keys=True).encode())
    for start in range(0, len(frame), 10000):
        text = frame.iloc[start:start + 10000].to_csv(index=False, header=False, lineterminator="\n",
                                                    float_format="%.17g", date_format="%Y-%m-%dT%H:%M:%S.%f")
        digest.update(text.encode("utf-8"))
    return digest.hexdigest()


def data_fingerprint(frames: dict[str, pd.DataFrame]) -> str:
    identities = {name: frame_fingerprint(frame) for name, frame in frames.items()}
    return hashlib.sha256(json.dumps(identities, sort_keys=True).encode()).hexdigest()
