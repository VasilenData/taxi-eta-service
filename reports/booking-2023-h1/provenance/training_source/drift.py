"""Drift monitoring: Population Stability Index of live requests against the training distribution.

PSI < 0.1 stable, 0.1 to 0.25 moderate shift, > 0.25 major shift (the usual industry thresholds).
"""
from __future__ import annotations

import numpy as np

NUMERIC_BINS = {"distance_miles": [0, 0.5, 1, 1.5, 2, 3, 5, 8, 12, 20, 101],
                "hour": list(range(0, 25)), "dow": list(range(0, 8)), "passengers": [1, 2, 3, 4, 5, 6, 7]}


def baseline_profile(df, features: list[str] | None = None) -> dict:
    """Training-time histogram per feature, stored in the model metadata."""
    prof = {}
    selected = set(df.columns if features is None else features)
    for col, edges in NUMERIC_BINS.items():
        if col not in selected:
            continue
        counts, _ = np.histogram(df[col], bins=edges)
        prof[col] = {"edges": edges, "share": (counts / counts.sum()).tolist()}
    for col in ("pickup_zone", "dropoff_zone"):
        if col not in selected:
            continue
        vc = df[col].value_counts(normalize=True)
        prof[col] = {"top": {int(k): float(v) for k, v in vc.head(50).items()}}
    return prof


def psi(expected, actual, eps: float = 1e-4) -> float:
    e = np.clip(np.asarray(expected, float), eps, None)
    a = np.clip(np.asarray(actual, float), eps, None)
    e, a = e / e.sum(), a / a.sum()
    return float(np.sum((a - e) * np.log(a / e)))


def feature_psi(profile: dict, rows: list[dict]) -> dict[str, float]:
    """PSI per feature for a batch of recent request feature dicts."""
    out = {}
    if not rows:
        return out
    for col, spec in profile.items():
        values = [r[col] for r in rows]
        if "edges" in spec:
            counts, _ = np.histogram(values, bins=spec["edges"])
            actual = counts / max(counts.sum(), 1)
            out[col] = psi(spec["share"], actual)
        else:
            top = {int(k): v for k, v in spec["top"].items()}
            keys = list(top) + ["other"]
            exp = [top[k] for k in top] + [max(1 - sum(top.values()), 0)]
            cnt = {k: 0 for k in keys}
            for v in values:
                cnt[v if v in top else "other"] += 1
            out[col] = psi(exp, [cnt[k] / len(values) for k in keys])
    return out


def status(value: float) -> str:
    return "stable" if value < 0.1 else "moderate" if value < 0.25 else "major"
