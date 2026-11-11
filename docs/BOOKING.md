# Booking-time trip duration protocol

The decision is an estimated completed-trip duration in minutes at booking, given a pickup zone, intended drop-off zone, planned NYC local pickup time and passenger count. Recorded taximeter distance is unavailable at that time and is excluded from the model and route baseline. No live traffic or route planner is integrated. Historical distance-assisted results remain a separate experiment.

## Verified run: 11 November 2026

The default `models/current/bundle.json` is the exact verified booking bundle from this run. It is byte-identical
to the report's model artifact and the local booking candidate; historical distance-model files remain untouched.

| June 2023 held-out sample | Booking LightGBM | Training-only baseline |
|---|---:|---:|
| Eligible test trips | 200,000 | 200,000 |
| MAE, minutes | 4.06775 | 5.57395 |
| Median absolute percentage error | 20.61% | 26.93% |
| Nominal 80% interval coverage | 80.5885% | 80.7260% |
| Mean interval width, minutes | 14.2062 | 18.9326 |

MAE was 27.02% lower than the baseline. April validation MAE was 3.81540 versus 5.20481 minutes, so the baseline
qualification gate passed independently of June. January-March supplied 300,000 training rows (100,000/month);
April and May each supplied 200,000. Best iteration was 325 of 600, with four threads.

Weaknesses remain visible: the 19,887 test trips on routes without 20 training observations had MAE 6.38 minutes,
and 13:00-16:59 hourly slices were around 5.00-5.14 minutes. A 14.21-minute average interval width is substantial.
These observations were not used to retune the model. All six source checksums, four phase fingerprints,
600,000 paired predictions and aggregate/slice metrics were independently reconciled after the run.

[Report](../reports/booking-2023-h1/index.html) | [Protocol and metrics](../reports/booking-2023-h1/report.json) |
[Held-out predictions](../reports/booking-2023-h1/predictions_test.csv.gz) |
[Artifact checksums](../reports/booking-2023-h1/manifest.json).

The displayed HTML had a separator-encoding correction after evaluation. [The presentation addendum](../reports/booking-2023-h1/presentation-edit.json) records old/new hashes; the exact measured source and original HTML are archived under the report's `provenance/` directory. Metrics, predictions and model bytes were not changed.

## Frozen temporal roles

| Phase | Default source months | Allowed use |
|---|---|---|
| Train | January–March 2023 | Fit LightGBM, route medians, hourly/global fallback and drift profile |
| Validation | April 2023 | Early stopping and a strict MAE gate against the training-only baseline |
| Calibration | May 2023 | Fit 10th/90th percentiles of actual/predicted ratios independently for both models |
| Test | June 2023 | Final paired error, interval coverage and failure slices only |

The test source is loaded only after fitting, validation selection and calibration. Calibration does not retrain the model. The learned model must have lower validation MAE than the baseline; a tie retains the baseline. An unsuccessful learned candidate still produces a report and candidate bundle but does not update the optional booking registry. This is a baseline qualification gate, not a comparison against an existing production champion. Repeated tuning after inspecting June would invalidate its role as an untouched test.

Ratios are expanded to include 1 so the interval contains the point prediction. The nominal central 80% interval is assessed on the future test month. It is an empirical error interval, not a guarantee of conditional or future coverage under shift. Report both coverage and width.

## Grain, cohort and identities

The booking CLI defaults to four LightGBM threads and records the actual setting. Use `--threads 2` for a more constrained host.

One source row is one observed TLC trip. There is no assumed global trip identifier or deduplication. Evidence keys combine source month with the Parquet physical row number; the manifest identifies the exact source bytes. Source rows are restricted to their declared pickup month. Labels whose drop-off occurs at or after the phase end are excluded before sampling, so future labels cannot cross a fit/selection/calibration cutoff.

The existing training cohort is retained: 1–240 minute completed trips, recorded distance 0.1–100 miles, positive fare, pickup/drop-off zones 1–265. Passenger counts retain the existing impute-to-one and clamp-to-1–6 rule. These eligibility rules depend on completed-trip information even though distance, fare and duration are not prediction features. Therefore measured accuracy describes this eligible completed-trip cohort, not every booking request, cancellation or extreme trip. Zones 264/265 retain the historical unknown-zone categories.

The default sample is at most 100,000 eligible rows from each training month and 200,000 from each later month. Sampling uses a seeded hash of physical source-row position with a row-number tie break; source order and the recorded DuckDB version matter. Each month has a deterministic seed derived from the declared base seed. Frames record actual sampled counts and value fingerprints. These samples are never labelled as the full monthly populations. Public monthly TLC publication lag makes this a retrospective experiment, not evidence of real-time training.

## Reproduce

Use the six complete `yellow_tripdata_2023-01.parquet` through `yellow_tripdata_2023-06.parquet` files from [TLC's public trip records](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page).

```bash
pip install -r requirements-dev.txt
PYTHONPATH=src python -m tripeta.booking --data-dir /path/to/taxi \
  --train 2023-01..2023-03 --valid 2023-04 --calibration 2023-05 --test 2023-06 \
  --train-sample 100000 --evaluation-sample 200000 --seed 42 --rounds 600 --threads 4 \
  --output reports/booking-2023-h1 --registry models/booking
```

Choose a new output directory for each run; previous evidence cannot be overwritten. The report includes source-file SHA-256 and counts, phase-frame fingerprints, Python/package versions, code hashes, model content SHA-256, model parameters, interval ratios, aggregate metrics and route/hour/passenger slices. Compressed CSVs preserve every validation, calibration and test prediction with its source-row identity. Validation interval columns use the later calibration fit and are diagnostic outputs, never selection inputs.

The training-only baseline uses a pickup/drop-off route median with at least 20 training samples, then hourly median, then the global training median. Unseen and under-supported routes are explicitly reported as a test slice.

## Publication and serving

The complete evidence set is staged, hashed and made visible by a directory rename. Failed generation does not expose a partial new report or alter earlier reports. A passed validation gate can then update a **separate** booking registry using the existing atomic bundle writer. Report publication and registry update are separate local boundaries; a registry write failure leaves the published evidence inspectable and the prior registry intact. This assumes one writer and a local filesystem, not distributed consensus or power-loss durability.

The old `models/current/model.txt` and metadata remain unchanged. To serve the new model explicitly:

```bash
TRIPETA_MODEL_DIR=models/booking PYTHONPATH=src uvicorn tripeta.api:app --port 8000
curl localhost:8000/predict -H 'content-type: application/json' \
  -d '{"pickup_zone":161,"dropoff_zone":237,"pickup_time":"2023-06-14T18:30:00","passengers":1}'
```

The loaded booster determines feature selection. Booking predictions are invariant to supplied recorded distance; legacy models still require that field. Drift profiles contain only the loaded model's features. `/model` exposes the feature contract, phase roles and hashes; `/ready` and predictions use the trained model's hash for version identity, with a fallback for legacy metadata.

Focused tests cover source-row identity, label availability, baseline support/fallback, changed-content fingerprints, strict phase order, final-test independence, actual booking-model distance invariance, both API contracts, drift selection and failed report publication. Fixture models are unit-test evidence, not public accuracy measurements.
