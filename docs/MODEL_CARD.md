# Model card: taxi trip duration

**What it predicts:** minutes from pickup to drop-off for a NYC yellow-taxi trip, given pickup zone, drop-off zone, pickup time, supplied distance and passengers. Historical training and evaluation use recorded taximeter distance.
**Model:** LightGBM gradient boosting on log duration, 900 trees, categorical zone ids. Fitted in 31 s on 3.2 million trips.
**Intended use:** a local trip-duration inference demonstration. Booking-time use would require a planned-route estimate; performance with that input has not been evaluated. Not for pricing, safety or legal decisions.

## Data
NYC TLC yellow-taxi records, January to October 2023, cleaned with this service's training-specific rules (duration 1 min to 4 h, distance 0.1 to 100 miles, valid zones, positive fare). 400,000 trips sampled per training month (Jan to Aug), September for early stopping, **October (3,350,420 trips) used for the recorded evaluation**.

## Recorded October 2023 evaluation

| | Mean absolute error | Median relative error |
|---|---|---|
| Baseline: distance divided by median speed for the hour | 9.33 min | 29.1% |
| **LightGBM model** | **3.52 min** | **16.4%** |

The recorded model error is 62% lower than the baseline. Average trip: 17.6 min. These stored metrics predate the current validation-only promotion rule; the earlier rule used test-month scores. The committed artifact has not been retrained or reevaluated under the revised policy, so these figures are not a fresh, untouched final-test result.

Error by trip length (mean absolute error, minutes): under 1 mile 1.95, 1 to 3 miles 2.93, 3 to 6 miles 4.59, 6 to 12 miles 5.64, over 12 miles 7.79.
Error by time of day: 00-05 2.47, 06-09 3.37, 10-15 4.04, 16-19 3.71, 20-23 2.96.

**Prediction range.** The 10th and 90th percentile of actual divided by predicted duration, measured on the validation month, give a range of 0.79x to 1.62x the prediction. In the recorded October evaluation it contains 83.7% of trips (target 80%). This is observed coverage for that dataset, not a guarantee for new requests.

## Limits
- Reported error is conditional on recorded taximeter distance. No route-planner integration or comparison using planned-route distances was evaluated, and a distance-feature ablation was not reported.
- Trained on 2023 data from one city. It knows nothing about road closures, weather or events.
- The upper tail is right-skewed (traffic jams): the error is largest for long trips and midday.
- No fairness analysis was done: zone ids are an input, so predictions differ by neighbourhood.

## Monitoring
`GET /drift` compares the last 5,000 requests with the training distribution using the Population Stability Index per feature (below 0.1 stable, 0.1 to 0.25 moderate, above 0.25 major). On real data: March, September and October 2023 all score below 0.011 (stable). A simulated airport-only stream scores 7.1 on pickup zone and 4.1 on distance (major).
Retraining uses a champion/challenger rule: current code promotes only when the challenger beats the champion on the validation month. The test month is reporting-only; the historical artifact above was produced before that policy change.
