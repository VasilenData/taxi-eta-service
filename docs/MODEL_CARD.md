# Model card: taxi trip duration

**What it predicts:** minutes from pickup to drop-off for a NYC yellow-taxi trip, given pickup zone, drop-off zone, pickup time, planned route distance and passengers.
**Model:** LightGBM gradient boosting on log duration, 900 trees, categorical zone ids. Fitted in 31 s on 3.2 million trips.
**Intended use:** an ETA estimate shown before or at booking. Not for pricing, safety or legal decisions.

## Data
NYC TLC yellow-taxi records, January to October 2023, cleaned with the same validity rules as [nyc-taxi-data-platform](https://github.com/VasilenData/nyc-taxi-data-platform) (duration 1 min to 4 h, distance 0.1 to 100 miles, valid zones, positive fare). 400,000 trips sampled per training month (Jan to Aug), September for early stopping, **October (3,350,420 trips) held out as the test month**.

## Results on the held-out future month

| | Mean absolute error | Median relative error |
|---|---|---|
| Baseline: distance divided by median speed for the hour | 9.33 min | 29.1% |
| **LightGBM model** | **3.52 min** | **16.4%** |

The model's error is 62% lower than the baseline. Average trip: 17.6 min.

Error by trip length (mean absolute error, minutes): under 1 mile 1.95, 1 to 3 miles 2.93, 3 to 6 miles 4.59, 6 to 12 miles 5.64, over 12 miles 7.79.
Error by time of day: 00-05 2.47, 06-09 3.37, 10-15 4.04, 16-19 3.71, 20-23 2.96.

**Prediction range.** The 10th and 90th percentile of actual divided by predicted duration, measured on the validation month, give a range of 0.79x to 1.62x the prediction. On the unseen test month it contains 83.7% of trips (target 80%), so it is slightly conservative.

## Limits
- Distance is an input, so the model needs a route planner upstream. Without distance, accuracy falls sharply.
- Trained on 2023 data from one city. It knows nothing about road closures, weather or events.
- The upper tail is right-skewed (traffic jams): the error is largest for long trips and midday.
- No fairness analysis was done: zone ids are an input, so predictions differ by neighbourhood.

## Monitoring
`GET /drift` compares the last 5,000 requests with the training distribution using the Population Stability Index per feature (below 0.1 stable, 0.1 to 0.25 moderate, above 0.25 major). On real data: March, September and October 2023 all score below 0.011 (stable). A simulated airport-only stream scores 7.1 on pickup zone and 4.1 on distance (major).
Retraining uses a champion/challenger rule: a new model replaces the registry only if it beats the current model on the same test month.
