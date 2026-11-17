[![gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/c/gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/Mnkjv/gta6-twitch-forecast/rank.svg)](https://make.fun/c/gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/Mnkjv/gta6-twitch-forecast/reward.svg)](https://make.fun/c/gta6-twitch-ccu-forecast-01)

# GTA VI launch-day Twitch CCU forecaster

Forecasts how many people watch Grand Theft Auto VI live on Twitch in each
hour of its release day (2026-11-19 UTC), as five quantiles per hour, from
the first hour's print alone — for the
[make.fun competition](https://make.fun/c/gta6-twitch-ccu-forecast-01).

## Model

A big console launch unlocks at midnight local time, country by country, so
hour 1 of the UTC release day only sees New Zealand / Australia / the UK
awake. Viewership then rises as Europe and the US unlock and wake, peaks in
the late UTC hours, and fades. How steeply it rises depends on how big the
launch already is at hour 1: a huge hour-1 print means the launch is already
global and stays flat; a small one grows many-fold.

`main.ts` fits that relationship at run time from the case file's own
comparables (past big launches' full release-day hourly curves):

- per forecast hour h = 2..24, OLS of `log(viewers_h / viewers_1)` on
  `log(viewers_1)` across comparables with hour-1 print ≥ 10,000;
- the target's median log-ratio from its own hour-1 print, clamped to the
  comparables' observed size range (no extrapolation past the biggest launch
  seen — this matters for GTA VI);
- quantile spread from the regression residuals: per-hour residual SD,
  smoothed as `sigma(h) = c + d * sqrt(h - 1)`, quantiles
  `p1 * exp(median + z_tau * sigma_h)`.

Deterministic (comparables in name order, no randomness, no network), stdlib
only, runs in under a second per case.

## Run locally

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=<kit>/input MAKEFUN_OUTPUT_DIR=out node main.ts
```

## Validation

Leave-one-out cross-validation on the kit's 37 usable comparables
(practice-like launches, hour-1 print 15k–400k): mean pinball loss ≈ 0.27
vs ≈ 0.34 for the house persistence baseline on the same targets; median
≈ 0.23. The four practice launches are all big Sony/Rockstar
rolling-midnight launches, the most predictable family in the set.
