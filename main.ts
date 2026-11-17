// gta6-twitch-ccu-forecast-01 entry (Node 22, standard library only).
//
// Rolling-midnight launch-day forecaster. A big console launch unlocks at
// midnight local time, country by country, so hour 1 of the UTC release day
// (00:00-01:00) only ever sees New Zealand / Australia / the UK awake: the
// day's curve is a rise from a small hour-1 print as Europe, then the US,
// unlock and wake, peaking in the late UTC hours (US daytime + EU evening)
// before fading. The shape of that rise depends on how big the launch
// already is at hour 1: a launch with a huge hour-1 print is already global
// and stays flat, while a small hour-1 print grows many-fold.
//
// The program fits that relationship at run time from the case file's own
// comparables (past big launches' full release-day hourly curves): for each
// forecast hour h = 2..24 it regresses log(viewers_h / viewers_1) on
// log(viewers_1) across the comparables (ordinary least squares), predicts
// the target's median log-ratio from its own hour-1 print, and takes the
// quantile spread from the regression residuals: the per-hour residual
// standard deviation, smoothed as sigma(h) = c + d * sqrt(h - 1). Quantiles
// are p1 * exp(median + z_tau * sigma_h). The hour-1 print is clamped to
// the comparables' observed range, so an unprecedentedly large launch (GTA
// VI) gets the flattest fitted curve rather than a wild extrapolation.
//
// Everything is deterministic: comparables are processed in name order, no
// randomness, no network, no clock. Output is {"hours": [{"hour": h, "q":
// [q05, q25, q50, q75, q95]}, ...]} for hours 2..24, one file per case.

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

/** The quantile levels the contest scores, in the order each row lists them. */
const LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95];
/** The standard normal's quantile at each level. */
const Z = [-1.6448536269514722, -0.6744897501960817, 0, 0.6744897501960817, 1.6448536269514722];
/** Smallest hour-1 print a comparable must have to train on (tried in order). */
const MIN_P1_STEPS = [10000, 5000, 1000, 100];
/** Minimum training launches before we fall back to a smaller threshold. */
const MIN_TRAIN = 8;
/** Floor / ceiling for the fitted log-spread, in natural-log units. */
const SIGMA_FLOOR = 0.05;
const SIGMA_CEIL = 1.5;
/** The house baseline's spread, used only if no comparables can train. */
const BASELINE_SIGMA = 0.15;

interface Comparable {
  name: string;
  day_start: string;
  hourly: Array<{ from: string; viewers: number }>;
}

interface CaseFile {
  levels: number[];
  known: Array<{ hour: number; viewers: number }>;
  forecast: Array<{ hour: number }>;
  comparables: Comparable[];
}

/** x rounded to 2 decimals (half up), like the house baseline. */
const round2 = (x: number): number => Math.round(x * 100) / 100;

/** Ordinary least squares of y on x; returns [intercept, slope]. */
const ols = (xs: number[], ys: number[]): [number, number] => {
  const n = xs.length;
  const mx = xs.reduce((s, x) => s + x, 0) / n;
  const my = ys.reduce((s, y) => s + y, 0) / n;
  let sxx = 0;
  let sxy = 0;
  for (let i = 0; i < n; i++) {
    sxx += (xs[i] - mx) * (xs[i] - mx);
    sxy += (xs[i] - mx) * (ys[i] - my);
  }
  const slope = sxx > 0 ? sxy / sxx : 0;
  return [my - slope * mx, slope];
};

/** Parse the release-day hourly curve (hours 1..24) of one comparable. */
const releaseDay = (c: Comparable): number[] | null => {
  const start = Date.parse(c.day_start);
  if (!Number.isFinite(start)) return null;
  const byHour = new Map<number, number>();
  for (const row of c.hourly ?? []) {
    const t = Date.parse(row.from);
    if (!Number.isFinite(t)) continue;
    const h = Math.floor((t - start) / 3_600_000) + 1;
    if (h >= 1 && h <= 24 && typeof row.viewers === 'number') byHour.set(h, row.viewers);
  }
  if (!byHour.has(1) || byHour.size < 23) return null;
  const p1 = byHour.get(1) as number;
  if (!(p1 > 0)) return null;
  const y: number[] = [];
  for (let h = 1; h <= 24; h++) y.push(Math.max(1, byHour.get(h) ?? 1));
  return y;
};

/** The house baseline's persistence forecast, used only as a fallback. */
const baselineQuantiles = (p1: number, h: number): number[] =>
  Z.map((z) => round2(Math.max(1, p1 * Math.exp(z * BASELINE_SIGMA * Math.sqrt(h - 1)))));

/** Forecast one case: 23 rows of five quantiles for hours 2..24. */
const forecastCase = (c: CaseFile): Array<{ hour: number; q: number[] }> => {
  const first = c.known.find((k) => k.hour === 1);
  if (first === undefined) throw new Error('case has no hour-1 print');
  const p1 = Math.max(1, first.viewers);

  // Training launches: past release days with a usable hour-1 print. The
  // GTA VI category's own pre-release history is not a launch day, so it
  // is left out. Comparables are sorted by name so the fit is stable.
  const curves: Array<{ p1: number; y: number[] }> = [];
  const sorted = [...(c.comparables ?? [])].sort((a, b) => (a.name < b.name ? -1 : a.name > b.name ? 1 : 0));
  for (const comp of sorted) {
    if (comp.name.includes('Grand Theft Auto VI')) continue;
    const y = releaseDay(comp);
    if (y !== null) curves.push({ p1: y[0], y });
  }
  // Use the largest hour-1 threshold that still leaves enough training
  // launches; with too few at every threshold, fall back to all curves.
  let train: Array<{ p1: number; y: number[] }> = curves;
  for (const minP1 of MIN_P1_STEPS) {
    const kept = curves.filter((t) => t.p1 >= minP1);
    if (kept.length >= MIN_TRAIN) {
      train = kept;
      break;
    }
  }
  if (train.length < MIN_TRAIN) {
    return c.forecast.map(({ hour }) => ({ hour, q: baselineQuantiles(p1, hour) }));
  }

  const logP1 = train.map((t) => Math.log(t.p1));
  const lo = Math.min(...logP1);
  const hi = Math.max(...logP1);
  // Clamp the target's size to the observed range: never extrapolate past
  // the biggest launch the comparables have seen.
  const lp = Math.min(Math.max(Math.log(p1), lo), hi);

  // Per-hour OLS of log-ratio on log hour-1 print, plus residual spread.
  const medians: number[] = [];
  const sigmas: number[] = [];
  for (let h = 2; h <= 24; h++) {
    const ratios = train.map((t) => Math.log(t.y[h - 1] / t.p1));
    const [a, b] = ols(logP1, ratios);
    medians.push(a + b * lp);
    const residuals = ratios.map((r, i) => r - (a + b * logP1[i]));
    const mean = residuals.reduce((s, r) => s + r, 0) / residuals.length;
    const variance = residuals.reduce((s, r) => s + (r - mean) * (r - mean), 0) / Math.max(1, residuals.length - 1);
    sigmas.push(Math.sqrt(variance));
  }
  // Smooth the spread over the day: sigma(h) = c + d * sqrt(h - 1).
  const sqrtH = Array.from({ length: 23 }, (_, i) => Math.sqrt(i + 1));
  const [sc, sd] = ols(sqrtH, sigmas);
  const sigmaAt = (h: number): number =>
    Math.min(SIGMA_CEIL, Math.max(SIGMA_FLOOR, sc + sd * Math.sqrt(h - 1)));

  return c.forecast.map(({ hour }) => {
    const median = medians[hour - 2];
    const sigma = sigmaAt(hour);
    const q = Z.map((z) => round2(Math.max(1, p1 * Math.exp(median + z * sigma))));
    // Enforce non-decreasing quantiles and the 1e9 cap after rounding.
    for (let j = 1; j < q.length; j++) q[j] = Math.max(q[j], q[j - 1]);
    return { hour, q: q.map((v) => Math.min(v, 1_000_000_000)) };
  });
};

mkdirSync(OUTPUT_DIR, { recursive: true });
// Every case file, in name order, so two runs write the same bytes.
const inputs = readdirSync(INPUT_DIR).filter((f) => f.endsWith('.in')).sort();
for (const file of inputs) {
  const id = file.slice(0, -'.in'.length);
  const c = JSON.parse(readFileSync(join(INPUT_DIR, file), 'utf8')) as CaseFile;
  if (c.levels.length !== LEVELS.length || c.levels.some((t, j) => t !== LEVELS[j])) {
    throw new Error(`${file} asks for levels ${JSON.stringify(c.levels)}, not ${JSON.stringify(LEVELS)}`);
  }
  const hours = forecastCase(c);
  writeFileSync(join(OUTPUT_DIR, `${id}.out`), `${JSON.stringify({ hours })}\n`);
}
